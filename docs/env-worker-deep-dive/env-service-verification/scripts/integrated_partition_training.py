"""Compare fixed and resident-sharing pools through full PSRL training and SMG."""

import argparse
import dataclasses
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import pandas as pd
import psutil
import ray
import yaml
from psrl.workers.env_service.overlay import build_lease_overlay, read_kernel_driver_version, resolve_driver_libs
from psrl.workers.env_service.probe import NvmlProbe
from psrl.workers.env_service.types import EnvDemand, EnvLease


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def runtime_sources():
    """Report actual imports inside a worker using the experiment runtime environment."""
    import psrl
    import transfer_queue
    import transformers
    import verl

    return {
        "psrl": psrl.__file__,
        "verl": verl.__file__,
        "transfer_queue": transfer_queue.__file__,
        "transformers": transformers.__file__,
    }


def gpu_layout(rollout_replicas):
    """Partition six physical GPUs while preserving the dedicated first-fit order."""
    if rollout_replicas not in (1, 2, 3):
        raise ValueError("One to three rollout replicas are supported in the six GPU experiment.")
    return {
        "training": [0, 1],
        "rollout": list(range(2, 2 + rollout_replicas)),
        "dedicated_environment": [gpu for gpu in (4, 5, 3) if gpu >= 2 + rollout_replicas],
    }


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    for name in ("psrl", "examples"):
        excluded = ("__pycache__", "outputs")
        shutil.copytree(args.repo / name, source / name, ignore=shutil.ignore_patterns(*excluded))
    scripts = Path(__file__).parent
    for name in ("integrated_partition_agent.py", "partition_worker.py", "integrated_partition_training.py"):
        shutil.copy2(scripts / name, source / name)
    shutil.copy2(args.repo / "examples/posttrain_bench/local_model.py", source / "task.py")
    shutil.copytree(args.repo / ".verification-deps/bootstrap", source / "bootstrap")
    shutil.copytree(
        args.repo / ".verification-deps/verl",
        source / "dependencies/verl",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        args.repo / ".verification-deps/psrl_state_grpc_proto",
        source / "dependencies/psrl_state_grpc_proto",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    rows = [
        {
            "data_source": "sciaccel_rl",
            "agent_name": "integrated_posttrain",
            "prompt": [{"role": "user", "content": "Adapt a language model and evaluate held-out likelihood."}],
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "updates": 896 if i % args.groups == args.groups - 1 else 448,
                "task_name": f"posttrain-{i}",
                "reward_key": "reward",
            },
        }
        for i in range((args.steps + 1) * args.groups)
    ]
    pd.DataFrame(rows).to_parquet(out / "tasks.parquet", index=False)
    save(
        out / "design.json",
        {
            "args": vars(args),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "comparison": "Same current source, fixed first-fit pool versus resident rollout spillover.",
            "steps_per_arm": args.steps,
            "training_episodes_per_arm": args.steps * args.groups * 2,
            "prefetch": "Staleness one allows at most one additional batch, reported separately.",
            "gpu_budget": gpu_layout(args.rollout_replicas),
            "command_limit_per_gpu": args.dedicated_cap,
            "shared_command_limit_per_gpu": args.shared_cap,
            "capacity_selection": json.loads(args.capacity_selection.read_text()) if args.capacity_selection else None,
            "command_memory_mib": 16384,
            "model_headroom_mib": 8192,
            "workload": "Bounded PostTrain adapter task with SMG learning rate selection and private NLL reward.",
            "limits": {
                "initialization_s": 1200,
                "arm_s": args.arm_timeout,
                "command_s": 240,
                "shared_command_s": 270,
                "episode_s": 720,
                "attempts_per_arm": 1,
            },
            "dependency_fix": "Isolated position_ids patch and tested package metadata optimization in both arms.",
        },
    )
    (out / "source.diff").write_text(subprocess.check_output(["git", "diff", "--", "psrl"], text=True))
    return out


def preflight(args, out):
    """Verify the current container CUDA path before allocating model workers."""
    from psrl.workers.env_service.probe import read_gpu_devices

    lease = EnvLease("probe", "probe", "probe", args.node_ip, (4,), EnvDemand(2, 16384, True, 16384), 0, 0)
    main = build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version()))["services"]["main"]
    name = "integrated-probe-" + uuid.uuid4().hex[:8]
    command = ["docker", "run", "--rm", "--name", name]
    for key, flag in (("devices", "--device"), ("volumes", "--volume"), ("environment", "--env")):
        for value in main[key]:
            command.extend([flag, value])
    for prefix in dict.fromkeys((Path(sys.executable).parent.parent, Path(sys.executable).resolve().parent.parent)):
        command.extend(["--volume", f"{prefix}:{prefix}:ro"])
    command.extend(
        [args.image, sys.executable, "-c", "import torch; print((torch.arange(32, device='cuda') + 1).sum().item())"]
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        save(
            out / "container-preflight.json",
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "devices": {key: dataclasses.asdict(value) for key, value in read_gpu_devices().items()},
            },
        )
        if result.returncode or "528" not in result.stdout:
            raise RuntimeError("Container CUDA preflight failed.")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)


def run_arm(args, out, mode):
    arm = out / mode
    arm.mkdir()
    source = out / "source"
    tag = f"integrated-{uuid.uuid4().hex[:8]}"
    probe = NvmlProbe()
    record = {"mode": mode, "tag": tag, "started_at": time.time(), "status": "initializing"}
    startup_clock = time.monotonic()

    def mark_startup(name):
        """Persist startup boundaries so infrastructure stalls remain attributable."""
        record.setdefault("startup_events", []).append(
            {"stage": name, "at": time.time(), "elapsed_s": time.monotonic() - startup_clock}
        )
        save(arm / "status.json", record)

    child = None
    owned = {}
    router = None
    try:
        mark_startup("resource_preflight_started")
        before = probe.read_gpu_snapshot(range(6))
        record["before"] = before
        if any(value["pids"] or value["used_mb"] > 64 for value in before.values()):
            raise RuntimeError("Experiment devices are occupied before launch.")
        subprocess.run(["docker", "info"], capture_output=True, timeout=8, check=True)
        runtime_paths = [source / "bootstrap", source / "dependencies", source]
        local_site = getattr(args, "local_python_site", None)
        if local_site:
            runtime_paths.insert(1, Path(local_site))
        expected = {
            "psrl": source / "psrl",
            "verl": source / "dependencies/verl",
            "transfer_queue": Path(importlib.util.find_spec("transfer_queue").origin).parent,
        }
        if local_site:
            expected["transformers"] = Path(local_site) / "transformers"
        driver_imports = {name: importlib.util.find_spec(name).origin for name in expected}
        if any(not Path(driver_imports[name]).is_relative_to(root) for name, root in expected.items()):
            raise RuntimeError(f"Driver module paths do not match the frozen launch: {driver_imports!r}.")
        record["driver_module_origins"] = driver_imports
        mark_startup("driver_module_paths_checked")
        mark_startup("ray_init_started")
        register_timeout = getattr(args, "ray_worker_register_timeout_s", 90)
        record["ray_worker_register_timeout_s"] = register_timeout
        info = ray.init(
            address="local",
            num_cpus=48,
            num_gpus=2 + args.rollout_replicas,
            include_dashboard=False,
            _node_ip_address=args.node_ip,
            _temp_dir=f"/tmp/{tag}",
            namespace=tag,
            log_to_driver=False,
            object_store_memory=2 * 1024**3,
            _system_config={"worker_register_timeout_seconds": register_timeout},
        )
        address = info.address_info["gcs_address"]
        record.update(ray_address=address, ray_tmp=f"/tmp/{tag}")
        mark_startup("ray_init_finished")
        record["ray_nodes"] = [
            {key: node[key] for key in ("NodeID", "NodeManagerAddress", "Resources")}
            for node in ray.nodes()
            if node["Alive"]
        ]
        if [node["NodeManagerAddress"] for node in record["ray_nodes"]] != [args.node_ip]:
            raise RuntimeError("Unexpected Ray node identity.")
        settings = dict(
            output=str(arm),
            source=str(source),
            env_model=str(args.env_model),
            data=str(args.data),
            bootstrap=str(source / "bootstrap"),
            tag=tag,
            image=args.image,
            router_name=tag,
        )
        save(arm / "settings.json", settings)
        (arm / "agent.yaml").write_text(
            yaml.safe_dump(
                [{"name": "integrated_posttrain", "_target_": "integrated_partition_agent.IntegratedPostTrainAgent"}]
            )
        )
        env = dict(os.environ)
        env.update(
            PSRL_WORKSPACE=str(source),
            PSRL_PATH=str(source),
            HF_MODEL_PATH=str(args.model),
            OUTPUT_DIR=str(arm),
            TRAIN_FILES=str(out / "tasks.parquet"),
            VAL_FILES=str(out / "tasks.parquet"),
            DATA_DIR=str(out),
            TRAIN_BATCH_SIZE=str(args.groups),
            TOTAL_TRAINING_STEPS=str(args.steps),
            MAX_PROMPT_LENGTH="512",
            MAX_RESPONSE_LENGTH="2048",
            MAX_TOKENS_PER_GPU="2560",
            MAX_TURNS="3",
            TRAIN_SP="1",
            TRAIN_FSDP="2",
            STALENESS="1",
            AGENT_NODE_IPS=args.node_ip,
            AGENT_LOOP_WORKERS="1",
            WANDB_MODE="disabled",
            SAVE_FREQ="-1",
            TEST_FREQ="-1",
            SCIACCEL_TMPDIR="/tmp",
            RAY_TMPDIR="/tmp",
            LOCAL_IP=args.node_ip,
            VLLM_HOST_IP=args.node_ip,
            RAY_ADDRESS=address,
            thinking_template="disable_thinking",
            OVERLONG_FILTERING="False",
            LR_WARMUP_STEPS="0",
            DATA_SEED="17",
            PARTITION_SETTINGS=str(arm / "settings.json"),
            AGENT_LOOP_CONFIG_PATH=str(arm / "agent.yaml"),
            DEFAULT_AGENT_LOOP="integrated_posttrain",
            PYTHONPATH=os.pathsep.join(map(str, runtime_paths)),
            MAX_CONCURRENT_SEQS_PER_INSTANCE="16",
            SERVER_MAX_CONCURRENCY="64",
        )
        mark_startup("worker_imports_started")
        record["worker_imports"] = ray.get(
            ray.remote(num_cpus=1)(runtime_sources)
            .options(runtime_env={"env_vars": {"PYTHONPATH": env["PYTHONPATH"]}})
            .remote(),
            timeout=180,
        )
        if any(not Path(record["worker_imports"][name]).is_relative_to(root) for name, root in expected.items()):
            raise RuntimeError(f"Worker module paths do not match the frozen launch: {record['worker_imports']!r}.")
        save(arm / "worker-imports.json", record["worker_imports"])
        mark_startup("worker_imports_finished")
        units = []
        layout = gpu_layout(args.rollout_replicas)
        for gpu in layout["dedicated_environment"] + layout["rollout"]:
            unit = dict(
                unit_id=f"gpu-{gpu}",
                node_ip=args.node_ip,
                cpu_quota=32,
                memory_mb=262144,
                gpu_indices=[gpu],
                gpu_memory_mb=92160,
                max_containers=16,
                gpu_reservation="command",
                command_gpu_pool="environment",
                max_gpu_commands=args.dedicated_cap,
            )
            if gpu in layout["rollout"]:
                unit.update(
                    shared_rollout_replica_rank=gpu - 2,
                    model_headroom_mb=8192,
                    max_gpu_commands=args.shared_cap,
                    command_gpu_pool="environment" if mode == "resident" else "unused-model",
                )
            units.append(unit)
        overrides = [
            "psrl.rollout_n=2",
            "psrl.agentic_rl.refill_failure_threshold=1",
            f"psrl.staleness_buffer_entries={args.groups}",
            f"psrl.deployment.n_rollout_instances={args.rollout_replicas}",
            "psrl.deployment.rollout_nnodes_per_instance=1",
            "psrl.deployment.rollout_ngpus_per_node_per_instance=1",
            "psrl.deployment.train_nnodes=1",
            "psrl.deployment.train_ngpus_per_node=2",
            "psrl.deployment.total_nnodes=1",
            "psrl.deployment.n_validate_instances=0",
            "psrl.colocate_validate_and_train=False",
            "psrl.env_service.enable=True",
            "psrl.env_service.node_policy=any",
            f"psrl.env_service.router_name={tag}",
            "psrl.env_service.default_gpu_memory_mb=16384",
            "psrl.env_service.command_placement_policy=first_fit",
            "psrl.env_service.gpu_sharing.max_command_s=270",
            "psrl.env_service.gpu_sharing.reuse_sleeping=False",
            "psrl.env_service.heartbeat_interval_s=5",
            "psrl.env_service.units=" + yaml.safe_dump(units, default_flow_style=True, width=100000).strip(),
            "gen_actor_rollout_ref.rollout.n=2",
            "gen_actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.45",
            "gen_actor_rollout_ref.rollout.enforce_eager=True",
            "++gen_actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True",
            "++gen_actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",
            "gen_actor_rollout_ref.rollout.agent.retry_limit=1",
            "gen_actor_rollout_ref.rollout.agent.raise_on_error=True",
            "train_actor_rollout_ref.actor.rollout_n=2",
            "train_actor_rollout_ref.rollout.n=2",
            "train_actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "trainer.nnodes=1",
            f"trainer.n_gpus_per_node={2 + args.rollout_replicas}",
            "trainer.logger=[console]",
            "trainer.resume_mode=disable",
            f"trainer.total_training_steps={args.steps}",
            "trainer.total_epochs=1",
            "data.shuffle=False",
            "reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=0",
            f"++ray_kwargs.ray_init.address={address}",
            "++ray_kwargs.ray_init.include_dashboard=False",
        ]
        if getattr(args, "live_partition_protocol", False):
            overrides.extend(
                [
                    "psrl.env_service.gpu_sharing.sync_overlap=True",
                    "psrl.env_service.gpu_sharing.sync_headroom_mb=16384",
                    f"psrl.env_service.elastic_partition.enable={args.elastic_partition}",
                    f"psrl.env_service.elastic_partition.borrowed_max_gpu_commands={args.dedicated_cap}",
                ]
            )
        for key in ("PYTHONPATH", "RAY_ADDRESS", "PARTITION_SETTINGS", "LOCAL_IP", "VLLM_HOST_IP"):
            overrides.append(f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}={env[key]}")
        command = ["bash", "examples/sciaccel_rl/fsdp_qwen35_4b.sh", *overrides]
        with (arm / "config-preflight.log").open("w") as config_log:
            # Validate composition here and let the actual workers load their business dependencies.
            mark_startup("config_preflight_started")
            subprocess.run(
                command + ["--cfg", "job"],
                cwd=source,
                env=env,
                stdout=config_log,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=True,
            )
            mark_startup("config_preflight_finished")
        record.update(
            command=command,
            environment={
                key: env[key]
                for key in ("PYTHONPATH", "HF_MODEL_PATH", "TRAIN_FILES", "STALENESS", "PARTITION_SETTINGS")
            },
        )
        save(arm / "status.json", record)
        with (arm / "launcher.log").open("w") as log, (arm / "telemetry.jsonl").open("w") as telemetry:
            mark_startup("driver_launch_started")
            child = subprocess.Popen(
                command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record["child_pid"] = child.pid
            mark_startup("driver_launched")
            deadline = (
                time.monotonic() + min(record["started_at"] + args.arm_timeout, args.deadline_epoch) - time.time()
            )
            init_deadline = time.monotonic() + record["started_at"] + 1200 - time.time()
            while child.poll() is None:
                for process in psutil.Process().children(recursive=True):
                    try:
                        owned[process.pid] = process.create_time()
                    except psutil.NoSuchProcess:
                        pass
                episodes = list(arm.glob("episodes/*/episode.json"))
                if episodes and record["status"] == "initializing":
                    record.update(status="training", initialization_s=time.time() - record["started_at"])
                    mark_startup("first_episode_observed")
                if time.monotonic() >= deadline or (
                    record["status"] == "initializing" and time.monotonic() >= init_deadline
                ):
                    raise TimeoutError(f"Budget exhausted during {record['status']}.")
                sample = {"time": time.time(), "episodes_started": len(episodes)}
                try:
                    sample["gpus"] = probe.read_gpu_snapshot(range(6))
                    if router is None:
                        actors = ray.util.list_named_actors(all_namespaces=True)
                        found = next((actor for actor in actors if actor["name"] == tag), None)
                        if found:
                            router = ray.get_actor(tag, namespace=found["namespace"])
                    if router:
                        sample["units"] = [
                            dataclasses.asdict(unit) for unit in ray.get(router.snapshot.remote(), timeout=3)
                        ]
                        sample["models"] = ray.get(router.shared_model_status.remote(), timeout=3)
                        if getattr(args, "live_partition_protocol", False):
                            sample["partition"] = ray.get(router.partition_control_status.remote(), timeout=3)
                except Exception as exc:
                    sample["telemetry_error"] = repr(exc)
                telemetry.write(json.dumps(sample, default=str) + "\n")
                telemetry.flush()
                time.sleep(2)
            record.update(exit_code=child.returncode, status="exited")
    except BaseException as exc:
        record.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    finally:
        for process in psutil.Process().children(recursive=True):
            try:
                owned[process.pid] = process.create_time()
            except psutil.NoSuchProcess:
                pass
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        if router:
            try:
                ray.get(router.shutdown_nodes.remote(), timeout=30)
            except Exception as exc:
                record["router_cleanup_error"] = repr(exc)
        ray.shutdown()
        remaining = []
        for pid, created in owned.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
                    process.terminate()
                    remaining.append(process)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(remaining, timeout=5)
        for process in alive:
            process.kill()
        ids = subprocess.check_output(["docker", "ps", "-aq"], text=True, timeout=10).split()
        record["removed_containers"] = []
        if ids:
            containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True, timeout=15))
            projects = {
                (container["Config"].get("Labels") or {}).get("com.docker.compose.project")
                for container in containers
                if any(Path(mount.get("Source", "/")).is_relative_to(arm) for mount in container.get("Mounts", []))
            }
            projects.discard(None)
            for container in containers:
                labels = container["Config"].get("Labels") or {}
                if labels.get("com.docker.compose.project") in projects:
                    subprocess.run(
                        ["docker", "rm", "-f", container["Id"]], capture_output=True, timeout=15, check=True
                    )
                    record["removed_containers"].append(container["Id"])
        for _ in range(30):
            after = probe.read_gpu_snapshot(range(6))
            if all(not value["pids"] and value["used_mb"] < 64 for value in after.values()):
                record["physical_release_verified"] = True
                break
            time.sleep(2)
        record.update(after=after, finished_at=time.time())
        save(arm / "status.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "output", "model", "env-model", "data"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--arm-timeout", type=int, default=3000)
    parser.add_argument("--total-timeout", type=int, default=6600)
    parser.add_argument("--ray-worker-register-timeout-s", type=int, choices=(30, 90), default=90)
    parser.add_argument("--dedicated-cap", type=int, default=2)
    parser.add_argument("--shared-cap", type=int, default=2)
    parser.add_argument("--capacity-selection", type=Path)
    parser.add_argument("--rollout-replicas", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--mode", choices=("fixed", "resident", "both"), default="both")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.capacity_selection:
        selected = json.loads(args.capacity_selection.read_text())
        if selected["groups"] != args.groups:
            parser.error("Capacity selection must use the same batch size as this experiment.")
        args.dedicated_cap = selected["selected_static_cap"]
        args.shared_cap = selected["shared_cap"]
    if not 1 <= args.dedicated_cap <= 5 or not 1 <= args.shared_cap <= 2:
        parser.error("Capacities exceed the validated experiment reservation budgets.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(2 + args.rollout_replicas))
    out = args.output.resolve() if args.resume else prepare(args)
    source = out / "source"
    if Path(__file__).resolve().parent != source:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(map(str, (source / "bootstrap", source / "dependencies", source)))
        environment["PSRL_WORKSPACE"] = str(source)
        command = [sys.executable, str(source / "integrated_partition_training.py")]
        for key, value in {**vars(args), "resume": True}.items():
            if value is None or value is False:
                continue
            command.append("--" + key.replace("_", "-"))
            if value is not True:
                command.append(str(value.resolve() if isinstance(value, Path) else value))
        os.chdir(source)
        os.execve(sys.executable, command, environment)
    args.deadline_epoch = (out / "design.json").stat().st_mtime + args.total_timeout
    preflight(args, out)
    results = []
    for mode in ("fixed", "resident") if args.mode == "both" else (args.mode,):
        status = out / mode / "status.json"
        if status.exists():
            previous = json.loads(status.read_text())
            if previous.get("exit_code") == 0 and previous.get("physical_release_verified"):
                results.append(previous)
                continue
            raise RuntimeError("A failed arm requires a new named attempt with preserved evidence.")
        result = run_arm(args, out, mode)
        results.append(result)
        save(out / "results.json", results)
        if result.get("exit_code") != 0 or not result.get("physical_release_verified"):
            raise RuntimeError(f"Arm {mode!r} did not complete and release its devices.")


if __name__ == "__main__":
    main()
