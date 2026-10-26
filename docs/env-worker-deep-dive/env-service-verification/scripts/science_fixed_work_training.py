"""Compare the historical commit and Router using an explicit science completion target."""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pandas as pd
import psutil
import ray
import yaml
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_gpu_devices,
    read_kernel_driver_version,
    resolve_driver_libs,
)
from psrl.workers.env_service.types import EnvDemand, EnvLease


def validate_reward_config():
    """Catch shortened-response configuration errors before creating GPU actors."""
    from omegaconf import OmegaConf
    from psrl.workers.reward.reward_loop.dapo import DAPORewardManager

    return DAPORewardManager(
        OmegaConf.create({}),
        None,
        None,
        max_resp_len=512,
        overlong_buffer_cfg=OmegaConf.create({"enable": False, "len": 0, "log": False, "penalty_factor": 1.0}),
    )


def prepare(args, out):
    """Prepare the explicitly selected native or CUDA pressure workload."""
    import shutil

    if args.workload == "native":
        from science_native_workload import prepare_native

        return prepare_native(args, out)

    frame = pd.read_parquet(args.dataset)
    names = [
        "sciaccel/laps-repair-coef-2d-rktmod-l17k1",
        "sciaccel/laps-repair-subtle-2d-rktmod-l17k1",
        "sciaccel/laps-repair-coef-2d-rktmod-l18k4",
        "sciaccel/laps-repair-subtle-2d-rktmod-l18k3",
    ]
    selected = frame.set_index("task_name").loc[names].reset_index()
    for i, row in selected.iterrows():
        info = dict(row.extra_info)
        task = out / "tasks" / Path(info["task_path"]).name
        if not task.exists():
            shutil.copytree(info["task_path"], task)
            cfg = (task / "task.toml").read_text().replace("[environment]", "[environment]\ngpus = 1")
            (task / "task.toml").write_text(cfg)
            shutil.copyfile(Path(__file__).with_name("cuda_pressure.py"), task / "environment/cuda_pressure.py")
            with (task / "environment/Dockerfile").open("a") as f:
                f.write("\nCOPY cuda_pressure.py /opt/verification/cuda_pressure.py\n")
        info.update(task_path=str(task), gpus=1, gpu_memory_mb=(args.pressure_gib + 1) * 1024)
        selected.at[i, "extra_info"] = info
        selected.at[i, "agent_name"] = f"science-pool-{i}"
    repeated = pd.concat([selected] * args.steps, ignore_index=True)
    for split in ("train", "eval"):
        directory = out / "data" / split
        directory.mkdir(parents=True, exist_ok=True)
        repeated.to_parquet(directory / "L1.parquet", index=False)
    devices = read_gpu_devices()
    lease = EnvLease(
        "overlay-only",
        "overlay-only",
        "pressure",
        args.node_ip,
        (args.environment_gpus[0],),
        EnvDemand(needs_gpu=True, gpu_memory_mb=(args.pressure_gib + 1) * 1024),
        0,
        0,
    )
    overlay = build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version()))
    overlay["services"]["main"].pop("labels", None)
    (out / "gpu-overlay.yaml").write_text(yaml.safe_dump(overlay))
    (out / "design.json").write_text(
        json.dumps(
            {
                "scope": (
                    f"Original LAPS repair with identical {args.pressure_gib} GiB / "
                    f"{args.pressure_hold} second CUDA pressure injection. "
                    "Not native GPU science workload."
                ),
                "task_names": names,
                "steps_per_arm": args.steps,
                "batch_size": 4,
                "rollout_n": args.rollout_n,
                "task_pools": 4,
                "accepted_rollouts": 4 * args.rollout_n * args.steps,
                "max_attempts_per_slot": 12,
                "common_active_attempt_limit": 4,
                "model_gpus": [0, 1, 2],
                "environment_gpus": args.environment_gpus,
                "unit_budget_mib": 81920,
                "task_demand_mib": (args.pressure_gib + 1) * 1024,
                "max_containers_per_unit": 4,
                "timeout_per_arm_s": args.timeout,
                "initialization_timeout_s": args.initialization_timeout,
                "work_timeout_s": args.work_timeout,
                "max_turns": args.turns,
                "modes": args.modes,
                "command_mode": (
                    "Persistent host leases, explicit foreground GPU reservations, device cgroup fencing"
                ),
                "task_timeout_s": args.task_timeout,
                "comparison": "0eeead8 + documented runtime fixes vs current Router; not a pure toggle ablation",
                "data_shuffle": False,
                "device_mapping": {str(i): dataclasses.asdict(devices[i]) for i in args.environment_gpus},
                "stopping_rule": (
                    f"{4 * args.rollout_n * args.steps} non-OOM graded rollouts, "
                    f"{args.steps} GRPO steps, or the recorded phase time budgets. "
                    "Failures are not completion."
                ),
                "baseline_repo": str(args.baseline_repo),
                "current_repo": str(args.repo),
                "baseline_gpu_mapping": (
                    f"Static mapping to GPU {args.environment_gpus[0]}, host libraries, no added lock"
                ),
            },
            indent=2,
        )
    )
    return ",".join(devices[i].path for i in args.environment_gpus)


def build_agent_configs(args, out, mode):
    """Validate the worker schema before allocating model or Ray resources."""
    from examples.sciaccel_rl.config import build_runtime_config

    arm = out / mode
    agent = [
        {
            "name": f"science-pool-{i}",
            "_target_": "science_fixed_work_agent.FixedWorkAgentLoop",
            "harbor": {
                "jobs_dir": str(arm / f"pool-{i}" / "harbor-jobs"),
                "agent_name": "science_fixed_work_agent:"
                + ("CommandPressureAgent" if mode == "command" else "FixedWorkPressureAgent"),
                "override_gpus": 0,
                "gpu_compose_override": str(out / "gpu-overlay.yaml"),
                "max_concurrent_episodes": 4,
                "max_observation_bytes": 4096,
                "memory_mb_override": 0,
            },
            "task_timeout_sec": args.task_timeout,
            "verifier_timeout_sec": 900,
        }
        for i in range(4)
    ]
    if args.workload == "native":
        for config in agent:
            config["_target_"] = "science_native_workload.NativeWorkAgentLoop"
            config["harbor"]["agent_name"] = "examples.sciaccel_rl.agent:TruncatingTerminus2"
            config["harbor"]["gpu_compose_override"] = ""
    for config in agent:
        build_runtime_config(config)
    return agent


def run_arm(args, out, mode, device):
    arm = out / mode
    arm.mkdir()
    evidence = {"mode": mode, "started_at": time.time(), "budget_seconds": args.timeout}
    token = uuid.uuid4().hex[:8]
    child = None
    owned = {}
    try:
        agent = build_agent_configs(args, out, mode)
        baseline = sample_gpu()
        evidence["gpu_baseline"] = baseline
        selected = {0, 1, 2, *args.environment_gpus}
        require_idle_gpus(baseline, selected)
        info = ray.init(
            address="local",
            num_cpus=32,
            num_gpus=3,
            include_dashboard=False,
            _node_ip_address=args.node_ip,
            _temp_dir=f"/tmp/ab-{token}",
            namespace=f"ab-{token}",
            log_to_driver=False,
        )
        evidence["ray_address"] = info.address_info["gcs_address"]
        evidence["ray_tmp"] = f"/tmp/ab-{token}"

        @ray.remote
        def sources():
            import psrl
            import transfer_queue
            import verl

            return {x.__name__: x.__file__ for x in (psrl, verl, transfer_queue)}

        evidence["worker_imports"] = ray.get(sources.remote(), timeout=120)
        evidence["ray_nodes"] = [
            {key: node[key] for key in ("NodeID", "NodeManagerAddress", "Alive", "Resources")} for node in ray.nodes()
        ]
        (arm / "agent.yaml").write_text(yaml.safe_dump(agent))
        selected_repo = args.baseline_repo if mode == "off" else args.repo
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            str(selected_repo) + ":" + ":".join(p for p in env["PYTHONPATH"].split(":") if p != str(args.repo))
        )
        env.update(
            PSRL_PATH=str(selected_repo),
            HF_MODEL_PATH=str(args.model),
            OUTPUT_DIR=str(arm),
            DATA_DIR=str(out / "data"),
            HINT_LEVEL="L1",
            TRAIN_BATCH_SIZE="4",
            TOTAL_TRAINING_STEPS=str(args.steps),
            MAX_RESPONSE_LENGTH=str(args.response_length),
            MAX_TOKENS_PER_GPU=str(args.response_length + 2048),
            MAX_TURNS=str(args.turns),
            TRAIN_SP="1",
            TRAIN_FSDP="2",
            STALENESS="0",
            AGENT_NODE_IPS=args.node_ip,
            AGENT_LOOP_WORKERS="1",
            WANDB_MODE="disabled",
            SAVE_FREQ="-1",
            TEST_FREQ="-1",
            SCIACCEL_TMPDIR="/tmp",
            RAY_TMPDIR="/tmp",
            LOCAL_IP=args.node_ip,
            VLLM_HOST_IP=args.node_ip,
            RAY_ADDRESS=evidence["ray_address"],
            thinking_template="disable_thinking",
            OVERLONG_FILTERING="False",
            LR_WARMUP_STEPS="0",
            DATA_SEED="17",
            PSRL_SCIACCEL_GPU_DEVICES=device,
            SCIENCE_PRESSURE_GIB=str(args.pressure_gib),
            SCIENCE_PRESSURE_HOLD=str(args.pressure_hold),
            SCIENCE_PRESSURE_AUDIT_DIR=str(out / "pressure-audit"),
        )
        overrides = [
            f"psrl.rollout_n={args.rollout_n}",
            "gen_actor_rollout_ref.rollout.agent.retry_limit=1",
            "gen_actor_rollout_ref.rollout.agent.raise_on_error=True",
            "psrl.agentic_rl.refill_failure_threshold=1",
            "psrl.staleness_buffer_entries=4",
            "psrl.deployment.n_rollout_instances=1",
            "psrl.deployment.rollout_nnodes_per_instance=1",
            "psrl.deployment.rollout_ngpus_per_node_per_instance=1",
            "psrl.deployment.train_nnodes=1",
            "psrl.deployment.train_ngpus_per_node=2",
            "psrl.deployment.total_nnodes=1",
            "psrl.deployment.n_validate_instances=0",
            "psrl.colocate_validate_and_train=False",
            f"psrl.env_service.enable={mode != 'off'}",
            "psrl.env_service.node_policy=any",
            f"psrl.env_service.default_gpu_memory_mb={(args.pressure_gib + 1) * 1024}",
            "data.shuffle=False",
            "psrl.env_service.router_name=science_training_ab_router",
            (
                "psrl.env_service.units=["
                + ",".join(
                    (
                        f"{{unit_id:pressure-{i},node_ip:{args.node_ip},cpu_quota:16,memory_mb:32768,"
                        f"gpu_indices:[{i}],gpu_memory_mb:81920,max_containers:4,"
                        f"gpu_reservation:{'command' if mode == 'command' else 'episode'}}}"
                    )
                    for i in args.environment_gpus
                )
                + "]"
            ),
            f"gen_actor_rollout_ref.rollout.n={args.rollout_n}",
            "gen_actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.55",
            "gen_actor_rollout_ref.rollout.enforce_eager=True",
            "++gen_actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True",
            "++gen_actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",
            f"gen_actor_rollout_ref.rollout.agent.agent_loop_config_path={arm / 'agent.yaml'}",
            f"train_actor_rollout_ref.actor.rollout_n={args.rollout_n}",
            f"train_actor_rollout_ref.rollout.n={args.rollout_n}",
            "train_actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "train_actor_rollout_ref.actor.optim.lr_warmup_steps=0",
            "trainer.nnodes=1",
            "trainer.n_gpus_per_node=3",
            "trainer.logger=[console]",
            "trainer.resume_mode=disable",
            f"trainer.total_training_steps={args.steps}",
            "trainer.total_epochs=1",
            "reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=0",
            f"++ray_kwargs.ray_init.address={evidence['ray_address']}",
            f"++ray_kwargs.ray_init.runtime_env.env_vars.RAY_ADDRESS={evidence['ray_address']}",
            f"++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={env['PYTHONPATH']}",
            '++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE="1"',
            f"++ray_kwargs.ray_init.runtime_env.env_vars.PSRL_SCIACCEL_GPU_DEVICES='{device}'",
            f'++ray_kwargs.ray_init.runtime_env.env_vars.SCIENCE_PRESSURE_GIB="{args.pressure_gib}"',
            f'++ray_kwargs.ray_init.runtime_env.env_vars.SCIENCE_PRESSURE_HOLD="{args.pressure_hold}"',
            f"++ray_kwargs.ray_init.runtime_env.env_vars.SCIENCE_PRESSURE_AUDIT_DIR={out / 'pressure-audit'}",
            "++ray_kwargs.ray_init.include_dashboard=False",
        ]
        if args.workload == "native":
            overrides = [
                x
                for x in overrides
                if not x.startswith("psrl.env_service.units=")
                and ".SCIENCE_PRESSURE_" not in x
                and ".PSRL_SCIACCEL_GPU_DEVICES=" not in x
            ]
            overrides.append(
                "psrl.env_service.units=["
                f"{{unit_id:native-cpu,node_ip:{args.node_ip},cpu_quota:16,memory_mb:32768,"
                "gpu_indices:[],gpu_memory_mb:0,max_containers:4}]"
            )
            for name in list(env):
                if name.startswith("SCIENCE_PRESSURE_") or name == "PSRL_SCIACCEL_GPU_DEVICES":
                    env.pop(name)
        if mode == "off":
            overrides = [x for x in overrides if not x.startswith("psrl.env_service.")]
        evidence["selected_repo"] = str(selected_repo)
        command = ["bash", "examples/sciaccel_rl/fsdp_qwen35_4b.sh", *overrides]
        evidence["command"] = command
        evidence["environment"] = {
            k: env[k]
            for k in (
                "PYTHONPATH",
                "HF_MODEL_PATH",
                "MAX_RESPONSE_LENGTH",
                "MAX_TURNS",
                "TRAIN_BATCH_SIZE",
                "thinking_template",
            )
        }
        (arm / "launch.json").write_text(json.dumps(evidence, indent=2))
        with (arm / "launcher.log").open("w") as log, (arm / "telemetry.jsonl").open("w") as telemetry:
            child = subprocess.Popen(
                command, cwd=selected_repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + args.timeout
            evidence["child_started_at"] = time.time()
            phase = "initialization"
            if args.initialization_timeout:
                deadline = time.monotonic() + args.initialization_timeout
            router = None
            while child.poll() is None and time.monotonic() < deadline:
                if phase == "initialization":
                    markers = list(arm.glob("pool-*/slot-starts/*.json"))
                    if markers:
                        first = min(json.loads(p.read_text())["started_at"] for p in markers)
                        evidence["work_started_at"] = first
                        phase = "training"
                        if args.work_timeout:
                            deadline = time.monotonic() + max(0, args.work_timeout - (time.time() - first))
                        (arm / "launch.json").write_text(json.dumps(evidence, indent=2))
                for process in psutil.Process().children(recursive=True):
                    try:
                        owned[process.pid] = process.create_time()
                    except psutil.NoSuchProcess:
                        pass
                sample = {"time": time.time()}
                sample.update(sample_gpu())
                if mode != "off":
                    try:
                        if router is None:
                            named = ray.util.list_named_actors(all_namespaces=True)
                            found = next((x for x in named if x["name"] == "science_training_ab_router"), None)
                            if found:
                                router = ray.get_actor(found["name"], namespace=found["namespace"])
                        if router:
                            sample["units"] = [
                                dataclasses.asdict(x) for x in ray.get(router.snapshot.remote(), timeout=2)
                            ]
                    except Exception as exc:  # noqa: BLE001 (Preserve telemetry if the actor is unavailable.)
                        sample["router_error"] = type(exc).__name__
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                time.sleep(1)
            if child.poll() is None:
                evidence["timed_out"] = True
                evidence["timeout_phase"] = phase
                os.killpg(child.pid, 15)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, 9)
            evidence["exit_code"] = child.wait(timeout=10)
    except Exception as exc:
        evidence["harness_error"] = f"{type(exc).__name__}: {exc}"
        evidence["exit_code"] = 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, 15)
            child.wait(timeout=20)
        ray.shutdown()
        leftovers = []
        for pid, created in owned.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
                    leftovers.append(process)
                    process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(leftovers, timeout=5)
        for process in alive:
            process.kill()
        psutil.wait_procs(alive, timeout=5)
        evidence["reclaimed_descendant_pids"] = [p.pid for p in leftovers]
        evidence["container_cleanup"] = cleanup_containers(out)
        if owned:
            selected = {0, 1, 2, *args.environment_gpus}
            evidence["gpu_release"] = wait_for_gpu_release(selected)
        evidence["finished_at"] = time.time()
        (arm / "launch.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"mode": mode, "exit_code": evidence.get("exit_code"), "out": str(arm)}), flush=True)
    return evidence.get("exit_code", 1)


def cleanup_containers(out):
    """Remove owned Compose siblings and networks identified by experiment bind mounts."""
    ids = subprocess.check_output(["docker", "ps", "-aq"], text=True, timeout=10).split()
    removed = []
    projects = set()
    observations = out / "containers.jsonl"
    if observations.exists():
        for line in observations.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            projects.update(c["labels"].get("com.docker.compose.project") for c in sample.get("containers", []))
    if ids:
        containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True, timeout=15))
        projects.update(
            (c["Config"].get("Labels") or {}).get("com.docker.compose.project")
            for c in containers
            if any(Path(m.get("Source", "/")).is_relative_to(out) for m in c.get("Mounts", []))
        )
        projects.discard(None)
        for container in containers:
            labels = container["Config"].get("Labels") or {}
            owned = any(Path(m.get("Source", "/")).is_relative_to(out) for m in container.get("Mounts", []))
            if owned or labels.get("com.docker.compose.project") in projects:
                subprocess.run(["docker", "rm", "-f", container["Id"]], check=True, timeout=20, capture_output=True)
                removed.append(container["Id"])
    networks = []
    for project in projects - {None}:
        found = subprocess.check_output(
            ["docker", "network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
            text=True,
            timeout=10,
        ).split()
        for network in found:
            subprocess.run(["docker", "network", "rm", network], check=True, timeout=15, capture_output=True)
            networks.append(network)
    return {"removed_ids": removed, "removed_networks": networks, "projects": sorted(projects - {None})}


def save_sources(args):
    """Keep exact source versions and runtime patches beside the measurements."""
    paths = [
        "examples/sciaccel_rl/runner.py",
        "examples/sciaccel_rl/agent_loop.py",
        "examples/sciaccel_rl/fsdp_qwen35_4b.sh",
        "psrl/trainer/main_ppo.py",
        "psrl/utils/common/http_utils.py",
    ]
    sources = {}
    for name, repo in (("off", args.baseline_repo), ("on", args.repo)):
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if name == "off" and head != "0eeead861ffafed6bf31c962d259fd847775dcf9":
            raise ValueError("OFF must use the user-specified historical commit.")
        diff = subprocess.check_output(["git", "-C", str(repo), "diff"], text=True)
        (args.output / f"{name}.patch").write_text(diff)
        sources[name] = {"head": head, "files": {}}
        for path in paths:
            content = (repo / path).read_bytes()
            sources[name]["files"][path] = hashlib.sha256(content).hexdigest()
    snapshot = args.output / "scripts-snapshot"
    snapshot.mkdir(exist_ok=True)
    for filename in (
        "science_fixed_work_training.py",
        "science_fixed_work_agent.py",
        "science_pressure_agent.py",
        "science_pressure_audit.py",
        "science_native_workload.py",
        "cuda_pressure.py",
    ):
        (snapshot / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2))


def sample_gpu():
    """Keep failed observations distinct from zero utilization and training failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode:
            return {"gpu_error": f"exit_code={result.returncode}"}
        return {"gpu": result.stdout.strip()}
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"gpu_error": type(exc).__name__}


def require_idle_gpus(sample, selected):
    """Reject occupied or missing devices before starting another expensive arm."""
    if "gpu_error" in sample:
        raise RuntimeError("GPU preflight did not return a usable observation.")
    readings = {int(line.split(",")[0]): float(line.split(",")[2]) for line in sample.get("gpu", "").splitlines()}
    if not selected or selected - readings.keys():
        raise RuntimeError("Selected GPUs are missing from the device inventory.")
    occupied = {index: readings[index] for index in selected if readings[index] > 64}
    if occupied:
        raise RuntimeError(f"Selected GPUs are already occupied in MiB: {occupied!r}.")


def wait_for_gpu_release(selected, timeout=30):
    """Bound teardown lag after owned processes exit without killing unrelated processes."""
    deadline = time.monotonic() + timeout
    observations = []
    while True:
        sample = sample_gpu()
        observations.append({"time": time.time(), **sample})
        try:
            require_idle_gpus(sample, selected)
            return {"released": True, "observations": observations}
        except RuntimeError as exc:
            if time.monotonic() >= deadline:
                return {"released": False, "error": str(exc), "observations": observations}
        time.sleep(min(1, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--environment-gpus", nargs="+", type=int, default=[6, 7])
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--pressure-gib", type=int, default=50)
    parser.add_argument("--pressure-hold", type=float, default=45)
    parser.add_argument("--response-length", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--initialization-timeout", type=int, default=0)
    parser.add_argument("--work-timeout", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--task-timeout", type=int, default=240)
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workload", choices=["cuda-pressure", "native"], default="cuda-pressure")
    parser.add_argument("--modes", nargs="+", default=["off", "on"])
    args = parser.parse_args()
    if args.workload == "native":
        if any(mode not in ("off", "on") for mode in args.modes):
            parser.error("Native CPU tasks use off/on modes without GPU command reservations.")
        args.environment_gpus = []
    if bool(args.initialization_timeout) != bool(args.work_timeout):
        parser.error("Set both phase timeouts or neither.")
    args.repo, args.model, args.output = args.repo.resolve(), args.model.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    save_sources(args)
    validate_reward_config()
    device = prepare(args, args.output)
    if args.prepare_only:
        raise SystemExit(0)
    codes = []
    for mode in args.modes:
        codes.append(run_arm(args, args.output, mode, device))
        if codes[-1]:
            break
        launch = json.loads((args.output / mode / "launch.json").read_text())
        if launch.get("gpu_release", {}).get("released") is False:
            raise RuntimeError("Physical GPU release failed. Do not start the next arm.")
    raise SystemExit(int(any(codes)))
