"""Compare bounded real science training with the resource ledger off and on."""

import argparse
import dataclasses
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
        OmegaConf.create({}), None, None, max_resp_len=512,
        overlong_buffer_cfg=OmegaConf.create({"enable": False, "len": 0, "log": False, "penalty_factor": 1.0}),
    )


def prepare(args, out):
    """Select original science tasks without changing instructions or rewards."""
    frame = pd.read_parquet(args.dataset)
    names = ["sciaccel/laps-repair-coef-2d-rktmod-l17k1",
             "sciaccel/laps-repair-subtle-2d-rktmod-l17k1"]
    selected = frame[frame.task_name.isin(names)].copy()
    assert set(selected.task_name) == set(names), "Both original tasks must exist."
    for info in selected.extra_info:
        assert Path(info["task_path"]).joinpath("task.toml").exists()
    repeated = pd.concat([selected] * args.steps, ignore_index=True)
    for split in ("train", "eval"):
        directory = out / "data" / split
        directory.mkdir(parents=True, exist_ok=True)
        repeated.to_parquet(directory / "L1.parquet", index=False)
    (out / "design.json").write_text(json.dumps({
        "task_names": names, "unique_tasks": 2, "rows": len(repeated),
        "steps_per_arm": args.steps, "batch_size": 2, "rollout_n": 2,
        "planned_episodes_per_arm": 4 * args.steps, "timeout_per_arm_s": args.timeout,
        "task_timeout_s": args.task_timeout, "max_turns": args.turns,
        "model": str(args.model), "seed": 17,
        "scope": "Original CPU science repair containers with GPU model training and inference.",
        "only_treatment": "psrl.env_service.enable", "env_concurrency": 4,
        "training_gpus": 2, "rollout_gpus": 1,
        "overlong_filtering": False,
        "note": "Truncated trajectories remain trainable in both arms. No convergence claim."
    }, indent=2))
    return ""


def run_arm(args, out, mode, device):
    arm = out / mode
    arm.mkdir()
    evidence = {"mode": mode, "started_at": time.time(), "budget_seconds": args.timeout}
    token = uuid.uuid4().hex[:8]
    child = None
    owned = {}
    try:
        baseline = sample_gpu()
        evidence["gpu_baseline"] = baseline
        selected = {int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")}
        require_idle_gpus(baseline, selected)
        info = ray.init(address="local", num_cpus=32, num_gpus=3, include_dashboard=False,
                        _node_ip_address=args.node_ip, _temp_dir=f"/tmp/ab-{token}",
                        namespace=f"ab-{token}", log_to_driver=False)
        evidence["ray_address"] = info.address_info["gcs_address"]
        evidence["ray_tmp"] = f"/tmp/ab-{token}"
        @ray.remote
        def sources():
            import psrl, verl, transfer_queue
            return {x.__name__: x.__file__ for x in (psrl, verl, transfer_queue)}
        evidence["worker_imports"] = ray.get(sources.remote(), timeout=120)
        agent = [{"name": "sciaccel", "_target_": "examples.sciaccel_rl.agent_loop.SciAccelAgentLoop",
                  "harbor": {"jobs_dir": str(arm / "harbor-jobs"),
                             "agent_name": "examples.sciaccel_rl.agent:TruncatingTerminus2", "max_concurrent_episodes": 4, "max_observation_bytes": 4096,
                             "memory_mb_override": 0},
                  "task_timeout_sec": args.task_timeout, "verifier_timeout_sec": 900}]
        (arm / "agent.yaml").write_text(yaml.safe_dump(agent))
        env = dict(os.environ)
        env.update(PSRL_PATH=str(args.repo), HF_MODEL_PATH=str(args.model), OUTPUT_DIR=str(arm),
                   DATA_DIR=str(out / "data"), HINT_LEVEL="L1", TRAIN_BATCH_SIZE="2", TOTAL_TRAINING_STEPS=str(args.steps),
                   MAX_RESPONSE_LENGTH="32768", MAX_TOKENS_PER_GPU="34816", MAX_TURNS=str(args.turns), TRAIN_SP="1", TRAIN_FSDP="2",
                   STALENESS="0", AGENT_NODE_IPS=args.node_ip, AGENT_LOOP_WORKERS="1", WANDB_MODE="disabled",
                   SAVE_FREQ="-1", TEST_FREQ="-1", SCIACCEL_TMPDIR="/tmp", RAY_TMPDIR="/tmp",
                   LOCAL_IP=args.node_ip, VLLM_HOST_IP=args.node_ip, RAY_ADDRESS=evidence["ray_address"],
                   thinking_template="disable_thinking", OVERLONG_FILTERING="False",
                   LR_WARMUP_STEPS="0", DATA_SEED="17")
        overrides = [
            "psrl.rollout_n=2", "psrl.agentic_rl.refill_failure_threshold=1", "psrl.staleness_buffer_entries=2", "psrl.deployment.n_rollout_instances=1",
            "psrl.deployment.rollout_nnodes_per_instance=1", "psrl.deployment.rollout_ngpus_per_node_per_instance=1",
            "psrl.deployment.train_nnodes=1", "psrl.deployment.train_ngpus_per_node=2",
            "psrl.deployment.total_nnodes=1", "psrl.deployment.n_validate_instances=0",
            "psrl.colocate_validate_and_train=False", f"psrl.env_service.enable={mode == 'on'}",
            "psrl.env_service.node_policy=cpu_only",
            "psrl.env_service.router_name=science_training_ab_router",
            (f"psrl.env_service.units=[{{unit_id:science-cpu,node_ip:{args.node_ip},cpu_quota:16,memory_mb:32768,max_containers:4}}]"),
            "gen_actor_rollout_ref.rollout.n=2", "gen_actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.55", "gen_actor_rollout_ref.rollout.enforce_eager=True",
            "++gen_actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True",
            "++gen_actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",
            f"gen_actor_rollout_ref.rollout.agent.agent_loop_config_path={arm / 'agent.yaml'}",
            "train_actor_rollout_ref.actor.rollout_n=2", "train_actor_rollout_ref.rollout.n=2", "train_actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "train_actor_rollout_ref.actor.optim.lr_warmup_steps=0",
            "trainer.nnodes=1", "trainer.n_gpus_per_node=3", "trainer.logger=[console]",
            "trainer.resume_mode=disable", f"trainer.total_training_steps={args.steps}", "trainer.total_epochs=1",
            "reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=0",
            f"++ray_kwargs.ray_init.address={evidence['ray_address']}",
            f"++ray_kwargs.ray_init.runtime_env.env_vars.RAY_ADDRESS={evidence['ray_address']}",
            f"++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={env['PYTHONPATH']}",
            '++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE="1"',
            "++ray_kwargs.ray_init.include_dashboard=False",
        ]
        command = ["bash", "examples/sciaccel_rl/fsdp_qwen35_4b.sh", *overrides]
        evidence["command"] = command
        evidence["environment"] = {k: env[k] for k in ("PYTHONPATH", "HF_MODEL_PATH", "MAX_RESPONSE_LENGTH", "MAX_TURNS", "TRAIN_BATCH_SIZE", "thinking_template")}
        (arm / "launch.json").write_text(json.dumps(evidence, indent=2))
        with (arm / "launcher.log").open("w") as log, (arm / "telemetry.jsonl").open("w") as telemetry:
            child = subprocess.Popen(command, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            deadline = time.monotonic() + args.timeout
            router = None
            while child.poll() is None and time.monotonic() < deadline:
                for process in psutil.Process().children(recursive=True):
                    try:
                        owned[process.pid] = process.create_time()
                    except psutil.NoSuchProcess:
                        pass
                sample = {"time": time.time()}
                sample.update(sample_gpu())
                if mode == "on":
                    try:
                        if router is None:
                            named = ray.util.list_named_actors(all_namespaces=True)
                            found = next((x for x in named if x["name"] == "science_training_ab_router"), None)
                            if found:
                                router = ray.get_actor(found["name"], namespace=found["namespace"])
                        if router:
                            sample["units"] = [dataclasses.asdict(x) for x in ray.get(router.snapshot.remote(), timeout=2)]
                    except Exception as exc:  # noqa: BLE001 - Preserve telemetry if the actor is unavailable.
                        sample["router_error"] = type(exc).__name__
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                time.sleep(1)
            if child.poll() is None:
                evidence["timed_out"] = True
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
        if owned:
            selected = {int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")}
            evidence["gpu_release"] = wait_for_gpu_release(selected)
        evidence["finished_at"] = time.time()
        (arm / "launch.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"mode": mode, "exit_code": evidence.get("exit_code"), "out": str(arm)}), flush=True)
    return evidence.get("exit_code", 1)


def sample_gpu():
    """Keep failed observations distinct from zero utilization and training failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
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
    readings = {int(line.split(",")[0]): float(line.split(",")[2])
                for line in sample.get("gpu", "").splitlines()}
    if not selected or selected - readings.keys():
        raise RuntimeError("Selected GPUs are missing from the device inventory.")
    occupied = {index: readings[index] for index in selected if readings[index] > 64}
    if occupied:
        raise RuntimeError(f"Selected GPUs are already occupied in MiB: {occupied!r}.")


def wait_for_gpu_release(selected, timeout=30):
    """Bound driver teardown lag after owned processes exit; never kill other users."""
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--task-timeout", type=int, default=600)
    parser.add_argument("--turns", type=int, default=32)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["off", "on"])
    args = parser.parse_args()
    args.repo, args.model, args.output = args.repo.resolve(), args.model.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    validate_reward_config()
    device = prepare(args, args.output)
    for mode in args.modes:
        if run_arm(args, args.output, mode, device):
            raise SystemExit(1)
