"""Run one bounded PSRL science training step per resource ledger setting."""

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
        OmegaConf.create({}),
        None,
        None,
        max_resp_len=512,
        overlong_buffer_cfg=OmegaConf.create({"enable": False, "len": 0, "log": False, "penalty_factor": 1.0}),
    )


def prepare(args, out):
    task = out / "budget-cuda"
    for folder in ("environment", "tests"):
        (task / folder).mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text("""schema_version = "1.4"
[task]
name = "verification/budget-cuda"
description = "Fixed 14 GiB CUDA workload for resource admission verification."
[environment]
cpus = 1
memory_mb = 1024
gpus = 1
build_timeout_sec = 90
network_mode = "no-network"
[agent]
timeout_sec = 150
[verifier]
timeout_sec = 30
""")
    (task / "instruction.md").write_text("Execute the fixed CUDA verification workload and preserve its result.\n")
    (task / "environment/Dockerfile").write_text(
        "FROM python:3.13-slim\nRUN mkdir -p /logs/artifacts /logs/verifier\nCOPY probe.py /app/probe.py\nWORKDIR /app\n"
    )
    probe = Path(__file__).with_name("cuda_driver_probe.py").read_text()
    probe = probe.replace('call("cuMemAlloc_v2",', 'allocation_start = time.time()\ncall("cuMemAlloc_v2",', 1)
    probe = probe.replace("64 * 1024 * 1024", "14 * 1024 * 1024 * 1024")
    probe = probe.replace("time.sleep(float(os.environ.get('PROBE_HOLD_SECONDS', '10')))", "time.sleep(15)")
    probe = probe.replace(
        "print(json.dumps(result), flush=True)",
        'result["allocation_start"] = allocation_start\nresult["held_from"] = time.time()',
    )
    probe += '\nresult["released_at"] = time.time()\nPath("/logs/artifacts/cuda-probe.json").write_text(json.dumps(result))\nprint(json.dumps(result), flush=True)\n'
    (task / "environment/probe.py").write_text(probe)
    (task / "tests/test.sh").write_text("""#!/bin/sh
set -eu
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('/logs/artifacts/cuda-probe.json').read_text())
assert p['kernel_output'] == 42 and p['device_count'] == 1
Path('/logs/verifier/reward.json').write_text(json.dumps({'reward': 1.0}))
PY
""")
    row = {
        "prompt": [{"role": "user", "content": "Run the GPU resource verification task."}],
        "data_source": "sciaccel_rl",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "task_name": "verification/budget-cuda",
        "extra_info": {"task_path": str(task), "reward_key": "reward", "gpus": 1, "gpu_memory_mb": 14336},
    }
    for split in ("train", "eval"):
        directory = out / "data" / split
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(directory / "L1.parquet", index=False)
    device = read_gpu_devices()[args.gpu]
    lease = EnvLease(
        "overlay-only",
        "overlay-only",
        "budget",
        args.node_ip,
        (args.gpu,),
        EnvDemand(needs_gpu=True, gpu_memory_mb=14336),
        0,
        0,
    )
    overlay = build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version()))
    overlay["services"]["main"].pop("labels")
    (out / "gpu-overlay.yaml").write_text(yaml.safe_dump(overlay))
    return device.path


def run_arm(args, out, mode, device):
    arm = out / mode
    arm.mkdir()
    evidence = {"mode": mode, "started_at": time.time(), "budget_seconds": args.timeout}
    token = uuid.uuid4().hex[:8]
    child = None
    owned = {}
    try:
        info = ray.init(
            address="local",
            num_cpus=24,
            num_gpus=2,
            include_dashboard=False,
            _node_ip_address=args.node_ip,
            _temp_dir=f"/tmp/ab-{token}",
            namespace=f"ab-{token}",
            log_to_driver=False,
        )
        evidence["ray_address"] = info.address_info["gcs_address"]
        evidence["ray_tmp"] = f"/tmp/ab-{token}"
        agent = [
            {
                "name": "sciaccel",
                "_target_": "examples.sciaccel_rl.agent_loop.SciAccelAgentLoop",
                "harbor": {
                    "jobs_dir": str(arm / "harbor-jobs"),
                    "agent_name": "science_budget_agent:BudgetAgent",
                    "max_concurrent_episodes": 4,
                    "override_gpus": 0,
                    "gpu_compose_override": str(out / "gpu-overlay.yaml"),
                },
                "task_timeout_sec": 150,
                "verifier_timeout_sec": 30,
            }
        ]
        (arm / "agent.yaml").write_text(yaml.safe_dump(agent))
        env = dict(os.environ)
        env.update(
            PSRL_PATH=str(args.repo),
            HF_MODEL_PATH=str(args.model),
            OUTPUT_DIR=str(arm),
            DATA_DIR=str(out / "data"),
            HINT_LEVEL="L1",
            TRAIN_BATCH_SIZE="1",
            TOTAL_TRAINING_STEPS="1",
            MAX_RESPONSE_LENGTH="512",
            MAX_TOKENS_PER_GPU="2560",
            MAX_TURNS="2",
            TRAIN_SP="1",
            TRAIN_FSDP="1",
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
            PSRL_SCIACCEL_GPU_DEVICES=device,
            thinking_template="multi_traj",
            OVERLONG_FILTERING="False",
            LR_WARMUP_STEPS="0",
            DATA_SEED="17",
        )
        overrides = [
            "psrl.rollout_n=4",
            "psrl.agentic_rl.refill_failure_threshold=1",
            "psrl.staleness_buffer_entries=1",
            "psrl.deployment.n_rollout_instances=1",
            "psrl.deployment.rollout_nnodes_per_instance=1",
            "psrl.deployment.rollout_ngpus_per_node_per_instance=1",
            "psrl.deployment.train_nnodes=1",
            "psrl.deployment.train_ngpus_per_node=1",
            "psrl.deployment.total_nnodes=1",
            "psrl.deployment.n_validate_instances=0",
            "psrl.colocate_validate_and_train=False",
            f"psrl.env_service.enable={mode == 'on'}",
            "psrl.env_service.node_policy=any",
            "psrl.env_service.default_gpu_memory_mb=14336",
            "psrl.env_service.router_name=budget_ab_router",
            (
                f"psrl.env_service.units=[{{unit_id:budget-gpu,node_ip:{args.node_ip},cpu_quota:4,memory_mb:4096,"
                f"gpu_indices:[{args.gpu}],gpu_memory_mb:56320,max_containers:4}}]"
            ),
            "gen_actor_rollout_ref.rollout.n=4",
            "gen_actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.15",
            "gen_actor_rollout_ref.rollout.enforce_eager=True",
            "++gen_actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True",
            "++gen_actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",
            f"gen_actor_rollout_ref.rollout.agent.agent_loop_config_path={arm / 'agent.yaml'}",
            "train_actor_rollout_ref.actor.rollout_n=4",
            "train_actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "trainer.nnodes=1",
            "trainer.n_gpus_per_node=2",
            "trainer.logger=[console]",
            "trainer.resume_mode=disable",
            "trainer.total_training_steps=1",
            "trainer.total_epochs=1",
            "reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=0",
            f"++ray_kwargs.ray_init.address={evidence['ray_address']}",
            f"++ray_kwargs.ray_init.runtime_env.env_vars.RAY_ADDRESS={evidence['ray_address']}",
            f"++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={env['PYTHONPATH']}",
            f"++ray_kwargs.ray_init.runtime_env.env_vars.PSRL_SCIACCEL_GPU_DEVICES={device}",
            '++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE="1"',
            "++ray_kwargs.ray_init.include_dashboard=False",
        ]
        command = ["bash", "examples/sciaccel_rl/fsdp_qwen35_4b.sh", *overrides]
        evidence["command"] = command
        (arm / "launch.json").write_text(json.dumps(evidence, indent=2))
        with (arm / "launcher.log").open("w") as log, (arm / "telemetry.jsonl").open("w") as telemetry:
            child = subprocess.Popen(
                command, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + args.timeout
            router = None
            while child.poll() is None and time.monotonic() < deadline:
                for process in psutil.Process().children(recursive=True):
                    try:
                        owned[process.pid] = process.create_time()
                    except psutil.NoSuchProcess:
                        pass
                sample = {"time": time.time()}
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={args.gpu}",
                        "--query-gpu=uuid,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                sample["gpu"] = result.stdout.strip()
                if mode == "on":
                    try:
                        if router is None:
                            named = ray.util.list_named_actors(all_namespaces=True)
                            found = next((x for x in named if x["name"] == "budget_ab_router"), None)
                            if found:
                                router = ray.get_actor(found["name"], namespace=found["namespace"])
                        if router:
                            sample["units"] = [
                                dataclasses.asdict(x) for x in ray.get(router.snapshot.remote(), timeout=2)
                            ]
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
        evidence["finished_at"] = time.time()
        (arm / "launch.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"mode": mode, "exit_code": evidence.get("exit_code"), "out": str(arm)}), flush=True)
    return evidence.get("exit_code", 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=720)
    parser.add_argument("--modes", nargs="+", default=["off", "on"])
    args = parser.parse_args()
    args.repo, args.model, args.output = args.repo.resolve(), args.model.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    validate_reward_config()
    device = prepare(args, args.output)
    for mode in args.modes:
        if run_arm(args, args.output, mode, device):
            raise SystemExit(1)
