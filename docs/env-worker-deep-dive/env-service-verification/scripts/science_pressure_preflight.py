"""Compare real science Harbor GPU admission without model inference or training."""

import argparse
import asyncio
import dataclasses
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import ray
from examples.sciaccel_rl.config import HarborConfig, SciAccelRuntimeConfig
from examples.sciaccel_rl.runner import run_harbor_episode
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand


async def run_arm(args, mode, router):
    arm = args.output / mode
    arm.mkdir()
    started = time.time()
    client = EnvServiceClient(router, args.node_ip, heartbeat_interval_s=5) if mode == "on" else None
    configs = [
        SciAccelRuntimeConfig(
            harbor=HarborConfig(
                jobs_dir=str(arm / f"pool-{i}" / "harbor-jobs"),
                agent_name="science_pressure_agent:PressureProbeAgent",
                override_gpus=0,
                gpu_compose_override=str(args.overlay),
                max_concurrent_episodes=4,
            ),
            task_timeout_sec=90,
            verifier_timeout_sec=30,
        )
        for i in range(2)
    ]
    demand = EnvDemand(cpu=1, memory_mb=1024, needs_gpu=True, gpu_memory_mb=52224)
    stop = asyncio.Event()

    async def sample():
        with (arm / "telemetry.jsonl").open("w") as stream:
            while not stop.is_set():
                proc = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--id=6,7",
                    "--query-gpu=uuid,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                )
                try:
                    output, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                    item = {"time": time.time(), "gpu": output.decode().strip()}
                except asyncio.TimeoutError:
                    if proc.returncode is None:
                        proc.kill()
                    await proc.communicate()
                    item = {"time": time.time(), "gpu_error": "TimeoutError"}
                if client:
                    item["units"] = [dataclasses.asdict(s) for s in await router.snapshot.remote()]
                stream.write(json.dumps(item) + "\n")
                stream.flush()
                await asyncio.sleep(0.5)

    monitor = asyncio.create_task(sample())
    results = []
    exit_code = 1
    try:
        jobs = [
            run_harbor_episode(
                task_path=str(args.task),
                model_base_url="http://127.0.0.1:1/v1",
                model_name="unused",
                config=configs[i // 2],
                needs_gpu=True,
                session_id=f"cuda-ab-{args.token}-{mode}-{i}",
                env_client=client,
                env_demand=demand if client else None,
                regrade_unverified=False,
                thinking_template="multi_traj",
            )
            for i in range(4)
        ]
        results = await asyncio.wait_for(asyncio.gather(*jobs), timeout=240)
        assert all(r.exception is None for r in results), results
        rewards = [r.reward for r in results]
        assert all(x == 1 for x in rewards) if mode == "on" else any(x == 0 for x in rewards), rewards
        exit_code = 0
    finally:
        stop.set()
        await monitor
        snapshot = [dataclasses.asdict(s) for s in await router.snapshot.remote()] if client else []
        (arm / "episode-results.json").write_text(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
        (arm / "launch.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "scope": "Real science runner and Harbor CUDA, no model inference or training",
                    "started_at": started,
                    "finished_at": time.time(),
                    "exit_code": exit_code,
                    "final_ledger": snapshot,
                },
                indent=2,
            )
        )
        (arm / "launcher.log").write_text("CUDA-only container comparison. Training was not run.\n")
    print(json.dumps({"mode": mode, "completed": len(results), "exit_code": exit_code}), flush=True)


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.token = uuid.uuid4().hex[:8]
    from psrl.workers.env_service.overlay import read_gpu_devices

    devices = read_gpu_devices()
    os.environ["PSRL_SCIACCEL_GPU_DEVICES"] = ",".join(devices[i].path for i in (6, 7))
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        _node_ip_address=args.node_ip,
        _temp_dir=f"/tmp/ca-{args.token}",
        namespace=f"ca-{args.token}",
        log_to_driver=False,
    )
    router = EnvRouter.remote(
        {
            "node_policy": "any",
            "units": [
                {
                    "unit_id": f"pressure-{i}",
                    "node_ip": args.node_ip,
                    "gpu_indices": [i],
                    "cpu_quota": 4,
                    "memory_mb": 4096,
                    "gpu_memory_mb": 81920,
                    "max_containers": 4,
                }
                for i in (6, 7)
            ],
        }
    )
    try:
        await asyncio.wait_for(router.ping.remote(), timeout=60)
        for mode in ("off", "on"):
            await run_arm(args, mode, router)
    finally:
        ray.kill(router)
        ray.shutdown()
        gpu = await asyncio.to_thread(
            subprocess.check_output,
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        (args.output / "cleanup-gpu.txt").write_text(gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    asyncio.run(main(parser.parse_args()))
