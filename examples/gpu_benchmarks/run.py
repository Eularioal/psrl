"""Run one bounded benchmark episode through a private EnvRouter."""

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import time
import uuid
from pathlib import Path


def gpu_snapshot(index):
    import pynvml as nv

    nv.nvmlInit()
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        return {
            "uuid": nv.nvmlDeviceGetUUID(handle),
            "name": nv.nvmlDeviceGetName(handle),
            "minor": nv.nvmlDeviceGetMinorNumber(handle),
            "total_mb": nv.nvmlDeviceGetMemoryInfo(handle, version=nv.nvmlMemory_v2).total / 1024**2,
            "used_mb": nv.nvmlDeviceGetMemoryInfo(handle, version=nv.nvmlMemory_v2).used / 1024**2,
            "pids": [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)],
        }
    finally:
        nv.nvmlShutdown()


async def run(args):
    import ray
    import tomllib
    from examples.sciaccel_rl.config import HarborConfig, SciAccelRuntimeConfig
    from examples.sciaccel_rl.runner import run_harbor_episode
    from psrl.workers.env_service.client import EnvServiceClient
    from psrl.workers.env_service.harbor_resources import demand_from_task
    from psrl.workers.env_service.router import EnvRouter

    args.task = args.task.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    tag = "bench-" + uuid.uuid4().hex[:10]
    config = tomllib.loads((args.task / "task.toml").read_text())
    demand = demand_from_task(str(args.task), gpu_memory_mb_override=args.gpu_memory_mb)
    record = {
        "tag": tag,
        "mode": "episode",
        "environment_backend": args.environment_backend,
        "agent": args.agent,
        "started": time.time(),
        "sources": {n: importlib.util.find_spec(n).origin for n in ("psrl", "verl", "harbor", "transfer_queue")},
        "provenance": json.loads((args.task / "provenance.json").read_text()),
    }
    before = gpu_snapshot(args.gpu)
    if before["pids"] or before["used_mb"] > 128:
        raise RuntimeError(f"Select an idle independent GPU: {before!r}.")
    record["gpu_before"] = before
    for environment in (config["environment"], config.get("verifier", {}).get("environment", {})):
        if environment.get("gpus", 0) > 1:
            raise ValueError("This verification launcher supports one GPU per environment.")
        gpu_types = environment.get("gpu_types", [])
        if gpu_types and not any(kind.rstrip("!") in before["name"] for kind in gpu_types):
            raise ValueError(f"The task requires {gpu_types!r}, but selected {before['name']!r}.")
    subprocess.run(["docker", "info"], check=True, timeout=10, stdout=subprocess.DEVNULL)
    router = None
    monitor = None
    record["samples"] = []

    async def sample_resources():
        while True:
            sample = {"time": time.time()}
            try:
                snapshot = await asyncio.wait_for(router.snapshot.remote(), 5)
                sample["ledger"] = [dataclasses.asdict(item) for item in snapshot]
                sample["gpu"] = await asyncio.to_thread(gpu_snapshot, args.gpu)
            except Exception as exc:
                sample["telemetry_error"] = repr(exc)
            record["samples"].append(sample)
            with (args.output / "samples.jsonl").open("a") as stream:
                stream.write(json.dumps(sample, default=str) + "\n")
            await asyncio.sleep(5)

    repo = Path(__file__).resolve().parents[2]
    record["source_sha256"] = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in (
            "examples/gpu_benchmarks/run.py",
            "examples/sciaccel_rl/runner.py",
            "psrl/workers/env_worker/harbor.py",
            "psrl/workers/env_service/managed_execution.py",
        )
    }
    try:
        ray.init(
            address="local",
            namespace=tag,
            _node_ip_address=args.node_ip,
            _temp_dir=f"/tmp/{tag}",
            num_cpus=4,
            num_gpus=0,
            include_dashboard=False,
            log_to_driver=False,
        )
        record["ray_nodes"] = ray.nodes()
        router = EnvRouter.options(name=tag).remote(
            {
                "node_policy": "any",
                "units": [
                    {
                        "unit_id": tag,
                        "node_ip": args.node_ip,
                        "cpu_quota": demand.cpu,
                        "memory_mb": demand.memory_mb,
                        "gpu_indices": [args.gpu],
                        "gpu_memory_mb": int(before["total_mb"]),
                        "max_containers": 1,
                        "gpu_reservation": "episode",
                    }
                ],
            }
        )
        await asyncio.wait_for(router.ping.remote(), 60)
        monitor = asyncio.create_task(sample_resources())
        client = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=5)
        runtime = SciAccelRuntimeConfig(
            harbor=HarborConfig(
                jobs_dir=str(args.output / "jobs"),
                agent_name=args.agent,
                environment_backend=args.environment_backend,
            ),
            task_timeout_sec=config["agent"]["timeout_sec"],
            verifier_timeout_sec=config["verifier"]["timeout_sec"],
        )
        async with asyncio.timeout(args.timeout):
            result = await run_harbor_episode(
                task_path=str(args.task),
                model_base_url=args.model_url,
                model_name=args.model,
                config=runtime,
                needs_gpu=True,
                session_id=tag,
                actor_id=tag,
                env_client=client,
                env_demand=demand,
                max_turns=args.max_turns,
                regrade_unverified=False,
            )
        record["result"] = dataclasses.asdict(result)
        record["passed"] = result.exception is None and bool(result.rewards)
    except BaseException as exc:
        record["error"] = repr(exc)
        raise
    finally:
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        if router is not None:
            try:
                snapshot = await asyncio.wait_for(router.snapshot.remote(), 10)
                record["ledger"] = [dataclasses.asdict(x) for x in snapshot]
                await asyncio.wait_for(router.shutdown_nodes.remote(), 30)
            except Exception as exc:
                record["cleanup_error"] = repr(exc)
            finally:
                ray.kill(router)
        if ray.is_initialized():
            ray.shutdown()
        record["containers_remaining"] = subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.actor_id={tag}"],
            text=True,
            timeout=10,
        ).strip()
        for _ in range(10):
            record["gpu_after"] = gpu_snapshot(args.gpu)
            if not record["gpu_after"]["pids"] and record["gpu_after"]["used_mb"] < 128:
                break
            await asyncio.sleep(1)
        record["finished"] = time.time()
        (args.output / "evidence.json").write_text(json.dumps(record, indent=2, default=str))
    if (
        not record.get("passed")
        or record.get("cleanup_error")
        or record["containers_remaining"]
        or record["gpu_after"]["pids"]
        or record["gpu_after"]["used_mb"] >= 128
    ):
        raise RuntimeError("Episode or physical cleanup failed. Inspect evidence.json.")


def build_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gpu-memory-mb", type=int, default=8192)
    parser.add_argument("--agent", default="oracle")
    parser.add_argument("--environment-backend", choices=["docker", "env_worker"], default="docker")
    parser.add_argument("--model-url", default="http://127.0.0.1:1/v1")
    parser.add_argument("--model", default="unused")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    return parser


def main():
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
