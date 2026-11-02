"""Compare resident inference through Env admission and the fixed historical runner."""

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import subprocess
import time
import uuid
from pathlib import Path

BASELINE = "0eeead861ffafed6bf31c962d259fd847775dcf9"


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str))


def resolve_pool_gpu_mb(configured, total_mb):
    """Use the selected device's capacity unless an explicit experiment quota is supplied."""
    budget = int(total_mb) if configured is None else configured
    if not 0 < budget <= total_mb:
        raise ValueError("GPU pool budget must be positive and no larger than physical capacity.")
    return budget


def gpu_snapshot(index):
    import pynvml as nv

    nv.nvmlInit()
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        memory = nv.nvmlDeviceGetMemoryInfo(handle, version=nv.nvmlMemory_v2)
        return {
            "uuid": nv.nvmlDeviceGetUUID(handle),
            "minor": nv.nvmlDeviceGetMinorNumber(handle),
            "used_mb": memory.used / 1024**2,
            "total_mb": memory.total / 1024**2,
            "pids": [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)],
        }
    finally:
        nv.nvmlShutdown()


def prepare(args):
    import yaml
    from psrl.workers.env_service.overlay import build_lease_overlay, read_kernel_driver_version, resolve_driver_libs
    from psrl.workers.env_service.types import EnvDemand, EnvLease

    args.root.mkdir(parents=True, exist_ok=True)
    task = args.root / "task"
    (task / "environment").mkdir(parents=True, exist_ok=True)
    (task / "tests").mkdir(exist_ok=True)
    save(task / "instruction.md", {"idle_s": args.resident_idle_s})
    (task / "task.toml").write_text(
        'version = "1.0"\n[environment]\n'
        f'docker_image = "{args.image}"\n'
        'cpus = 4\nmemory_mb = 32768\ngpus = 1\nnetwork_mode = "no-network"\n'
        "[agent]\ntimeout_sec = 400\n[verifier]\ntimeout_sec = 30\n"
    )
    verifier = task / "tests/test.sh"
    verifier.write_text(
        "#!/bin/sh\nset -eu\npython - <<'PY'\n"
        "import json\nfrom pathlib import Path\n"
        "items = [json.loads(Path(f'/logs/artifacts/request-{i}.json').read_text()) for i in range(2)]\n"
        "assert items[0]['server_pid'] == items[1]['server_pid']\n"
        "assert all(x['response']['choices'][0]['message']['content'].strip() for x in items)\n"
        "Path('/logs/verifier/reward.json').write_text(json.dumps({'reward': 1.0}))\nPY\n"
    )
    verifier.chmod(0o755)
    demand = EnvDemand(4, 32768, True, 20480)
    lease = EnvLease("static", "static", "probe", args.node_ip, (args.gpu,), demand, 0, 0)
    overlay = build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version()))
    overlay["services"]["main"].pop("labels", None)
    (args.root / "gpu-overlay.yaml").write_text(yaml.safe_dump(overlay))
    save(args.root / "design.json", vars(args))


async def run(args):
    import ray
    from examples.sciaccel_rl.config import HarborConfig, SciAccelRuntimeConfig
    from examples.sciaccel_rl.runner import run_harbor_episode

    output = args.root / args.mode
    output.mkdir(exist_ok=False)
    tag = "resident-" + uuid.uuid4().hex[:10]
    source = Path(inspect.getfile(run_harbor_episode))
    repo = source.parents[2]
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if args.mode == "off" and head != BASELINE:
        raise ValueError(f"OFF must import the fixed baseline, got {head!r}.")
    (output / "source.patch").write_bytes(subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD"]))
    record = {
        "mode": args.mode,
        "tag": tag,
        "head": head,
        "runner": str(source),
        "runner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "started_at": time.time(),
        "pool_gpu_mb_override": args.pool_gpu_mb,
        "request_gpu_mb": 20480,
        "sources": {n: importlib.util.find_spec(n).origin for n in ("psrl", "verl", "harbor", "transfer_queue")},
        "episodes": [],
        "episode_count": args.episodes,
        "warm_episodes": args.warm_episodes,
        "resident_idle_s": args.resident_idle_s,
        "max_attempts": args.attempts,
        "identity_backend": args.gpu_identity_backend,
    }
    before = gpu_snapshot(args.gpu)
    record["gpu_before"] = before
    pool_gpu_mb = resolve_pool_gpu_mb(args.pool_gpu_mb, before["total_mb"])
    record["pool_gpu_mb"] = pool_gpu_mb
    if before["pids"] or before["used_mb"] > 128:
        raise RuntimeError(f"Experiment GPU is occupied: {before!r}.")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, timeout=10)
    router, monitor = None, None
    kwargs = {}
    config_kwargs = {}
    try:
        if args.mode == "on":
            from psrl.workers.env_service.client import EnvServiceClient
            from psrl.workers.env_service.harbor_resources import demand_from_task
            from psrl.workers.env_service.router import EnvRouter

            demand = demand_from_task(str(args.root / "task"), gpu_memory_mb_override=20480)
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
                    "gpu_identity_backend": args.gpu_identity_backend,
                    "units": [
                        {
                            "unit_id": tag,
                            "node_ip": args.node_ip,
                            "cpu_quota": args.episodes * demand.cpu,
                            "memory_mb": args.episodes * demand.memory_mb,
                            "gpu_indices": [args.gpu],
                            "gpu_memory_mb": pool_gpu_mb,
                            "max_containers": args.episodes,
                            "gpu_reservation": "episode",
                        }
                    ],
                }
            )
            await asyncio.wait_for(router.ping.remote(), 60)
            client = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=5)
            kwargs = {"env_client": client, "env_demand": demand}
            config_kwargs["environment_backend"] = "env_worker"

        async def sample():
            with (output / "samples.jsonl").open("w") as stream:
                while True:
                    item = {"time": time.time()}
                    try:
                        item["gpu"] = await asyncio.to_thread(gpu_snapshot, args.gpu)
                        if router is not None:
                            item["ledger"] = [
                                dataclasses.asdict(x) for x in await asyncio.wait_for(router.snapshot.remote(), 5)
                            ]
                    except Exception as exc:
                        item["telemetry_error"] = repr(exc)
                    stream.write(json.dumps(item) + "\n")
                    stream.flush()
                    await asyncio.sleep(1)

        monitor = asyncio.create_task(sample())
        runtime = SciAccelRuntimeConfig(
            harbor=HarborConfig(
                jobs_dir=str(output / "jobs"),
                agent_name="resident_agent:ResidentInferenceAgent",
                override_gpus=0,
                gpu_compose_override=str(args.root / "gpu-overlay.yaml") if args.mode == "off" else "",
                max_concurrent_episodes=args.episodes,
                **config_kwargs,
            ),
            task_timeout_sec=400,
            verifier_timeout_sec=30,
        )

        async def episode(index, attempt=1):
            item = {"index": index, "attempt": attempt, "requested_at": time.time()}
            record["episodes"].append(item)
            result = await run_harbor_episode(
                task_path=str(args.root / "task"),
                model_base_url="http://127.0.0.1:1/v1",
                model_name="unused",
                config=runtime,
                needs_gpu=True,
                session_id=f"{tag}-{index}-{attempt}",
                actor_id=tag,
                regrade_unverified=False,
                **kwargs,
            )
            item.update(finished_at=time.time(), result=dataclasses.asdict(result))
            item["accepted"] = result.exception is None and result.reward == 1
            item["cuda_oom"] = "CUDA out of memory" in str(result.exception)
            save(output / "result.json", record)
            return item

        record["work_started_at"] = time.time()
        async with asyncio.timeout(900):
            tasks = [asyncio.create_task(episode(index)) for index in range(args.warm_episodes)]
            try:
                if tasks:
                    async with asyncio.timeout(300):
                        while True:
                            ready = 0
                            for path in (output / "jobs").rglob("resident.json"):
                                try:
                                    ready += "ready_at" in json.loads(path.read_text())
                                except json.JSONDecodeError:
                                    pass  # A writer may be between truncate and flush.
                            if ready == args.warm_episodes:
                                break
                            if any(task.done() for task in tasks):
                                raise RuntimeError("Warm episode ended before the arrival gate opened.")
                            await asyncio.sleep(0.5)
                record["burst_submitted_at"] = time.time()
                tasks.extend(asyncio.create_task(episode(i)) for i in range(args.warm_episodes, args.episodes))
                results = await asyncio.gather(*tasks)
                for attempt in range(2, args.attempts + 1):
                    failed = [item for item in results if not item["accepted"]]
                    if not failed:
                        break
                    if not all(item["cuda_oom"] for item in failed):
                        raise RuntimeError("An episode failed for a non-OOM reason. No automatic retry.")
                    tasks = [asyncio.create_task(episode(item["index"], attempt)) for item in failed]
                    results = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        record["work_finished_at"] = time.time()
    except BaseException as exc:
        record["error"] = repr(exc)
        raise
    finally:
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        if router is not None:
            try:
                record["ledger_after"] = [
                    dataclasses.asdict(x) for x in await asyncio.wait_for(router.snapshot.remote(), 10)
                ]
                await asyncio.wait_for(router.shutdown_nodes.remote(), 30)
            except Exception as exc:
                record["cleanup_error"] = repr(exc)
            finally:
                ray.kill(router)
        if ray.is_initialized():
            ray.shutdown()
        record["containers_remaining"] = subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.actor_id={tag}"], text=True, timeout=10
        ).strip()
        for _ in range(10):
            record["gpu_after"] = gpu_snapshot(args.gpu)
            if not record["gpu_after"]["pids"] and record["gpu_after"]["used_mb"] < 128:
                break
            await asyncio.sleep(1)
        record["finished_at"] = time.time()
        save(output / "result.json", record)
    accepted = {item["index"] for item in record["episodes"] if item["accepted"]}
    assert len(accepted) == args.episodes, "Not all logical episodes completed within the attempt budget."
    assert (
        not record.get("cleanup_error")
        and not record["containers_remaining"]
        and not record["gpu_after"]["pids"]
        and record["gpu_after"]["used_mb"] < 128
    ), "Physical cleanup failed."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image", default="psrl-inference-resident:20261102")
    parser.add_argument(
        "--pool-gpu-mb", type=int, help="Explicit pressure-test quota in MiB. Defaults to measured device capacity."
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--warm-episodes", type=int, default=0)
    parser.add_argument("--resident-idle-s", type=float, default=8)
    parser.add_argument("--attempts", type=int, choices=[1, 2], default=1)
    parser.add_argument("--gpu-identity-backend", choices=["procfs", "ebpf"], default="procfs")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--mode", choices=["on", "off"], default="on")
    args = parser.parse_args()
    if not 0 <= args.warm_episodes < args.episodes or not 0 <= args.resident_idle_s <= 60:
        parser.error("Require 0 <= warm episodes < episodes and 0 <= idle seconds <= 60.")
    if args.prepare:
        prepare(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
