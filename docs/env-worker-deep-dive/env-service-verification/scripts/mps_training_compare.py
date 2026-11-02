"""Compare equal fixed GPU training work through real EnvRouter Docker environments."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import subprocess
import time
import uuid
from pathlib import Path

import ray
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.overlay import build_lease_overlay, read_kernel_driver_version, resolve_driver_libs
from psrl.workers.env_service.probe import NvmlProbe
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand
from psrl.workers.env_worker.harbor import EnvWorkerEnvironment


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str))
    temporary.replace(path)


async def arm(args, mode, iteration, probe):
    root = args.output / f"{iteration}-{mode}"
    root.mkdir()
    tag = "mps-train-" + uuid.uuid4().hex[:10]
    record = {"mode": mode, "iteration": iteration, "tag": tag, "passed": False, "started": time.time()}
    write(root / "ownership.json", record)
    before = probe.read_gpu_snapshot([args.gpu])[args.gpu]
    if before["pids"] or before["used_mb"] > 64:
        raise RuntimeError("Selected GPU is not idle.")
    backend = "mps" if mode == "mps" else "direct"
    router = EnvRouter.options(name=tag).remote(
        {
            "node_policy": "any",
            "gpu_identity_backend": "ebpf",
            "prewarm": {"timeout_s": 60},
            "units": [
                {
                    "unit_id": tag,
                    "node_ip": args.node_ip,
                    "cpu_quota": 9,
                    "memory_mb": 33792,
                    "gpu_indices": [args.gpu],
                    "gpu_memory_mb": int(before["total_mb"]),
                    "max_containers": 2,
                    "gpu_reservation": "episode",
                    "gpu_execution_backend": backend,
                    "mps_image": args.image if backend == "mps" else "",
                }
            ],
        }
    )
    client = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=3)
    environments, handles, jobs = [], [], []
    stop = asyncio.Event()

    async def sample():
        with (root / "gpu.jsonl").open("w") as stream:
            while not stop.is_set():
                row = {"time": time.time()}
                try:
                    row["gpu"] = await asyncio.wait_for(asyncio.to_thread(probe.read_gpu_snapshot, [args.gpu]), 5)
                except Exception as exc:
                    row["missing"] = repr(exc)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                await asyncio.sleep(0.5)

    async def await_file(index, name):
        async with asyncio.timeout(240):
            while not (root / str(index) / name).exists():
                status = environments[index].gpu_job_status(jobs[index])
                if status["state"] in ("failed", "cancelled", "completed"):
                    raise RuntimeError(status)
                await asyncio.sleep(0.1)

    monitor = asyncio.create_task(sample())
    try:
        await asyncio.wait_for(router.wait_prewarm.remote(), 65)
        for i in range(2):
            local = root / str(i)
            local.mkdir()
            handle = await client.acquire(f"{tag}-{i}", EnvDemand(4, 16384, True, 40000), max_wait_s=30)
            handles.append(handle)
            overlay = build_lease_overlay(handle.lease, resolve_driver_libs(read_kernel_driver_version()))
            overlay["services"]["main"]["volumes"] += [
                f"{local.resolve()}:/perf:rw",
                f"{args.source.resolve()}:/workload:ro",
            ]
            definition = local / "definition"
            definition.mkdir()
            env = EnvWorkerEnvironment(
                environment_dir=definition,
                environment_name=tag,
                session_id=f"{tag}-{i}",
                task_env_config=EnvironmentConfig(docker_image=args.image, cpus=4, memory_mb=16384),
                trial_paths=TrialPaths(local / "trial"),
                network_policy=NetworkPolicy(network_mode="no-network"),
                actor_id=tag,
                lease_overlay=overlay,
                gpu_execution={
                    "router_name": tag,
                    "lease_id": handle.lease.lease_id,
                    "mode": "episode",
                    "backend": backend,
                    "gpu_memory_mb": 40000,
                },
            )
            environments.append(env)
            await env.start(force_build=False)
            await env.upload_file(Path(__file__).with_name("mps_train_workload.py"), "/tmp/mps_train_workload.py")
        record["containers_ready"] = time.time()
        # All arms prepare in the same order before either measured training segment starts.
        for i, env in enumerate(environments):
            profile = " --profile" if mode != "serial" and iteration == 1 else ""
            command = "python3 /tmp/mps_train_workload.py --source /workload/train.py --output /perf" + profile
            jobs.append(
                env.start_gpu_job(
                    command,
                    cwd="/workload",
                    timeout_sec=480,
                    wait_timeout_sec=40,
                    env={
                        "PYTHONPATH": "/workload",
                        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor",
                        "TRITON_CACHE_DIR": "/tmp/triton",
                    },
                )["job_id"]
            )
            await await_file(i, "train-ready.json")
        record["prepared"] = time.time()
        if mode == "mps":
            # Check actual MPS client membership after CUDA initialization, not just configuration.
            domain = subprocess.check_output(
                ["docker", "ps", "-q", "--filter", f"label=psrl.env_mps_unit={tag}"], text=True
            ).strip()
            record["mps_clients"] = subprocess.check_output(
                ["docker", "exec", "-i", domain, "nvidia-cuda-mps-control"], input="ps\n", text=True, timeout=10
            )
            if len([line for line in record["mps_clients"].splitlines() if line and not line.startswith("PID")]) != 2:
                raise RuntimeError("Two training clients did not join MPS.")
        for i in range(2):
            (root / str(i) / "train-go").touch()
            if mode == "serial":
                await await_file(i, "measured.json")
        for i in range(2):
            await await_file(i, "measured.json")
        if mode != "serial" and iteration == 1:
            for i in range(2):
                await await_file(i, "profile-ready.json")
            for i in range(2):
                (root / str(i) / "profile-go").touch()
        record["jobs"] = [
            await env.wait_gpu_job(job, timeout_sec=90, released=True) for env, job in zip(environments, jobs)
        ]
        if any(job["state"] != "completed" or job["exit_code"] != 0 for job in record["jobs"]):
            raise RuntimeError(record["jobs"])
        record["workloads"] = [json.loads((root / str(i) / "result.json").read_text()) for i in range(2)]
        for result in record["workloads"]:
            assert result["passed"] and len([s for s in result["steps"] if 3 <= s["step"] < 13]) == 10
        record["training_span_s"] = max(r["train_end"] for r in record["workloads"]) - min(
            r["train_start"] for r in record["workloads"]
        )
        record["training_seconds"] = [r["train_end"] - r["train_start"] for r in record["workloads"]]
        record["passed"] = True
    except BaseException as exc:
        record["error"] = repr(exc)
        try:
            record["node_capacity_on_error"] = await asyncio.wait_for(router.node_capacity.remote(), 5)
        except Exception as diagnostic_error:
            record["diagnostic_error"] = repr(diagnostic_error)
        raise
    finally:
        for env in reversed(environments):
            try:
                await env.stop(delete=True)
            except Exception as exc:
                record.setdefault("cleanup_errors", []).append(repr(exc))
        for handle in handles:
            await handle.release()
        try:
            await router.shutdown_nodes.remote()
        except Exception as exc:
            record.setdefault("cleanup_errors", []).append(repr(exc))
        ray.kill(router)
        stop.set()
        await monitor
        for _ in range(20):
            record["after"] = probe.read_gpu_snapshot([args.gpu])
            if not record["after"][args.gpu]["pids"] and record["after"][args.gpu]["used_mb"] < 64:
                break
            await asyncio.sleep(0.5)
        record["remaining"] = subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.actor_id={tag}"], text=True
        ).strip()
        record["wall_seconds"] = time.time() - record["started"]
        write(root / "result.json", record)
    if record.get("cleanup_errors") or record["remaining"] or record["after"][args.gpu]["pids"]:
        raise RuntimeError("Physical cleanup is unconfirmed.")
    return record


async def main(args):
    args.output.mkdir(parents=True, exist_ok=args.resume)
    subprocess.run(["docker", "info"], check=True, timeout=10, stdout=subprocess.DEVNULL)
    metadata = {
        "sources": {n: importlib.util.find_spec(n).origin for n in ("psrl", "verl", "transfer_queue")},
        "image": subprocess.check_output(
            ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
        ).strip(),
        "workload_sha256": hashlib.sha256((args.source / "train.py").read_bytes()).hexdigest(),
        "scope": "Fixed autoresearch updates. Not native LAPS, official benchmark scoring, or historical OFF.",
    }
    write(args.output / "metadata.json", metadata)
    probe = NvmlProbe()
    records = []
    try:
        for iteration, order in enumerate([["serial", "plain", "mps"], ["mps", "plain", "serial"]]):
            for mode in order:
                if args.arm and args.arm != f"{iteration}-{mode}":
                    continue
                previous = args.output / f"{iteration}-{mode}" / "result.json"
                if args.resume and previous.exists():
                    completed = json.loads(previous.read_text())
                    clean_gpu = all(not gpu["pids"] and gpu["used_mb"] < 64 for gpu in completed["after"].values())
                    if (
                        completed["passed"]
                        and not completed.get("cleanup_errors")
                        and not completed["remaining"]
                        and clean_gpu
                    ):
                        records.append(completed)
                        continue
                    previous.parent.rename(
                        previous.parent.with_name(previous.parent.name + "-failed-" + uuid.uuid4().hex[:6])
                    )
                ray.init(
                    address="local",
                    namespace="mps-training",
                    _node_ip_address=args.node_ip,
                    _temp_dir="/tmp/mp-" + uuid.uuid4().hex[:8],
                    num_cpus=3,
                    num_gpus=0,
                    include_dashboard=False,
                    log_to_driver=False,
                )
                assert [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]] == [args.node_ip]
                try:
                    records.append(await arm(args, mode, iteration, probe))
                finally:
                    ray.shutdown()
                write(args.output / "summary.json", records)
    finally:
        probe.close()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arm", choices=[f"{i}-{mode}" for i in range(2) for mode in ("serial", "plain", "mps")])
    asyncio.run(asyncio.wait_for(main(parser.parse_args()), 1500))
