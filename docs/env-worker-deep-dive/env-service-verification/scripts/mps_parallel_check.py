"""Verify MPS queueing and independent cleanup through real Router and Docker providers."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import uuid
from pathlib import Path

import ray
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.probe import NvmlProbe
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand
from psrl.workers.env_worker.harbor import EnvWorkerEnvironment


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    tag = "mps-parallel-" + uuid.uuid4().hex[:8]
    evidence = {"tag": tag, "passed": False, "mode": args.mode, "provider": args.provider}
    repo = Path(__file__).resolve().parents[4]
    evidence["sources"] = {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (repo / "psrl/workers/env_service").glob("*.py")
    }
    (args.output / "ownership.json").write_text(json.dumps(evidence))
    probe = NvmlProbe()
    before = probe.read_gpu_snapshot([args.gpu])[args.gpu]
    if before["pids"] or before["used_mb"] > 32:
        raise RuntimeError("Selected GPU is occupied.")
    ray.init(
        address="local",
        namespace=tag,
        _node_ip_address=args.node_ip,
        _temp_dir=f"/tmp/{tag}",
        num_cpus=3,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
    )
    assert [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]] == [args.node_ip]
    router = EnvRouter.options(name=tag).remote(
        {
            "node_policy": "any",
            "gpu_identity_backend": "ebpf",
            "units": [
                {
                    "unit_id": tag,
                    "node_ip": args.node_ip,
                    "cpu_quota": 5,
                    "memory_mb": 4096,
                    "gpu_indices": [args.gpu],
                    "gpu_memory_mb": 2304,
                    "max_containers": 2,
                    "gpu_reservation": args.mode,
                    "gpu_execution_backend": "mps",
                    "mps_thread_percentage": args.thread_percent,
                    "mps_image": args.image,
                }
            ],
        }
    )
    client = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=3)
    envs, handles, paths = [], [], []
    try:
        cls = EnvWorkerEnvironment if args.provider == "native" else ManagedDockerEnvironment
        for i in range(2 if args.mode == "command" else 1):
            handle = await client.acquire(f"{tag}-{i}", EnvDemand(2, 1024, True, 2048), max_wait_s=30)
            handles.append(handle)
            overlay = build_lease_overlay(handle.lease, resolve_driver_libs(read_kernel_driver_version()))
            root = args.output / str(i)
            root.mkdir()
            kwargs = dict(
                environment_dir=root,
                environment_name=tag,
                session_id=f"{tag}-{i}",
                task_env_config=EnvironmentConfig(docker_image=args.image, cpus=2, memory_mb=1024),
                trial_paths=TrialPaths(root / "trial"),
                network_policy=NetworkPolicy(network_mode="no-network"),
                actor_id=tag,
                lease_overlay=overlay,
                gpu_execution={
                    "router_name": tag,
                    "lease_id": handle.lease.lease_id,
                    "mode": args.mode,
                    "backend": "mps",
                    "gpu_memory_mb": 2048,
                },
            )
            if args.provider == "compose":
                path = write_overlay(overlay)
                paths.append(path)
                kwargs["extra_docker_compose"] = [path]
                await cls.prepare_runtime()
            env = cls(**kwargs)
            envs.append(env)
            await env.start(force_build=False)
            await env.upload_file(Path(__file__).with_name("mps_cuda_client.py"), "/tmp/client.py")
            await env.exec("mkdir -p /tmp/out; echo retained > /tmp/state", timeout_sec=10)
            denied = await env.exec(
                "python3 /tmp/client.py --name denied --output /tmp/out --seconds 0.1", timeout_sec=10
            )
            assert denied.return_code != 0, "Unreserved GPU access must be denied."
            denied_mps = await env.exec(
                f"CUDA_MPS_PIPE_DIRECTORY={handle.lease.mps_pipe} python3 /tmp/client.py "
                "--name denied-mps --output /tmp/out --seconds 0.1",
                timeout_sec=10,
            )
            assert denied_mps.return_code != 0, "The MPS pipe must not bypass device admission."
        a, b = envs[0], envs[-1]

        async def start(env, name):
            from examples.sciaccel_rl.gpu_tools import dispatch_gpu_tool

            request = {
                "op": "start",
                "command": f"python3 /tmp/client.py --name {name} --output /tmp/out --seconds 12",
                "gpu_memory_mb": 1024,
                "timeout_sec": 25,
                "wait_timeout_sec": 40,
            }
            return (await dispatch_gpu_tool(env, "psrl-gpu " + json.dumps(request)))["job_id"]

        first, second = await start(a, "first"), await start(a, "second")
        async with asyncio.timeout(30):
            while any(a.gpu_job_status(j)["state"] != "running" for j in (first, second)):
                if any(a.gpu_job_status(j)["state"] == "failed" for j in (first, second)):
                    raise RuntimeError([a.gpu_job_status(j) for j in (first, second)])
                await asyncio.sleep(0.05)
        third = await start(b, "third")
        async with asyncio.timeout(15):
            while True:
                outputs = [await a.exec(f"cat /tmp/out/{name}.json", timeout_sec=3) for name in ("first", "second")]
                if all(output.return_code == 0 for output in outputs):
                    values = [json.loads(output.stdout) for output in outputs]
                    if all(value.get("launches", 0) > 50 and value["status"] == "running" for value in values):
                        evidence["cuda_overlap"] = values
                        break
                await asyncio.sleep(0.05)
        if args.expected_sms:
            assert all(value["visible_sms"] == args.expected_sms for value in evidence["cuda_overlap"])
        await asyncio.sleep(0.5)
        evidence["telemetry"] = [await a.gpu_job_telemetry(first), await a.gpu_job_telemetry(second)]
        assert all(
            any((d["peak_memory_mb"] or 0) > 0 for d in item["telemetry"]["devices"].values())
            for item in evidence["telemetry"]
        ), evidence["telemetry"]
        if args.faults:
            queued = await start(b, "cancel-queued")
            await asyncio.sleep(0.1)
            evidence["queued_cancel"] = await b.cancel_gpu_job(queued)
            assert evidence["queued_cancel"]["state"] == "cancelled"
        evidence["before_cancel"] = [a.gpu_job_status(first), a.gpu_job_status(second), b.gpu_job_status(third)]
        assert b.gpu_job_status(third)["state"] == "queued", "Third command must wait for budget."
        evidence["cancel"] = await a.cancel_gpu_job(first)
        assert evidence["cancel"]["state"] == "cancelled", evidence["cancel"]
        async with asyncio.timeout(15):
            while b.gpu_job_status(third)["state"] != "running":
                if b.gpu_job_status(third)["state"] == "failed":
                    raise RuntimeError(b.gpu_job_status(third))
                await asyncio.sleep(0.05)
        assert a.gpu_job_status(second)["state"] == "running", "Cancelling a peer must preserve running work."
        evidence["results"] = [
            await a.wait_gpu_job(second, timeout_sec=30, released=True),
            await b.wait_gpu_job(third, timeout_sec=30, released=True),
        ]
        assert all(r["state"] == "completed" and r["exit_code"] == 0 for r in evidence["results"]), evidence
        assert (await a.exec("cat /tmp/state", timeout_sec=5)).stdout.strip() == "retained"
        if args.faults:
            timed = a.start_gpu_job(
                "python3 /tmp/client.py --name timed --output /tmp/out --seconds 12 & wait",
                gpu_memory_mb=1024,
                timeout_sec=2,
                wait_timeout_sec=10,
            )["job_id"]
            evidence["timeout"] = await a.wait_gpu_job(timed, timeout_sec=20, released=True)
            assert evidence["timeout"]["state"] == "failed" and "TimeoutError" in evidence["timeout"]["error"]
            result = await a.exec("cat /tmp/out/timed.json", timeout_sec=5)
            value = json.loads(result.stdout)
            assert value["launches"] > 0, "Timed-out child must have executed CUDA."
            evidence["timeout_cuda"] = value
            recovered = a.start_gpu_job(
                "python3 /tmp/client.py --name recovered --output /tmp/out --seconds 0.2",
                gpu_memory_mb=1024,
                timeout_sec=10,
                wait_timeout_sec=10,
            )["job_id"]
            evidence["after_timeout"] = await a.wait_gpu_job(recovered, timeout_sec=15, released=True)
            assert evidence["after_timeout"]["exit_code"] == 0
        if args.limits:
            capped = a.start_gpu_job(
                "python3 /tmp/client.py --name capped --output /tmp/out --seconds 0.1 --mib 512",
                gpu_memory_mb=256,
                timeout_sec=10,
                wait_timeout_sec=10,
            )["job_id"]
            evidence["memory_cap"] = await a.wait_gpu_job(capped, timeout_sec=15, released=True)
            assert evidence["memory_cap"]["exit_code"] == 1
            value = json.loads(evidence["memory_cap"]["stdout"])
            assert value["cuda_error"] == 2, "Client memory cap must report CUDA out of memory."
        evidence["passed"] = True
    finally:
        for env in reversed(envs):
            try:
                await env.stop(delete=True)
            except Exception as exc:
                if args.faults and "GPU jobs failed:" in str(exc) and "TimeoutError" in str(exc):
                    evidence["reported_execution_failure_on_close"] = str(exc)
                else:
                    evidence.setdefault("cleanup_errors", []).append(repr(exc))
        for handle in handles:
            await handle.release()
        try:
            await router.shutdown_nodes.remote()
        except Exception as exc:
            evidence.setdefault("cleanup_errors", []).append(repr(exc))
        ray.kill(router)
        ray.shutdown()
        for path in paths:
            path.unlink(missing_ok=True)
        evidence["after"] = probe.read_gpu_snapshot([args.gpu])
        evidence["remaining"] = subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.actor_id={tag}"], text=True, timeout=10
        ).strip()
        (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str))
    assert not evidence.get("cleanup_errors") and not evidence["remaining"], evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thread-percent", type=int, default=100)
    parser.add_argument("--expected-sms", type=int)
    parser.add_argument("--limits", action="store_true")
    parser.add_argument("--faults", action="store_true")
    parser.add_argument("--mode", choices=("command", "episode"), default="command")
    parser.add_argument("--provider", choices=("native", "compose"), default="native")
    asyncio.run(asyncio.wait_for(run(parser.parse_args()), 180))
