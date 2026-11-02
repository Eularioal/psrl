"""Verify both container providers against one managed GPU execution contract."""

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


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=10).strip()


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    tag = "shared-exec-" + uuid.uuid4().hex[:10]
    (args.output / "ownership.json").write_text(json.dumps({"actor_id": tag, "gpu": args.gpu}))
    probe = NvmlProbe()
    before = probe.read_gpu_snapshot([args.gpu])
    if before[args.gpu]["used_mb"] > 64 or before[args.gpu]["pids"]:
        raise RuntimeError("Selected GPU is occupied.")
    docker("info", "--format", "{{.ServerVersion}}")
    evidence = {"tag": tag, "before": before, "cases": [], "passed": False}
    repo = Path(__file__).resolve().parents[4]
    evidence["source_sha256"] = {
        path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
        for path in (
            "psrl/workers/env_service/managed_execution.py",
            "psrl/workers/env_service/docker.py",
            "psrl/workers/env_worker/harbor.py",
            "psrl/workers/env_worker/sandbox.py",
        )
    }
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
            "units": [
                {
                    "unit_id": tag,
                    "node_ip": args.node_ip,
                    "cpu_quota": 2,
                    "memory_mb": 2048,
                    "gpu_indices": [args.gpu],
                    "gpu_memory_mb": 2048,
                    "max_containers": 1,
                    "gpu_reservation": "command",
                }
            ],
        }
    )
    client = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=3)
    env, handle = None, None
    paths = []
    try:
        for name, cls in [("compose", ManagedDockerEnvironment), ("native", EnvWorkerEnvironment)]:
            handle = await client.acquire(name, EnvDemand(1, 512, True, 512), max_wait_s=30)
            overlay = build_lease_overlay(handle.lease, resolve_driver_libs(read_kernel_driver_version()))
            binding = {"router_name": tag, "lease_id": handle.lease.lease_id, "mode": "command"}
            root = args.output / name
            root.mkdir()
            kwargs = dict(
                environment_dir=root,
                environment_name=tag,
                session_id=f"{tag}-{name}",
                task_env_config=EnvironmentConfig(docker_image=args.image, cpus=1, memory_mb=512),
                trial_paths=TrialPaths(root / "trial"),
                network_policy=NetworkPolicy(network_mode="no-network"),
                actor_id=tag,
                lease_overlay=overlay,
                gpu_execution=binding,
            )
            if name == "compose":
                path = write_overlay(overlay)
                paths.append(path)
                kwargs["extra_docker_compose"] = [path]
                await cls.prepare_runtime()
            env = cls(**kwargs)
            await env.start(force_build=False)
            await env.exec("mkdir -p /logs/artifacts; echo retained > /tmp/state", timeout_sec=10)
            await env.upload_file(Path(__file__).with_name("cuda_driver_probe.py"), "/tmp/probe.py")
            command = "PROBE_HOLD_SECONDS=0.1 python3 /tmp/probe.py"
            denied = await env.exec(command, timeout_sec=10)
            assert denied.return_code != 0, "Unreserved CUDA must be denied."
            direct = await env.exec_gpu(command, timeout_sec=15, wait_timeout_sec=20)
            assert direct.return_code == 0 and json.loads(direct.stdout)["kernel_output"] == 42
            expected_uuid = uuid.UUID(before[args.gpu]["uuid"].removeprefix("GPU-")).hex
            assert json.loads(direct.stdout)["uuid_hex"] == expected_uuid, "Wrong physical device."
            job = env.start_gpu_job(command, timeout_sec=15, wait_timeout_sec=20)
            result = await env.wait_gpu_job(job["job_id"], timeout_sec=30, released=True)
            assert result["state"] == "completed", result
            cancelled = env.start_gpu_job("sleep 30", timeout_sec=40, wait_timeout_sec=20)
            async with asyncio.timeout(15):
                while env.gpu_job_status(cancelled["job_id"])["state"] != "running":
                    await asyncio.sleep(0.05)
            status = await env.cancel_gpu_job(cancelled["job_id"])
            assert status["state"] == "cancelled", status
            retained = await env.exec("cat /tmp/state", timeout_sec=5)
            assert retained.stdout.strip() == "retained", "Cancelling a job must preserve its container."
            after = await env.exec(command, timeout_sec=10)
            assert after.return_code != 0, "Returned CUDA permissions must be revoked."
            evidence["cases"].append(
                {
                    "provider": name,
                    "direct": json.loads(direct.stdout),
                    "job": result,
                    "cancel": status,
                    "state_retained": True,
                }
            )
            await env.stop(delete=True)
            env = None
            await handle.release()
            handle = None
        evidence["passed"] = True
    finally:
        if env:
            await env.stop(delete=True)
        if handle:
            await handle.release()
        await router.shutdown_nodes.remote()
        ray.kill(router)
        ray.shutdown()
        for path in paths:
            path.unlink(missing_ok=True)
        evidence["remaining_containers"] = docker("ps", "-aq", "--filter", f"label=psrl.actor_id={tag}")
        evidence["after"] = probe.read_gpu_snapshot([args.gpu])
        (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str))
    assert not evidence["remaining_containers"], "Owned containers must be removed."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(run(args), 240))
