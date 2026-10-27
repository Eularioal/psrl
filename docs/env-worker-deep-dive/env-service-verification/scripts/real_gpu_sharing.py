"""Verify resident inference and confirmed sleep windows with real Docker CUDA commands."""

import argparse
import asyncio
import importlib.util
import json
import math
import time
import uuid
from dataclasses import asdict
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
from real_gpu_lending import InferenceOnlyServer
from real_gpu_preference import docker, save


async def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tag = f"gpu-share-{uuid.uuid4().hex[:8]}"
    probe = NvmlProbe()
    before = probe.read_gpu_snapshot([0])
    assert not before[0]["pids"] and before[0]["used_mb"] < 64, "GPU must be idle before the experiment."
    evidence = {
        "passed": False,
        "tag": tag,
        "before": before,
        "scope": "Real vLLM inference and Harbor CUDA commands. No training, NIXL or SMG routing.",
        "sources": {
            name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "vllm", "transfer_queue")
        },
        "devices": {k: asdict(v) for k, v in probe.read_gpu_devices().items() if k == 0},
    }
    started = time.monotonic()
    server = router = env = handle = overlay_path = None
    docker("info", "--format", "{{.ServerVersion}}")
    evidence["image"] = docker("image", "inspect", args.image, "--format", "{{.Id}}")
    ray.init(
        address="local",
        num_cpus=6,
        num_gpus=1,
        include_dashboard=False,
        log_to_driver=True,
        namespace=tag,
        _node_ip_address=args.node_ip,
        _temp_dir=f"/tmp/{tag}",
        object_store_memory=256 * 1024**2,
    )
    save(out / "ownership.json", {"tag": tag, "ray_temp": f"/tmp/{tag}"})

    async def active_command():
        async with asyncio.timeout(20):
            while True:
                reports = await router.gpu_job_reports.remote(handle.lease.lease_id)
                if any(r["state"] == "active" for r in reports):
                    return reports
                await asyncio.sleep(0.1)

    async def workflow():
        nonlocal server, router, env, handle, overlay_path
        server = ray.remote(num_cpus=2, num_gpus=1, max_concurrency=16)(InferenceOnlyServer).remote(
            str(args.model), sharing=True
        )
        evidence["worker"] = await server.ready.remote()
        assert evidence["worker"]["node_ip"] == args.node_ip, "Wrong execution node."
        evidence["generation_before"] = await server.query.remote()
        resident = probe.read_gpu_snapshot([0])
        profiles = await server.collective_rpc.remote("env_sharing_profile")
        headroom = max(p["transient_mb"] for p in profiles) + 2048
        capacity = math.ceil(resident[0]["used_mb"] + headroom + 1024)
        evidence.update(
            resident_snapshot=resident, profiles=profiles, model_headroom_mb=headroom, unit_capacity_mb=capacity
        )
        config = {
            "node_policy": "prefer_other",
            "slot_ttl_s": 120,
            "gpu_sharing": {"safety_margin_mb": 1024, "max_command_s": 30, "reuse_sleeping": True},
            "units": [
                {
                    "unit_id": tag,
                    "node_ip": args.node_ip,
                    "cpu_quota": 2,
                    "memory_mb": 1024,
                    "gpu_indices": [0],
                    "gpu_memory_mb": capacity,
                    "gpu_reservation": "command",
                    "max_containers": 2,
                    "shared_rollout_replica_rank": 0,
                    "model_headroom_mb": headroom,
                }
            ],
        }
        evidence["config"] = config
        router = EnvRouter.options(name=tag).remote(config, [args.node_ip])
        await router.set_gpu_placement.remote({args.node_ip: [0]})
        evidence["binding"] = await router.bind_shared_rollout_gpu.remote(tag, server)
        client = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=2)
        handle = await client.acquire("normal", EnvDemand(1, 512, True, 512), max_wait_s=10)
        oversized = await client.acquire("oversized", EnvDemand(1, 512, True, capacity), max_wait_s=10)
        try:
            denied = await router.acquire_gpu_command.remote(oversized.lease.lease_id, "unused", "oversized")
            evidence["overbudget"] = denied
            assert denied["status"] == "wait", "Overbudget work must not receive GPU access."
        finally:
            await oversized.release()
        overlay = build_lease_overlay(handle.lease, resolve_driver_libs(read_kernel_driver_version()))
        overlay_path = write_overlay(overlay)
        (out / "environment").mkdir()
        env = ManagedDockerEnvironment(
            environment_dir=out / "environment",
            environment_name=tag,
            session_id=uuid.uuid4().hex[:12],
            trial_paths=TrialPaths(out / "trial"),
            task_env_config=EnvironmentConfig(docker_image=args.image, cpus=1, memory_mb=512),
            extra_docker_compose=[overlay_path],
            lease_overlay=overlay,
            gpu_execution={"router_name": tag, "lease_id": handle.lease.lease_id, "mode": "command"},
            network_policy=NetworkPolicy(network_mode="no-network"),
        )
        await env.start(force_build=False)
        await env.exec("mkdir -p /logs/artifacts", timeout_sec=10)
        await env.upload_file(Path(__file__).with_name("cuda_driver_probe.py"), "/tmp/probe.py")
        for sleeping in (False, True):
            phase = "sleeping" if sleeping else "resident"
            if sleeping:
                await server.nixl_sleep.remote(1)
            command = asyncio.create_task(
                env.exec_gpu("PROBE_HOLD_SECONDS=6 python3 /tmp/probe.py", timeout_sec=20, wait_timeout_sec=10)
            )
            await active_command()
            if sleeping:
                wake = asyncio.ensure_future(server.nixl_wake_up.remote())
                await asyncio.sleep(0.2)
                assert not wake.done(), "Wake must wait until the active CUDA command returns."
                evidence["wake_fenced"] = True
            else:
                async with asyncio.timeout(10):
                    while len(probe.read_gpu_snapshot([0])[0]["pids"]) < 2:
                        await asyncio.sleep(0.1)
                evidence["during_resident_command"] = probe.read_gpu_snapshot([0])
                evidence["concurrent_generation"] = await server.query.remote()
                assert not await server.is_sleeping.remote(), "Resident sharing must keep inference awake."
            result = await command
            assert result.return_code == 0, f"CUDA command failed: {result.stderr!r}."
            parsed = json.loads(result.stdout.strip())
            assert parsed["kernel_output"] == 42, "CUDA kernel must execute successfully."
            assert parsed["uuid_hex"] == uuid.UUID(before[0]["uuid"].removeprefix("GPU-")).hex, "Wrong GPU."
            evidence[phase] = parsed
            if sleeping:
                await asyncio.wait_for(wake, 20)
            save(out / "evidence.json", evidence)
        evidence["generation_after"] = await server.query.remote()
        assert evidence["generation_after"]["token_ids"] == evidence["generation_before"]["token_ids"], (
            "Weights must survive the sleep window."
        )
        evidence["reports"] = await router.gpu_job_reports.remote(handle.lease.lease_id)
        assert [r["admission_memory"]["model_mode"] for r in evidence["reports"]] == ["resident", "sleeping"], (
            "Both sharing paths must be exercised."
        )
        evidence["passed"] = True

    try:
        await asyncio.wait_for(workflow(), timeout=420)
    finally:
        evidence["cleanup_errors"] = []
        for name, action in (
            ("env", lambda: env.stop(delete=True) if env else asyncio.sleep(0)),
            ("lease", lambda: handle.release() if handle else asyncio.sleep(0)),
            ("engine", lambda: server.stop.remote() if server else asyncio.sleep(0)),
            ("router", lambda: router.shutdown_nodes.remote() if router else asyncio.sleep(0)),
        ):
            try:
                await asyncio.wait_for(action(), 30)
            except Exception as exc:
                evidence["cleanup_errors"].append({name: repr(exc)})
        for actor in (server, router):
            if actor is not None:
                ray.kill(actor, no_restart=True)
        ray.shutdown()
        if overlay_path:
            overlay_path.unlink(missing_ok=True)
        owned = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={tag}").split()
        if owned:
            docker("rm", "-f", *owned)
        for _ in range(40):
            after = probe.read_gpu_snapshot([0])
            if not after[0]["pids"] and after[0]["used_mb"] <= before[0]["used_mb"] + 16:
                break
            await asyncio.sleep(0.5)
        evidence["after"] = after
        evidence["containers_after"] = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={tag}")
        evidence["passed"] &= not after[0]["pids"] and after[0]["used_mb"] <= before[0]["used_mb"] + 16
        evidence["elapsed_s"] = time.monotonic() - started
        save(out / "evidence.json", evidence)
        probe.close()
    assert evidence["passed"], "Sharing or cleanup verification failed."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
