"""Check dedicated device preference with real Ray, Harbor and bounded CUDA pressure."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import subprocess
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
from resident_gpu_pressure import ResidentPressure


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=10).strip()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


async def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tag = f"gpu-pref-{uuid.uuid4().hex[:10]}"
    indices = (args.protected_gpu, args.dedicated_gpu)
    if len(set(indices)) != 2:
        raise ValueError("The probe requires two distinct GPUs.")
    probe = NvmlProbe()
    before = probe.read_gpu_snapshot(indices)
    if any(reading["pids"] or reading["used_mb"] > 64 for reading in before.values()):
        raise RuntimeError("Both probe GPUs must have no compute processes and less than 64 MiB application usage.")
    evidence = {
        "passed": False,
        "scope": "One real node and two GPUs. Protected ownership is an experimental label, not a running model.",
        "before": before,
        "devices": {index: asdict(device) for index, device in probe.read_gpu_devices().items() if index in indices},
        "sources": {
            name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "harbor")
        },
        "image": docker("image", "inspect", args.image, "--format", "{{.Id}}"),
        "cases": [],
        "tag": tag,
    }
    repo = Path(__file__).resolve().parents[4]
    evidence["source_sha256"] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (repo / "psrl/workers/env_service").glob("*.py")
    }
    docker("info", "--format", "{{.ServerVersion}}")
    started = time.monotonic()
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
        namespace=tag,
        _node_ip_address=args.node_ip,
        _temp_dir=f"/tmp/{tag}",
        object_store_memory=128 * 1024**2,
        _memory=512 * 1024**2,
    )
    assert [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]] == [args.node_ip], "Wrong Ray node identity."
    units = [
        {
            "unit_id": f"{tag}-{role}",
            "node_ip": args.node_ip,
            "cpu_quota": 1,
            "memory_mb": 512,
            "gpu_indices": [index],
            "gpu_memory_mb": 4096,
            "max_containers": 1,
            "gpu_reservation": "command",
        }
        for role, index in zip(("protected", "dedicated"), indices, strict=True)
    ]
    router = EnvRouter.options(name=tag).remote(
        {"node_policy": "prefer_other", "units": units, "default_gpu_memory_mb": 2048, "slot_ttl_s": 120},
        [args.node_ip],
    )
    pressures = []
    await router.set_gpu_placement.remote({args.node_ip: [args.protected_gpu]})
    save(out / "ownership.json", {"tag": tag, "units": units, "ray_temp": f"/tmp/{tag}"})

    async def run_case(name, expected):
        path = out / name
        path.mkdir()
        client = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=2)
        handle = await client.acquire(name, EnvDemand(1, 512, True, 2048), max_wait_s=20)
        lease = handle.lease
        overlay = build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version()))
        overlay_path = write_overlay(overlay)
        (path / "environment").mkdir()
        environment = ManagedDockerEnvironment(
            environment_dir=path / "environment",
            environment_name="gpu-preference",
            session_id=uuid.uuid4().hex[:12],
            trial_paths=TrialPaths(path / "trial"),
            task_env_config=EnvironmentConfig(docker_image=args.image, cpus=1, memory_mb=512),
            extra_docker_compose=[overlay_path],
            lease_overlay=overlay,
            gpu_execution={"router_name": tag, "lease_id": lease.lease_id, "mode": "command"},
            network_policy=NetworkPolicy(network_mode="no-network"),
        )
        record = {"name": name, "chosen_gpu": lease.gpu_indices, "memory_before": probe.read_gpu_snapshot(indices)}
        evidence["cases"].append(record)
        try:
            assert lease.gpu_indices == (expected,), f"Unexpected GPU choice for {name!r}."
            await environment.start(force_build=False)
            await environment.exec("mkdir -p /logs/artifacts", timeout_sec=10)
            await environment.upload_file(Path(__file__).with_name("cuda_driver_probe.py"), "/tmp/probe.py")
            output = await environment.exec_gpu(
                "PROBE_HOLD_SECONDS=0 python3 /tmp/probe.py", timeout_sec=30, wait_timeout_sec=20
            )
            record["exit_code"] = output.return_code
            record["stdout"] = output.stdout
            record["stderr"] = output.stderr
            assert output.return_code == 0, "The container CUDA kernel must succeed."
            result = json.loads(output.stdout.strip())
            assert result["uuid_hex"] == uuid.UUID(before[expected]["uuid"].removeprefix("GPU-")).hex, (
                "Wrong GPU UUID."
            )
            assert result["kernel_output"] == 42, "CUDA readback must equal 42."
        finally:
            try:
                await environment.stop(delete=True)
            finally:
                await handle.release()
                overlay_path.unlink(missing_ok=True)
            save(out / "evidence.json", evidence)

    def pressure(name, gpu, gib):
        path = out / name
        path.mkdir()
        holder = ResidentPressure(gpu, gib, path, lifetime=240)
        pressures.append(holder)
        return holder, holder.start()

    try:
        async with asyncio.timeout(210):
            _, evidence["protected_pressure"] = await asyncio.to_thread(pressure, "protected-pressure", indices[0], 1)
            await run_case("dedicated-first", indices[1])
            holder, evidence["dedicated_pressure"] = await asyncio.to_thread(
                pressure, "dedicated-pressure", indices[1], 3
            )
            await run_case("fallback-under-pressure", indices[0])
            evidence["dedicated_pressure_stop"] = await asyncio.to_thread(holder.stop)
            await run_case("dedicated-restored", indices[1])
            snapshots = await router.snapshot.remote()
            evidence["leases_after"] = sum(s.leases_active for s in snapshots)
            assert evidence["leases_after"] == 0, "All experiment leases must be released."
            evidence["passed"] = True
    finally:
        for holder in pressures:
            if holder.process is not None and holder.process.poll() is None:
                await asyncio.to_thread(holder.stop)
        try:
            await asyncio.wait_for(router.shutdown_nodes.remote(), timeout=30)
        finally:
            ray.kill(router, no_restart=True)
            ray.shutdown()
            for unit in units:
                owned = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={unit['unit_id']}").split()
                if owned:
                    docker("rm", "-f", *owned)
            deadline = time.monotonic() + 20
            while True:
                final = probe.read_gpu_snapshot(indices)
                if all(
                    not reading["pids"] and reading["used_mb"] <= before[index]["used_mb"] + 16
                    for index, reading in final.items()
                ):
                    break
                if time.monotonic() >= deadline:
                    evidence["passed"] = False
                    break
                await asyncio.sleep(0.5)
            evidence["after"] = final
            evidence["containers_after"] = {
                unit["unit_id"]: docker("ps", "-aq", "--filter", f"label=psrl.env_unit={unit['unit_id']}")
                for unit in units
            }
            evidence["elapsed_s"] = time.monotonic() - started
            save(out / "evidence.json", evidence)
            probe.close()
    assert evidence["passed"], "GPU preference or physical cleanup verification failed."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--protected-gpu", type=int, required=True)
    parser.add_argument("--dedicated-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
