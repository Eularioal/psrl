"""Check actual Harbor Compose start and exec_gpu without loading a language model."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gpu_command_handoff import Remote
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.command_gpu import GpuCommandScheduler
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.types import EnvDemand, EnvUnit


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    unit = EnvUnit("harbor-command-probe", "local", 4, 4096, (args.gpu,), 2, 81920, gpu_reservation="command")
    node = EnvNodeService(None, "local")
    readings = await node.gpu_lending_snapshot(unit.gpu_indices)
    assert readings[args.gpu]["used_mb"] < 64, "Probe GPU must be idle."
    ledger = EnvUnitLedger([unit], NodeRules.build(policy="any"))
    lease = ledger.try_admit("probe", EnvDemand(2, 2048, True, 52224), "local").lease

    async def get_node(ip):
        return Remote(node)

    scheduler = GpuCommandScheduler(ledger, get_node)
    router = SimpleNamespace(
        acquire_gpu_command=SimpleNamespace(remote=scheduler.acquire),
        release_gpu_command=SimpleNamespace(remote=scheduler.release),
        cancel_gpu_command=SimpleNamespace(remote=scheduler.cancel),
    )
    overlay = build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version()))
    overlay_path = write_overlay(overlay)
    (args.output / "environment").mkdir(exist_ok=True)
    env = ManagedDockerEnvironment(
        environment_dir=args.output / "environment",
        environment_name="phase-probe",
        session_id=uuid.uuid4().hex[:12],
        trial_paths=TrialPaths(args.output / "trial"),
        task_env_config=EnvironmentConfig(docker_image=args.image, cpus=2, memory_mb=2048),
        extra_docker_compose=[overlay_path],
        lease_overlay=overlay,
        gpu_commands={"router_name": "probe", "lease_id": lease.lease_id},
        network_policy=NetworkPolicy(network_mode="no-network"),
    )
    evidence = {"scope": "Real Harbor Compose and CUDA, in-process scheduler/node RPC adapter"}
    try:
        async with asyncio.timeout(180):
            await env.start(force_build=False)
            await env.upload_file(Path(__file__).with_name("cuda_pressure.py"), "/tmp/probe.py")
            command = "python3 /tmp/probe.py --gib 50 --hold 1 --output /tmp/result.json"
            with patch("ray.get_actor", return_value=router):
                evidence["phases"] = []
                for _ in range(0 if args.background_leak else 2):
                    denied = await env.exec(command, timeout_sec=15)
                    assert denied.return_code != 0, "GPU access must be denied before grant."
                    result = await env.exec_gpu(command, timeout_sec=30, wait_timeout_sec=30)
                    assert result.return_code == 0, result
                    evidence["phases"].append({"timing": env.last_gpu_command, "cuda": json.loads(result.stdout)})
            if args.background_leak:
                source = Path(__file__).with_name("cuda_pressure.py").read_text()
                source = source.replace(
                    'result["allocated_at"] = time.time()',
                    'result["allocated_at"] = time.time()\n        Path("/tmp/gpu-ready").touch()',
                )
                leak = args.output / "cuda_leak.py"
                leak.write_text(source)
                await env.upload_file(leak, "/tmp/leak.py")
                shell = (
                    "python3 /tmp/leak.py --gib 1 --hold 30 --output /tmp/leak.json > /tmp/leak.log 2>&1 & "
                    "for i in $(seq 1 50); do test -f /tmp/gpu-ready && exit 0; sleep 0.1; done; exit 1"
                )
                with patch("ray.get_actor", return_value=router):
                    try:
                        await env.exec_gpu(shell, timeout_sec=15, wait_timeout_sec=30)
                    except RuntimeError as exc:
                        evidence["expected_return_failure"] = str(exc)
                    else:
                        raise AssertionError("A background CUDA owner must prevent quota return.")
                evidence["quarantined"] = unit.unit_id in ledger._blocked_units
                evidence["retained_gpu_quota_mib"] = ledger.snapshot()[0].declared_gpu_memory_mb
                assert evidence["quarantined"] and evidence["retained_gpu_quota_mib"] == 52224
            cpu = await env.exec("echo cpu-still-alive", timeout_sec=10)
            assert cpu.stdout.strip() == "cpu-still-alive"
            evidence["passed"] = True
    finally:
        await env.stop(delete=True)
        await scheduler.finish(lease.lease_id, lease.unit_id)
        ledger.release(lease.lease_id)
        overlay_path.unlink(missing_ok=True)
        evidence["final_gpu"] = await node.gpu_lending_snapshot(unit.gpu_indices)
        (args.output / "result.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-leak", action="store_true")
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
