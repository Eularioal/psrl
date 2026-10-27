"""
Verify delayed CUDA allocation with the production scheduler and three owned containers.
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from gpu_command_handoff import Remote, command
from psrl.workers.env_service.command_gpu import GpuCommandScheduler
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    command_security_options,
    read_kernel_driver_version,
    resolve_driver_libs,
)
from psrl.workers.env_service.types import EnvDemand, EnvUnit
from resident_gpu_pressure import ResidentPressure


async def run(args, report):
    node = EnvNodeService(None, "local")
    holder = None
    leases, containers = [], []
    unit = EnvUnit("memory-check", "local", 6, 6144, (args.gpu,), 3, 90 * 1024, gpu_reservation="command")
    ledger = EnvUnitLedger([unit], NodeRules.build(policy="any"), lease_ttl_s=300)

    async def get_node(_ip):
        return Remote(node)

    scheduler = GpuCommandScheduler(ledger, get_node)
    try:
        await node.check_gpu_command_support()
        initial = await node.gpu_lending_snapshot((args.gpu,))
        report["initial"] = initial
        if initial[args.gpu]["used_mb"] > 64:
            raise RuntimeError("Selected GPU is occupied. No experiment resources were allocated.")
        libs = resolve_driver_libs(read_kernel_driver_version())
        probe = Path(__file__).with_name("cuda_pressure.py").resolve()
        for i in range(3):
            lease = ledger.try_admit(str(i), EnvDemand(2, 2048, True, 15 * 1024), "local").lease
            leases.append(lease)
            main = build_lease_overlay(lease, libs)["services"]["main"]
            main.update(command_security_options())
            argv = [
                "docker",
                "run",
                "-d",
                "--network=none",
                "--cpus=2",
                "--memory=2g",
                "--name",
                f"psrl-memory-{uuid.uuid4().hex[:10]}",
            ]
            for key, flag in (
                ("devices", "--device"),
                ("volumes", "-v"),
                ("labels", "--label"),
                ("environment", "-e"),
                ("cap_drop", "--cap-drop"),
                ("security_opt", "--security-opt"),
            ):
                for value in main.get(key, []):
                    argv += [flag, value]
            argv += ["-v", f"{probe}:/probe.py:ro", "--entrypoint", "sleep", args.image, "infinity"]
            _, cid, _ = await command(*argv)
            containers.append(cid.strip())
        report["containers"] = containers
        holder = ResidentPressure(args.gpu, 70, args.output, lifetime=180)
        report["resident"] = await asyncio.to_thread(holder.start)
        report["events"] = []
        first = await scheduler.acquire(leases[0].lease_id, containers[0], "0")
        assert first["status"] == "granted", "The first reservation must fit."
        for i in (1, 2):
            result = await scheduler.acquire(leases[i].lease_id, containers[i], str(i))
            report["events"].append({"kind": "before_first_allocation", "request": i, "result": result})
            assert result["status"] == "wait", "Pending memory must block followers before CUDA starts."
        for i, lease in enumerate(leases):
            result = first if i == 0 else await scheduler.acquire(lease.lease_id, containers[i], str(i))
            assert result["status"] == "granted", "Confirmed physical return must admit the next request."
            if i < 2:
                following = await scheduler.acquire(leases[i + 1].lease_id, containers[i + 1], str(i + 1))
                assert following["status"] == "wait", "Refill must preserve the newly granted reservation."
            _, stdout, _ = await command(
                "docker",
                "exec",
                containers[i],
                "python3",
                "/probe.py",
                "--gib",
                "14",
                "--hold",
                "2",
                "--output",
                "/tmp/pressure.json",
            )
            phase = json.loads(stdout.splitlines()[-1])
            report["events"].append({"kind": "cuda_finished", "request": i, "phase": phase})
            assert phase["status"] == "ok" and phase["kernel_output"] == 42, "Real CUDA allocation must succeed."
            await scheduler.release(lease.lease_id, str(i), result["epoch"])
        report["requests"] = [scheduler.reports(lease.lease_id) for lease in leases]
        assert holder.process.poll() is None, "Resident pressure must persist throughout all commands."
        report["passed"] = True
    finally:
        cleanup_errors = []
        for lease in leases:
            try:
                await scheduler.finish(lease.lease_id, lease.unit_id)
                ledger.release(lease.lease_id)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if holder is not None:
            report["resident_cleanup"] = await asyncio.to_thread(holder.stop)
        report["final"] = await node.gpu_lending_snapshot((args.gpu,))
        report["cleanup_errors"] = cleanup_errors
        await node.close()
        assert not cleanup_errors and report["final"][args.gpu]["used_mb"] <= 64, (
            "Owned containers and GPU allocations must be released."
        )


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "started_at": time.time(),
        "scope": "Real Docker/CUDA. Direct production node service, no Ray or training.",
    }
    try:
        async with asyncio.timeout(120):
            await run(args, report)
    except BaseException as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = time.time()
        (args.output / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
