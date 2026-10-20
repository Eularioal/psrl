"""Bounded real CUDA comparison using production ledger, node fences and scheduler."""

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path
from types import SimpleNamespace

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


async def command(*args, check=True):
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if check and process.returncode:
        raise RuntimeError(f"{args[:2]}: {stderr.decode()} {stdout.decode()}")
    return process.returncode, stdout.decode(), stderr.decode()


class Remote:
    """Use the production node directly without a Ray cluster for this small probe."""

    def __init__(self, node):
        self.node = node

    def __getattr__(self, name):
        return SimpleNamespace(remote=getattr(self.node, name))


async def run(args, mode):
    unit = EnvUnit("probe", "local", 8, 16384, (args.gpu,), 4, 81920, gpu_reservation=mode)
    ledger = EnvUnitLedger([unit], NodeRules.build(policy="any"), lease_ttl_s=300)
    node = EnvNodeService(None, "local")
    await node.check_gpu_command_support()
    snapshot = await node.gpu_lending_snapshot((args.gpu,))
    if snapshot[args.gpu]["used_mb"] > 64:
        raise RuntimeError("Selected GPU is already occupied.")

    async def get_node(ip):
        return Remote(node)

    scheduler = GpuCommandScheduler(ledger, get_node)
    libs = resolve_driver_libs(read_kernel_driver_version())
    events, leases = [], []
    start = time.monotonic()

    def event(task, kind, **extra):
        events.append(
            dict(
                task=task,
                event=kind,
                t=time.monotonic() - start,
                units=[dataclasses.asdict(s) for s in ledger.snapshot()],
                **extra,
            )
        )

    async def task(name):
        event(name, "queued")
        while True:
            admission = ledger.try_admit(name, EnvDemand(2, 2048, True, 52224), "local")
            if admission.lease is not None:
                break
            await asyncio.sleep(0.1)
        lease = admission.lease
        leases.append(lease)
        main = build_lease_overlay(lease, libs)["services"]["main"]
        if mode == "command":
            main.update(command_security_options())
        argv = ["docker", "run", "-d", "--network=none", "--cpus=2", "--memory=2g"]
        for key, flag in [
            ("devices", "--device"),
            ("volumes", "-v"),
            ("labels", "--label"),
            ("environment", "-e"),
            ("cap_drop", "--cap-drop"),
            ("security_opt", "--security-opt"),
        ]:
            for value in main.get(key, []):
                argv += [flag, value]
        pressure = Path(__file__).with_name("cuda_pressure.py").resolve()
        argv += ["-v", f"{pressure}:/probe.py:ro", "--entrypoint", "sleep", args.image, "infinity"]
        _, cid, _ = await command(*argv)
        cid = cid.strip()
        event(name, "container_started", container=cid, lease=lease.lease_id)
        await command("docker", "exec", cid, "sh", "-c", "echo retained > /tmp/task-state")
        probe = [
            "docker",
            "exec",
            cid,
            "python3",
            "/probe.py",
            "--gib",
            "50",
            "--hold",
            "3",
            "--output",
            "/tmp/probe.json",
        ]
        if mode == "command":
            denied = await command(*probe, check=False)
            if denied[0] == 0:
                raise AssertionError("CUDA succeeded without a GPU grant.")
            event(name, "initial_cuda_denied", output=denied[1])
        for phase in range(2):
            request = f"{name}-{phase}"
            event(name, "gpu_requested", phase=phase)
            if mode == "command":
                while True:
                    admission = await scheduler.acquire(lease.lease_id, cid, request)
                    if admission["status"] == "granted":
                        break
                    await asyncio.sleep(0.1)
            event(name, "gpu_granted", phase=phase)
            _, output, _ = await command(*probe)
            event(name, "cuda_finished", phase=phase, result=json.loads(output))
            if mode == "command":
                await scheduler.release(lease.lease_id, request, admission["epoch"])
                denied = await command(*probe, check=False)
                if denied[0] == 0:
                    raise AssertionError("CUDA succeeded after GPU return.")
                event(name, "returned_cuda_denied", phase=phase, output=denied[1])
            event(name, "gpu_released", phase=phase)
            if phase == 0:
                event(name, "cpu_gap_start")
                _, output, _ = await command("docker", "exec", cid, "sh", "-c", "sleep 12; cat /tmp/task-state")
                assert output.strip() == "retained", "Container state must survive GPU return."
                event(name, "cpu_gap_end")
        if mode == "command":
            await scheduler.finish(lease.lease_id, lease.unit_id)
        else:
            await node.remove_gpu_lease(lease.lease_id)
        ledger.release(lease.lease_id)
        event(name, "task_finished")

    try:
        async with asyncio.timeout(180):
            await asyncio.gather(task("A"), task("B"))
    finally:
        for lease in leases:
            await node.remove_gpu_lease(lease.lease_id)
        final = await node.gpu_lending_snapshot((args.gpu,))
        result = dict(mode=mode, duration_s=time.monotonic() - start, events=events, gpu_final=final)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{mode}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(mode=mode, duration_s=result["duration_s"], gpu_final=final)), flush=True)


async def main(args):
    for mode in args.modes:
        await run(args, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["episode", "command"])
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
