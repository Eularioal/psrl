"""Probe concurrent MPS control requests using an owned Docker service."""

import argparse
import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from psrl.workers.env_service.parallel_gpu import MpsDomain
from psrl.workers.env_service.probe import NvmlProbe


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    probe = NvmlProbe()
    before = probe.read_gpu_snapshot([args.gpu])[args.gpu]
    if before["pids"] or before["used_mb"] > 32:
        raise RuntimeError("Selected GPU is occupied.")
    unit = SimpleNamespace(
        unit_id="mps-control-" + uuid.uuid4().hex[:12], gpu_indices=(args.gpu,), mps_image=args.image
    )
    domain = MpsDomain(SimpleNamespace(_gpu_probe=lambda: probe), unit)
    record = {"unit": unit.unit_id, "before": before, "rounds": []}
    (args.output / "ownership.json").write_text(json.dumps({"daemon": domain.daemon, "pipe": str(domain.pipe)}))

    async def request(index, traced=False):
        started = time.monotonic()
        try:
            if traced:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "strace",
                        "-f",
                        "-tt",
                        "-o",
                        str(args.output / f"control-{index}.strace"),
                        "/usr/bin/nvidia-cuda-mps-control",
                    ],
                    input="get_server_list\n",
                    text=True,
                    capture_output=True,
                    timeout=10,
                    env={**os.environ, "CUDA_MPS_PIPE_DIRECTORY": str(domain.pipe)},
                )
                return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            return {
                "stdout": await asyncio.to_thread(domain.control, "get_server_list"),
                "seconds": time.monotonic() - started,
            }
        except Exception as exc:
            return {"error": repr(exc), "seconds": time.monotonic() - started}

    try:
        await domain.register(SimpleNamespace(lease_id="diagnostic"))
        record["serial"] = [await request(i) for i in range(8)]
        for batch in range(4):
            record["rounds"].append(await asyncio.gather(*(request(batch * 8 + i) for i in range(8))))
        if args.trace:
            record["traced"] = await asyncio.gather(*(request(i, traced=True) for i in range(16)))
        logs = subprocess.run(
            ["docker", "exec", domain.daemon, "cat", "/var/log/nvidia-mps/control.log"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        (args.output / "daemon.log").write_text(logs.stdout + logs.stderr)
        record["after_probe"] = await request(999)
        record["container_state"] = subprocess.check_output(
            ["docker", "inspect", domain.daemon, "--format", "{{json .State}}"], text=True, timeout=10
        )
    finally:
        domain.leases.clear()
        await domain.close()
        record["after"] = probe.read_gpu_snapshot([args.gpu])[args.gpu]
        probe.close()
        (args.output / "result.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    asyncio.run(asyncio.wait_for(run(parser.parse_args()), 90))
