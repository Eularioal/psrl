"""Execute the bounded feedback controller through native EnvWorker."""

import argparse
import asyncio
import dataclasses
import json
import shlex
import time
from pathlib import Path

from flashinfer_env_probe import gpu_snapshot, run


async def experiment(environment, python, root, record, router, gpu):
    command = shlex.join([python, "/work/worker.py"])
    started = time.monotonic()
    protocol = json.loads((root / "protocol.json").read_text())
    result = await environment.exec_gpu(
        command, timeout_sec=protocol["limits"]["command_s"], wait_timeout_sec=30, overlap_return=False
    )
    (root / "worker.stdout").write_text(result.stdout or "")
    (root / "worker.stderr").write_text(result.stderr or "")
    record["command_s"] = time.monotonic() - started
    record["return_code"] = result.return_code
    record["command_timing"] = environment.last_gpu_command
    if result.return_code:
        raise RuntimeError("Feedback experiment failed. Inspect worker.stderr and partial measurements.")
    evidence = json.loads((root / "measurements.json").read_text())
    assert evidence["passed"], "All preregistered comparisons must complete."
    record["ledger_after_command"] = [dataclasses.asdict(x) for x in await router.snapshot.remote()]
    record["gpu_after_command"] = gpu_snapshot(gpu)
    assert not record["gpu_after_command"]["pids"], "All experiment GPU processes must exit."
    assert all(
        x["gpu_commands_active"] == 0 and x["declared_gpu_memory_mb"] == 0 for x in record["ledger_after_command"]
    ), "Command accounting must return to zero."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    args = parser.parse_args()
    args.budget_s = json.loads((args.root / "protocol.json").read_text())["limits"]["env_s"]
    asyncio.run(run(args, experiment))
