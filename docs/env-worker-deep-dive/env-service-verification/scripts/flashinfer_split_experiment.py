"""Run a bounded whole versus retained-evaluator split check through native Env."""

import argparse
import asyncio
import json
import shlex
import time
from pathlib import Path

from flashinfer_env_probe import run


async def experiment(environment, python, root, record, router, gpu):
    record["split_runs"] = []
    for index, mode in enumerate(("whole", "split", "split", "whole")):
        output = f"split-{index}.json"
        command = shlex.join(
            [python, "/work/flashinfer_split_worker.py", "--mode", mode, "--output", f"/work/{output}"]
        )
        started = time.monotonic()
        result = await environment.exec_gpu(command, timeout_sec=100, wait_timeout_sec=20, overlap_return=False)
        (root / f"split-{index}.stderr").write_text(result.stderr or "")
        assert result.return_code == 0, f"Official evaluation failed: {result.stderr!r}."
        evidence = json.loads((root / output).read_text())
        evidence["command_with_return_s"] = time.monotonic() - started
        record["split_runs"].append(evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    parser.add_argument("--budget-s", type=int, default=360)
    asyncio.run(run(parser.parse_args(), experiment=experiment))
