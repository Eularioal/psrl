"""Separate command overhead from process reuse on official FlashInfer workloads."""

import argparse
import asyncio
import dataclasses
import json
import shlex
import time
from pathlib import Path

from flashinfer_env_probe import gpu_snapshot, run


async def experiment(environment, python, root, record, router, gpu):
    groups = []
    plans = ("separate", "one_command_fresh", "one_process", "one_process", "one_command_fresh", "separate")
    pairs = [(size, solution) for size in (1, 16254) for solution in ("flashinfer_wrapper_0af255", "torch_reference")]
    for index, mode in enumerate(plans):
        ordered = pairs if index < 3 else list(reversed(pairs))
        commands, outputs = [], []
        for number, (size, solution) in enumerate(ordered if mode != "one_process" else [(None, None)]):
            name = f"group-{index}-{number}.json"
            argv = [python, "/work/flashinfer_batch_worker.py", "--output", f"/work/{name}"]
            if size is not None:
                argv += ["--batch", str(size), "--solution", solution]
            elif index >= 3:
                argv.append("--reverse")
            commands.append(shlex.join(argv))
            outputs.append(name)
        if mode == "one_command_fresh":
            commands = [" && ".join(commands)]
        group = {"mode": mode, "index": index, "commands": [], "outputs": outputs}
        started = time.monotonic()
        for number, command in enumerate(commands):
            before = time.monotonic()
            result = await environment.exec_gpu(command, timeout_sec=240, wait_timeout_sec=30, overlap_return=False)
            (root / f"group-{index}-command-{number}.stderr").write_text(result.stderr or "")
            group["commands"].append({"wall_s": time.monotonic() - before, "timing": environment.last_gpu_command})
            if result.return_code:
                raise RuntimeError(f"Experiment command failed in group {index}: {result.stderr!r}.")
        group["wall_s"] = time.monotonic() - started
        group["results"] = [json.loads((root / name).read_text()) for name in outputs]
        rows = [row for output in group["results"] for row in output["rows"]]
        actual = {(r["workload"]["axes"]["batch_size"], r["solution"]) for r in rows}
        if actual != set(pairs) or len(rows) != 4 or not all(r["passed"] for r in group["results"]):
            raise RuntimeError("Experiment returned incomplete or incorrect results.")
        group["ledger"] = [dataclasses.asdict(x) for x in await router.snapshot.remote()]
        group["gpu_after"] = gpu_snapshot(gpu)
        if any(x["declared_gpu_memory_mb"] or x["gpu_commands_active"] for x in group["ledger"]):
            raise RuntimeError("GPU command quota has not been returned.")
        if group["gpu_after"]["pids"] or group["gpu_after"]["used_mb"] > 128:
            raise RuntimeError("GPU has not returned to idle between groups.")
        groups.append(group)
        (root / "groups.json").write_text(json.dumps(groups, indent=2, default=str))
        print(json.dumps({"completed_group": index, "mode": mode, "wall_s": group["wall_s"]}), flush=True)
    record["experiment_groups"] = len(groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    args = parser.parse_args()
    args.budget_s = 900
    asyncio.run(run(args, experiment))
