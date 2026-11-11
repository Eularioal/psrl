"""Collect official candidate measurements for bounded offline screening analysis."""

import argparse
import asyncio
import dataclasses
import json
import shlex
import time
from pathlib import Path


def collect():
    import torch
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    torch.manual_seed(20261109)
    traces = TraceSet.from_path("/work/subset")
    definition = "rmsnorm_h2048"
    original = list(traces.solutions[definition])
    workloads = list(traces.workloads[definition])
    expected = {(t.workload.uuid, s.name) for t in workloads for s in original}
    assert len(expected) == 8, "Require two shapes and four candidates."
    config = BenchmarkConfig.default(timeout_seconds=60)
    benchmark = Benchmark(traces, config)
    record = {"config": config.model_dump(mode="json"), "rounds": []}
    try:
        for index in range(13):
            offset = index % len(original)
            traces.solutions[definition] = original[offset:] + original[:offset]
            traces.workloads[definition] = workloads if index % 2 == 0 else list(reversed(workloads))
            started = time.monotonic()
            output = benchmark.run_all(dump_traces=False)
            rows = [row.model_dump(mode="json") for values in output.traces.values() for row in values]
            passed = (
                len(rows) == len(expected)
                and {(r["workload"]["uuid"], r["solution"]) for r in rows} == expected
                and all(r["evaluation"]["status"] == "PASSED" for r in rows)
            )
            record["rounds"].append(
                {"index": index, "warmup": index == 0, "wall_s": time.monotonic() - started, "rows": rows}
            )
            record["passed"] = passed and index == 12
            Path("/work/measurements.json").write_text(json.dumps(record, indent=2))
            print(json.dumps({"round": index, "passed": passed, "wall_s": record["rounds"][-1]["wall_s"]}), flush=True)
            if not passed:
                raise RuntimeError("Candidate portability or correctness failed. No silent exclusions are allowed.")
    finally:
        benchmark.close()


async def experiment(environment, python, root, record, router, gpu):
    from flashinfer_env_probe import gpu_snapshot

    command = shlex.join([python, "/work/flashinfer_racing_probe.py", "--worker"])
    started = time.monotonic()
    result = await environment.exec_gpu(command, timeout_sec=420, wait_timeout_sec=30, overlap_return=False)
    (root / "worker.stdout").write_text(result.stdout or "")
    (root / "worker.stderr").write_text(result.stderr or "")
    record["collection_command_s"] = time.monotonic() - started
    record["command_timing"] = environment.last_gpu_command
    if result.return_code:
        raise RuntimeError("Collection failed. Inspect worker logs and partial measurements.")
    data = json.loads((root / "measurements.json").read_text())
    assert data["passed"] and len(data["rounds"]) == 13, "Require complete measurement evidence."
    record["ledger_after_command"] = [dataclasses.asdict(x) for x in await router.snapshot.remote()]
    record["gpu_after_command"] = gpu_snapshot(gpu)
    assert not record["gpu_after_command"]["pids"], "GPU processes must exit at the command boundary."
    assert all(
        x["gpu_commands_active"] == 0 and x["declared_gpu_memory_mb"] == 0 for x in record["ledger_after_command"]
    ), "The command must return its GPU quota."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--node-ip")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    args = parser.parse_args()
    if args.worker:
        collect()
    else:
        if not all((args.root, args.upstream, args.node_ip)):
            parser.error("Host execution requires root, upstream and node-ip.")
        from flashinfer_env_probe import run

        args.budget_s = 600
        asyncio.run(run(args, experiment))
