"""Evaluate fixed candidate workloads in a fresh or reused official worker."""

import argparse
import json
import time
from pathlib import Path


def main(args):
    started = time.monotonic()
    import torch
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet
    from flashinfer_bench.data import EvaluationStatus

    imported = time.monotonic()
    torch.manual_seed(20261109)
    traces = TraceSet.from_path("/work/subset")
    definition = "rmsnorm_h2048"
    if args.reverse:
        traces.workloads[definition].reverse()
        traces.solutions[definition].reverse()
    if args.batch is not None:
        traces.workloads[definition] = [
            t for t in traces.workloads[definition] if t.workload.axes["batch_size"] == args.batch
        ]
    config = BenchmarkConfig.default(timeout_seconds=90, solutions=[args.solution] if args.solution else None)
    expected = {
        (t.workload.uuid, s.name)
        for t in traces.workloads[definition]
        for s in traces.solutions[definition]
        if not args.solution or s.name == args.solution
    }
    assert expected, "Expected workload set must be nonempty."
    benchmark = Benchmark(traces, config)
    try:
        output = benchmark.run_all(dump_traces=False)
        rows = [t for values in output.traces.values() for t in values]
        passed = (
            len(rows) == len(expected)
            and {(t.workload.uuid, t.solution) for t in rows} == expected
            and all(t.evaluation.status == EvaluationStatus.PASSED for t in rows)
        )
        result = {
            "passed": passed,
            "import_s": imported - started,
            "evaluation_s": time.monotonic() - imported,
            "config": config.model_dump(mode="json"),
            "rows": [t.model_dump(mode="json") for t in rows],
        }
    finally:
        benchmark.close()
    result["worker_total_s"] = time.monotonic() - started
    args.output.write_text(json.dumps(result, indent=2))
    assert passed, "Every expected official correctness evaluation must pass."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
