"""Measure official workload boundaries with one evaluator retained across splits."""

import argparse
import json
import time
from pathlib import Path


def main(args):
    started = time.monotonic()
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet
    from flashinfer_bench.data import EvaluationStatus

    imported = time.monotonic()
    traces = TraceSet.from_path("/work/subset")
    definition = "rmsnorm_h2048"
    workloads = list(traces.workloads[definition])
    expected = {(w.workload.uuid, s.name) for w in workloads for s in traces.solutions[definition]}
    benchmark = Benchmark(traces, BenchmarkConfig.default(timeout_seconds=90))
    original = benchmark._runner.run_workload
    boundaries = []

    def measured(definition, workload, solutions, config, root):
        before = time.monotonic()
        result = original(definition, workload, solutions, config, root)
        boundaries.append({"uuid": workload.uuid, "seconds": time.monotonic() - before})
        return result

    benchmark._runner.run_workload = measured
    rows = []
    try:
        parts = [workloads] if args.mode == "whole" else [workloads[:1], workloads[1:]]
        for part in parts:
            benchmark._trace_set.workloads[definition] = part
            result = benchmark.run_all(dump_traces=False)
            rows.extend(t for values in result.traces.values() for t in values)
    finally:
        benchmark.close()
    assert len(rows) == len(expected), "Splitting must preserve the number of evaluations."
    assert {(t.workload.uuid, t.solution) for t in rows} == expected, "Every expected pair must be evaluated."
    assert all(t.evaluation.status == EvaluationStatus.PASSED for t in rows), "Official correctness must pass."
    args.output.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "import_s": imported - started,
                "total_s": time.monotonic() - started,
                "boundaries": boundaries,
                "rows": [t.model_dump(mode="json") for t in rows],
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("whole", "split"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
