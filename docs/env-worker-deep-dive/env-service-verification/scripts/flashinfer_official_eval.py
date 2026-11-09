"""Run a pinned FlashInfer workload subset with the official evaluator."""

import json
import time
from pathlib import Path


def main():
    import torch
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet
    from flashinfer_bench.data import EvaluationStatus

    torch.manual_seed(20261105)
    traces = TraceSet.from_path("/work/subset")
    config = BenchmarkConfig.default(timeout_seconds=120)
    expected = {
        (definition, workload.workload.uuid, solution.name)
        for definition, workloads in traces.workloads.items()
        for workload in workloads
        for solution in traces.solutions[definition]
    }
    if not expected or len(expected) > 3:
        raise ValueError("This bounded pilot requires one to three evaluations.")
    benchmark = Benchmark(traces, config)
    started = time.monotonic()
    try:
        result = benchmark.run_all(dump_traces=False)
        rows = [trace for values in result.traces.values() for trace in values]
        actual = {(row.definition, row.workload.uuid, row.solution) for row in rows}
        evidence = {
            "config": config.model_dump(mode="json"),
            "device": torch.cuda.get_device_name(),
            "elapsed_s": time.monotonic() - started,
            "expected_count": len(expected),
            "results": [row.model_dump(mode="json") for row in rows],
            "passed": actual == expected
            and len(rows) == len(expected)
            and all(row.evaluation.status == EvaluationStatus.PASSED for row in rows),
            "scope": "Official random inputs at captured production shapes; not production tensor dumps.",
            "speedup_reference": "Official mathematical reference, not another optimized production kernel.",
        }
        Path("/work/official-results.json").write_text(json.dumps(evidence, indent=2))
        if not evidence["passed"]:
            raise RuntimeError("Official evaluation failed or returned incomplete results.")
    finally:
        benchmark.close()


if __name__ == "__main__":
    main()
