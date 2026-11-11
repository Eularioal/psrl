"""Check a combined operator subset with one persistent official evaluator."""

import argparse
import json
from pathlib import Path


def main(root, output):
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    traces = TraceSet.from_path(root)
    benchmark = Benchmark(
        traces,
        BenchmarkConfig.default(timeout_seconds=15, profile_baseline=False, warmup_runs=0, iterations=1, num_trials=1),
    )
    try:
        result = benchmark.run_all(dump_traces=False)
        rows = [row.model_dump(mode="json") for group in result.traces.values() for row in group]
        (output / "operator-results.json").write_text(json.dumps(rows, indent=2, default=str))
        (output / "reward.json").write_text(json.dumps({"reward": 0.0, "quick_check": 1, "evaluations": len(rows)}))
    finally:
        benchmark.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    main(args.root, args.output)
