"""Check a combined operator subset with one persistent official evaluator."""

import argparse
import json
from pathlib import Path


def main(root, output):
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    traces = TraceSet.from_path(root)
    config = BenchmarkConfig.default(
        timeout_seconds=15, profile_baseline=False, warmup_runs=0, iterations=1, num_trials=1
    )
    benchmark = Benchmark(traces, config)
    try:
        rows = []
        missing = []
        for name in traces.definitions:
            config.definitions = [name]
            config.timeout_seconds = 60 if traces.definitions[name].op_type == "sampling" else 15
            result = benchmark.run_all(dump_traces=False)
            records = [row.model_dump(mode="json") for group in result.traces.values() for row in group]
            rows.extend(records)
            if not records:
                missing.append(name)
                benchmark.close()
                benchmark = Benchmark(traces, config)
            (output / "operator-results.json").write_text(json.dumps(rows, indent=2, default=str))
            (output / "missing-definitions.json").write_text(json.dumps(missing))
        (output / "reward.json").write_text(json.dumps({"reward": 0.0, "quick_check": 1, "evaluations": len(rows)}))
    finally:
        benchmark.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    main(args.root, args.output)
