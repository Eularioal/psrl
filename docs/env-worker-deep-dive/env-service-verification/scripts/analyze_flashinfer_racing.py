"""Replay a predeclared screening rule without claiming counterfactual wall time."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median


def analyze(root):
    data = json.loads((root / "measurements.json").read_text())
    status = json.loads((root / "status.json").read_text())
    assert data["passed"] and len(data["rounds"]) == 13, "Require complete collection."
    assert status["status"] == "passed" and not status["cleanup_errors"], "Require successful cleanup."
    samples = {}
    for block in data["rounds"]:
        assert len(block["rows"]) == 8, "Every block must contain all candidate workloads."
        for row in block["rows"]:
            assert row["evaluation"]["status"] == "PASSED", "Every candidate must pass correctness."
            shape = row["workload"]["axes"]["batch_size"]
            candidate = row["solution"]
            samples.setdefault(shape, {}).setdefault(candidate, {})[block["index"]] = row["evaluation"]["performance"][
                "latency_ms"
            ]
    results = {}
    for shape, candidates in samples.items():

        def med(name, indices, values=candidates):
            return median(values[name][index] for index in indices)

        names = sorted(candidates)
        assert len(names) == 4
        kept = sorted(names, key=lambda n: (med(n, range(1, 5)), n))[:2]
        winners = {
            "four_rounds_only": min(names, key=lambda n: (med(n, range(1, 5)), n)),
            "uniform_eight": min(names, key=lambda n: (med(n, range(1, 9)), n)),
            "screen_then_refine": min(kept, key=lambda n: (med(n, range(1, 9)), n)),
            "holdout": min(names, key=lambda n: (med(n, range(9, 13)), n)),
        }
        best_holdout = med(winners["holdout"], range(9, 13))
        results[shape] = {
            "ranking_counts": [
                {"order": order, "count": count}
                for order, count in Counter(
                    tuple(sorted(names, key=lambda n: (candidates[n][index], n))) for index in range(1, 13)
                ).items()
            ],
            "kept": kept,
            "winners": winners,
            "decision_evaluation_counts": {"four_rounds_only": 16, "uniform_eight": 32, "screen_then_refine": 24},
            "holdout_relative_loss_percent": {
                k: 100 * (med(v, range(9, 13)) / best_holdout - 1) for k, v in winners.items() if k != "holdout"
            },
            "candidate_summary": {
                n: {
                    "screen_median_us": med(n, range(1, 5)) * 1000,
                    "holdout_median_us": med(n, range(9, 13)) * 1000,
                    "all_median_us": med(n, range(1, 13)) * 1000,
                    "all_min_us": min(candidates[n][i] for i in range(1, 13)) * 1000,
                    "all_max_us": max(candidates[n][i] for i in range(1, 13)) * 1000,
                    "round_wins": sum(n == min(names, key=lambda v: (candidates[v][i], v)) for i in range(1, 13)),
                }
                for n in names
            },
        }
    return {
        "scope": "Real measurements plus offline replay. Holdout medians are noisy references, not ground truth.",
        "results": results,
        "actual_evaluations": 104,
        "measurements": data,
        "formal_measurement_s": sum(r["wall_s"] for r in data["rounds"][1:]),
        "warmup_s": data["rounds"][0]["wall_s"],
        "actual_collection_command_s": status["collection_command_s"],
        "provenance": json.loads((root / "manifest.json").read_text()),
        "measurement_sha256": hashlib.sha256((root / "measurements.json").read_bytes()).hexdigest(),
        "cleanup": {
            "containers_remaining": status["containers_remaining"],
            "gpu_pids": status["gpu_after"]["pids"],
            "gpu_used_mb": status["gpu_after"]["used_mb"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["results"], indent=2))
