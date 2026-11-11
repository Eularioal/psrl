"""Explain recorded GPU experiments using CPU analysis only."""

import json
from pathlib import Path


def main():
    evidence = Path(__file__).resolve().parents[2] / "experiments/gpu-benchmarks"
    batching = json.loads((evidence / "flashinfer-batching-evidence.json").read_text())
    screening = json.loads((evidence / "flashinfer-screening-evidence.json").read_text())
    print("Recorded results only. This script launches no GPU work.")
    for mode, row in batching["summary"].items():
        mean = sum(row["wall_s"]) / len(row["wall_s"])
        print(f"{mode}: mean group time {mean:.2f} s")
    rounds = screening["measurements"]["rounds"]
    actual = sum(len(block["rows"]) for block in rounds)
    print(f"Collected evaluations: {len(rounds)} blocks x 8 pairs = {actual}")
    for shape, result in screening["results"].items():
        print(f"Batch {shape}:")
        for policy, count in result["decision_evaluation_counts"].items():
            print(f"  {policy}: uses {count} observations, winner {result['winners'][policy]}")
    print("Observation counts are offline replay budgets, not measured time savings.")


if __name__ == "__main__":
    main()
