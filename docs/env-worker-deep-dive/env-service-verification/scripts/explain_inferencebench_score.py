"""Explain simulated InferenceBench metrics and the real local PSRL reward gate."""

import json
import math
import runpy
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[4]
    compute_score = runpy.run_path(str(repo / "examples/gpu_benchmarks/reward.py"))["compute_score"]
    cases = []
    for name, correct, ttft, contamination in (
        ("reference", 400, 0.4, 0),
        ("candidate_pass", 390, 0.2, 0),
        ("candidate_quality_fail", 360, 0.2, 0),
        ("candidate_flagged", 390, 0.2, 1),
    ):
        # Steps 4 to 6 use teaching inputs, with complete and valid requests assumed.
        quality_ratio = (correct / 500) / (400 / 500)
        quality_pass = quality_ratio >= 0.95
        metrics = {
            "reward": 1 / ttft if quality_pass else 0.0,
            "eval_valid": 1,
            "speed_pass": 1,
            "quality_pass": int(quality_pass),
            "judge_valid": 1,
            "contamination_detected": contamination,
            "disallowed_use_detected": 0,
        }
        # Step 7 calls the actual PSRL gate without importing the training stack.
        result = compute_score("inference_bench", "", "", {"harbor_rewards": metrics})
        cases.append(
            {
                "case": name,
                "quality_ratio": quality_ratio,
                "simulated_harbor_reward": metrics["reward"],
                "actual_psrl_score": result["score"],
            }
        )
    print(
        json.dumps(
            {
                "mode": "teaching_simulation_with_real_local_reward_gate",
                "cases": cases,
                "simulated_scenario_c": round(math.prod((8, 27, 64)) ** (1 / 3), 6),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
