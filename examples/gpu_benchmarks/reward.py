"""Preserve benchmark metrics while rejecting invalid numerical rewards."""

import math


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    metrics = extra_info.get("harbor_rewards") or {}
    key = extra_info.get("reward_key", "reward")
    score = float(metrics.get(key, 0.0))
    if not math.isfinite(score):
        raise ValueError("Benchmark reward must be finite.")
    if data_source == "re_bench" and (metrics.get("official_evaluation") != 1 or metrics.get("judge_valid") != 1):
        score = 0.0
    if data_source == "re_bench_local" and (
        metrics.get("local_evaluation") != 1 or metrics.get("judge_valid") != 1 or metrics.get("eval_valid") != 1
    ):
        score = 0.0
    if data_source == "re_bench_smoke":
        score = 0.0
    if data_source == "inference_bench" and (
        metrics.get("judge_valid") != 1
        or metrics.get("quality_pass") != 1
        or metrics.get("eval_valid") != 1
        or metrics.get("contamination_detected", 1) != 0
        or metrics.get("disallowed_use_detected", 1) != 0
    ):
        score = 0.0
    return {"score": score, "reward_extra_info": {"benchmark": data_source, "metrics": metrics}}
