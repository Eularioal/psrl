"""Preserve benchmark metrics while rejecting invalid numerical rewards."""

import math


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    metrics = extra_info.get("harbor_rewards") or {}
    key = extra_info.get("reward_key", "reward")
    score = float(metrics.get(key, 0.0))
    if not math.isfinite(score):
        raise ValueError("Benchmark reward must be finite.")
    if data_source == "kernel_bench_local" and (
        metrics.get("local_evaluation") != 1 or metrics.get("bounded_inputs") != 1
    ):
        score = 0.0
    if data_source == "re_bench" and (metrics.get("official_evaluation") != 1 or metrics.get("judge_valid") != 1):
        score = 0.0
    if data_source == "re_bench_local" and (
        metrics.get("local_evaluation") != 1 or metrics.get("judge_valid") != 1 or metrics.get("eval_valid") != 1
    ):
        score = 0.0
    if data_source == "mle_bench_local" and (
        metrics.get("local_evaluation") != 1 or metrics.get("valid_submission") != 1
    ):
        score = 0.0
    if data_source == "inference_bench_local" and (
        metrics.get("local_evaluation") != 1 or metrics.get("quality_pass") != 1 or metrics.get("eval_valid") != 1
    ):
        score = 0.0
    if data_source == "re_bench_smoke":
        score = 0.0
    if data_source == "fastkernels_local" and (
        metrics.get("local_evaluation") != 1
        or metrics.get("eval_valid") != 1
        or metrics.get("cuda") != 1
        or metrics.get("cases", 0) < 1
        or metrics.get("skipped", 0) != 0
    ):
        score = 0.0
    if data_source in ("mlgym_rl_local", "mlgym_vision_local", "mlgym_nlp_local", "posttrain_bench_local") and (
        metrics.get("local_evaluation") != 1 or metrics.get("eval_valid") != 1 or metrics.get("cuda") != 1
    ):
        score = 0.0
    if data_source == "posttrain_bench_judged_local" and (
        metrics.get("local_evaluation") != 1
        or metrics.get("eval_valid") != 1
        or metrics.get("cuda") != 1
        or metrics.get("judge_valid") != 1
        or metrics.get("generated_answers", 0) < 1
        or metrics.get("judge_calls", 0) < 1
    ):
        score = 0.0
    if data_source == "swe_serve_local" and (
        metrics.get("local_evaluation") != 1 or metrics.get("eval_valid") != 1 or metrics.get("tests") != 2
    ):
        score = 0.0
    if data_source == "swe_serve_regression_local" and (
        metrics.get("local_evaluation") != 1 or metrics.get("eval_valid") != 1 or metrics.get("tests", 0) < 2
    ):
        score = 0.0
    if data_source == "swe_serve_projection_local" and (
        metrics.get("local_evaluation") != 1
        or metrics.get("eval_valid") != 1
        or metrics.get("cuda") != 1
        or metrics.get("tests") != 3
    ):
        score = 0.0
    if data_source == "re_bench_cumsum_local" and (
        metrics.get("local_evaluation") != 1 or metrics.get("eval_valid") != 1 or metrics.get("correctness") != 1
    ):
        score = 0.0
    if data_source in (
        "re_bench_mlm_local",
        "re_bench_embedding_local",
        "re_bench_scaling_local",
        "re_bench_foundry_local",
        "re_bench_rust_local",
        "flashinfer_bench_local",
    ) and (metrics.get("local_evaluation") != 1 or metrics.get("eval_valid") != 1):
        score = 0.0
    if data_source == "re_bench_foundry_local" and metrics.get("weights_match") != 1:
        score = 0.0
    if data_source == "re_bench_rust_local" and metrics.get("gpu_inference_calls", 0) < 1:
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
