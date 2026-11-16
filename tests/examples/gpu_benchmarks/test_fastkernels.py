"""Prevent skipped upstream scenarios from counting as functional GPU coverage."""

from examples.fastkernels_bench.verify import score
from examples.gpu_benchmarks.reward import compute_score


def test_skipped_empty_and_wrong_identity_are_not_valid():
    case = {
        "op": "fastkernels.tasks.baseline.L1.linear:Linear",
        "status": "PASSED",
        "speedup": 1.2,
        "candidate_ms": 0.2,
        "baseline_ms": 0.24,
    }
    assert score({"scenarios": [case]}, "Linear")["eval_valid"] == 1
    for report in (
        {},
        {"scenarios": [{**case, "status": "SKIPPED"}]},
        {"scenarios": [{**case, "op": "different:Matmul"}]},
        {"scenarios": [{**case, "baseline_ms": 0}]},
    ):
        assert score(report, "Linear")["eval_valid"] == 0
    for key in ("eval_valid", "cuda", "local_evaluation", "cases"):
        metrics = {"reward": 2, "eval_valid": 1, "cuda": 1, "local_evaluation": 1, "cases": 1, "skipped": 0}
        metrics[key] = 0
        result = compute_score("fastkernels_local", "", None, {"harbor_rewards": metrics})
        assert result["score"] == 0
