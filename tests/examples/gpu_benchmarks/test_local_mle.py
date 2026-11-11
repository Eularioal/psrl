"""Check held-out prediction completeness for the local GPU competition."""

from examples.mle_bench.local_grade import grade


def test_local_grader_rejects_duplicate_and_unexpected_ids(tmp_path):
    submission = tmp_path / "submission.csv"
    for body in ("id,label\na,1\na,1\n", "id,label\na,1\nc,2\n", "id,label\na,NaN\nb,2\n"):
        submission.write_text(body)
        assert grade(submission, {"a": 1, "b": 2})["valid_submission"] == 0, "Reject malformed predictions."
    submission.write_text("id,label\nb,2\na,3\n")
    assert grade(submission, {"a": 1, "b": 2})["reward"] == 0.5, "Grade by identity instead of row position."


def test_local_rewards_require_verified_protocol():
    from examples.gpu_benchmarks.reward import compute_score

    for source in ("mle_bench_local", "inference_bench_local"):
        result = compute_score(source, "", "", {"harbor_rewards": {"reward": 1}})
        assert result["score"] == 0, "A positive metric alone must not bypass local verification."


def test_mix_rejects_smoke_before_publishing_dataset(tmp_path):
    import json

    import pytest
    from examples.gpu_benchmarks.mix import prepare

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("[environment]\ngpus=1\n")
    (task / "provenance.json").write_text(json.dumps({"mode": "feedback-smoke"}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "task_path": str(task),
                    "data_source": "re_bench_local",
                    "environment_backend": "env_worker",
                    "gpu_memory_mb": 49152,
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="Smoke"):
        prepare(manifest, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists(), "Invalid protocols must not publish training data."
