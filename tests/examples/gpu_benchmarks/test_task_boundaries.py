"""Check invalid submissions and owned service cleanup without GPU dependencies."""

import json
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
from examples.gpu_benchmarks.mix import prepare
from examples.gpu_benchmarks.reward import compute_score
from examples.inference_bench import local_verify
from examples.mle_bench.local_grade import grade
from examples.re_bench import verify


@pytest.mark.parametrize(
    "body",
    [b"id,label\na,1,extra\nb,2\n", b"id,label\na,1\nb,2\nc,3\n", b'id,label\na,"1\nb,2\n', b"\xff"],
)
def test_malformed_csv_is_a_scored_invalid_submission(tmp_path, body):
    submission = tmp_path / "submission.csv"
    submission.write_bytes(body)
    result = grade(submission, {"a": 1, "b": 2})
    assert result["reward"] == 0 and result["valid_submission"] == 0, "Reject malformed candidate data."


def test_missing_re_submission_is_zero_without_claiming_judge_success(tmp_path, monkeypatch):
    monkeypatch.setenv("REBENCH_MODE", "bounded-local")
    monkeypatch.setenv("REBENCH_JUDGE_BACKEND", "local")
    verify.main(tmp_path / "missing", tmp_path)
    rewards = json.loads((tmp_path / "reward.json").read_text())
    assert rewards["eval_valid"] == 1 and rewards["valid_submission"] == 0, "Grade missing submissions."
    assert rewards["judge_valid"] == 0 and rewards["official_evaluation"] == 0, "Do not invent judge results."
    score = compute_score("re_bench_local", "", "", {"harbor_rewards": rewards})
    assert score["score"] == 0, "Preserve the local reward gate."


def test_re_protocol_misconfiguration_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("REBENCH_MODE", "bounded-local")
    monkeypatch.setenv("REBENCH_JUDGE_BACKEND", "replicate")
    with pytest.raises(ValueError, match="local judge"):
        verify.main(tmp_path, tmp_path)
    assert not (tmp_path / "reward.json").exists(), "Do not grade a broken verifier configuration."


@pytest.mark.parametrize(
    "provenance,source",
    [
        ({"benchmark": "mle_bench_local"}, "mle_bench"),
        ({"benchmark": "inference_bench_local"}, "inference_bench"),
        ({"benchmark": "re_bench", "judge": {"backend": "local"}}, "re_bench"),
    ],
)
def test_mix_rejects_local_tasks_labelled_as_official(tmp_path, provenance, source):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("[environment]\ngpus=1\n")
    (task / "provenance.json").write_text(json.dumps(provenance))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [{"task_path": str(task), "environment_backend": "docker", "data_source": source, "gpu_memory_mb": 1}]
        )
    )
    with pytest.raises(ValueError, match="scoring identity"):
        prepare(manifest, tmp_path / "output")
    assert not (tmp_path / "output").exists(), "Reject protocol mismatches before publishing a dataset."


def test_missing_inference_submission_is_scored_without_starting_service(tmp_path):
    local_verify.main(tmp_path / "missing.sh", tmp_path)
    reward = json.loads((tmp_path / "reward.json").read_text())
    assert reward["eval_valid"] == 1 and reward["quality_pass"] == 0, "A missing service must receive zero."


def test_service_cleanup_kills_descendants_after_shell_exit(monkeypatch):
    signals = []
    monkeypatch.setattr(local_verify.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    service = SimpleNamespace(pid=123, wait=lambda **kwargs: 0)
    local_verify.stop_service(service)
    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)], "Clean the group after its leader exits."


def test_service_cleanup_reaps_a_real_cpu_process():
    service = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        local_verify.stop_service(service, timeout=1)
        assert service.poll() is not None, "Reap the owned CPU process."
    finally:
        if service.poll() is None:
            service.kill()
            service.wait(timeout=2)
