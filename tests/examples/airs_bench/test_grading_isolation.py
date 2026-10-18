"""Verify the public data boundary and trusted scoring input contract."""

from pathlib import Path

import pytest
import yaml
from examples.airs_bench.config import AirsBenchRuntimeConfig
from examples.airs_bench.grading import build_verifier_spec, parse_metrics, task_config
from psrl.environments.mlgym_env import build_sandbox_spec

pytestmark = pytest.mark.cpu_test


def prepared_task(tmp_path):
    root = tmp_path / "airs_configs"
    (root / "tasks").mkdir(parents=True)
    assets = root / "data" / "tiny"
    assets.mkdir(parents=True)
    (assets / "evaluate.py").write_text("print('trusted evaluator')\n")
    task = {"id": "tiny", "name": "Tiny", "description": "Predict labels.", "evaluation_paths": ["evaluate.py"]}
    (root / "tasks" / "tiny.yaml").write_text(yaml.safe_dump(task))
    data = tmp_path / "airs_prepared" / "tiny"
    for split in ("train", "test", "test_with_labels"):
        (data / split).mkdir(parents=True)
    return AirsBenchRuntimeConfig(data_root=str(tmp_path)), data


def test_agent_cannot_mount_labels_or_inherit_proxy_credentials(tmp_path, monkeypatch):
    runtime, data = prepared_task(tmp_path)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:password@proxy")
    agent = build_sandbox_spec(runtime, str(data), {})
    verifier, command = build_verifier_spec(runtime, str(data), "tasks/tiny.yaml", {})
    assert {Path(host).name for host, _, _ in agent.mounts} == {"train", "test"}
    assert all(Path(host) != data for host, _, _ in agent.mounts)
    assert not agent.env
    assert agent.network == verifier.network == "none"
    assert (str(data), "/home/agent/workspace/data", "ro") in verifier.mounts
    assert "--submission-file" in command
    assert "/opt/airs-verifier/evaluate.py" in command
    assert agent.auxiliary_cpus == verifier.cpus
    assert agent.auxiliary_memory == verifier.memory


def test_public_split_symlink_is_rejected(tmp_path):
    runtime, data = prepared_task(tmp_path)
    (data / "test").rmdir()
    (data / "test").symlink_to(data / "test_with_labels")
    with pytest.raises(ValueError, match="symlink"):
        build_sandbox_spec(runtime, str(data), {})


@pytest.mark.parametrize(
    "output", ['{"Accuracy": NaN}', '{"Accuracy": true}', '{"other": 1}', '{"Accuracy": 1}\nnoise']
)
def test_untrusted_or_incomplete_metric_values_are_rejected(output):
    with pytest.raises(ValueError):
        parse_metrics(output, "Accuracy")


def test_logs_before_final_metrics_are_supported():
    assert parse_metrics('Loading dataset...\n{"Accuracy": 0.5}\n', "Accuracy") == {"Accuracy": 0.5}


def test_code_execution_task_uses_an_isolated_judge_and_reserves_executor(tmp_path):
    runtime, data = prepared_task(tmp_path)
    path = tmp_path / "airs_configs" / "tasks" / "tiny.yaml"
    task = yaml.safe_load(path.read_text())
    task["id"] = "CodeGenerationAPPSPassAt5"
    path.write_text(yaml.safe_dump(task))
    (tmp_path / "airs_configs/data/tiny").rename(tmp_path / "airs_configs/data/CodeGenerationAPPSPassAt5")
    assert task_config(runtime, "tasks/tiny.yaml")[0]["id"] == task["id"]
    verifier, command = build_verifier_spec(runtime, str(data), "tasks/tiny.yaml", {})
    agent = build_sandbox_spec(runtime, str(data), {}, verifier_spec=verifier)
    assert command == ""
    assert verifier.code_execution is not None
    assert agent.auxiliary_cpus == runtime.verifier_cpus + runtime.apps_candidate_cpus
    assert int(agent.auxiliary_memory) == 4 * 1024**3 + 512 * 1024**2


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import nonexistent_airs_verifier_dependency", RuntimeError),
        ("import argparse; argparse.ArgumentParser().error('Invalid CSV')", ValueError),
    ],
)
async def test_grader_dependency_failure_is_not_a_bad_submission(source, expected):
    import subprocess
    import sys
    from types import SimpleNamespace

    from examples.airs_bench.grading import grade_submission
    from psrl.workers.env_worker.sandbox import ExecResult

    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=3)

    async def evaluate(*args):
        return ExecResult(completed.stdout + completed.stderr, completed.returncode, False, 0)

    with pytest.raises(expected):
        await grade_submission(SimpleNamespace(evaluate=evaluate), None, "trusted", "Accuracy", 1)


@pytest.mark.parametrize("output", ['{"Accuracy": NaN}', '{"other": 1}', "not-json"])
async def test_invalid_verifier_metrics_are_infrastructure_failures(output):
    from types import SimpleNamespace

    from examples.airs_bench.grading import grade_submission
    from psrl.workers.env_worker.sandbox import ExecResult

    async def evaluate(*args):
        return ExecResult(output, 0, False, 0)

    with pytest.raises(RuntimeError, match="invalid metrics"):
        await grade_submission(SimpleNamespace(evaluate=evaluate), None, "", "Accuracy", 1)
