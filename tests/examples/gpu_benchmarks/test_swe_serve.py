"""Check full inventory import and private verifier mounts."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml
from examples.swe_serve.prepare import prepare


def checkout(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "task_list.csv").write_text("task_id,task_dir,gpus\na,gpu-a,1\nb,gpu-b,1\nc,cpu-c,0\n")
    for name in ("gpu-a", "gpu-b", "cpu-c"):
        task = root / "tasks" / name
        for part in ("environment", "tests", "solution"):
            (task / part).mkdir(parents=True)
        (task / "instruction.md").write_text("Solve the public task.")
        (task / "tests/test.sh").write_text("private-test-marker")
        (task / "solution/solve.sh").write_text("oracle-marker")
        (task / "task.toml").write_text('[environment]\ngpus = 1\n[verifier]\nenvironment_mode = "separate"\n')
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "Fixture"],
    ):
        subprocess.run(command, cwd=root, check=True)
    return root


def test_all_gpu_tasks_keep_private_tests_and_provenance(tmp_path):
    source = checkout(tmp_path)
    output = tmp_path / "prepared"
    result = prepare(source, output, image_map={"gpu-a": "test:fixture"})
    assert result["gpu_tasks_imported"] == 2, "Every GPU task must be imported."
    assert result["runtime_verified_count"] == 0, "Import must not imply runtime verification."
    rows = pd.read_parquet(output / "tasks.parquet")
    assert len(rows) == 2, "CPU tasks must not enter the GPU dataset."
    for row in rows.itertuples():
        task = Path(row.extra_info["task_path"])
        agent = yaml.safe_load((task / "environment/docker-compose.yaml").read_text())
        verifier = yaml.safe_load((task / "tests/docker-compose.yaml").read_text())
        assert all("/tests" not in mount for mount in agent["services"]["main"]["volumes"]), (
            "Private tests must never be mounted into the agent."
        )
        assert f"{task / 'tests'}:/tests:ro" in verifier["services"]["main"]["volumes"], (
            "The independent verifier must receive its immutable test files."
        )
        assert (task / "tests/test.sh").read_text() == "private-test-marker", "Tests must remain unchanged."
        assert (task / "tests/test.sh").stat().st_mode & 0o111, "Harbor must be able to execute the test entry."
        record = json.loads((task / "provenance.json").read_text())
        assert "tests/test.sh" in record["upstream_files"], "Private test hashes must be recorded."
    train = pd.read_parquet(output / "train.parquet")
    val = pd.read_parquet(output / "val.parquet")
    ids = lambda frame: {row["upstream_task_id"] for row in frame.extra_info}
    assert not ids(train) & ids(val), "Training and validation identities must be disjoint."
    assert ids(train) | ids(val) == {"a", "b"}, "Splitting must retain all imported tasks."


def test_reject_unknown_override_and_dirty_checkout(tmp_path):
    source = checkout(tmp_path)
    with pytest.raises(ValueError, match="unknown image"):
        prepare(source, tmp_path / "unknown", image_map={"missing": "test:fixture"})
    (source / "task_list.csv").write_text("changed")
    with pytest.raises(ValueError, match="clean upstream"):
        prepare(source, tmp_path / "dirty")


def test_mlgym_reward_requires_independent_cuda_evidence():
    from examples.gpu_benchmarks.reward import compute_score

    metrics = {"reward": 0.7, "local_evaluation": 1, "eval_valid": 1, "cuda": 1}
    for missing in ("local_evaluation", "eval_valid", "cuda"):
        incomplete = {key: value for key, value in metrics.items() if key != missing}
        result = compute_score("mlgym_rl_local", "", "", {"harbor_rewards": incomplete})
        assert result["score"] == 0, "Incomplete verifier evidence must not earn reward."
    result = compute_score("mlgym_rl_local", "", "", {"harbor_rewards": metrics})
    assert result["score"] == 0.7, "Verified finite reward must be preserved."


def test_prebuilt_speed_task_requires_private_pristine_baseline(tmp_path):
    source = checkout(tmp_path)
    tests = source / "tasks/gpu-a/tests"
    (tests / "Dockerfile").write_text("FROM test:base\nRUN mv /code /base\nCOPY speed_check /speed-check\n")
    (tests / "speed_check").mkdir()
    (tests / "speed_check/bench.py").write_text("print('private timing harness')")
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "Speed fixture"],
        cwd=source,
        check=True,
    )
    output = tmp_path / "missing-base"
    with pytest.raises(ValueError, match="pristine /base"):
        prepare(source, output, image_map={"gpu-a": "test:fixture"})
    assert not output.exists(), "Do not publish a speed task without its independent baseline."
    base = tmp_path / "pristine"
    base.mkdir()
    output = tmp_path / "complete"
    prepare(source, output, image_map={"gpu-a": "test:fixture"}, base_map={"gpu-a": base})
    task = output / "tasks/gpu-a"
    agent = yaml.safe_load((task / "environment/docker-compose.yaml").read_text())["services"]["main"]
    verifier = yaml.safe_load((task / "tests/docker-compose.yaml").read_text())["services"]["main"]
    assert f"{base}:/base:ro" in verifier["volumes"], "The verifier needs the immutable pre-patch baseline."
    assert f"{task / 'tests/speed_check'}:/speed-check:ro" in verifier["volumes"], "Keep the private timing harness."
    assert all(":/base:" not in item and ":/speed-check:" not in item for item in agent["volumes"]), (
        "Do not expose verifier-owned baseline assets to the agent."
    )
