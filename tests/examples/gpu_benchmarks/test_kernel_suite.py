"""Check complete task enumeration and managed dataset publication on CPU."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
from examples.kernel_bench.prepare_suite import discover, prepare_suite


def test_suite_publishes_unique_level_tasks_and_explicit_validation_boundary(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    reference = "class Model: pass\ndef get_inputs(): return []\ndef get_init_inputs(): return []\n"
    for level in (1, 2):
        folder = repo / "KernelBench" / f"level{level}"
        folder.mkdir(parents=True)
        (folder / "1_example.py").write_text(reference)
    monkeypatch.setattr("examples.kernel_bench.prepare_suite.provenance", lambda _: {"revision": "test"})
    monkeypatch.setattr("examples.gpu_benchmarks.prepare.provenance", lambda _: {"revision": "test"})
    args = argparse.Namespace(
        repo=repo,
        output=tmp_path / "suite",
        levels=[1, 2, 1],
        image="test:local",
        verifier_image=None,
        python="/usr/local/bin/python",
        mount_python_prefix=False,
        backend="triton",
        timeout=120,
        gpu_memory_mb=4096,
    )
    out = prepare_suite(args)
    rows = pd.read_parquet(out / "dataset/tasks.parquet")
    assert len(rows) == 2, "Publish every distinct level and task pair."
    assert len({row["task_path"] for row in rows.extra_info}) == 2, "Keep task paths distinct."
    inventory = json.loads((out / "inventory.json").read_text())
    assert all(not item["gpu_verified"] for item in inventory["tasks"]), "Do not invent GPU verification."
    for item in json.loads((out / "manifest.json").read_text()):
        task = Path(item["task_path"])
        metadata = json.loads((task / "provenance.json").read_text())
        assert metadata["benchmark"] == "kernel_bench", "Preserve the scoring identity."
        assert (task / "tests/evaluate.py").is_file(), "Include the independent verifier."


def test_suite_rejects_incompatible_reference_before_creating_output(tmp_path):
    folder = tmp_path / "KernelBench/level1"
    folder.mkdir(parents=True)
    (folder / "bad.py").write_text("class Model: pass\n")
    with pytest.raises(ValueError, match="interface"):
        discover(tmp_path, [1])
    with pytest.raises(ValueError, match="No tasks"):
        discover(tmp_path, [2])
