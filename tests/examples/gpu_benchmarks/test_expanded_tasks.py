"""Check expanded real datasets and protected operator identities without a GPU."""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from examples.flashinfer_bench.verify import main as verify
from examples.gpu_benchmarks import evaluate
from examples.gpu_benchmarks.prepare import prepare as prepare_gpu
from examples.gpu_benchmarks.reward import compute_score
from examples.kernel_bench import quick_check
from examples.mle_bench.local import prepare
from pydantic import BaseModel


@pytest.mark.parametrize("dataset,features,classes", [("wine", 13, 3), ("iris", 4, 3), ("breast-cancer", 30, 2)])
def test_real_dataset_preparation_hides_labels(tmp_path, dataset, features, classes):
    task = prepare(argparse.Namespace(output=tmp_path / dataset, dataset=dataset, image="test:local", python="python"))
    metadata = json.loads((task / "provenance.json").read_text())
    assert metadata["features"] == features and metadata["classes"] == classes, "Keep real dataset dimensions."
    with (task / "public/test.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert "label" not in reader.fieldnames, "Keep test labels private."
        ids = {row["id"] for row in reader}
    assert ids == set(json.loads((task / "private/labels.json").read_text())), "Cover every protected test identity."


def test_verifier_uses_protected_operator_identity(tmp_path):
    definitions = tmp_path / "traces/definitions"
    definitions.mkdir(parents=True)
    (definitions / "gemm_example.json").write_text("{}")
    submission = tmp_path / "solution.json"
    submission.write_text(json.dumps({"definition": "rmsnorm_h2048", "sources": [{"content": "wrong"}]}))
    verify(submission, tmp_path, definitions.parent)
    assert json.loads((tmp_path / "reward.json").read_text())["reward"] == 0, "Reject a different operator identity."


def test_quick_worker_reuses_process_and_reaps_owned_group(tmp_path, monkeypatch):
    def control(item, destination):
        destination.write_text(json.dumps({"pid": os.getpid()}))

    monkeypatch.setattr(quick_check, "check_one", control)
    parent, child = mp.get_context("fork").Pipe()
    worker = mp.get_context("fork").Process(target=quick_check.worker_loop, args=(child,))
    worker.start()
    try:
        pids = []
        for index in range(2):
            destination = tmp_path / f"{index}.json"
            parent.send(({}, destination))
            assert parent.poll(3), "Return bounded feedback from the CPU control."
            parent.recv()
            pids.append(json.loads(destination.read_text())["pid"])
        assert pids == [worker.pid, worker.pid], "Reuse the worker across independent requests."
    finally:
        quick_check.stop_worker(worker)
        parent.close()
        child.close()
    assert not worker.is_alive(), "Reap the owned worker."


def test_bounded_profile_is_explicit_and_preserves_numerical_parameters():
    source = "batch_size=4096\ndim=393216\neps=1e-6\naxis=1\ndef get_inputs(): return batch_size, dim\n"
    bounded, overrides = quick_check.bounded_source(source)
    context = {}
    exec(bounded, context)
    assert context["get_inputs"]() == (2, 32), "Bound explicit dimensions in the diagnostic copy."
    assert context["eps"] == 1e-6 and context["axis"] == 1, "Preserve tolerance and axis parameters."
    assert overrides == {"batch_size": 2, "dim": 32}, "Expose every applied dimension override."


@pytest.mark.parametrize("bounded", [False, True])
def test_kernel_failure_metadata_remains_serializable(tmp_path, monkeypatch, bounded):
    class Result(BaseModel):
        correctness: bool = False
        runtime: float = -1
        ref_runtime: float = -1
        metadata: dict[str, Any]

    result = Result(metadata={"runtime_error": RuntimeError("Invalid candidate shape.")})
    kernel = ModuleType("kernelbench")
    kernel_eval = ModuleType("kernelbench.eval")
    kernel_eval.eval_kernel_against_ref = lambda *args, **kwargs: result
    monkeypatch.setitem(sys.modules, "kernelbench", kernel)
    monkeypatch.setitem(sys.modules, "kernelbench.eval", kernel_eval)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda name: name))
    for name in ("artifacts", "verifier"):
        (tmp_path / name).mkdir()
    (tmp_path / "artifacts/submission.py").write_text("invalid")
    reference = tmp_path / "reference.py"
    reference.write_text("dim = 4096")
    monkeypatch.setitem(sys.modules, "quick_check", quick_check)
    monkeypatch.setattr(
        evaluate,
        "Path",
        lambda name: tmp_path / name.removeprefix("/logs/") if name.startswith("/logs/") else Path(name),
    )
    spec = {"benchmark": "kernel_bench", "reference": str(reference), "backend": "cuda"}
    if bounded:
        spec["input_profile"] = "bounded_module_dimensions"
    metrics = evaluate.evaluate(spec)
    from harbor.models.verifier.result import VerifierResult

    assert VerifierResult(rewards=metrics).rewards == metrics, "Accept numeric metrics in the real Harbor schema."
    assert metrics["bounded_inputs"] == int(bounded), "Identify the evaluation profile numerically."
    assert metrics["reward"] == 0, "Keep the failed candidate score."
    data = json.loads((tmp_path / "verifier/kernel_result.json").read_text())
    assert data["metadata"]["runtime_error"] == "Invalid candidate shape.", "Keep readable failure metadata."


def test_bounded_task_publishes_distinct_protocol(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    reference = repo / "KernelBench/level1/example.py"
    reference.parent.mkdir(parents=True)
    reference.write_text(
        "dim=4096\nclass Model: pass\ndef get_inputs(): return []\ndef get_init_inputs(): return []\n"
    )
    monkeypatch.setattr("examples.gpu_benchmarks.prepare.provenance", lambda _: {"revision": "test"})
    args = argparse.Namespace(
        repo=repo,
        output=tmp_path / "task",
        benchmark="kernel_bench",
        bounded_inputs=True,
        image="test:local",
        python="python",
        level=1,
        problem="example.py",
        backend="cuda",
        timeout=60,
        gpu_memory_mb=4096,
    )
    task = prepare_gpu(args)
    metadata = json.loads((task / "provenance.json").read_text())
    assert metadata["benchmark"] == "kernel_bench_local", "Separate canonical and bounded task identities."
    assert (task / "tests/quick_check.py").is_file(), "Package the same profile implementation for the verifier."
    assert "dim = 32" in (task / "solution/submission.py").read_text(), (
        "Match the reference control to local dimensions."
    )
    score = compute_score("kernel_bench_local", "", "", {"harbor_rewards": {"reward": 1}})
    assert score["score"] == 0, "Reject a reward without the bounded evaluation marker."
    metrics = {"reward": 1, "local_evaluation": 1, "bounded_inputs": 1}
    score = compute_score("kernel_bench_local", "", "", {"harbor_rewards": metrics})
    assert score["score"] == 1, "Accept a bounded evaluation with valid numeric markers."
