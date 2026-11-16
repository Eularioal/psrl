"""
Exercise original PostTrain verification on CPU substitutes at its explicit CUDA boundary.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import safetensors.torch
import torch
from bounded_worker import Slot, Study, file_manifest

SOURCE = Path(__file__).resolve().parents[5] / "examples/posttrain_bench/local_model.py"


@pytest.fixture
def cpu_protocol(tmp_path, monkeypatch):
    """
    Preserve the real verify function while replacing only device and base loading boundaries.
    """
    model_path = tmp_path / "base"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"bounded-test"}')
    (model_path / "model.safetensors").write_bytes(b"private CPU fixture")
    data = tmp_path / "rows.json"
    data.write_text(json.dumps([{"prompt": "P", "answer": "A"}]))
    calls = {"model": 0, "tokenizer": 0}

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.norm = torch.nn.LayerNorm(2)

        def cuda(self):
            return self

        def forward(self, **kwargs):
            return SimpleNamespace(loss=self.model.norm.weight.mean())

    class ModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["model"] += 1
            return Tiny()

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["tokenizer"] += 1
            return SimpleNamespace(encode=lambda text, **kwargs: [1, 2])

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = TokenizerLoader
    transformers.AutoModelForCausalLM = ModelLoader
    transformers.__version__ = "CPU-protocol-substitute"
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    original_tensor = torch.tensor
    original_load = safetensors.torch.load_file

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path, **kwargs: original_load(path, device="cpu"))
    candidates = []
    for name, values in (("A", [1.0, 1.0]), ("B", [2.0, 4.0])):
        path = tmp_path / name
        path.mkdir()
        safetensors.torch.save_file({"model.norm.weight": original_tensor(values)}, path / "adapter.safetensors")
        candidates.append(path)
    return SimpleNamespace(model=model_path, data=data, candidates=candidates, calls=calls)


@pytest.mark.parametrize("mode,builds", [("fresh", 3), ("resident", 1)])
def test_original_verify_replaces_complete_candidate_after_a_b_a(cpu_protocol, mode, builds):
    fixture = cpu_protocol
    study = Study(mode, fixture.model, SOURCE)
    try:
        rows = [study.request(fixture.data, fixture.candidates[index]) for index in (0, 1, 0)]
    finally:
        study.close()
    assert fixture.calls == {"model": builds, "tokenizer": builds}, "Count actual pretrained loader invocations."
    assert [row["metrics"]["answer_nll"] for row in rows] == [1.0, 3.0, 1.0], (
        "The original verify function must score each fully replaced candidate."
    )
    assert rows[0]["candidate_tensor_id"] == rows[2]["candidate_tensor_id"] != rows[1]["candidate_tensor_id"], (
        "Returning to A must restore the exact candidate tensor identity."
    )
    assert rows[0]["metrics"] == rows[2]["metrics"], "Candidate B must not contaminate repeated A verification."


def test_configuration_and_source_content_invalidate_resident_base(cpu_protocol, tmp_path):
    fixture = cpu_protocol
    source = tmp_path / "execute.py"
    source.write_bytes(SOURCE.read_bytes())
    study = Study("resident", fixture.model, source)
    try:
        study.request(fixture.data, fixture.candidates[0])
        (fixture.model / "config.json").write_text('{"model_type":"changed-test"}')
        changed_config = study.request(fixture.data, fixture.candidates[1])
        source.write_text(source.read_text() + "\nUNRELATED_BOUNDARY_MARKER = 1\n")
        changed_source = study.request(fixture.data, fixture.candidates[0])
    finally:
        study.close()
    assert fixture.calls["model"] == 3, "Changed source or configuration must reconstruct the base."
    assert not changed_config["cache_hits"]["model"] and not changed_source["cache_hits"]["model"], (
        "A previous loaded model cannot satisfy a different explicit source identity."
    )


def test_manifest_detects_same_size_and_timestamp_content_change(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"old-weights")
    import os

    stamp = path.stat().st_mtime_ns
    before = file_manifest(tmp_path)
    path.write_bytes(b"new-weights")
    os.utime(path, ns=(stamp, stamp))
    assert file_manifest(tmp_path)["sha256"] != before["sha256"], "File metadata cannot substitute for base content."


def test_slot_never_reuses_a_different_model_identity():
    slot, created, released = Slot(), [], []

    def build(name):
        created.append(name)
        return {"model": name}

    a, first = slot.get("base-A", lambda: build("A"), lambda: released.append(True))
    repeated, hit = slot.get("base-A", lambda: build("unused"), lambda: released.append(True))
    b, switched = slot.get("base-B", lambda: build("B"), lambda: released.append(True))
    assert a is repeated and hit and not first and not switched, "Reuse requires an identical model identity."
    assert b != a and created == ["A", "B"] and len(released) == 2, (
        "Replace the previous base before building another."
    )
