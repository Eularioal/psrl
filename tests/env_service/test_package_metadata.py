"""Check aliases, stale records and bounded probes in wheel metadata discovery."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_path = Path(__file__).resolve().parents[2] / "patch/transformers/fast_package_metadata.py"
_spec = importlib.util.spec_from_file_location("fast_metadata", _path)
metadata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metadata)


class Distribution:
    def __init__(self, name, files, existing, declared=None):
        self.metadata = {"Name": name}
        self.record = "\n".join(f"{name},," for name in files)
        self.existing = set(existing)
        self.declared = declared
        self.probes = []

    def read_text(self, name):
        return self.declared if name == "top_level.txt" else self.record

    @property
    def files(self):
        pytest.fail("Wheel discovery must not evaluate the eager files property.")

    def locate_file(self, name):
        self.probes.append(name)
        return SimpleNamespace(exists=lambda: name in self.existing)


def test_metadata_names_preserve_aliases_namespaces_and_missing_files(monkeypatch):
    a = Distribution(
        "pillow", ["PIL/missing.py", "PIL/Image.py", "gone.py", "pillow.dist-info/METADATA"], ["PIL/Image.py"]
    )
    b = Distribution("namespace-extra", ["PIL/extra.py"], ["PIL/extra.py"])
    c = Distribution("declared", [], [], "editable\n")
    monkeypatch.setattr(metadata.importlib.metadata, "distributions", lambda: [a, b, c])
    assert metadata.packages_distributions() == {"PIL": ["pillow", "namespace-extra"], "editable": ["declared"]}, (
        "Aliases and namespaces must retain every owning distribution."
    )
    assert c.probes == [], "Declared names do not require filesystem probes."


def test_large_wheel_checks_only_until_first_existing_file(monkeypatch):
    files = [f"torch/file{i}.py" for i in range(10000)]
    dist = Distribution("torch", files, files)
    monkeypatch.setattr(metadata.importlib.metadata, "distributions", lambda: [dist])
    assert metadata.packages_distributions() == {"torch": ["torch"]}, "The import name must be inferred."
    assert dist.probes == [files[0]], "One existing file proves the package is present."
