"""Check that vision task preparation cannot expose held-out answers to agents."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from examples.mlgym_bench.download_vision import SOURCES
from examples.mlgym_bench.prepare_vision import prepare


def test_private_labels_are_only_available_to_independent_verifier(tmp_path):
    data = tmp_path / "data"
    for name in SOURCES:
        task = data / name
        (task / "public").mkdir(parents=True)
        (task / "private").mkdir()
        (task / "public/images.npz").write_bytes(b"fixture")
        (task / "public/train_labels.json").write_text("[0]")
        (task / "private/test_labels.json").write_text("[7]")
        (task / "assets.json").write_text(json.dumps({"dataset": name}))
    output = tmp_path / "prepared"
    prepare(data, output, sys.executable, "test:fixture")
    rows = pd.read_parquet(output / "tasks.parquet")
    assert len(rows) == 3, "All three vision task identities must be retained."
    for name in SOURCES:
        task = output / name
        agent = yaml.safe_load((task / "environment/docker-compose.yaml").read_text())["services"]["main"]
        verifier = yaml.safe_load((task / "tests/docker-compose.yaml").read_text())["services"]["main"]
        private = data / name / "private"
        assert all(
            Path(volume.split(":")[0]) not in (private, *private.parents)
            and volume.split(":")[1] not in ("/data/private", "/tests")
            for volume in agent["volumes"]
        ), "The agent must not see private labels or verifier code."
        assert f"{data / name / 'private'}:/data/private:ro" in verifier["volumes"], (
            "The verifier must receive immutable held-out labels."
        )
        assert not list((task / "public").rglob("*test_labels*")), "Public baseline assets must not copy answers."
    (data / next(iter(SOURCES)) / "private/test_labels.json").unlink()
    incomplete = tmp_path / "incomplete"
    with pytest.raises(FileNotFoundError):
        prepare(data, incomplete, sys.executable, "test:fixture")
    assert not incomplete.exists(), "Missing evaluation data must not leave a runnable partial suite."
