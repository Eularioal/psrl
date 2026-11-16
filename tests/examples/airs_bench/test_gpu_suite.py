"""Check GPU admission and public input boundaries for bounded AIRS tasks."""

import sys
from pathlib import Path

import pytest
from datasets import Dataset
from examples.airs_bench.config import AirsBenchRuntimeConfig
from examples.airs_bench.prepare.prepare_gpu_suite import baseline_spec
from psrl.environments.mlgym_env import build_sandbox_spec

pytestmark = pytest.mark.cpu_test


def test_gpu_sandbox_preserves_private_label_boundary(tmp_path):
    for split in ("train", "test", "test_with_labels"):
        (tmp_path / split).mkdir()
    runtime = AirsBenchRuntimeConfig(sandbox_gpus=1, sandbox_python=sys.executable)
    spec = build_sandbox_spec(runtime, str(tmp_path), {})
    assert spec.gpus == 1
    assert all("test_with_labels" not in source for source, _, _ in spec.mounts)
    assert all(mode == "ro" for _, _, mode in spec.mounts)
    assert str(Path(sys.executable).parent.parent) in {source for source, _, _ in spec.mounts}


@pytest.mark.parametrize("gpus", [-1, 2, True])
def test_invalid_gpu_demand_is_rejected(gpus):
    with pytest.raises(ValueError, match="sandbox_gpus"):
        AirsBenchRuntimeConfig(sandbox_gpus=gpus)


@pytest.mark.parametrize("name", ["gpu_agent_config.yaml", "mlgym_agent_config.yaml"])
def test_shipped_configs_create_callable_agent_factories(name, monkeypatch):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from psrl.workers.agent_loop.loops.mlgym_agent_loop import MLGymAgentLoop

    monkeypatch.setenv("AIRS_DATA_ROOT", "/data/airs")
    monkeypatch.setenv("MLGYM_REPO", "/opt/mlgym")
    monkeypatch.setenv("AIRS_AGENT_CONFIG", "/opt/psrl/examples/airs_bench/agent_config.yaml")
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(root / "examples/airs_bench/config" / name)[0]
    factory = instantiate(config, _partial_=True)
    assert callable(factory), "The PSRL worker requires a callable Hydra partial."
    assert factory.func is MLGymAgentLoop


def test_numeric_baseline_uses_only_public_features():
    train = Dataset.from_list([{"text": "a", "label": 0}, {"text": "b", "label": 1}])
    test = Dataset.from_list([{"text": "new"}])
    spec = baseline_spec("SentimentAnalysisYelpReviewFullAccuracy", train, test)
    assert spec["train_x"] == [{"text": "a"}, {"text": "b"}]
    assert spec["test_x"] == [{"text": "new"}]
    assert spec["train_y"] == [0, 1]


def test_rideshare_forecast_flattens_series_without_using_future_as_input():
    train = Dataset.from_list(
        [{"target": [[1.0] * 48, [2.0] * 48], "label_target": [[1.0] * 48 + [3.0] * 48, [2.0] * 48 + [4.0] * 48]}]
    )
    test = Dataset.from_list([{"target": [[5.0] * 48, [6.0] * 48]}])
    spec = baseline_spec("TimeSeriesForecastingRideshareMAE", train, test)
    assert len(spec["test_x"]) == 2
    assert spec["train_y"] == [[3.0] * 48, [4.0] * 48]
    assert spec["train_x"] == [{"target": [1.0] * 48}, {"target": [2.0] * 48}]
