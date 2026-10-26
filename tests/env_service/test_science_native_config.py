"""Validate native science configuration without starting Ray or loading models."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.sciaccel_rl.config import build_runtime_config
from omegaconf.errors import ValidationError


@pytest.mark.parametrize("mode", ["off", "on"])
def test_native_agent_config_uses_valid_empty_gpu_override(tmp_path, monkeypatch, mode):
    scripts = Path(__file__).resolve().parents[2] / "docs/env-worker-deep-dive/env-service-verification/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("science_fixed_work_training")
    configs = module.build_agent_configs(SimpleNamespace(workload="native", task_timeout=300), tmp_path, mode)
    assert len(configs) == 4, "All four task pools must use the validated native configuration."
    for config in configs:
        runtime = build_runtime_config(config)
        assert runtime.harbor.gpu_compose_override == "", "No GPU overlay must use the schema's empty string."
        assert runtime.harbor.agent_name == "examples.sciaccel_rl.agent:TruncatingTerminus2", (
            "Native tasks must execute the original agent without pressure injection."
        )


def test_null_gpu_override_reproduces_worker_validation_failure():
    with pytest.raises(ValidationError, match="gpu_compose_override"):
        build_runtime_config({"harbor": {"gpu_compose_override": None}})
