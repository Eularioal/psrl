"""Check cluster discovery and environment injection without launching model workers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("initialized", [False, True])
def test_task_runner_inherits_current_cluster_address(monkeypatch, initialized):
    from psrl.trainer import main_ppo

    config = OmegaConf.create(
        {
            "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {"EXISTING": "value"}}}},
            "transfer_queue": {"enable": True},
            "global_profiler": {"tool": None},
        }
    )
    OmegaConf.set_struct(config, True)
    ray = MagicMock()
    ray.is_initialized.return_value = initialized
    ray.get_runtime_context.return_value = SimpleNamespace(gcs_address="127.0.0.1:23456")
    monkeypatch.setattr(main_ppo, "ray", ray)
    monkeypatch.setattr(main_ppo, "get_ppo_ray_runtime_env", lambda: {})
    monkeypatch.setattr(main_ppo, "_reserve_excess_nodes", lambda config: [])
    runner = MagicMock()

    main_ppo.run_ppo(config, task_runner_class=runner)

    runner.options.assert_called_once_with(runtime_env={"env_vars": {"RAY_ADDRESS": "127.0.0.1:23456"}})
    if initialized:
        ray.init.assert_not_called()
    else:
        assert ray.init.call_args.kwargs["runtime_env"]["env_vars"] == {
            "EXISTING": "value",
            "TRANSFER_QUEUE_ENABLE": "1",
        }, "Runtime variables were lost or could not be injected into structured config."
    assert "TRANSFER_QUEUE_ENABLE" not in config.ray_kwargs.ray_init.runtime_env.env_vars
