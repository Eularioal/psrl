"""Keep configuration inspection independent of the model execution stack."""

import os
import subprocess
import sys

import yaml


def test_config_cli_does_not_import_model_runtime():
    """Compose overrides without loading model implementations or PPO execution.

    Hydra still imports veRL's config package, whose package initializer loads
    torch, tokenizer utilities and transfer_queue. Allow those existing
    dependencies; forbid model implementations and the PPO execution helpers.
    """
    program = """
import importlib.abc
import runpy
import sys

class BlockModelRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith('transformers.models')
                or fullname.startswith('verl.trainer.ppo')
                or fullname == 'psrl.utils.config'):
            raise RuntimeError(f'Configuration inspection imported {fullname!r}.')

sys.meta_path.insert(0, BlockModelRuntime())
sys.argv = ['main_ppo', '--cfg', 'job', 'trainer.total_training_steps=2']
runpy.run_module('psrl.trainer.main_ppo', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(result.stdout)
    assert config["trainer"]["total_training_steps"] == 2
    assert "actor_rollout_ref" in config
