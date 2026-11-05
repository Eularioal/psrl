"""Check verifier delegation and conservative command configuration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from examples.kernel_bench.verifier import ManagedVerifierEnvironment, validate_units


def unit(**updates):
    return dict(gpu_indices=[0], gpu_reservation="command", max_containers=1, **updates)


@pytest.mark.parametrize(
    "change",
    [
        {"gpu_execution_backend": "mps"},
        {"shared_rollout_replica_rank": 0},
        {"rollout_replica_rank": 0},
        {"max_containers": 2},
        {"gpu_indices": [0, 1]},
        {"gpu_reservation": "episode"},
    ],
)
def test_unsafe_performance_pool_rejected(change):
    with pytest.raises(ValueError, match="dedicated"):
        validate_units([{**unit(), **change}])


def test_valid_pool():
    validate_units([unit()])
    with pytest.raises(ValueError):
        validate_units([])


@pytest.mark.asyncio
async def test_verifier_reserves_and_waits_for_return():
    env = SimpleNamespace(
        _gpu_execution={"mode": "command"},
        exec_gpu=AsyncMock(return_value=SimpleNamespace(return_code=0)),
        upload_dir=AsyncMock(),
    )
    proxy = ManagedVerifierEnvironment(env, 120)
    await proxy.upload_dir("a", "b")
    env.exec_gpu.assert_not_called()
    await proxy.exec("run evaluator", env={"TEST": "1"}, user="root")
    env.exec_gpu.assert_awaited_once_with(
        "run evaluator",
        timeout_sec=120,
        wait_timeout_sec=120,
        overlap_return=False,
        env={"TEST": "1"},
        user="root",
    )
    env.exec_gpu.return_value.return_code = 1
    result = await proxy.exec("failed evaluator")
    assert result.return_code == 1, "Preserve Harbor exit-code handling for preparation commands."


@pytest.mark.asyncio
async def test_cancellation_propagates():
    import asyncio

    env = SimpleNamespace(_gpu_execution={"mode": "command"}, exec_gpu=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await ManagedVerifierEnvironment(env, 120).exec("cancelled")


def test_harbor_factory_preserves_preprovisioned_tests():
    from examples.kernel_bench.verifier import VERIFIER_PATH
    from harbor.models.trial.config import VerifierConfig
    from harbor.verifier.factory import VerifierFactory

    env = SimpleNamespace(_gpu_execution={"mode": "command"})
    task = SimpleNamespace(
        config=SimpleNamespace(verifier=SimpleNamespace(environment_mode="separate", timeout_sec=120))
    )
    verifier = VerifierFactory.create_verifier_from_config(
        VerifierConfig(import_path=VERIFIER_PATH),
        task=task,
        trial_paths=SimpleNamespace(),
        environment=env,
        skip_tests_upload=True,
    )
    assert verifier._skip_tests_upload, "Do not upload over immutable verifier tests."
    assert verifier.environment.environment is env
    task.config.verifier.environment_mode = "same"
    with pytest.raises(ValueError, match="separate"):
        VerifierFactory.create_verifier_from_config(
            VerifierConfig(import_path=VERIFIER_PATH),
            task=task,
            trial_paths=SimpleNamespace(),
            environment=env,
        )
