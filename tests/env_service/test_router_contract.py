"""Validate named router reuse without starting a Ray cluster."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from psrl.workers.env_service.router import EnvRouterCore
from psrl.workers.env_service.router_contract import router_contract, verify_router_contract


def test_contract_is_order_independent_and_snapshots_nested_config():
    config = {"health": {"max_failures": 3}, "units": []}
    expected = router_contract(config, ["b", "a", "a"])
    assert expected == router_contract({"units": [], "health": {"max_failures": 3}}, ["a", "b"])
    config["health"]["max_failures"] = 9
    with pytest.raises(ValueError, match="contract mismatch"):
        verify_router_contract(expected, router_contract(config, ["a", "b"]))


@pytest.mark.parametrize(
    "changed",
    [
        {"default_gpu_memory_mb": 40000},
        {"units": [{"gpu_execution_backend": "mps"}]},
        {"units": [{"gpu_indices": [1]}]},
        {"node_policy": "any"},
    ],
)
def test_contract_rejects_changed_budget_backend_devices_or_policy(changed):
    with pytest.raises(ValueError, match="contract mismatch"):
        verify_router_contract(router_contract({}), router_contract(changed))


def test_contract_rejects_changed_nodes_and_protocol():
    original = router_contract({}, ["a"])
    with pytest.raises(ValueError):
        verify_router_contract(original, router_contract({}, ["b"]))
    actual = router_contract({}, ["a"])
    actual["configuration"]["protocol_version"] += 1
    with pytest.raises(ValueError):
        verify_router_contract(original, actual)


@pytest.mark.asyncio
async def test_service_contract_cannot_be_changed_through_returned_value():
    core = EnvRouterCore({})
    original = await core.startup_contract()
    returned = await core.startup_contract()
    returned["configuration"]["config"]["units"] = ["changed"]
    assert await core.startup_contract() == original


def test_factory_rejects_mismatch_before_prewarm(monkeypatch):
    from psrl.workers.env_service import router

    handle = SimpleNamespace(
        startup_contract=SimpleNamespace(remote=Mock(return_value="contract")),
        begin_prewarm=SimpleNamespace(remote=Mock()),
    )
    actor = Mock()
    actor.options.return_value.remote.return_value = handle
    monkeypatch.setattr(router, "EnvRouter", actor)
    monkeypatch.setattr(router.ray, "get", lambda ref: router_contract({"slot_ttl_s": 9}))
    with pytest.raises(ValueError, match="contract mismatch"):
        router.create_env_router({"slot_ttl_s": 10})
    handle.begin_prewarm.remote.assert_not_called()


def test_factory_rejects_service_without_contract(monkeypatch):
    from psrl.workers.env_service import router

    actor = Mock()
    actor.options.return_value.remote.return_value = SimpleNamespace()
    monkeypatch.setattr(router, "EnvRouter", actor)
    with pytest.raises(ValueError, match="no startup contract"):
        router.create_env_router({})
