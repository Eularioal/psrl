"""Verify startup sharing, failure propagation and cleanup without allocating a GPU."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from psrl.workers.env_service import node, router


def make_router(monkeypatch, *, timeout=1):
    config = {
        "node_policy": "any",
        "prewarm": {"timeout_s": timeout},
        "units": [{"unit_id": "u", "node_ip": "127.0.0.1", "cpu_quota": 2, "memory_mb": 1024}],
    }
    instance = router.RayEnvRouter(config)
    service = SimpleNamespace(
        ready=SimpleNamespace(remote=AsyncMock(return_value="http://node")),
        prepare=SimpleNamespace(remote=AsyncMock()),
        close=SimpleNamespace(remote=AsyncMock()),
    )
    created, killed = [], []

    def create(ip):
        created.append(ip)
        return service

    monkeypatch.setattr(instance, "_create_node", create)
    monkeypatch.setattr(router.ray, "kill", lambda actor, **kwargs: killed.append(actor))
    return instance, service, created, killed


@pytest.mark.asyncio
async def test_prewarm_shares_startup_and_survives_cancelled_consumer(monkeypatch):
    instance, service, created, killed = make_router(monkeypatch)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def prepare(indices):
        entered.set()
        await finish.wait()

    service.prepare.remote.side_effect = prepare
    await instance.begin_prewarm()
    await entered.wait()
    waiter = asyncio.create_task(instance._gpu_command_node("127.0.0.1"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not instance._node_services["127.0.0.1"].done(), "A consumer must not cancel shared startup."
    finish.set()
    await instance.wait_prewarm()
    assert await instance._gpu_command_node("127.0.0.1") is service, "Admission must reuse the prepared node."
    assert created == ["127.0.0.1"], "Concurrent consumers must create one service."
    assert not instance._ledger._leases, "Prewarming must not allocate task resources."
    await instance.shutdown_nodes()
    assert killed == [service], "Shutdown must release the owned service."
    service.close.remote.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_failure_reaches_admission_and_reclaims_actor(monkeypatch):
    instance, service, created, killed = make_router(monkeypatch)
    service.prepare.remote.side_effect = RuntimeError("Docker is unavailable.")
    with pytest.raises(RuntimeError, match="Docker"):
        await instance.wait_prewarm()
    with pytest.raises(RuntimeError, match="Docker"):
        await instance._gpu_command_node("127.0.0.1")
    assert len(created) == 1 and killed == [service], "A failed startup must not leave an actor or silently retry."
    assert not instance._ledger._leases, "A failed probe must not consume capacity."
    await instance.shutdown_nodes()


@pytest.mark.asyncio
async def test_startup_timeout_is_bounded_and_reclaims_actor(monkeypatch):
    instance, service, _, killed = make_router(monkeypatch, timeout=0.05)

    async def blocked(indices):
        await asyncio.Event().wait()

    service.prepare.remote.side_effect = blocked
    with pytest.raises(TimeoutError):
        await instance.wait_prewarm()
    assert killed == [service], "Timeout must reclaim a partially initialized actor."
    await instance.shutdown_nodes()


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_startup(monkeypatch):
    instance, service, _, killed = make_router(monkeypatch)
    entered = asyncio.Event()

    async def blocked(indices):
        entered.set()
        await asyncio.Event().wait()

    service.prepare.remote.side_effect = blocked
    await instance.begin_prewarm()
    await entered.wait()
    await instance.shutdown_nodes()
    assert killed == [service], "Shutdown must reclaim the actor even before readiness."
    with pytest.raises(RuntimeError, match="closing"):
        await instance.begin_prewarm()


@pytest.mark.asyncio
async def test_prepare_uses_owned_probe_without_storing_gpu_availability(monkeypatch):
    service = node.EnvNodeService(None, "127.0.0.1")
    calls = []
    service._nvml_probe = SimpleNamespace(
        read_gpu_devices=lambda: calls.append("identity"),
        read_gpu_snapshot=lambda indices: calls.append(tuple(indices)),
    )
    ping = AsyncMock()
    monkeypatch.setattr(node, "ping_docker", ping)
    await service.prepare((2,))
    assert calls == ["identity", (2,)], "Warmup must initialize the same probe used by live commands."
    assert not service._pending, "Warmup must not create a cleanup lease."
    ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_continues_after_one_service_close_fails(monkeypatch):
    instance, service, _, killed = make_router(monkeypatch)
    await instance.wait_prewarm()
    service.close.remote.side_effect = RuntimeError("Driver close failed.")
    other = SimpleNamespace(close=SimpleNamespace(remote=AsyncMock()))
    instance._node_handles["another-owned-node"] = other
    with pytest.raises(RuntimeError, match="shutdown errors"):
        await instance.shutdown_nodes()
    assert killed == [service, other], "One close failure must not leak other owned services."


@pytest.mark.asyncio
async def test_shutdown_refuses_to_abandon_active_leases(monkeypatch):
    instance, service, _, killed = make_router(monkeypatch)
    await instance.wait_prewarm()
    instance._ledger._leases["active"] = object()
    with pytest.raises(RuntimeError, match="active environment leases"):
        await instance.shutdown_nodes()
    assert not killed, "Live leases still require their node cleanup service."
    instance._ledger._leases.clear()
    await instance.shutdown_nodes()
    assert killed == [service], "Released leases must permit shutdown."


@pytest.mark.parametrize("enabled", [False, True])
def test_factory_only_joins_prewarm_submission(monkeypatch, enabled):
    handle = SimpleNamespace(
        ping=SimpleNamespace(remote=Mock(return_value="ping-ref")),
        begin_prewarm=SimpleNamespace(remote=Mock(return_value="submission-ref")),
        wait_prewarm=SimpleNamespace(remote=Mock()),
    )
    actor = Mock()
    actor.options.return_value.remote.return_value = handle
    get = Mock()
    monkeypatch.setattr(router, "EnvRouter", actor)
    monkeypatch.setattr(router.ray, "get", get)
    assert router.create_env_router({"prewarm": {"enable": enabled}}) is handle
    assert [call.args[0] for call in get.call_args_list] == (
        ["ping-ref", "submission-ref"] if enabled else ["ping-ref"]
    ), "Factory must wait for submission only, leaving node startup concurrent with model work."
    handle.wait_prewarm.remote.assert_not_called()
