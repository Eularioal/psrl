"""Verify episode ownership credits across allocation, release and container replacement."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.env_service import episode_memory
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.types import EnvDemand, EnvUnit


def ledger(devices=(0,), capacity=(96,)):
    now = [0.0]
    unit = EnvUnit("u", "node", 16, 16384, devices, 8, sum(capacity), gpu_memory_per_device_mb=capacity)
    return EnvUnitLedger([unit], NodeRules.build("any"), clock=lambda: now[0]), unit, now


def test_realized_episode_reservations_are_not_double_counted():
    book, unit, _ = ledger()
    leases = [book.try_admit(str(i), EnvDemand(needs_gpu=True, gpu_memory_mb=20), "node").lease for i in range(3)]
    book.report_measurement("node", {0: 42}, owned_mb_by_device={lease.lease_id: {0: 14} for lease in leases})
    assert book._effective_gpu_mb(unit, 0) == 60, "Realized reservations must be credited once."
    assert book._effective_gpu_by_device(unit, 0) == (60,), "Per-device admission must consume the same credits."
    assert book.try_admit("fourth", EnvDemand(needs_gpu=True, gpu_memory_mb=20), "node").status == "granted"
    assert book.snapshot()[0].owned_gpu_memory_mb == 42, "Attribution must be observable."


def test_raw_measurement_and_expiry_invalidate_ownership():
    book, unit, now = ledger()
    lease = book.try_admit("a", EnvDemand(needs_gpu=True, gpu_memory_mb=20), "node").lease
    book.report_measurement("node", {0: 14}, owned_mb_by_device={lease.lease_id: {0: 14}})
    book.report_measurement("node", {0: 14})
    assert book._effective_gpu_mb(unit, 0) == 34, "A new unpaired observation must not retain old credits."
    book.report_measurement("node", {0: 14}, owned_mb_by_device={lease.lease_id: {0: 14}})
    now[0] = 1000
    assert book._episode_credits(unit, now[0]) == {}, "Stale observations cannot provide credits."
    assert book._effective_gpu_mb(unit, now[0]) == 20, "Outstanding promises remain charged."


def test_release_and_foreign_leases_cannot_credit_a_new_reservation():
    book, unit, _ = ledger()
    lease = book.try_admit("a", EnvDemand(needs_gpu=True, gpu_memory_mb=20), "node").lease
    book.release(lease.lease_id)
    book.try_admit("b", EnvDemand(needs_gpu=True, gpu_memory_mb=20), "node")
    book.report_measurement("node", {0: 14}, owned_mb_by_device={lease.lease_id: {0: 14}})
    assert book._effective_gpu_mb(unit, 0) == 34, "A retired owner's memory remains unattributed."


def test_partial_allocation_external_usage_and_reallocation():
    book, unit, _ = ledger()
    lease = book.try_admit("a", EnvDemand(needs_gpu=True, gpu_memory_mb=20), "node").lease
    for allocated in (14, 6, 0, 24):
        book.report_measurement("node", {0: 30 + allocated}, owned_mb_by_device={lease.lease_id: {0: allocated}})
        assert book._effective_gpu_mb(unit, 0) == 30 + max(20, allocated), "External usage cannot pay down a promise."


def test_per_device_credits_cannot_hide_one_full_device():
    book, unit, _ = ledger((0, 1), (48, 48))
    demand = EnvDemand(needs_gpu=True, gpu_memory_mb=40, gpu_memory_per_device_mb=(20, 20))
    lease = book.try_admit("a", demand, "node").lease
    book.report_measurement("node", {0: 40, 1: 14}, owned_mb_by_device={lease.lease_id: {0: 14, 1: 14}})
    assert book._effective_gpu_by_device(unit, 0) == (46, 20), "Credits must remain on their observed device."
    assert book.try_admit("b", demand, "node").status == "wait", "Aggregate headroom cannot rescue a full device."
    with pytest.raises(ValueError, match="contained"):
        book.report_measurement("node", {0: 10, 1: 14}, owned_mb_by_device={lease.lease_id: {0: 14}})


def container(name, lease="lease", started="first"):
    return {
        "Id": name,
        "State": {"Pid": 1, "StartedAt": started, "Running": True},
        "Config": {"Labels": {"psrl.env_lease": lease, "psrl.env_unit": "u"}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_episode_aggregates_agent_and_verifier_and_rechecks_identity(monkeypatch, restart):
    rows = [{"Id": name, "State": "running", "Labels": container(name)["Config"]["Labels"]} for name in ("a", "v")]
    monkeypatch.setattr(episode_memory, "list_unit_containers", AsyncMock(return_value=rows))
    inspect = AsyncMock(
        side_effect=[
            container("a"),
            container("v"),
            container("a"),
            container("v", started="restart" if restart else "first"),
        ]
    )
    monkeypatch.setattr(episode_memory, "inspect_container", inspect)
    monkeypatch.setattr(episode_memory, "container_identity", lambda c: c["Id"])
    closed = []
    monkeypatch.setattr(
        episode_memory,
        "TaskGpuTelemetry",
        lambda probe, fence, devices, **kwargs: SimpleNamespace(
            indices=devices, close_identity=lambda: closed.append(fence)
        ),
    )
    monkeypatch.setattr(
        episode_memory,
        "read_command_memory",
        lambda *args: {
            "devices": {0: {"used_mb": 30}},
            "owned_mb": {"a": 10, "v": 10},
            "owned_mb_by_device": {"a": {0: 10}, "v": {0: 10}},
            "attribution_errors": {},
        },
    )
    service = episode_memory.EpisodeMemory(None, None)
    result = await service.snapshot((0,), [("lease", "u", (0,))])
    assert result["owned_mb_by_device"] == {"lease": {0: 10 if restart else 20}}, (
        "Only stable containers can credit a lease."
    )
    assert ("v" in result["attribution_errors"]) == restart, "The replacement must be observable."
    assert closed == (["v"] if restart else []), "Only the replacement should be unbound before cleanup."
    await service.close()
    assert sorted(closed) == ["a", "v"], "All identity registrations must close."


@pytest.mark.asyncio
async def test_missing_identity_keeps_fresh_measurement_and_reports_no_credit(monkeypatch):
    node = EnvNodeService(None, "node")
    monkeypatch.setattr(node, "_gpu_identity", AsyncMock(side_effect=RuntimeError("host identity unavailable")))
    monkeypatch.setattr(node, "gpu_lending_snapshot", AsyncMock(return_value={0: {"used_mb": 42}}))
    sample = await node.episode_gpu_memory_snapshot((0,), [("lease", "u", (0,))])
    assert sample["devices"][0]["used_mb"] == 42, "Attribution failure must not replace real usage with zero."
    assert sample["owned_mb_by_device"] == {}, "Unsupported identity must not credit reservations."
    assert sample["attribution_errors"]["identity"] == "host identity unavailable", "Keep the fallback reason."


@pytest.mark.asyncio
async def test_no_live_episodes_close_old_registrations_before_fresh_sample(monkeypatch):
    node = EnvNodeService(None, "node")
    node._episode_memory = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(node, "gpu_lending_snapshot", AsyncMock(return_value={0: {"used_mb": 14}}))
    sample = await node.episode_gpu_memory_snapshot((0,), [])
    node._episode_memory.close.assert_awaited_once()
    assert sample["devices"][0]["used_mb"] == 14 and not sample["owned_mb_by_device"], (
        "Memory remaining after lease retirement is still measured, without ownership credit."
    )
