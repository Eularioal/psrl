"""Check resident headroom, model priority and physical return of shared commands."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.env_service.types import EnvUnit
from psrl.workers.gen.gpu_handoff import GpuControlGate, GpuHandoffMixin, gpu_control

from tests.env_service.test_gpu_commands import Remote, setup


class Model(GpuHandoffMixin):
    def __init__(self):
        self._gpu_control = GpuControlGate()
        self._gpu_sync_active = False
        self._gpu_weights_valid = True
        self.sleeping = False
        self.engine = SimpleNamespace(is_sleeping=AsyncMock(side_effect=lambda: self.sleeping))
        self.collective_rpc = AsyncMock(return_value=[{"transient_mb": 8}])
        self.wakes = 0

    @gpu_control
    async def wake(self):
        self.sleeping = False
        self.wakes += 1


@pytest.mark.asyncio
async def test_model_priority_and_existing_sleep_window():
    model = Model()
    assert await model.acquire_env_command("a") == "resident", "Awake inference must remain awake."
    wake = asyncio.create_task(model.wake())
    await asyncio.sleep(0)
    assert not wake.done(), "Control must wait for physical command return."
    assert await model.acquire_env_command("b") is None, "Queued model control must win over new commands."
    await model.release_env_command("a")
    await asyncio.wait_for(wake, 1)
    model.sleeping = True
    assert await model.acquire_env_command("c") == "sleeping", "Already sleeping capacity must be reusable."
    assert model.wakes == 1 and model.sleeping, "Acquisition must neither wake nor sleep the model."
    await model.release_env_command("c")
    assert model.sleeping, "Return must preserve the coordinator's sleep state."


@pytest.mark.asyncio
async def test_missing_profile_and_insufficient_margin_fail_closed():
    model = Model()
    with pytest.raises(ValueError, match="below"):
        await model.validate_env_sharing(10, 4)
    assert (await model.validate_env_sharing(12, 4))["required_headroom_mb"] == 12, "Profile must include margin."
    model.collective_rpc.return_value = [{"transient_mb": None}]
    with pytest.raises(ValueError, match="profiling"):
        await model.validate_env_sharing(9999, 4)


def shared_setup():
    ledger, scheduler, node, leases, clock = setup()
    ledger._units["u"] = replace(ledger._units["u"], shared_rollout_replica_rank=0, model_headroom_mb=35)
    model = Model()
    scheduler.shared_servers["u"] = Remote(model)
    scheduler.shared_margin_mb = 4
    return ledger, scheduler, node, leases, model


@pytest.mark.asyncio
async def test_headroom_rejects_resident_then_sleep_allows_without_wake():
    ledger, scheduler, node, (lease, _, _), model = shared_setup()
    assert (await scheduler.acquire(lease.lease_id, "c", "r"))["status"] == "wait", "51+35 exceeds 80."
    assert not model._gpu_control.shared_commands, "Rejected work must not block model control."
    model.sleeping = True
    grant = await scheduler.acquire(lease.lease_id, "c", "r")
    assert grant["status"] == "granted", "51+4 fits while the model is confirmed asleep."
    wake = asyncio.create_task(model.wake())
    await asyncio.sleep(0)
    node.return_gate = asyncio.Event()
    returned = asyncio.create_task(scheduler.release(lease.lease_id, "r", grant["epoch"]))
    await node.return_started.wait()
    assert not wake.done(), "Logical release must not unblock the model before GPU acknowledgement."
    node.return_gate.set()
    await returned
    await asyncio.wait_for(wake, 1)
    assert not ledger._leases[lease.lease_id].gpu_active, "Capacity must be returned."


@pytest.mark.asyncio
async def test_failed_cleanup_keeps_model_fenced():
    _, scheduler, node, (lease, _, _), model = shared_setup()
    model.sleeping = True
    grant = await scheduler.acquire(lease.lease_id, "c", "r")
    node.busy = True
    with pytest.raises(RuntimeError, match="CUDA"):
        await scheduler.release(lease.lease_id, "r", grant["epoch"])
    assert model._gpu_control.shared_commands, "Uncertain ownership must not permit a model wake."
    scheduler.shared_deadlines["r"].cancel()


@pytest.mark.asyncio
async def test_deadline_removes_only_own_lease_before_model_return():
    _, scheduler, node, (lease, _, _), model = shared_setup()
    model.sleeping = True
    scheduler.shared_max_command_s = 0.01
    node.confirm_gpu_return = AsyncMock()
    await scheduler.acquire(lease.lease_id, "c", "r")
    await asyncio.wait_for(scheduler.shared_deadlines["r"], 1)
    assert not node.active and not model._gpu_control.shared_commands, "Deadline must physically release first."
    assert lease.gpu_closing, "A timed out environment must not accept more work."
    node.confirm_gpu_return.assert_awaited_once()


def test_shared_units_require_command_scope_and_explicit_margin():
    with pytest.raises(ValueError, match="command mode"):
        EnvUnit("u", "n", 1, 100, (0,), 1, 100, shared_rollout_replica_rank=0)
