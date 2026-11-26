"""Verify that the experimental gate only relaxes resident weight synchronization."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.gen.gpu_handoff import GpuControlGate, GpuHandoffMixin, gpu_control


class Model(GpuHandoffMixin):
    def __init__(self, enabled=True):
        self.psrl_config = SimpleNamespace(
            env_service=SimpleNamespace(gpu_sharing={"sync_overlap": enabled}), ps_mode="nixl_cpu"
        )
        self._gpu_weights_valid = True
        self._gpu_sync_active = False
        self._gpu_control = GpuControlGate()
        self.engine = SimpleNamespace(is_sleeping=AsyncMock(return_value=False))

    @gpu_control
    async def pause_for_sync(self):
        self._gpu_sync_active = True

    @gpu_control
    async def sleep(self):
        return "slept"


@pytest.mark.asyncio
async def test_resident_sync_preserves_owner_and_blocks_new_commands():
    model = Model()
    assert await model.acquire_env_command("a") == "resident"
    await asyncio.wait_for(model.pause_for_sync(), 1)
    assert model._gpu_control.shared_commands == {"a": "resident"}
    assert await model.acquire_env_command("b") is None
    assert model._gpu_control.control_events[-1]["shared_at_entry"] == {"a": "resident"}
    sleep = asyncio.create_task(model.sleep())
    await asyncio.sleep(0)
    assert not sleep.done()
    await model.release_env_command("a")
    assert await asyncio.wait_for(sleep, 1) == "slept"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["disabled", "sleeping", "reservation", "invalid", "other_transfer"])
async def test_unsupported_sync_still_waits(case):
    model = Model(enabled=case != "disabled")
    model._gpu_control.shared_commands["a"] = "sleeping" if case == "sleeping" else "resident"
    if case == "reservation":
        model._gpu_control.reservation = "lease"
    if case == "invalid":
        model._gpu_weights_valid = False
    if case == "other_transfer":
        model.psrl_config.ps_mode = "cpu"
    pending = asyncio.create_task(model.pause_for_sync())
    await asyncio.sleep(0)
    assert not pending.done()
    await model.release_env_command("a")
    await asyncio.wait_for(pending, 1)
