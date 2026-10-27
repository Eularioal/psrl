"""Verify that masked groups are refilled without entering optimizer batches."""

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from psrl.utils.transferqueue_utils import PayloadState
from psrl.workers.agent_loop import manager as manager_module
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager
from tensordict import TensorDict


def make_manager():
    manager = object.__new__(PSRL_AgentLoopManager)
    manager._failed_train_group_ids = set()
    manager._consecutive_group_failures = 0
    manager._group_failure_reasons = Counter()
    manager._refill_breaker_diagnosis = None
    manager.refill_failure_threshold = 2
    manager._train_buffer_waiters = {}
    manager._train_chunk_waiters = {}
    manager._val_buffer_waiters = {}
    manager._resolved_train_chunks = {}
    manager.ps_manager_handle = SimpleNamespace(clear_reserved_entries=SimpleNamespace(remote=AsyncMock()))
    manager._refill_failed_group = AsyncMock(return_value=2)
    return manager


@pytest.mark.parametrize("nested", [False, True])
def test_empty_groups_refill_then_trip_breaker(monkeypatch, nested):
    mask = torch.zeros(2, 4)
    if nested:
        mask = torch.nested.as_nested_tensor([torch.zeros(3), torch.zeros(7)], layout=torch.jagged)
    read = AsyncMock(return_value=TensorDict({"response_mask": mask}, batch_size=[2]))
    clear = AsyncMock()
    monkeypatch.setattr(manager_module.tq, "async_kv_batch_get", read)
    monkeypatch.setattr(manager_module, "async_clear_payload", clear)
    manager = make_manager()

    async def run():
        waiter = asyncio.get_running_loop().create_future()
        manager._train_buffer_waiters[0] = [waiter]
        assert not await manager._keep_trainable_group(1, ["2", "3"]), "Empty group was accepted."
        assert not waiter.done(), "Transient filtering ended training."
        assert not await manager._keep_trainable_group(2, ["4", "5"]), "Empty refill was accepted."
        with pytest.raises(RuntimeError, match="no_trainable_tokens=2"):
            await waiter

    asyncio.run(run())
    manager._refill_failed_group.assert_awaited_once()
    assert manager._failed_train_group_ids == {1, 2}, "Late arrivals could be readmitted."
    clear.assert_awaited_with(keys=["4", "5"], partition_id="train", state=PayloadState.DROPPED)
    assert manager._group_failure_reasons[TerminateReason.NO_TRAINABLE_TOKENS.value] == 2


def test_mixed_group_preserves_masked_siblings(monkeypatch):
    mask = torch.nested.as_nested_tensor([torch.zeros(3), torch.tensor([0.0, 1.0, 1.0])], layout=torch.jagged)
    read = AsyncMock(return_value=TensorDict({"response_mask": mask}, batch_size=[2]))
    clear = AsyncMock()
    monkeypatch.setattr(manager_module.tq, "async_kv_batch_get", read)
    monkeypatch.setattr(manager_module, "async_clear_payload", clear)
    manager = make_manager()
    assert asyncio.run(manager._keep_trainable_group(1, ["2", "3"])), "Trainable group was discarded."
    clear.assert_not_awaited()
    manager._refill_failed_group.assert_not_awaited()
