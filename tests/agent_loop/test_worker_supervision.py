"""Check completion acknowledgement, retry state and fatal waiter notification."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf
from psrl.workers.agent_loop import worker as worker_module
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

Worker = worker_module.PSRL_AgentLoopWorker.__ray_metadata__.modified_class


def batch(size=1):
    value = TensorDict({}, batch_size=[size])
    for key, values in {
        "uid": list(range(size)),
        "parent_id": list(range(size)),
        "validate": [False] * size,
        "global_steps": [0] * size,
    }.items():
        tu.assign_non_tensor_stack(value, key, values)
    return value


def manager():
    value = object.__new__(PSRL_AgentLoopManager)
    value._refill_breaker_diagnosis = None
    value._train_buffer_waiters = {}
    value._train_chunk_waiters = {}
    value._val_buffer_waiters = {}
    value._resolved_train_chunks = {}
    return value


@pytest.mark.asyncio
async def test_batch_acknowledgement_waits_for_execution_and_joins_cancelled_cleanup():
    value = object.__new__(Worker)
    value._programs_closing = False
    value.agent_programs = set()
    value.config = OmegaConf.create({"gen_actor_rollout_ref": {"rollout": {"agent": {"default_agent_loop": "test"}}}})
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def run(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.01)
            cleaned.set()

    value._run_agent_loop = run
    call = asyncio.create_task(value.add_agent_program(batch()))
    await entered.wait()
    assert not call.done(), "Enqueuing must not acknowledge execution completion."
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert cleaned.is_set() and not value.agent_programs, "Cancellation must join owned cleanup."


@pytest.mark.asyncio
async def test_successful_retry_does_not_rethrow_a_previous_exception(monkeypatch):
    value = object.__new__(Worker)
    value.config = OmegaConf.create(
        {
            "gen_actor_rollout_ref": {
                "rollout": {
                    "agent": {
                        "retry_limit": 2,
                        "raise_on_error": True,
                    }
                }
            },
            "data": {},
        }
    )
    for field in (
        "rollout_gateway_url",
        "session_router_url",
        "reward_manager",
        "tokenizer",
        "processor",
        "dataset_cls",
    ):
        setattr(value, field, None)
    value.agent_loop_manager = SimpleNamespace(notify_group_failed=SimpleNamespace(remote=AsyncMock()))
    value.ps_manager_handle = SimpleNamespace(
        update_request_status=SimpleNamespace(remote=AsyncMock(return_value=True)),
        abort_requests=SimpleNamespace(remote=AsyncMock()),
    )
    value.postprocess_output = AsyncMock()
    loop = SimpleNamespace(
        run_with_termination_handling=AsyncMock(
            side_effect=[
                RuntimeError("Transient startup failure"),
                (object(), TerminateReason.FINISHED),
            ]
        )
    )
    monkeypatch.setitem(worker_module.AGENT_LOOP_REGISTRY, "test", {})
    monkeypatch.setattr(worker_module.hydra.utils, "instantiate", lambda **kwargs: lambda **kw: loop)
    monkeypatch.setattr(worker_module, "rollout_trace_attr", lambda **kw: nullcontext())
    monkeypatch.setattr(worker_module, "log_dual_events", lambda *args, **kw: nullcontext())
    await value._run_agent_loop_inner("test", batch(), 0, 0)
    assert loop.run_with_termination_handling.await_count == 2, "A transient failure must retry once."
    value.postprocess_output.assert_awaited_once()
    value.agent_loop_manager.notify_group_failed.remote.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_rpc_failure_reaches_existing_and_late_waiters():
    value = manager()
    futures = [asyncio.get_running_loop().create_future() for _ in range(3)]
    value._train_buffer_waiters[0] = [futures[0]]
    value._train_chunk_waiters[(0, 0)] = [futures[1]]
    value._val_buffer_waiters[0] = [futures[2]]
    value.agent_loop_workers = [
        SimpleNamespace(
            add_agent_program=SimpleNamespace(
                remote=AsyncMock(side_effect=RuntimeError("Worker disappeared during execution"))
            )
        )
    ]
    await value._supervise_worker_batch(0, batch())
    for future in futures:
        with pytest.raises(RuntimeError, match="Worker disappeared"):
            await future
    with pytest.raises(RuntimeError, match="Worker disappeared"):
        value._raise_if_refill_breaker_tripped()
    value._retry_data = AsyncMock()
    assert await value._refill_failed_group(1, "test") == 0, "A latched failure must prevent refill."
    value._retry_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_failure_during_buffer_creation_cannot_register_a_stranded_waiter():
    value = manager()

    async def ensure(buffer_id):
        value._trip_refill_breaker("Worker died while buffer was being created.")

    value.ps_manager_handle = SimpleNamespace(ensure_train_buffer_exists=SimpleNamespace(remote=ensure))
    with pytest.raises(RuntimeError, match="Worker died"):
        await asyncio.wait_for(value.wait_for_training_batch(0), timeout=1)
    assert not value._train_buffer_waiters, "Failure must be checked after remote buffer creation."


@pytest.mark.asyncio
async def test_manager_shutdown_cancels_supervision_and_joins_worker_cleanup(monkeypatch):
    from psrl.workers.agent_loop import manager as manager_module

    value = manager()
    reference = asyncio.get_running_loop().create_future()
    cancelled = []
    monkeypatch.setattr(manager_module.ray, "cancel", cancelled.append)
    value.agent_loop_workers = [
        SimpleNamespace(
            add_agent_program=SimpleNamespace(remote=lambda batch: reference),
            stop_agent_programs=SimpleNamespace(remote=AsyncMock()),
        )
    ]
    task = asyncio.create_task(value._supervise_worker_batch(0, batch()))
    value._worker_dispatch_tasks = {task}
    value.train_dispatch_task = value.val_dispatch_task = value.collect_task = None
    value._shutdown_distributed_post_pool = AsyncMock()
    await asyncio.sleep(0)
    await value.stop_busy_loop()
    assert task.cancelled() and cancelled == [reference], "Shutdown must cancel the remote execution owner."
    value.agent_loop_workers[0].stop_agent_programs.remote.assert_awaited_once()
    value._shutdown_distributed_post_pool.assert_awaited_once()
    assert value._refill_breaker_diagnosis is None, "Expected shutdown cancellation is not an execution failure."
