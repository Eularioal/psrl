"""Distinguish the owning trajectory deadline from downstream operation failures."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason


@pytest.mark.asyncio
@pytest.mark.parametrize("outer", [True, False])
async def test_termination_uses_the_timer_that_actually_expired(outer):
    async def run(prompt):
        if not outer:
            raise TimeoutError("Inner operation expired.")
        await asyncio.sleep(1)

    loop = SimpleNamespace(
        config=SimpleNamespace(
            gen_actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(agent=SimpleNamespace(trajectory_timeout=0.01 if outer else 1))
            )
        ),
        run=run,
    )
    output, reason = await AgentLoopBase.run_with_termination_handling(loop, {})
    expected = TerminateReason.TRAJECTORY_TIMEOUT if outer else TerminateReason.OPERATION_TIMEOUT
    assert output is None and reason is expected, "The outer and inner timers must remain distinguishable."
    assert reason.needs_worker_retry() and not reason.is_successful, "Timeouts must not create trainable output."


@pytest.mark.asyncio
async def test_external_cancellation_is_not_converted_to_timeout():
    loop = SimpleNamespace(
        config=SimpleNamespace(
            gen_actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(agent=SimpleNamespace(trajectory_timeout=1)))
        ),
        run=AsyncMock(side_effect=asyncio.CancelledError),
    )
    with pytest.raises(asyncio.CancelledError):
        await AgentLoopBase.run_with_termination_handling(loop, {})
