"""Check shared execution without conflating job completion and session release."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.env_service.gpu_execution import GpuExecutionBackend, reservation_policy
from psrl.workers.env_service.gpu_jobs import GpuJobManager


def episode_backend():
    return GpuExecutionBackend(reservation_policy({"mode": "episode", "router_name": "r", "lease_id": "lease"}))


@pytest.mark.asyncio
async def test_episode_jobs_share_execution_but_keep_the_session_reservation():
    backend = episode_backend()
    runs = []

    async def execute(job):
        async def run():
            runs.append(job.command)
            return SimpleNamespace(return_code=0, stdout=job.command, stderr="")

        await backend.execute(run, AsyncMock(return_value="container"), timeout_sec=1, wait_timeout_sec=1, job=job)
        await backend.wait_return()

    jobs = GpuJobManager(execute)
    for command in ("first", "second"):
        identifier = jobs.start(command)["job_id"]
        result = await jobs.wait(identifier, released=True)
        assert result["state"] == "completed" and result["stdout"] == command
        assert result["reservation"]["reservation_mode"] == "episode"
        assert result["reservation"]["capacity_released"] is False
        assert "released_at" not in result["reservation"]
    assert runs == ["first", "second"]
    await jobs.close()


@pytest.mark.asyncio
async def test_episode_cancellation_waits_for_process_cleanup_without_releasing_session():
    backend = episode_backend()
    running, cleaning, acknowledge = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def run():
        running.set()
        await asyncio.Event().wait()

    async def cancel():
        cleaning.set()
        await acknowledge.wait()

    execution = asyncio.create_task(
        backend.execute(run, AsyncMock(return_value="container"), timeout_sec=1, wait_timeout_sec=1, cancel=cancel)
    )
    await running.wait()
    execution.cancel()
    await cleaning.wait()
    assert not execution.done()
    acknowledge.set()
    with pytest.raises(asyncio.CancelledError):
        await execution
    assert backend.last_record["capacity_released"] is False
    assert not backend.lock.locked()


@pytest.mark.asyncio
async def test_local_admission_deadline_does_not_cancel_current_owner():
    backend = episode_backend()
    running, finished = asyncio.Event(), asyncio.Event()

    async def run():
        running.set()
        await finished.wait()

    active = asyncio.create_task(
        backend.execute(run, AsyncMock(return_value="container"), timeout_sec=1, wait_timeout_sec=1)
    )
    await running.wait()
    following = AsyncMock()
    with pytest.raises(TimeoutError):
        await backend.execute(following, AsyncMock(), timeout_sec=1, wait_timeout_sec=0.01)
    following.assert_not_awaited()
    assert not active.done()
    finished.set()
    await active


def test_unknown_reservation_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        reservation_policy({"mode": "automatic", "router_name": "r", "lease_id": "lease"})


@pytest.mark.asyncio
async def test_prepared_process_cleanup_slack_does_not_violate_execution_budget():
    backend = episode_backend()
    run = AsyncMock(return_value="finished")
    result = await backend.execute(run, AsyncMock(return_value="container"), timeout_sec=3605, wait_timeout_sec=3600)
    assert result == "finished"
