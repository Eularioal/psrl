"""Check shared execution without conflating job completion and session release."""

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.env_service.gpu_execution import GpuExecutionBackend, GpuOperationTimeout, reservation_policy
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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["container_lookup", "gpu_admission", "command_execution", "return_request"])
async def test_timeout_retains_phase_and_cleanup_evidence(phase):
    backend = episode_backend()
    lookup = AsyncMock(return_value="container")
    run = AsyncMock(return_value="output")
    backend.policy.acquire = AsyncMock()
    backend.policy.finish = AsyncMock(return_value=False)
    backend.policy.cancel = AsyncMock()
    operations = {
        "container_lookup": lookup,
        "gpu_admission": backend.policy.acquire,
        "command_execution": run,
        "return_request": backend.policy.finish,
    }
    operations[phase].side_effect = TimeoutError("Inner deadline expired.")
    with pytest.raises(GpuOperationTimeout) as error:
        await backend.execute(run, lookup, timeout_sec=1, wait_timeout_sec=1)
    record = error.value.record
    restored = pickle.loads(pickle.dumps(error.value))
    assert restored.phase == phase and restored.record == record, "Remote exception transport must retain evidence."
    assert error.value.phase == phase, "The innermost operation must identify its phase."
    assert record["failure"]["phase"] == phase, "Cleanup must not replace the original failure."
    assert all("elapsed_s" in stage for stage in record["phases"]), "Failed phases also need timing."
    assert backend.policy.cancel.await_count == (phase != "container_lookup"), "Scheduler requests need cancellation."
    assert not backend.lock.locked(), "Failure must release the local execution lock."


@pytest.mark.asyncio
async def test_expired_execution_timer_is_not_reported_as_admission():
    backend = episode_backend()
    cancel = AsyncMock()
    with pytest.raises(GpuOperationTimeout) as error:
        await backend.execute(
            lambda: asyncio.sleep(1),
            AsyncMock(return_value="container"),
            timeout_sec=0.01,
            wait_timeout_sec=1,
            cancel=cancel,
        )
    assert error.value.phase == "command_execution", "The actual timer must retain its stage."
    cancel.assert_awaited_once()
