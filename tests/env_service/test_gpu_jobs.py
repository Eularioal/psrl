"""Check job ownership, observation timeouts, overlap and cancellation boundaries."""

import asyncio
from types import SimpleNamespace

import pytest
from examples.sciaccel_rl.gpu_tools import dispatch_gpu_tool
from psrl.workers.env_service.gpu_jobs import GpuJobManager


@pytest.mark.asyncio
async def test_output_and_physical_return_have_independent_boundaries():
    running, output, returned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def execute(job):
        job.transition("running")
        running.set()
        await output.wait()
        job.output = SimpleNamespace(stdout="ready", stderr="", return_code=0)
        job.transition("returning")
        job.output_ready.set()
        await returned.wait()

    jobs = GpuJobManager(execute)
    first = jobs.start("gpu program")["job_id"]
    await running.wait()
    assert (await jobs.wait(first, timeout_sec=0.01))["state"] == "running"
    assert not jobs.jobs[first].task.cancelled()
    output.set()
    result = await jobs.wait(first)
    assert result["stdout"] == "ready" and result["state"] == "returning"
    second = jobs.start("next program")["job_id"]
    await asyncio.sleep(0)
    assert jobs.status(second)["state"] == "queued"
    returned.set()
    assert (await jobs.wait(second, released=True))["state"] == "completed"
    await jobs.close()


@pytest.mark.asyncio
async def test_queued_cancellation_and_deadline_do_not_release_active_job():
    running, finish = asyncio.Event(), asyncio.Event()

    async def execute(job):
        running.set()
        await finish.wait()

    jobs = GpuJobManager(execute)
    first = jobs.start("first")["job_id"]
    await running.wait()
    second = jobs.start("cancelled")["job_id"]
    assert (await jobs.cancel(second))["state"] == "cancelled"
    third = jobs.start("expires", wait_timeout_sec=0.01)["job_id"]
    assert (await jobs.wait(third, released=True))["state"] == "failed"
    assert not jobs.jobs[first].task.done()
    finish.set()
    await jobs.wait(first, released=True)
    with pytest.raises(RuntimeError, match="failed"):
        await jobs.close()


@pytest.mark.asyncio
async def test_tool_start_does_not_interpret_shell_on_host():
    seen = []

    def start(command, **kwargs):
        seen.append(command)
        return {"job_id": "job", "state": "queued"}

    env = SimpleNamespace(start_gpu_job=start)
    result = await dispatch_gpu_tool(env, 'psrl-gpu {"op":"start","command":"echo $(secret)"}')
    assert result["state"] == "queued" and seen == ["echo $(secret)"]
    with pytest.raises(ValueError, match="Unknown"):
        await dispatch_gpu_tool(env, 'psrl-gpu {"op":"shell"}')


@pytest.mark.asyncio
async def test_failed_release_cannot_be_reported_as_completed():
    async def execute(job):
        job.output = SimpleNamespace(stdout="partial", stderr="", return_code=0)
        job.output_ready.set()
        raise RuntimeError("Device still owned.")

    jobs = GpuJobManager(execute)
    job_id = jobs.start("background leak")["job_id"]
    result = await jobs.wait(job_id, released=True)
    assert result["state"] == "failed" and "Device still owned" in result["error"]


@pytest.mark.asyncio
async def test_cancellation_during_return_waits_for_physical_acknowledgement():
    from unittest.mock import AsyncMock

    from psrl.workers.env_service.docker import ManagedDockerEnvironment

    env = object.__new__(ManagedDockerEnvironment)
    entered, acknowledged = asyncio.Event(), asyncio.Event()

    async def returning():
        entered.set()
        await acknowledged.wait()

    env.exec_gpu = AsyncMock()
    env.wait_for_gpu_return = returning
    jobs = GpuJobManager(env._execute_gpu_job)
    job_id = jobs.start("program")["job_id"]
    await entered.wait()
    cancelling = asyncio.create_task(jobs.cancel(job_id))
    await asyncio.sleep(0.01)
    assert not cancelling.done()
    acknowledged.set()
    assert (await cancelling)["state"] == "cancelled"


def test_process_group_cancellation_reaps_foreground_children(tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    import time
    import uuid
    from pathlib import Path

    from psrl.workers.env_service import gpu_job_process

    job_id = uuid.uuid4().hex
    directory = Path("/tmp/psrl-gpu-jobs") / job_id
    script = gpu_job_process.__file__
    process = subprocess.Popen([sys.executable, script, "run", job_id, "sleep 60 & wait"])
    try:
        deadline = time.monotonic() + 3
        while not (directory / "process.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        record = json.loads((directory / "process.json").read_text())
        subprocess.run([sys.executable, script, "cancel", job_id], check=True, timeout=3)
        assert process.wait(timeout=3) != 0
        assert gpu_job_process.identity(record["pid"]) != record["birth"]
    finally:
        if process.poll() is None:
            subprocess.run([sys.executable, script, "cancel", job_id], check=True, timeout=3)
            process.wait(timeout=3)
        shutil.rmtree(directory, ignore_errors=True)


def test_process_options_explicitly_set_directory_and_environment(tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    import uuid
    from pathlib import Path

    from psrl.workers.env_service import gpu_job_process

    job_id = uuid.uuid4().hex
    options = {"cwd": str(tmp_path), "env": {"GPU_JOB_TEST_VALUE": "retained"}}
    try:
        result = subprocess.run(
            [
                sys.executable,
                gpu_job_process.__file__,
                "run",
                job_id,
                'printf "%s|%s" "$PWD" "$GPU_JOB_TEST_VALUE"',
                json.dumps(options),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert result.stdout == f"{tmp_path}|retained"
    finally:
        shutil.rmtree(Path("/tmp/psrl-gpu-jobs") / job_id, ignore_errors=True)
