"""Process cancellation tests use no model, Ray actor or Docker daemon."""

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.airs_bench.process_runner import run_in_process

pytestmark = pytest.mark.cpu_test


def blocked_runner(payload):
    Path(payload["marker"]).write_text(str(os.getpid()))
    time.sleep(30)
    return {"unexpected": True}


def rpc_runner(payload):
    future = asyncio.run_coroutine_threadsafe(
        payload["sandbox_handle"].exec("echo test", 1),
        payload["event_loop"],
    )
    return {"reply": future.result(timeout=5)}


def grading_error_runner(payload):
    future = asyncio.run_coroutine_threadsafe(payload["sandbox_handle"].evaluate(None, "", 1), payload["event_loop"])
    try:
        future.result(timeout=5)
    except Exception as error:
        return {"error_type": type(error).__name__}
    return {"error_type": None}


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
async def test_grading_error_category_survives_process_boundary(error_type):
    async def evaluate(*args):
        raise error_type("Injected grading error.")

    handle = SimpleNamespace(sandbox_id="test", evaluate=evaluate)
    result = await asyncio.wait_for(run_in_process({}, handle, grading_error_runner), 20)
    assert result == {"error_type": error_type.__name__}, "The agent must distinguish a bad CSV from infrastructure."


async def test_timeout_kills_the_running_process(tmp_path):
    marker = tmp_path / "started"
    task = asyncio.create_task(
        run_in_process({"marker": str(marker)}, SimpleNamespace(sandbox_id="test"), blocked_runner)
    )
    try:
        async with asyncio.timeout(20):
            while not marker.exists():
                await asyncio.sleep(0.02)
        pid = int(marker.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_process_bridges_sandbox_calls_and_returns_result():
    async def execute(command, timeout_s, no_output_timeout_s):
        assert command == "echo test"
        return "test"

    handle = SimpleNamespace(sandbox_id="test", exec=execute)
    result = await asyncio.wait_for(run_in_process({}, handle, rpc_runner), 20)
    assert result == {"reply": "test"}
