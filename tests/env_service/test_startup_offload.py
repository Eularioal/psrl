"""Check loop progress, cancellation ownership, and validation during startup probes."""

import asyncio
import functools
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from harbor.environments.docker.docker import DockerEnvironment
from psrl.workers.env_service.docker import ManagedDockerEnvironment


async def wait_started(event):
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


def test_kernel_probe_is_shared_and_cancellation_joins_it(monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    calls = []

    @functools.cache
    def probe():
        calls.append(threading.get_ident())
        entered.set()
        assert finish.wait(3), "The event loop must run while the probe waits."
        return False

    monkeypatch.setattr(DockerEnvironment, "_egress_control_kernel_support", staticmethod(probe))

    async def run():
        jobs = [asyncio.create_task(ManagedDockerEnvironment.prepare_runtime()) for _ in range(4)]
        try:
            await wait_started(entered)
            jobs[0].cancel()
            await asyncio.sleep(0.02)
            assert not jobs[0].done(), "Cancellation must join the temporary probe container."
            assert len(calls) == 1, "Concurrent jobs must share the cold probe."
            assert calls[0] != threading.get_ident(), "The probe must leave the event loop."
        finally:
            finish.set()
            results = await asyncio.gather(*jobs, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert results[1:] == [None] * 3
        assert DockerEnvironment._egress_control_kernel_support() is False
        assert len(calls) == 1, "Harbor constructors must reuse the original cached result."

    asyncio.run(run())


@pytest.mark.parametrize("daemon_os", ["linux", "windows", None])
def test_daemon_query_is_fresh_off_loop_and_validation_is_preserved(monkeypatch, daemon_os):
    entered, finish = threading.Event(), threading.Event()
    threads = []

    def detect():
        threads.append(threading.get_ident())
        entered.set()
        assert finish.wait(3), "The event loop must run while Docker info waits."
        return daemon_os

    async def start(self, force_build):
        self._validate_daemon_mode()
        return "started"

    monkeypatch.setattr(DockerEnvironment, "_detect_daemon_os", staticmethod(detect))
    monkeypatch.setattr(DockerEnvironment, "start", start)
    env = object.__new__(ManagedDockerEnvironment)
    env._gpu_lending = None
    env._is_windows_container = False

    async def run():
        job = asyncio.create_task(env.start(False))
        try:
            await wait_started(entered)
            assert not job.done()
            assert threads[0] != threading.get_ident()
        finally:
            finish.set()
        if daemon_os == "windows":
            with pytest.raises(RuntimeError, match="daemon is running"):
                await job
        else:
            assert await job == "started"
            assert await env.start(False) == "started"
            assert len(threads) == 2, "Every start must query the current daemon mode."

    asyncio.run(run())


def test_cancelled_overlay_query_cannot_leave_a_file(monkeypatch):
    from examples.sciaccel_rl import runner

    entered, finish = threading.Event(), threading.Event()

    def build(*args):
        entered.set()
        assert finish.wait(3), "The event loop must run while inventory is queried."
        return {"services": {"main": {}}}

    write = Mock()
    monkeypatch.setattr(runner, "build_lease_overlay", build)
    monkeypatch.setattr(runner, "write_overlay", write)

    async def run():
        paths, kwargs = [], {}
        task = asyncio.create_task(runner._apply_env_lease(SimpleNamespace(gpu_indices=()), paths, kwargs, "", ""))
        try:
            await wait_started(entered)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            finish.set()
        assert paths == [] and kwargs == {}

    asyncio.run(run())
    write.assert_not_called()
