"""Bounded CPU reproductions for shell corruption and reservation leaks."""

import asyncio
import os
import signal
import subprocess
import time
from types import SimpleNamespace

import pytest
from psrl.workers.env_worker.coordinator import EnvWorkerCoordinator, acquire_sandbox
from psrl.workers.env_worker.sandbox import SandboxSpec
from psrl.workers.env_worker.worker import ContainerShell, EnvWorker

pytestmark = pytest.mark.cpu_test


class LocalBashWorker(EnvWorker):
    """Replace only Docker process creation and removal with a real local Bash."""

    async def create_sandbox(self, spec, sandbox_id="sandbox"):
        process = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
        self._shells[sandbox_id] = ContainerShell(process, sandbox_id)
        await self.exec(sandbox_id, "echo ready", 2)
        return sandbox_id

    async def destroy_sandbox(self, sandbox_id):
        shell = self._shells.get(sandbox_id)
        if shell is not None and shell.process.poll() is None:
            os.killpg(shell.process.pid, signal.SIGKILL)
        await super().destroy_sandbox(sandbox_id)


def worker():
    return LocalBashWorker(0, 1, 0, 2)


async def test_timeout_destroys_shell_before_any_following_observation():
    instance = worker()
    await instance.create_sandbox(SandboxSpec(image="local"))
    try:
        first = await instance.exec("sandbox", "echo A-start; sleep 0.6; echo A-end", 0.1)
        assert first.timed_out
        assert "A-start" in first.stdout
        assert "A-end" not in first.stdout
        with pytest.raises(KeyError):
            await instance.exec("sandbox", "echo B-output; false", 1)
        assert not instance._shells
    finally:
        await instance.destroy_sandbox("sandbox")


async def test_persistent_state_nonzero_exit_and_marker_spoof():
    instance = worker()
    await instance.create_sandbox(SandboxSpec(image="local"))
    try:
        await instance.exec("sandbox", "export AIRS_STATE=kept", 1)
        result = await instance.exec("sandbox", "echo ///PROCESS-DONE:0:PROCESS-DONE///; echo $AIRS_STATE; false", 1)
        assert result.exit_code == 1
        assert "kept" in result.stdout
        assert "///PROCESS-DONE:0:PROCESS-DONE///" in result.stdout
    finally:
        await instance.destroy_sandbox("sandbox")


async def test_process_exit_does_not_wait_for_startup_deadline():
    instance = worker()
    await instance.create_sandbox(SandboxSpec(image="local"))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exited with code 7"):
        await instance.exec("sandbox", "exit 7", 5)
    assert time.monotonic() - started < 1
    assert not instance._shells


async def test_continuous_output_cannot_bypass_deadline():
    instance = worker()
    await instance.create_sandbox(SandboxSpec(image="local"))
    started = time.monotonic()
    try:
        result = await instance.exec("sandbox", "while true; do echo spam; done", 0.1)
        assert result.timed_out
        assert time.monotonic() - started < 1
    finally:
        await instance.destroy_sandbox("sandbox")


async def test_full_stdin_pipe_respects_deadline_and_keeps_the_event_loop_alive():
    instance = worker()
    await instance.create_sandbox(SandboxSpec(image="local"))
    shell = instance._shells["sandbox"]
    loop = asyncio.get_running_loop()
    await shell.write("sleep 2\n", loop.time() + 1)
    ticked = asyncio.Event()

    async def tick():
        await asyncio.sleep(0.02)
        ticked.set()

    task = asyncio.create_task(tick())
    started = time.monotonic()
    try:
        result = await instance.exec("sandbox", "#" + "x" * 500000, 0.1)
        assert result.timed_out, "A full stdin pipe must consume the command deadline."
        assert ticked.is_set(), "Other sandbox work must progress while stdin is full."
        assert time.monotonic() - started < 1, "The write waited for the sleeping shell."
    finally:
        await task
        await instance.destroy_sandbox("sandbox")


async def test_output_flood_is_bounded_before_observation_truncation(monkeypatch):
    from psrl.workers.env_worker import worker as module

    monkeypatch.setattr(module, "_SHELL_OUTPUT_LIMIT", 65536)
    instance = worker()
    await instance.create_sandbox(SandboxSpec(image="local"))
    try:
        with pytest.raises(RuntimeError, match="output exceeds"):
            await instance.exec("sandbox", "head -c 1000000 /dev/zero", 2)
        assert not instance._shells, "Output overflow must invalidate the sandbox."
    finally:
        await instance.destroy_sandbox("sandbox")


async def test_background_output_cannot_keep_drain_running_forever(monkeypatch):
    from psrl.workers.env_worker import worker as module

    monkeypatch.setattr(module, "_SHELL_OUTPUT_LIMIT", 65536)
    instance = worker()
    await instance.create_sandbox(SandboxSpec(image="local"))
    shell = instance._shells["sandbox"]
    monkeypatch.setattr(shell, "_read_available", lambda: b"x" * 8192)
    try:
        with pytest.raises(RuntimeError, match="stale output exceeds"):
            await instance.exec("sandbox", "echo next", 1)
        assert not instance._shells, "An unbounded stale stream must invalidate the sandbox."
    finally:
        await instance.destroy_sandbox("sandbox")


async def test_utf8_character_split_between_pipe_reads_is_preserved(monkeypatch):
    instance = worker()
    await instance.create_sandbox(SandboxSpec(image="local"))
    shell = instance._shells["sandbox"]
    encoded = "中文".encode()
    chunks = iter([encoded[:2], encoded[2:4], encoded[4:] + b"\n///PROCESS-DONE:test:0:PROCESS-DONE///\n"])
    monkeypatch.setattr(shell, "_read_available", lambda: next(chunks, b""))
    try:
        output = await shell.read_until(asyncio.get_running_loop().time() + 1, 1, "test")
        assert output.startswith("中文\n"), "Pipe chunk boundaries must not corrupt UTF-8."
    finally:
        await instance.destroy_sandbox("sandbox")


def remote(obj):
    return SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(obj, name))
            for name in dir(obj)
            if not name.startswith("_") and callable(getattr(obj, name))
        }
    )


class OwnedWorker:
    def __init__(self):
        self.live = set()
        self.fail_delete = False

    async def create_sandbox(self, spec, sandbox_id):
        self.live.add(sandbox_id)
        return sandbox_id

    async def destroy_sandbox(self, sandbox_id):
        if self.fail_delete:
            raise RuntimeError("Injected deletion failure.")
        self.live.discard(sandbox_id)


async def test_expired_owner_frees_container_and_capacity_but_failed_delete_does_not():
    coordinator = EnvWorkerCoordinator()
    instance = OwnedWorker()
    await coordinator.register_worker(remote(instance), 0, 1, 0, "local", 9, 36 * 1024**3)
    spec = SandboxSpec(image="local", cpus=8, memory="32g", auxiliary_cpus=1, auxiliary_memory="4g")
    handle = await acquire_sandbox(remote(coordinator), spec, 1)
    handle._heartbeat_task.cancel()
    coordinator._reservations[handle.sandbox_id].last_seen -= spec.idle_timeout_s + 1
    instance.fail_delete = True
    assert await coordinator.reap_expired() == []
    assert coordinator._slots[0].used_cpu == 1
    assert instance.live == {handle.sandbox_id}
    instance.fail_delete = False
    assert await coordinator.reap_expired() == [handle.sandbox_id]
    assert not instance.live
    assert coordinator._slots[0].used_cpu == 0
    await handle.destroy()
    assert coordinator._slots[0].used_cpu == 0
    await coordinator.shutdown()


async def test_many_waiters_do_not_prevent_release_and_cancelled_wait_leaks_nothing():
    coordinator = EnvWorkerCoordinator()
    instance = OwnedWorker()
    await coordinator.register_worker(remote(instance), 0, 1, 0, "local")
    spec = SandboxSpec(image="local")
    handle = await acquire_sandbox(remote(coordinator), spec, 1)
    waiters = [asyncio.create_task(acquire_sandbox(remote(coordinator), spec, 0.1)) for _ in range(80)]
    await asyncio.sleep(0.02)
    await asyncio.wait_for(handle.destroy(), 0.1)
    results = await asyncio.gather(*waiters, return_exceptions=True)
    assert all(isinstance(result, TimeoutError) for result in results)
    assert not coordinator._reservations
    assert not instance.live
    await coordinator.shutdown()


async def test_resource_budget_rejects_impossible_demands_without_waiting():
    coordinator = EnvWorkerCoordinator()
    await coordinator.register_worker(remote(OwnedWorker()), 0, 56, 0, "local", 8, 32 * 1024**3)
    with pytest.raises(ValueError, match="budget"):
        await acquire_sandbox(remote(coordinator), SandboxSpec(image="x", cpus=8, auxiliary_cpus=1), 1)
    await coordinator.shutdown()


async def test_stalled_cleanup_does_not_block_other_expired_owners(monkeypatch):
    from psrl.workers.env_worker import coordinator as module

    monkeypatch.setattr(module, "_CLEANUP_TIMEOUT_S", 0.1)
    coordinator = EnvWorkerCoordinator()
    instance = OwnedWorker()
    original_destroy = instance.destroy_sandbox
    blocked = None

    async def destroy(sandbox_id):
        if sandbox_id == blocked:
            await asyncio.Event().wait()
        await original_destroy(sandbox_id)

    instance.destroy_sandbox = destroy
    await coordinator.register_worker(remote(instance), 0, 2, 0, "local")
    spec = SandboxSpec(image="local", idle_timeout_s=0.1)
    first = await coordinator.reserve(spec, "blocked")
    second = await coordinator.reserve(spec, "healthy")
    blocked = first.sandbox_id
    for handle in (first, second):
        await instance.create_sandbox(spec, handle.sandbox_id)
        coordinator._reservations[handle.sandbox_id].last_seen -= 1
    try:
        reclaimed = await asyncio.wait_for(coordinator.reap_expired(), 0.5)
        assert reclaimed == ["healthy"], "A stuck worker must not prevent independent cleanup."
        assert coordinator._slots[0].used_cpu == 1, "Unconfirmed deletion must remain charged."
        assert instance.live == {"blocked"}, "Only the confirmed deletion may release capacity."
    finally:
        blocked = None
        await coordinator.shutdown()
