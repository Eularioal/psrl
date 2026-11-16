"""Check bounded client cleanup and missing monitoring data without GPU access."""

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "docs/env-worker-deep-dive/env-service-verification/scripts/warm_router_study"
)


def load_helper(name):
    spec = importlib.util.spec_from_file_location("test_warm_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_expired_cleanup_does_not_start_a_new_coroutine():
    guard = load_helper("runtime_guard")
    ran = []

    async def work():
        ran.append(True)

    coroutine = work()
    with pytest.raises(TimeoutError):
        await guard.await_cleanup(coroutine, time.monotonic() - 1)
    assert ran == [] and coroutine.cr_frame is None


async def test_uncooperative_cleanup_returns_at_deadline_without_claiming_completion():
    guard = load_helper("runtime_guard")
    release = asyncio.Event()
    tasks = []

    async def work():
        tasks.append(asyncio.current_task())
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await guard.await_cleanup(work(), time.monotonic() + 0.03)
        assert time.monotonic() - started < 0.3
        assert tasks and not tasks[0].done()
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_cancel_join_deduplicates_clients_and_reports_a_noncooperative_one_as_pending():
    guard = load_helper("runtime_guard")
    release = asyncio.Event()

    async def work():
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(work())
    await asyncio.sleep(0)
    started = time.monotonic()
    try:
        result = await guard.cancel_and_join([task, task, asyncio.current_task()], deadline=time.monotonic() + 0.03)
        assert result == dict(cancel_requested=1, joined=False, pending=1)
        assert time.monotonic() - started < 0.3
    finally:
        release.set()
        await task


def test_archive_failure_preserves_the_runtime_evidence(monkeypatch, tmp_path):
    guard = load_helper("runtime_guard")
    runtime = tmp_path / "runtime"
    service = runtime / "service-0"
    service.mkdir(parents=True)
    evidence = service / "progress.json"
    evidence.write_text('{"completed": 64}')
    archive = tmp_path / "archive"
    archive.mkdir()

    def fail_copy(*args):
        raise OSError("The archive destination is unavailable.")

    monkeypatch.setattr(guard.shutil, "copy2", fail_copy)
    with pytest.raises(OSError):
        guard.archive_runtime(runtime, archive)
    assert evidence.read_text() == '{"completed": 64}'


def test_sampler_keeps_na_and_missing_gpu_distinct_from_zero(monkeypatch, tmp_path):
    module = load_helper("gpu_sampler")
    calls = []

    def query(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="1, N/A, 100, 1200, 150\n")

    monkeypatch.setattr(module.subprocess, "run", query)
    rows = module.GpuSampler([1, 3], tmp_path / "samples.jsonl")._sample()
    assert rows[0]["status"] == "partial" and rows[0]["utilization_gpu_pct"] is None
    assert rows[1]["status"] == "missing" and "utilization_gpu_pct" not in rows[1]
    assert calls[0][1]["timeout"] == 3
    assert "--id=1,3" in calls[0][0]


def test_sampler_real_cli_timeout_is_bounded_and_records_missing_observations(monkeypatch, tmp_path):
    module = load_helper("gpu_sampler")
    executable = tmp_path / "nvidia-smi"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    started = time.monotonic()
    rows = module.GpuSampler([1, 3], tmp_path / "samples.jsonl")._sample()
    assert 2.5 < time.monotonic() - started < 5
    assert [row["status"] for row in rows] == ["timeout", "timeout"]
    assert all("utilization_gpu_pct" not in row for row in rows)


def test_sampler_shutdown_wait_has_a_five_second_cap(tmp_path):
    module = load_helper("gpu_sampler")
    sampler = module.GpuSampler([1, 3], tmp_path / "samples.jsonl")
    waits = []
    sampler._thread = SimpleNamespace(join=lambda timeout: waits.append(timeout), is_alive=lambda: True)
    assert sampler.stop()["status"] == "stop_timeout"
    assert waits == [5]
