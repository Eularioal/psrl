"""Reject foreign activity, stale process samples and unsupported GPU measurements."""

from types import SimpleNamespace

import pytest
from examples.sciaccel_rl.gpu_grading import apply_gpu_activity_gate
from psrl.workers.env_service.task_telemetry import TaskGpuTelemetry, activity_verified, process_identities


def reading(samples, memory=None, status="available"):
    return {
        0: {
            "uuid": "gpu",
            "memory_mb_by_pid": memory or {},
            "utilization_status": status,
            "samples": samples,
        }
    }


def test_busy_neighbor_does_not_activate_cpu_task_reward():
    task = TaskGpuTelemetry(None, None, (0,))
    task.started_us = task.since_us = 100
    task.consume(
        reading([{"pid": 99, "timestamp_us": t, "sm_util": 95} for t in (101, 102, 103)], {99: 4000, 42: 100}),
        {42: (1, 90)},
    )
    report = task.summary()
    assert report["devices"][0]["peak_memory_mb"] == 100
    assert report["devices"][0]["peak_sm_util"] is None
    assert not activity_verified(report)
    rewards = apply_gpu_activity_gate(
        {"reward": 1, "reward_gpu": 1},
        [{"tag": "check", "state": "completed", "telemetry": report}],
        ["check"],
    )
    assert rewards["reward"] == 1 and rewards["reward_gpu"] == 0


def test_owned_samples_are_not_counted_twice_and_pid_reuse_is_rejected():
    task = TaskGpuTelemetry(None, None, (0,))
    task.started_us = task.since_us = 100
    samples = [{"pid": 42, "timestamp_us": t, "sm_util": 50} for t in (110, 120, 130)]
    task.consume(reading(samples), {42: (1, 115)})
    assert task.summary()["devices"][0]["observed_samples"] == 2
    task.consume(reading(samples), {42: (1, 115)})
    assert task.summary()["devices"][0]["observed_samples"] == 2
    task.consume(reading([{"pid": 42, "timestamp_us": 140, "sm_util": 50}]), {42: (1, 115)})
    assert activity_verified(task.summary())


def test_unknown_utilization_is_not_zero_or_a_success():
    task = TaskGpuTelemetry(None, None, (0,))
    task.consume(reading([], {42: 100}, "unsupported"), {42: (1, 0)})
    report = task.summary()
    assert report["devices"][0]["peak_memory_mb"] == 100
    assert report["devices"][0]["peak_sm_util"] is None
    assert not activity_verified(report)


def test_every_required_check_needs_its_own_completed_report():
    task = TaskGpuTelemetry(None, None, (0,))
    task.started_us = task.since_us = 100
    task.consume(reading([{"pid": 42, "timestamp_us": t, "sm_util": 80} for t in (110, 120, 130)]), {42: (1, 90)})
    report = {"tag": "a", "state": "completed", "telemetry": task.summary()}
    assert apply_gpu_activity_gate({"reward": 0.8}, [report], ["a", "b"])["reward_gpu"] == 0
    assert apply_gpu_activity_gate({"reward": 0.8}, [report], ["a"])["reward_gpu"] == 0.8


def test_kernel_identity_is_stable_when_wall_clock_conversion_drifts(monkeypatch):
    task = TaskGpuTelemetry(SimpleNamespace(read_process_snapshot=lambda *args: reading([], {42: 100})), None, (0,))
    identities = iter([{42: (123, 1000000)}, {42: (123, 1000002)}])
    monkeypatch.setattr(task, "check_proc_root", lambda: None)
    monkeypatch.setattr(task, "_owners", lambda pids: next(identities))
    task.sample()
    assert task.summary()["devices"][0]["peak_memory_mb"] == 100


def test_process_replaced_during_query_is_not_attributed(monkeypatch):
    task = TaskGpuTelemetry(SimpleNamespace(read_process_snapshot=lambda *args: reading([], {42: 100})), None, (0,))
    identities = iter([{42: (123, 1000000)}, {42: (124, 1010000)}])
    monkeypatch.setattr(task, "check_proc_root", lambda: None)
    monkeypatch.setattr(task, "_owners", lambda pids: next(identities))
    task.sample()
    assert task.summary()["devices"][0]["peak_memory_mb"] == 0


def test_host_proc_matches_exact_cgroup_and_descendants(tmp_path):
    from pathlib import Path

    for pid, group in ((42, "/docker/ours"), (43, "/docker/ours/child"), (44, "/docker/ours-neighbor")):
        root = tmp_path / str(pid)
        root.mkdir()
        (root / "cgroup").write_text(f"11:devices:{group}\n")
        (root / "stat").write_text(f"{pid} (worker name) " + " ".join(["S", *(["0"] * 18), "123"]))
    assert set(process_identities([42, 43, 44], tmp_path, Path("/docker/ours"))) == {42, 43}


@pytest.mark.asyncio
async def test_missing_host_proc_stops_sampler_and_rejects_grading(tmp_path):
    task = TaskGpuTelemetry(None, None, (0,))
    task.proc_root = tmp_path
    await task._sample()
    assert task.stop_event.is_set() and task.summary()["errors"]
    with pytest.raises(RuntimeError, match="host process attribution"):
        apply_gpu_activity_gate(
            {"reward": 1},
            [
                {"tag": "a", "state": "completed", "telemetry": task.summary()},
            ],
            ["a"],
        )
