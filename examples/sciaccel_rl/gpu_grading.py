"""Gate native CUDA rewards on host observations attributed to each required check."""

import json
from pathlib import Path

import tomllib
from psrl.workers.env_service.task_telemetry import activity_verified

LAPS_CUDA_CHECKS = ("aw-2d-256", "aw-2d-512", "aw-128")


def is_laps_cuda(task_path):
    path = Path(task_path)
    manifest = path / "task.toml"
    if manifest.exists():
        name = tomllib.loads(manifest.read_text()).get("task", {}).get("name", path.name)
    else:
        name = path.name
    return name.rsplit("/", 1)[-1] == "laps-accel-cuda"


def save_gpu_reports(directory, reports, rewards):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "gpu-job-reports.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps({"jobs": reports, "training_rewards": rewards}, indent=2))
    temporary.replace(destination)


def apply_gpu_activity_gate(rewards, reports, checks):
    """Preserve numerical correctness while replacing whole-device activity attribution."""
    for report in reports:
        if report["tag"] in checks:
            telemetry = report.get("telemetry")
            if not telemetry or telemetry.get("errors"):
                raise RuntimeError("GPU activity cannot be graded without host process attribution.")
            if any(device["utilization_status"] == "unsupported" for device in telemetry["devices"].values()):
                raise RuntimeError("GPU activity cannot be graded because process utilization is unsupported.")
    verified = {
        report["tag"]
        for report in reports
        if report["state"] == "completed" and activity_verified(report["telemetry"])
    }
    active = float(bool(checks) and set(checks) <= verified)
    return {**rewards, "gpu_active": active, "reward_gpu": float(rewards.get("reward", 0)) * active}
