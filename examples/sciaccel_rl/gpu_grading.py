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


def save_gpu_reports(directory, reports, rewards, *, mode=None):
    if mode is not None:
        validate_attribution_mode(mode)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "gpu-job-reports.json"
    temporary = destination.with_suffix(".tmp")
    payload = {"jobs": reports, "training_rewards": rewards}
    if mode is not None:
        payload["gpu_attribution"] = {
            "mode": mode,
            "enforced": mode == "required",
            "score_source": "task_verifier_unverified" if mode == "observe" else "host_process_attribution",
            "preflight": "skipped_by_policy" if mode == "observe" else "required",
            "risk": "Whole-device activity can include neighboring tasks." if mode == "observe" else None,
        }
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(destination)


def validate_attribution_mode(mode):
    if mode not in ("required", "observe"):
        raise ValueError(f"Unknown GPU attribution mode {mode!r}.")


def apply_gpu_activity_gate(rewards, reports, checks, *, mode="required"):
    """Preserve numerical correctness while replacing whole-device activity attribution."""
    validate_attribution_mode(mode)
    if mode == "observe":
        return {**rewards, "gpu_attribution_enforced": 0.0}
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
