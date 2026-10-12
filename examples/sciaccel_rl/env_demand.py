"""
Translate a SciAccel task into the `EnvDemand` that Gate A charges against a unit.

Reads `[environment]` straight from `task.toml` with `tomllib`, the same fields
`prepare/build_dataset.py` already reads, so admission can run before Harbor exists in
the picture. The design document (section 7.4) notes that Harbor validates `[task] name`
as `org/name`, which a bare parse of the file sidesteps.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

from psrl.workers.env_service.types import EnvDemand

psrl_logger = logging.getLogger("psrl.sciaccel_rl.env_demand")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def demand_from_task(
    task_path: str,
    *,
    needs_gpu_hint: bool = False,
    gpu_memory_mb_override: int = 0,
) -> EnvDemand:
    """
    Build the demand for one episode.

    Harbor declares CPUs, host memory and a GPU count, but no device memory, so the device
    memory is either supplied by the caller (for example a per-task value in the dataset's
    `extra_info`) or left at zero for the router to fill with its configured default.

    Args:
        task_path (str): Directory containing `task.toml`.
        needs_gpu_hint (bool): The dataset row's own claim that the task needs a GPU. Used
            when `task.toml` is unreadable, and ORed with the file otherwise, so a row that
            says it needs a GPU is never silently run without one.
        gpu_memory_mb_override (int): Device memory to charge. Zero means "router default".

    Returns:
        EnvDemand: The demand. A missing or malformed `task.toml` degrades to one CPU and
        no memory accounting, with a warning, because refusing to run an episode over an
        unreadable resource hint would turn a bookkeeping gap into lost training data.
    """
    environment: dict = {}
    toml_path = Path(task_path) / "task.toml"
    try:
        environment = tomllib.loads(toml_path.read_text(encoding="utf-8")).get("environment", {}) or {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        psrl_logger.warning("Cannot read %s (%s); charging a default one-CPU demand.", toml_path, exc)

    cpus = _as_float(environment.get("cpus"), default=1.0)
    memory_mb = int(_as_float(environment.get("memory_mb"), default=0.0))
    needs_gpu = needs_gpu_hint or int(_as_float(environment.get("gpus"), default=0.0)) > 0
    return EnvDemand(
        cpu=max(cpus, 0.0) or 1.0,
        memory_mb=max(memory_mb, 0),
        needs_gpu=needs_gpu,
        gpu_memory_mb=int(gpu_memory_mb_override) if needs_gpu else 0,
    )


def _as_float(value: object, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
