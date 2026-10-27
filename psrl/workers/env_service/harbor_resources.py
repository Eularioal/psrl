"""
Translate a Harbor task into one resource plan shared by dispatch and admission.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path

import tomllib
import yaml

from psrl.workers.env_service.container_budget import ContainerBudget
from psrl.workers.env_service.types import EnvDemand

psrl_logger = logging.getLogger("psrl.env_service.harbor_resources")
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
        EnvDemand: Host limits for main services and sidecars across both environments.
            Missing resource declarations use a bounded one CPU and 2048 MB main service.
    """
    task: dict = {}
    toml_path = Path(task_path) / "task.toml"
    try:
        task = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        psrl_logger.warning("Cannot read %s (%s). Applying bounded default container resources.", toml_path, exc)

    environment = task.get("environment", {}) or {}
    cpus = _as_float(environment.get("cpus"), default=1.0)
    memory_mb = int(_as_float(environment.get("memory_mb"), default=0.0))
    verifier_environments = [task.get("verifier", {}).get("environment", {})] + [
        step.get("verifier", {}).get("environment", {}) for step in task.get("steps", [])
    ]
    needs_gpu = needs_gpu_hint or any(
        int(_as_float(env.get("gpus"), default=0.0)) > 0 for env in [environment, *verifier_environments]
    )
    demand = EnvDemand(
        cpu=max(cpus, 0.0) or 1.0,
        memory_mb=max(memory_mb, 0),
        needs_gpu=needs_gpu,
        gpu_memory_mb=int(gpu_memory_mb_override) if needs_gpu else 0,
    )
    return with_container_budget(task_path, demand, task=task)


def with_container_budget(task_path: str, demand: EnvDemand, *, task: dict | None = None) -> EnvDemand:
    """Charge both environment sets and their bounded sidecars before Harbor creates them."""
    if demand.container_budget is not None:
        return demand
    if task is None:
        try:
            task = tomllib.loads((Path(task_path) / "task.toml").read_text())
        except (OSError, tomllib.TOMLDecodeError):
            task = {}
    verifiers = [task.get("verifier", {})] + [step.get("verifier", {}) for step in task.get("steps", [])]
    separate = any(v.get("environment_mode") == "separate" or v.get("environment") for v in verifiers)
    environments = [task.get("environment", {})] + [v.get("environment", {}) for v in verifiers]
    cpu = max([demand.cpu, 1.0] + [_as_float(e.get("cpus"), 1.0) for e in environments])
    memory = max([demand.memory_mb or 2048] + [int(_as_float(e.get("memory_mb"), 0)) for e in environments])
    sidecars = 0
    for path in Path(task_path).rglob("docker-compose.yaml"):
        document = yaml.load(path.read_text(), Loader=yaml.BaseLoader) or {}
        sidecars = max(sidecars, len(set(document.get("services", {})) - {"main"}))
    # Harbor can add one network service to either environment after admission.
    settings = task.get("metadata", {}).get("env_service", {})
    budget = ContainerBudget(
        cpu,
        memory,
        sidecars + 1,
        2 if separate else 1,
        sidecar_cpu=float(settings.get("sidecar_cpu", 0.25)),
        sidecar_memory_mb=int(settings.get("sidecar_memory_mb", 128)),
    )
    return replace(demand, cpu=budget.cpu, memory_mb=budget.memory_mb, container_budget=budget)


def _as_float(value: object, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
