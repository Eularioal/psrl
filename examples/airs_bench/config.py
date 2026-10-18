"""
Runtime configuration for the AIRS-Bench recipe.

Mirrors the structured-config pattern used by `examples/mem_agent/config.py`, so
values can be supplied from the launch script and merged onto typed defaults.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULT_IMAGE = "psrl/airs-bench-agent:latest"
DEFAULT_DATA_ROOT = os.getenv("AIRS_DATA_ROOT", "data/airs-bench")
DEFAULT_MLGYM_REPO = os.getenv("MLGYM_REPO", "external/MLGym")
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_AGENT_CONFIG = str(_THIS_DIR / "agent_config.yaml")


@dataclass
class AirsBenchRuntimeConfig:
    """
    Per-episode settings owned by the AIRS-Bench recipe.

    Attributes:
        image (str): Sandbox container image.
        sandbox_cpus (float): Docker CPU limit for one sandbox.
        sandbox_memory (str): Docker memory limit for one sandbox.
        data_root (str): Root holding airs_prepared and airs_configs.
        mlgym_repo (str): Path to the read-only MLGym checkout.
        agent_config_path (str): MLGym agent config. Must pin DefaultHistoryProcessor.
        episode_timeout_s (float): Wall-clock cap for one whole episode.
        per_action_timeout_s (float): Cap for one agent action. Covers MLGym's
            3600 second training_timeout.
        startup_timeout_s (float): Cap for sandbox shell readiness.
    """

    image: str = DEFAULT_IMAGE
    sandbox_cpus: float = 8.0
    sandbox_memory: str = "32g"
    data_root: str = DEFAULT_DATA_ROOT
    mlgym_repo: str = DEFAULT_MLGYM_REPO
    agent_config_path: str = DEFAULT_AGENT_CONFIG
    episode_timeout_s: float = 10800.0
    per_action_timeout_s: float = 3600.0
    startup_timeout_s: float = 60.0
    lease_timeout_s: float = 180.0
    verifier_cpus: float = 1.0
    verifier_memory: str = "4g"
    sandbox_network: str = "none"
    apps_candidate_cpus: float = 1.0
    apps_candidate_memory: str = "512m"
    apps_case_timeout_s: float = 4.0
    apps_candidate_timeout_s: float = 11.0

    def __post_init__(self) -> None:
        for name in (
            "sandbox_cpus",
            "verifier_cpus",
            "episode_timeout_s",
            "per_action_timeout_s",
            "startup_timeout_s",
            "lease_timeout_s",
            "apps_candidate_cpus",
            "apps_case_timeout_s",
            "apps_candidate_timeout_s",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"AIRS runtime {name!r} must be finite and positive.")


def build_runtime_config(value: DictConfig | dict[str, Any] | None) -> AirsBenchRuntimeConfig:
    """
    Merge YAML values onto the structured AIRS-Bench schema.

    Args:
        value (DictConfig | dict[str, Any] | None): Overrides from configuration.

    Returns:
        AirsBenchRuntimeConfig: Fully populated runtime config.
    """
    raw = OmegaConf.create(value or {})
    merged = OmegaConf.merge(OmegaConf.structured(AirsBenchRuntimeConfig), raw)
    config: AirsBenchRuntimeConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    return config
