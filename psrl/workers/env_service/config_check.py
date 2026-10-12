"""
Startup validation of `psrl.env_service`, so a bad config fails at launch, not mid-run.

A pure function over a plain mapping, so it is testable without Hydra, Ray or verl.
"""

from __future__ import annotations

from collections.abc import Mapping

from psrl.workers.env_service.node_policy import parse_policy
from psrl.workers.env_service.types import EnvUnit, validate_units


def validate_env_service_config(cfg: Mapping) -> list[EnvUnit]:
    """
    Validate an enabled `psrl.env_service` section and return its parsed units.

    Args:
        cfg (Mapping): The `psrl.env_service` section as a plain mapping.

    Returns:
        list[EnvUnit]: The declared units.

    Raises:
        ValueError: With a message naming the offending key, on the first problem found.
    """
    parse_policy(cfg.get("node_policy", "cpu_only"))  # raises with the valid choices

    raw_units = list(cfg.get("units") or ())
    if not raw_units:
        raise ValueError(
            "psrl.env_service.enable=True requires at least one entry in psrl.env_service.units, "
            "because a container can only be admitted to a declared unit."
        )
    units: list[EnvUnit] = []
    for position, raw in enumerate(raw_units):
        try:
            units.append(EnvUnit.from_mapping(raw))
        except (KeyError, TypeError, ValueError) as exc:
            detail = f"missing key {exc}" if isinstance(exc, KeyError) else str(exc)
            raise ValueError(f"psrl.env_service.units[{position}] is invalid: {detail}.") from exc
    validate_units(units)

    default_gpu_mb = int(cfg.get("default_gpu_memory_mb", 24000))
    if default_gpu_mb <= 0:
        raise ValueError(f"psrl.env_service.default_gpu_memory_mb must be positive, got {default_gpu_mb}.")

    ttl = float(cfg.get("slot_ttl_s", 1800))
    beat = float(cfg.get("heartbeat_interval_s", 60))
    if ttl <= 0 or beat <= 0:
        raise ValueError(f"psrl.env_service.slot_ttl_s and heartbeat_interval_s must be positive, got {ttl}, {beat}.")
    if beat * 2 > ttl:
        raise ValueError(
            f"psrl.env_service.heartbeat_interval_s={beat} must be at most half of slot_ttl_s={ttl}, "
            f"or a live lease can be reclaimed between two heartbeats."
        )

    poll = float(cfg.get("poll_interval_s", 1.0))
    poll_max = float(cfg.get("max_poll_interval_s", 5.0))
    if poll <= 0 or poll_max < poll:
        raise ValueError(
            f"psrl.env_service needs 0 < poll_interval_s <= max_poll_interval_s, got {poll} and {poll_max}."
        )

    # A GPU unit that cannot fit even one default-sized task would reject every GPU episode.
    for unit in units:
        if unit.has_gpu and unit.gpu_memory_mb < default_gpu_mb:
            raise ValueError(
                f"Unit {unit.unit_id!r} has gpu_memory_mb={unit.gpu_memory_mb}, below "
                f"default_gpu_memory_mb={default_gpu_mb}, so it could never host a GPU episode."
            )
    return units
