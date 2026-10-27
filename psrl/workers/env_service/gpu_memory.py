"""
Charge device usage and the unallocated portion of each live reservation.
"""

import math
from collections.abc import Mapping


def accounted_gpu_memory_mb(
    measured_mb: float,
    reservations: Mapping[str, float],
    owned_mb: Mapping[str, float] | None = None,
) -> float:
    """
    Add outstanding promises to a fresh device measurement without double charging owners.

    Ownership credits must come from the same observation as `measured_mb`.
    Missing ownership earns no credit, even when aggregate usage has increased.
    """
    owned_mb = owned_mb or {}
    values = [measured_mb, *reservations.values(), *owned_mb.values()]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("GPU memory observations and reservations must be finite and nonnegative.")
    if owned_mb.keys() - reservations.keys() or sum(owned_mb.values()) > measured_mb:
        raise ValueError("GPU ownership credits must be disjoint and contained in measured usage.")
    return measured_mb + sum(
        max(0.0, amount - owned_mb.get(lease_id, 0.0)) for lease_id, amount in reservations.items()
    )
