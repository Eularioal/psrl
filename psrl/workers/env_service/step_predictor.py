"""Estimate fixed-step work from completed wall-clock progress chunks."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Literal


@dataclass(frozen=True)
class StepForecast:
    """Keep missing or overdue observations distinct from confirmed completion."""

    status: Literal["estimated", "complete", "insufficient_history", "overrun"]
    estimate_s: float | None
    lower_s: float | None
    upper_s: float | None
    per_step_estimate_s: float | None
    per_step_lower_s: float | None
    per_step_upper_s: float | None
    sample_chunks: int
    next_report_upper_s: float | None


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def predict_remaining(
    chunk_steps: Sequence[int],
    chunk_wall_s: Sequence[float],
    completed_steps: int,
    total_steps: int,
    elapsed_since_report_s: float = 0,
    *,
    report_every_steps: int = 16,
    window_chunks: int = 8,
    lower_factor: float = 0.8,
    upper_factor: float = 1.3,
    report_grace_s: float = 0.05,
    min_chunks: int = 2,
) -> StepForecast:
    """
    Forecast remaining work from the median of recent chunk wall times per step.

    A chunk includes computation and the preceding progress publication. Its report
    timestamp marks the boundary before publishing the next observation. The padded
    10th/90th percentiles are empirical intervals, not confidence or worst-case bounds.

    Args:
        chunk_steps (Sequence[int]): Completed update counts for each observed chunk.
        chunk_wall_s (Sequence[float]): Matching observed wall durations in seconds.
        completed_steps (int): Total updates completed at the reported boundary.
        total_steps (int): Fixed total update count for the command.
        elapsed_since_report_s (float): Wall time since that boundary, including publication.
        report_every_steps (int): Maximum updates before the next report is published.
        window_chunks (int): Number of recent chunks retained for estimation.
        lower_factor (float): Multiplier at or below one for the lower empirical rate.
        upper_factor (float): Multiplier at or above one for the upper empirical rate.
        report_grace_s (float): Additional wall seconds allowed for the next report.
        min_chunks (int): Minimum observed chunks before a forecast is available.

    Returns:
        StepForecast: A point estimate and interval, or the reason they remain unknown.
    """
    if type(completed_steps) is not int or type(total_steps) is not int:
        raise ValueError("Step counts must be integers.")
    if completed_steps < 0 or total_steps < completed_steps:
        raise ValueError("Step counts must satisfy zero <= completed <= total.")
    for count in (report_every_steps, window_chunks, min_chunks):
        if type(count) is not int or count < 1:
            raise ValueError("Report and observation counts must be positive integers.")
    if min_chunks > window_chunks:
        raise ValueError("Minimum history cannot exceed the observation window.")
    if not isfinite(elapsed_since_report_s) or elapsed_since_report_s < 0:
        raise ValueError("Observation age must be finite and nonnegative.")
    if not isfinite(report_grace_s) or report_grace_s < 0:
        raise ValueError("Report grace must be finite and nonnegative.")
    if not isfinite(lower_factor) or not 0 < lower_factor <= 1:
        raise ValueError("Lower padding must be finite and in (0, 1].")
    if not isfinite(upper_factor) or upper_factor < 1:
        raise ValueError("Upper padding must be finite and at least one.")
    if len(chunk_steps) != len(chunk_wall_s):
        raise ValueError("Chunk update counts and wall durations must have equal lengths.")
    if any(type(count) is not int or count < 1 for count in chunk_steps):
        raise ValueError("Observed chunk update counts must be positive integers.")
    if any(not isfinite(elapsed) or elapsed <= 0 for elapsed in chunk_wall_s):
        raise ValueError("Observed chunk wall durations must be finite and positive.")
    if sum(chunk_steps) > completed_steps:
        raise ValueError("Observed chunks cannot contain more updates than the reported completion count.")

    pending = total_steps - completed_steps
    if pending == 0:
        return StepForecast("complete", 0, 0, 0, None, None, None, 0, None)
    samples = [wall / count for count, wall in zip(chunk_steps, chunk_wall_s, strict=True)][-window_chunks:]
    if len(samples) < min_chunks:
        return StepForecast("insufficient_history", None, None, None, None, None, None, len(samples), None)

    point = median(samples)
    lower = _quantile(samples, 0.1) * lower_factor
    upper = _quantile(samples, 0.9) * upper_factor
    next_steps = min(report_every_steps, pending)
    next_upper = next_steps * upper + report_grace_s
    if elapsed_since_report_s >= next_upper:
        return StepForecast("overrun", None, None, None, point, lower, upper, len(samples), next_upper)

    later_steps = pending - next_steps
    lower_remaining = max(0, next_steps * lower - elapsed_since_report_s) + later_steps * lower
    point_remaining = max(0, next_steps * point - elapsed_since_report_s) + later_steps * point
    upper_remaining = next_upper - elapsed_since_report_s + later_steps * upper
    return StepForecast(
        "estimated",
        point_remaining,
        lower_remaining,
        upper_remaining,
        point,
        lower,
        upper,
        len(samples),
        next_upper,
    )
