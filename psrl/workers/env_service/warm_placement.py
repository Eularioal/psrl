"""Compare model reuse with rebuilding using explicitly uncertain duration estimates."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class DurationInterval:
    """Represent estimated elapsed seconds, without claiming a statistical guarantee."""

    lower_s: float
    upper_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.lower_s) or not isfinite(self.upper_s):
            raise ValueError("Duration bounds must be finite.")
        if self.lower_s < 0 or self.upper_s < self.lower_s:
            raise ValueError("Duration bounds must be nonnegative and ordered.")

    def __add__(self, other: "DurationInterval") -> "DurationInterval":
        return DurationInterval(self.lower_s + other.lower_s, self.upper_s + other.upper_s)


@dataclass(frozen=True)
class DeviceSnapshot:
    """
    Describe a serial device queue and the available model cache budget.

    `busy=True` with `active_remaining=None` means the active command is unknown.
    Every queued duration includes all service needed before the new request,
    including any rebuild. This module neither reserves memory nor dispatches work.
    """

    device_id: str
    resident_models: frozenset[str]
    free_cache_bytes: int
    busy: bool = False
    active_remaining: DurationInterval | None = None
    queued_durations: tuple[DurationInterval | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("Device identifier must be nonempty.")
        if not isinstance(self.free_cache_bytes, int) or self.free_cache_bytes < 0:
            raise ValueError("Free cache capacity must be a nonnegative integer.")
        if not self.busy and self.active_remaining is not None:
            raise ValueError("An idle device cannot have an active remaining duration.")
        object.__setattr__(self, "resident_models", frozenset(self.resident_models))
        object.__setattr__(self, "queued_durations", tuple(self.queued_durations))


@dataclass(frozen=True)
class PlacementRequest:
    """
    Describe one command with a reusable model and a relative completion budget.

    `model_bytes` is additional cache capacity required on a cold device.
    `execution_duration` excludes rebuild and queue waiting. A missing estimate
    remains unknown. `deadline_s` counts seconds from this snapshot, when supplied.
    """

    model_id: str
    model_bytes: int
    rebuild_duration: DurationInterval | None
    execution_duration: DurationInterval | None
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("Model identifier must be nonempty.")
        if not isinstance(self.model_bytes, int) or self.model_bytes < 0:
            raise ValueError("Model cache size must be a nonnegative integer.")
        if self.deadline_s is not None and (not isfinite(self.deadline_s) or self.deadline_s < 0):
            raise ValueError("Completion deadline must be finite and nonnegative.")


@dataclass(frozen=True)
class DeviceEstimate:
    """Keep waiting, model rebuilding, and execution costs separately inspectable."""

    device_id: str
    resident: bool
    memory_fits: bool
    waiting: DurationInterval | None
    rebuild: DurationInterval | None
    execution: DurationInterval | None
    completion: DurationInterval | None
    deadline_state: Literal["within", "miss", "uncertain", "unknown"]


@dataclass(frozen=True)
class PlacementDecision:
    """Return a proposal that the caller must revalidate before admission."""

    action: Literal["reuse", "wait", "rebuild", "defer", "reject"]
    device_id: str | None
    reason: str
    estimates: tuple[DeviceEstimate, ...]

    @property
    def selected(self) -> DeviceEstimate | None:
        """Return the selected cost breakdown, if a device was selected."""
        return next((estimate for estimate in self.estimates if estimate.device_id == self.device_id), None)


def estimate_device(request: PlacementRequest, device: DeviceSnapshot) -> DeviceEstimate:
    """Estimate completion after the active command and all already queued commands."""
    resident = request.model_id in device.resident_models
    memory_fits = resident or device.free_cache_bytes >= request.model_bytes
    waiting = device.active_remaining if device.busy else DurationInterval(0, 0)
    for queued in device.queued_durations:
        waiting = waiting + queued if waiting is not None and queued is not None else None
    rebuild = DurationInterval(0, 0) if resident else request.rebuild_duration
    execution = request.execution_duration
    completion = None
    if memory_fits and waiting is not None and rebuild is not None and execution is not None:
        completion = waiting + rebuild + execution
    deadline_state = "unknown"
    if completion is not None:
        if request.deadline_s is None or completion.upper_s <= request.deadline_s:
            deadline_state = "within"
        elif completion.lower_s > request.deadline_s:
            deadline_state = "miss"
        else:
            deadline_state = "uncertain"
    return DeviceEstimate(
        device.device_id, resident, memory_fits, waiting, rebuild, execution, completion, deadline_state
    )


def choose_placement(
    request: PlacementRequest,
    devices: Sequence[DeviceSnapshot],
    *,
    switch_margin_s: float = 0,
) -> PlacementDecision:
    """
    Rebuild only when its estimated upper finish beats the hot lower finish.

    Overlapping intervals retain model affinity. Unknown hot waits defer rather
    than becoming zero or implying that a known cold estimate is faster. Without
    a hot replica, choose a fully estimated cold device that meets the deadline.
    All interval comparisons are conditional on the supplied estimates remaining
    valid, and never guarantee minimum idle time or actual deadline completion.
    """
    if not isfinite(switch_margin_s) or switch_margin_s < 0:
        raise ValueError("Switch margin must be finite and nonnegative.")
    if len({device.device_id for device in devices}) != len(devices):
        raise ValueError("Device identifiers must be unique.")
    estimates = tuple(estimate_device(request, device) for device in devices)

    def decision(
        action: Literal["reuse", "wait", "rebuild", "defer", "reject"],
        selected: DeviceEstimate | None,
        reason: str,
    ) -> PlacementDecision:
        return PlacementDecision(action, selected.device_id if selected is not None else None, reason, estimates)

    fitting = [estimate for estimate in estimates if estimate.memory_fits]
    if not fitting:
        return decision("reject", None, "No device has sufficient model cache capacity.")
    hot = [estimate for estimate in fitting if estimate.resident]
    known_hot = [estimate for estimate in hot if estimate.completion is not None]
    cold = [estimate for estimate in fitting if not estimate.resident and estimate.deadline_state == "within"]
    best_cold = min(cold, key=lambda estimate: (estimate.completion.upper_s, estimate.device_id), default=None)
    if known_hot:
        best_hot = min(known_hot, key=lambda estimate: (estimate.completion.upper_s, estimate.device_id))
        if best_cold is not None and (best_cold.completion.upper_s + switch_margin_s < best_hot.completion.lower_s):
            return decision("rebuild", best_cold, "Rebuilding finishes before the hot lower estimate with margin.")
        if best_hot.deadline_state == "within":
            action = "wait" if best_hot.waiting.upper_s > 0 else "reuse"
            return decision(action, best_hot, "Model affinity avoids an unproven rebuild benefit.")
    elif hot:
        return decision("defer", None, "Hot completion is unknown, so a rebuild advantage cannot be established.")
    elif best_cold is not None:
        return decision("rebuild", best_cold, "No hot replica exists, so a known feasible cold device is selected.")
    if all(estimate.deadline_state == "miss" for estimate in fitting):
        return decision("reject", None, "Every fitting device exceeds the completion deadline even optimistically.")
    return decision("defer", None, "Available estimates cannot establish a feasible completion.")


@dataclass(frozen=True)
class ProgressEstimate:
    """Expose a fixed step forecast or the reason it must remain unknown."""

    remaining: DurationInterval | None
    per_step: DurationInterval | None
    status: Literal["complete", "estimated", "insufficient_history", "overrun"]


def estimate_remaining(
    sample_step_durations_s: Sequence[float],
    completed_steps: int,
    total_steps: int,
    elapsed_step_s: float = 0,
    *,
    lower_factor: float = 0.8,
    upper_factor: float = 1.5,
) -> ProgressEstimate:
    """
    Forecast a bounded command whose remaining steps resemble the measured steps.

    Bounds use the smallest and largest completed sample with explicit padding.
    They describe an empirical forecast, not a worst case bound. `completed_steps`
    excludes the ongoing step. `elapsed_step_s` is time inside that ongoing step,
    excluding rebuild. An unobserved completion at the upper bound invalidates the
    forecast. Arbitrary shell commands need another progress source or stay unknown.
    """
    if not isinstance(completed_steps, int) or not isinstance(total_steps, int):
        raise ValueError("Step counts must be integers.")
    if completed_steps < 0 or total_steps < completed_steps:
        raise ValueError("Step counts must satisfy zero <= completed <= total.")
    if not isfinite(elapsed_step_s) or elapsed_step_s < 0:
        raise ValueError("Elapsed step duration must be finite and nonnegative.")
    if not isfinite(lower_factor) or not 0 < lower_factor <= 1:
        raise ValueError("Lower padding factor must be finite and in (0, 1].")
    if not isfinite(upper_factor) or upper_factor < 1:
        raise ValueError("Upper padding factor must be finite and at least one.")
    if any(not isfinite(sample) or sample <= 0 for sample in sample_step_durations_s):
        raise ValueError("Measured step durations must be finite and positive.")
    pending = total_steps - completed_steps
    if pending == 0:
        if elapsed_step_s:
            raise ValueError("A completed command cannot have an ongoing step.")
        return ProgressEstimate(DurationInterval(0, 0), None, "complete")
    if not sample_step_durations_s:
        return ProgressEstimate(None, None, "insufficient_history")
    per_step = DurationInterval(
        min(sample_step_durations_s) * lower_factor, max(sample_step_durations_s) * upper_factor
    )
    if elapsed_step_s >= per_step.upper_s:
        return ProgressEstimate(None, per_step, "overrun")
    remaining = DurationInterval(
        max(0, per_step.lower_s - elapsed_step_s) + (pending - 1) * per_step.lower_s,
        per_step.upper_s * pending - elapsed_step_s,
    )
    return ProgressEstimate(remaining, per_step, "estimated")
