"""Propose serial-queue model placement without reserving or dispatching GPU work."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from .step_predictor import StepForecast
from .warm_placement import DurationInterval

DecisionMode = Literal["point", "interval"]
DispatchStrategy = Literal["affinity", "always_copy", "predictive"]
PlacementAction = Literal["reuse", "wait", "rebuild", "defer", "reject"]
ReasonCode = Literal[
    "no_declared_capacity",
    "known_cold_copy_requested",
    "predictive_margin_met",
    "unknown_hot_completion_affinity_fallback",
    "affinity_retained",
    "deadline_uncertain",
    "no_feasible_hot_replica",
    "predicted_deadline_miss",
    "cold_estimate_unavailable",
]


def _nonnegative_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number.")


def _nonnegative_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


@dataclass(frozen=True)
class TimeEstimate:
    """Retain a point and empirical interval without implying confidence coverage."""

    point_s: float
    interval: DurationInterval
    statistical_guarantee: Literal[False] = field(default=False, init=False)

    def __post_init__(self) -> None:
        _nonnegative_number(self.point_s, "Point duration")
        if not isinstance(self.interval, DurationInterval):
            raise ValueError("An empirical DurationInterval is required.")
        _nonnegative_number(self.interval.lower_s, "Lower duration")
        _nonnegative_number(self.interval.upper_s, "Upper duration")
        if not self.interval.lower_s <= self.point_s <= self.interval.upper_s:
            raise ValueError("The point must lie inside its empirical interval.")

    def __add__(self, other: "TimeEstimate") -> "TimeEstimate":
        if not isinstance(other, TimeEstimate):
            return NotImplemented
        return TimeEstimate(self.point_s + other.point_s, self.interval + other.interval)


def estimate_from_forecast(forecast: StepForecast) -> TimeEstimate | None:
    """Keep missing or overdue progress unknown instead of turning it into zero."""
    if not isinstance(forecast, StepForecast):
        raise ValueError("A StepForecast is required.")
    if forecast.status not in {"estimated", "complete", "insufficient_history", "overrun"}:
        raise ValueError("The forecast status is invalid.")
    if forecast.status not in {"estimated", "complete"}:
        return None
    for value in (forecast.estimate_s, forecast.lower_s, forecast.upper_s):
        _nonnegative_number(value, "Forecast duration")
    return TimeEstimate(forecast.estimate_s, DurationInterval(forecast.lower_s, forecast.upper_s))


@dataclass(frozen=True)
class ModelCost:
    """Describe one immutable model identity and the command's additional workspace."""

    model_id: str
    model_bytes: int
    rebuild: TimeEstimate | None
    execution: TimeEstimate | None
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        _identifier(self.model_id, "Model identity")
        _nonnegative_integer(self.model_bytes, "Model bytes")
        _nonnegative_integer(self.workspace_bytes, "Workspace bytes")
        for estimate in (self.rebuild, self.execution):
            if estimate is not None and not isinstance(estimate, TimeEstimate):
                raise ValueError("Model durations must be TimeEstimate values or unknown.")


@dataclass(frozen=True)
class QueueSnapshot:
    """
    Describe serial service and guaranteed capacity when the new command starts.

    Available bytes are the reservation budget minus resident and other retained
    state, before this command's workspace. Queued costs include all prior service.
    Resident model identities must remain valid through that prior service. This
    pure snapshot is not measured free memory or an admission guarantee.
    """

    device_id: str
    resident_models: frozenset[str]
    available_gpu_bytes: int
    busy: bool = False
    active_remaining: TimeEstimate | None = None
    queued_service: tuple[TimeEstimate | None, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.device_id, "Device identity")
        _nonnegative_integer(self.available_gpu_bytes, "Available GPU bytes")
        if type(self.busy) is not bool:
            raise ValueError("Busy must be a boolean.")
        if self.active_remaining is not None and not isinstance(self.active_remaining, TimeEstimate):
            raise ValueError("Active remaining service must be a TimeEstimate or unknown.")
        if not self.busy and self.active_remaining is not None:
            raise ValueError("An idle device cannot have active remaining service.")
        if isinstance(self.resident_models, (str, bytes)):
            raise ValueError("Resident models must be a collection of identities.")
        if isinstance(self.queued_service, (str, bytes)):
            raise ValueError("Queued service must be a collection of estimates.")
        try:
            models, queued = frozenset(self.resident_models), tuple(self.queued_service)
        except TypeError as exc:
            raise ValueError("Resident models and queued service must be iterable.") from exc
        for model in models:
            _identifier(model, "Resident model identity")
        if any(value is not None and not isinstance(value, TimeEstimate) for value in queued):
            raise ValueError("Queued service must contain TimeEstimate values or unknown.")
        object.__setattr__(self, "resident_models", models)
        object.__setattr__(self, "queued_service", queued)


def required_gpu_bytes(cost: ModelCost, device: QueueSnapshot) -> int:
    """Charge workspace on every device and model bytes only on a cold device."""
    if not isinstance(cost, ModelCost) or not isinstance(device, QueueSnapshot):
        raise ValueError("A ModelCost and QueueSnapshot are required.")
    return cost.workspace_bytes + (0 if cost.model_id in device.resident_models else cost.model_bytes)


def fits_capacity(cost: ModelCost, device: QueueSnapshot) -> bool:
    """Check the same declared capacity boundary used by every strategy."""
    return required_gpu_bytes(cost, device) <= device.available_gpu_bytes


@dataclass(frozen=True)
class DeviceCostEstimate:
    """Expose each cost component and declared capacity independently."""

    device_id: str
    resident: bool
    required_gpu_bytes: int
    available_gpu_bytes: int
    capacity_fits: bool
    waiting: TimeEstimate | None
    rebuild: TimeEstimate | None
    execution: TimeEstimate | None
    completion: TimeEstimate | None
    deadline_state: Literal["within", "miss", "uncertain", "unknown"]


@dataclass(frozen=True)
class SchedulingDecision:
    """
    Return an advisory placement and the estimates actually used to compare it.

    An unknown hot completion may select wait as an affinity fallback. Its missing
    completion stays unknown. Callers must serialize that queue and recheck actual
    admission before executing. This object never reserves or releases resources.
    """

    action: PlacementAction
    device_id: str | None
    reason_code: ReasonCode
    mode: DecisionMode
    strategy: DispatchStrategy
    switch_margin_s: float
    estimates: tuple[DeviceCostEstimate, ...]
    statistical_guarantee: Literal[False] = field(default=False, init=False)

    @property
    def selected(self) -> DeviceCostEstimate | None:
        return next((value for value in self.estimates if value.device_id == self.device_id), None)


def _estimate(cost: ModelCost, device: QueueSnapshot, deadline_s: float | None) -> DeviceCostEstimate:
    zero = TimeEstimate(0, DurationInterval(0, 0))
    waiting = device.active_remaining if device.busy else zero
    for queued in device.queued_service:
        waiting = waiting + queued if waiting is not None and queued is not None else None
    resident = cost.model_id in device.resident_models
    rebuild = zero if resident else cost.rebuild
    completion = None
    capacity_fits = fits_capacity(cost, device)
    if capacity_fits and waiting is not None and rebuild is not None and cost.execution is not None:
        completion = waiting + rebuild + cost.execution
    deadline_state: Literal["within", "miss", "uncertain", "unknown"] = "unknown"
    if completion is not None:
        if deadline_s is None or completion.interval.upper_s <= deadline_s:
            deadline_state = "within"
        elif completion.interval.lower_s > deadline_s:
            deadline_state = "miss"
        else:
            deadline_state = "uncertain"
    return DeviceCostEstimate(
        device.device_id,
        resident,
        required_gpu_bytes(cost, device),
        device.available_gpu_bytes,
        capacity_fits,
        waiting,
        rebuild,
        cost.execution,
        completion,
        deadline_state,
    )


@dataclass(frozen=True)
class WarmPlacementController:
    """Compare affinity with known cold copies using explicit decision semantics."""

    mode: DecisionMode = "point"
    switch_margin_s: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"point", "interval"}:
            raise ValueError("Mode must be point or interval.")
        _nonnegative_number(self.switch_margin_s, "Switch margin")

    def decide(
        self,
        cost: ModelCost,
        devices: Sequence[QueueSnapshot],
        *,
        strategy: DispatchStrategy = "predictive",
        device_costs: Mapping[str, ModelCost] | None = None,
        deadline_s: float | None = None,
    ) -> SchedulingDecision:
        """
        Propose one device without mutating the queue, model cache or GPU budget.

        Point mode compares point completion plus a switch margin. Interval mode
        requires cold upper plus margin below the selected hot lower estimate.
        Neither mode supplies a statistical or worst-case completion guarantee.
        Per-device costs must describe the same model and workspace requirements.
        """
        if not isinstance(cost, ModelCost):
            raise ValueError("A ModelCost is required.")
        if not isinstance(strategy, str) or strategy not in {"affinity", "always_copy", "predictive"}:
            raise ValueError("Unknown dispatch strategy.")
        if deadline_s is not None:
            _nonnegative_number(deadline_s, "Deadline")
        if not isinstance(devices, Sequence) or any(not isinstance(device, QueueSnapshot) for device in devices):
            raise ValueError("Devices must contain QueueSnapshot values.")
        identifiers = {device.device_id for device in devices}
        if len(identifiers) != len(devices):
            raise ValueError("Device identities must be unique.")
        overrides = {} if device_costs is None else device_costs
        if not isinstance(overrides, Mapping) or set(overrides) - identifiers:
            raise ValueError("Per-device costs must map known device identities.")
        for value in overrides.values():
            if not isinstance(value, ModelCost) or (value.model_id, value.model_bytes, value.workspace_bytes) != (
                cost.model_id,
                cost.model_bytes,
                cost.workspace_bytes,
            ):
                raise ValueError("Per-device costs must retain the same model, model bytes and workspace bytes.")
        estimates = tuple(_estimate(overrides.get(device.device_id, cost), device, deadline_s) for device in devices)

        def decision(action: PlacementAction, selected: DeviceCostEstimate | None, reason: ReasonCode):
            return SchedulingDecision(
                action,
                selected.device_id if selected is not None else None,
                reason,
                self.mode,
                strategy,
                self.switch_margin_s,
                estimates,
            )

        def score(estimate: DeviceCostEstimate) -> float:
            duration = estimate.completion
            if duration is None:
                raise ValueError("Only known completion estimates can be ranked.")
            return duration.point_s if self.mode == "point" else duration.interval.upper_s

        fitting = [value for value in estimates if value.capacity_fits]
        if not fitting:
            return decision("reject", None, "no_declared_capacity")
        hot = [value for value in fitting if value.resident and value.deadline_state != "miss"]
        known_hot = [value for value in hot if value.completion is not None]
        best_hot = min(known_hot, key=lambda value: (score(value), value.device_id), default=None)
        if best_hot is None:
            best_hot = min(hot, key=lambda value: value.device_id, default=None)
        cold = [
            value
            for value in fitting
            if not value.resident and value.completion is not None and value.deadline_state == "within"
        ]
        best_cold = min(cold, key=lambda value: (score(value), value.device_id), default=None)
        if best_hot is not None:
            if strategy == "always_copy" and best_cold is not None:
                return decision("rebuild", best_cold, "known_cold_copy_requested")
            if strategy == "predictive" and best_cold is not None and best_hot.completion is not None:
                hot_finish = best_hot.completion.point_s
                cold_finish = best_cold.completion.point_s
                if self.mode == "interval":
                    hot_finish = best_hot.completion.interval.lower_s
                    cold_finish = best_cold.completion.interval.upper_s
                if cold_finish + self.switch_margin_s < hot_finish:
                    return decision("rebuild", best_cold, "predictive_margin_met")
            if best_hot.completion is None:
                return decision("wait", best_hot, "unknown_hot_completion_affinity_fallback")
            if best_hot.deadline_state == "within":
                action = "wait" if best_hot.waiting.interval.upper_s > 0 else "reuse"
                return decision(action, best_hot, "affinity_retained")
            return decision("defer", None, "deadline_uncertain")
        if best_cold is not None:
            return decision("rebuild", best_cold, "no_feasible_hot_replica")
        if all(value.deadline_state == "miss" for value in fitting):
            return decision("reject", None, "predicted_deadline_miss")
        return decision("defer", None, "cold_estimate_unavailable")
