"""Gate warm placement on measured costs and trustworthy command progress."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from statistics import median
from typing import Literal

from .command_prediction import CommandRuntimePredictor, CommandSpec, StepProgress, WorkloadKey
from .warm_dispatch import (
    ModelCost,
    QueueSnapshot,
    SchedulingDecision,
    TimeEstimate,
    WarmPlacementController,
    estimate_from_forecast,
)
from .warm_placement import DurationInterval


def _number(value: object, name: str, *, positive: bool = False) -> None:
    if type(value) not in (int, float) or not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")


def _bytes(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("GPU budgets must be nonnegative integer bytes.")


@dataclass(frozen=True)
class PlacementConfig:
    """Disable placement by default and make observation independent of activation."""

    mode: Literal["off", "observe", "auto"] = "off"
    min_profile_samples: int = 3
    max_profile_age_s: float = 3600
    max_profile_rate_ratio: float = 1.5
    switch_margin_s: float = 2.0
    decision_mode: Literal["point", "interval"] = "point"

    def __post_init__(self) -> None:
        if self.mode not in {"off", "observe", "auto"}:
            raise ValueError("Placement mode must be off, observe or auto.")
        if type(self.min_profile_samples) is not int or self.min_profile_samples < 1:
            raise ValueError("Profile sample minimum must be a positive integer.")
        _number(self.max_profile_age_s, "Profile age", positive=True)
        _number(self.max_profile_rate_ratio, "Profile spread", positive=True)
        if self.max_profile_rate_ratio < 1:
            raise ValueError("Profile spread must be at least one.")
        WarmPlacementController(self.decision_mode, self.switch_margin_s)


@dataclass(frozen=True)
class PlacementRequest:
    """Declare a structured command and its retained model and peak workspace."""

    spec: CommandSpec
    model_bytes: int
    workspace_bytes: int = 0
    home_device_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, CommandSpec):
            raise ValueError("A structured command specification is required.")
        _bytes(self.model_bytes)
        _bytes(self.workspace_bytes)
        if self.home_device_id is not None and (
            not isinstance(self.home_device_id, str) or not self.home_device_id.strip()
        ):
            raise ValueError("Home device must be a nonempty identity.")
        if not self.spec.relocatable and self.home_device_id is None:
            raise ValueError("A command with nontransferable state must declare its home device.")


@dataclass(frozen=True)
class CommandCostProfile:
    """
    Bind independent full service measurements to one workload and command size.

    Rebuild measures reconstructing the declared transferable model. Execution
    includes preparation, computation and the tail until the service is available.
    Samples must come from separate successful commands, not repeated polling.
    The provider remains responsible for the truth of these measured declarations.
    """

    workload: WorkloadKey
    model_id: str
    total_steps: int
    model_bytes: int
    workspace_bytes: int
    rebuild_samples_s: tuple[float, ...]
    execution_samples_s: tuple[float, ...]
    observed_at_s: float

    def __post_init__(self) -> None:
        CommandSpec("profile-validation", self.workload, self.model_id, self.total_steps)
        _bytes(self.model_bytes)
        _bytes(self.workspace_bytes)
        _number(self.observed_at_s, "Profile timestamp")
        for samples in (self.rebuild_samples_s, self.execution_samples_s):
            if not isinstance(samples, tuple):
                raise ValueError("Profile samples must be immutable tuples.")
            for sample in samples:
                _number(sample, "Service sample", positive=True)


@dataclass(frozen=True)
class QueuedWork:
    """Retain the request and its device-specific measured service profile."""

    request: PlacementRequest
    profile: CommandCostProfile | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, PlacementRequest):
            raise ValueError("Queue entries must contain PlacementRequest values.")
        if self.profile is not None and not isinstance(self.profile, CommandCostProfile):
            raise ValueError("Queue profiles must be CommandCostProfile values or unknown.")


@dataclass(frozen=True)
class DeviceState:
    """
    Snapshot one exclusively managed serial service and its retained model cache.

    Available bytes are a provider reservation budget after resident models and
    external reservations, before command workspace. Models must survive this
    queue. This contract does not represent arbitrary unowned GPU free memory.
    """

    device_id: str
    device_class: str
    resident_models: frozenset[str]
    available_gpu_bytes: int
    active: QueuedWork | None = None
    progress: StepProgress | None = None
    queued: tuple[QueuedWork, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.device_id, self.device_class):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Device identities and classes must be nonempty strings.")
        _bytes(self.available_gpu_bytes)
        if not isinstance(self.resident_models, frozenset) or any(
            not isinstance(model, str) or not model.strip() for model in self.resident_models
        ):
            raise ValueError("Resident models must be a frozenset of nonempty identities.")
        if self.active is not None and not isinstance(self.active, QueuedWork):
            raise ValueError("Active work must be a QueuedWork value or unknown.")
        if not isinstance(self.queued, tuple) or any(not isinstance(work, QueuedWork) for work in self.queued):
            raise ValueError("Queued work must be an immutable tuple of QueuedWork values.")
        if self.active is None and self.progress is not None:
            raise ValueError("An idle service cannot report active command progress.")
        works = ((self.active,) if self.active is not None else ()) + self.queued
        if any(work.request.spec.workload.device_class != self.device_class for work in works):
            raise ValueError("Active and queued command identities must match the device class.")


@dataclass(frozen=True)
class GuardedDispatchDecision:
    """Explain activation or fallback without inventing a duration for unknown work."""

    action: str
    device_id: str | None
    reason: str
    prediction_used: bool
    assessments: tuple[tuple[str, str], ...] = ()
    comparison: SchedulingDecision | None = None


class PredictionAwareScheduler:
    """Plan placement while leaving atomic enqueue and physical admission to the owner."""

    def __init__(
        self,
        config: PlacementConfig | None = None,
        predictor: CommandRuntimePredictor | None = None,
    ) -> None:
        if config is None:
            config = PlacementConfig()
        if not isinstance(config, PlacementConfig):
            raise ValueError("Placement configuration is required.")
        if predictor is not None and not isinstance(predictor, CommandRuntimePredictor):
            raise ValueError("A CommandRuntimePredictor is required.")
        self.config = config
        self.predictor = predictor if predictor is not None else CommandRuntimePredictor()

    def _profile(
        self,
        request: PlacementRequest,
        device_class: str,
        profile: CommandCostProfile | None,
        now_s: float,
        *,
        need_rebuild: bool,
    ) -> tuple[ModelCost | None, str]:
        if request.spec.total_steps is None:
            return None, "unknown_total_steps"
        if profile is None:
            return None, "profile_unavailable"
        if not isinstance(profile, CommandCostProfile):
            raise ValueError("Profiles must contain CommandCostProfile values.")
        expected = replace(request.spec.workload, device_class=device_class)
        if (
            profile.workload != expected
            or profile.model_id != request.spec.model_id
            or profile.total_steps != request.spec.total_steps
            or (profile.model_bytes, profile.workspace_bytes) != (request.model_bytes, request.workspace_bytes)
        ):
            return None, "profile_identity_mismatch"
        if not 0 <= now_s - profile.observed_at_s < self.config.max_profile_age_s:
            return None, "profile_expired"

        def estimate(samples: tuple[float, ...]) -> TimeEstimate | None:
            if len(samples) < self.config.min_profile_samples:
                return None
            if max(samples) / min(samples) > self.config.max_profile_rate_ratio:
                return None
            return TimeEstimate(median(samples), DurationInterval(min(samples) * 0.8, max(samples) * 1.3))

        execution, rebuild = estimate(profile.execution_samples_s), estimate(profile.rebuild_samples_s)
        if execution is None or (need_rebuild and rebuild is None):
            return None, "profile_insufficient_or_unstable"
        return ModelCost(profile.model_id, profile.model_bytes, rebuild, execution, profile.workspace_bytes), "ready"

    def _view(
        self, device: DeviceState, now_s: float, *, observe: bool
    ) -> tuple[QueueSnapshot | None, tuple[tuple[str, str], ...]]:
        models = set(device.resident_models)
        retained_sizes: dict[str, int] = {}
        remaining = device.available_gpu_bytes
        active_remaining = None
        reasons: list[tuple[str, str]] = []
        queued_service = []
        works = ((device.active,) if device.active is not None else ()) + device.queued
        for work in works:
            request = work.request
            model_id = request.spec.model_id
            if model_id in retained_sizes and retained_sizes[model_id] != request.model_bytes:
                raise ValueError("One immutable model identity cannot declare different retained sizes.")
            retained_sizes[model_id] = request.model_bytes
            cold = model_id not in models
            if cold:
                remaining -= request.model_bytes
            if remaining < request.workspace_bytes:
                return None, ((device.device_id, "prior_queue_exceeds_budget"),)
            if work is device.active:
                if observe:
                    assessment = self.predictor.assess(request.spec, device.progress, now_s=now_s)
                    reasons.append((device.device_id, assessment.reason))
                    if assessment.eligible and assessment.forecast is not None:
                        active_remaining = estimate_from_forecast(assessment.forecast)
            else:
                cost, reason = self._profile(request, device.device_class, work.profile, now_s, need_rebuild=cold)
                reasons.append((device.device_id, f"queued:{reason}"))
                service = None if cost is None else cost.execution
                if cold and service is not None:
                    service = service + cost.rebuild
                queued_service.append(service)
            models.add(model_id)
        return (
            QueueSnapshot(
                device.device_id,
                frozenset(models),
                remaining,
                busy=device.active is not None,
                active_remaining=active_remaining,
                queued_service=tuple(queued_service),
            ),
            tuple(reasons),
        )

    @staticmethod
    def _fallback(
        request: PlacementRequest,
        devices: tuple[DeviceState, ...],
        views: Mapping[str, QueueSnapshot],
        reason: str,
        assessments: tuple[tuple[str, str], ...],
    ) -> GuardedDispatchDecision:
        legal = []
        for device in devices:
            view = views.get(device.device_id)
            if view is None:
                continue
            if not request.spec.relocatable and device.device_id != request.home_device_id:
                continue
            needed = request.workspace_bytes + (
                0 if request.spec.model_id in view.resident_models else request.model_bytes
            )
            if needed <= view.available_gpu_bytes:
                legal.append(device)
        if not legal:
            return GuardedDispatchDecision("reject", None, "no_declared_capacity", False, assessments)
        selected = min(
            legal,
            key=lambda device: (
                request.spec.model_id not in views[device.device_id].resident_models,
                device.device_id != request.home_device_id,
                int(device.active is not None) + len(device.queued),
                device.device_id,
            ),
        )
        hot = request.spec.model_id in views[selected.device_id].resident_models
        busy = selected.active is not None or bool(selected.queued)
        action = "wait" if busy else ("reuse" if hot else "rebuild")
        return GuardedDispatchDecision(action, selected.device_id, reason, False, assessments)

    def plan(
        self,
        request: PlacementRequest,
        devices: tuple[DeviceState, ...],
        profiles: Mapping[str, CommandCostProfile],
        *,
        now_s: float,
    ) -> GuardedDispatchDecision:
        """Inspect every needed queue component before making a time comparison."""
        _number(now_s, "Current time")
        if not isinstance(request, PlacementRequest):
            raise ValueError("A PlacementRequest is required.")
        if not isinstance(devices, tuple) or any(not isinstance(device, DeviceState) for device in devices):
            raise ValueError("Devices must be an immutable tuple of DeviceState values.")
        if len({device.device_id for device in devices}) != len(devices):
            raise ValueError("Device identities must be unique.")
        if not isinstance(profiles, Mapping) or set(profiles) - {device.device_id for device in devices}:
            raise ValueError("Cost profiles must map known device identities.")
        views = {}
        reasons: list[tuple[str, str]] = []
        model_contracts = {request.spec.model_id: (request.spec.workload.model_version, request.model_bytes)}
        for device in devices:
            works = ((device.active,) if device.active is not None else ()) + device.queued
            for work in works:
                queued_request = work.request
                contract = (queued_request.spec.workload.model_version, queued_request.model_bytes)
                previous = model_contracts.setdefault(queued_request.spec.model_id, contract)
                if previous != contract:
                    raise ValueError("Immutable model identities must preserve their version and retained size.")
            view, checks = self._view(device, now_s, observe=self.config.mode != "off")
            reasons.extend(checks)
            if view is not None:
                views[device.device_id] = view

        def fallback(reason: str) -> GuardedDispatchDecision:
            return self._fallback(request, devices, views, reason, tuple(reasons))

        if self.config.mode != "auto":
            return fallback(f"mode_{self.config.mode}")
        if not request.spec.relocatable:
            return fallback("state_not_relocatable")
        if request.spec.total_steps is None:
            return fallback("unknown_total_steps")
        costs = {}
        candidates = []
        for device in devices:
            view = views.get(device.device_id)
            if view is None:
                continue
            needed = request.workspace_bytes + (
                0 if request.spec.model_id in view.resident_models else request.model_bytes
            )
            if needed > view.available_gpu_bytes:
                continue
            cost, reason = self._profile(
                request,
                device.device_class,
                profiles.get(device.device_id),
                now_s,
                need_rebuild=request.spec.model_id not in view.resident_models,
            )
            reasons.append((device.device_id, reason))
            if cost is None:
                return fallback(reason)
            if (view.busy and view.active_remaining is None) or any(value is None for value in view.queued_service):
                return fallback("queue_time_unpredictable")
            costs[device.device_id] = cost
            candidates.append(view)
        if not candidates:
            return fallback("no_declared_capacity")
        comparison = WarmPlacementController(self.config.decision_mode, self.config.switch_margin_s).decide(
            next(iter(costs.values())),
            tuple(candidates),
            device_costs=costs,
        )
        return GuardedDispatchDecision(
            comparison.action, comparison.device_id, comparison.reason_code, True, tuple(reasons), comparison
        )
