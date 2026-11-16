"""Gate fixed-step command forecasts using progress integrity and completed work."""

from collections import OrderedDict, deque
from dataclasses import dataclass, fields
from hashlib import sha256
from math import isfinite
from statistics import median
from typing import TypeAlias

from .step_predictor import StepForecast, predict_remaining


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class WorkloadKey:
    """Identify execution conditions whose timing history may be shared."""

    code_version: str
    model_version: str
    input_signature: str
    device_class: str
    runtime_version: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(getattr(self, field.name), str) or not getattr(self, field.name).strip()
            for field in fields(self)
        ):
            raise ValueError("Workload identity fields must be nonempty strings.")


@dataclass(frozen=True)
class CommandSpec:
    """Declare immutable command identity and an optional fixed amount of work."""

    command_id: str
    workload: WorkloadKey
    model_id: str
    total_steps: int | None
    relocatable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.command_id, str) or not self.command_id.strip():
            raise ValueError("Command identity must be a nonempty string.")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("Model identity must be a nonempty string.")
        if not isinstance(self.workload, WorkloadKey):
            raise ValueError("Command workload must be a WorkloadKey.")
        if self.total_steps is not None and (
            type(self.total_steps) is not int or self.total_steps < 1 or not _finite_number(self.total_steps)
        ):
            raise ValueError("Fixed command work must be a positive integer.")
        if type(self.relocatable) is not bool:
            raise ValueError("Command relocatability must be a boolean.")


@dataclass(frozen=True)
class StepProgress:
    """Carry bounded recent chunks and an adapter-normalized report age in seconds."""

    command_id: str
    workload: WorkloadKey
    total_steps: int
    completed_steps: int
    chunk_steps: tuple[int, ...]
    chunk_wall_s: tuple[float, ...]
    report_age_s: float


@dataclass(frozen=True)
class PredictabilityConfig:
    """Bound progress variability, independent calibration evidence and memory."""

    min_chunks: int = 3
    warmup_chunks: int = 1
    max_rate_ratio: float = 1.5
    max_drift_ratio: float = 1.25
    min_completed_commands: int = 3
    history_window: int = 8
    max_relative_error: float = 0.25
    absolute_error_floor_s: float = 0.25
    history_ttl_s: float = 3600
    max_profiles: int = 128
    max_pending: int = 256

    def __post_init__(self) -> None:
        for name in ("min_chunks", "min_completed_commands", "history_window", "max_profiles", "max_pending"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.warmup_chunks) is not int or self.warmup_chunks < 0:
            raise ValueError("warmup_chunks must be a nonnegative integer.")
        if self.min_completed_commands > self.history_window:
            raise ValueError("Calibration minimum cannot exceed the history window.")
        for name in ("max_rate_ratio", "max_drift_ratio"):
            value = getattr(self, name)
            if not _finite_number(value) or value < 1:
                raise ValueError(f"{name} must be finite and at least one.")
        if not _finite_number(self.max_relative_error) or self.max_relative_error < 0:
            raise ValueError("max_relative_error must be finite and nonnegative.")
        for name in ("absolute_error_floor_s", "history_ttl_s"):
            value = getattr(self, name)
            if not _finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True)
class PredictabilityAssessment:
    """Explain whether the current forecast has sufficient trustworthy evidence."""

    eligible: bool
    reason: str
    forecast: StepForecast | None
    calibration_commands: int
    rate_ratio: float | None
    drift_ratio: float | None
    reference_time_s: float | None = None


ProfileKey: TypeAlias = tuple[WorkloadKey, int | None, str]


@dataclass(frozen=True)
class _ProgressState:
    completed_steps: int
    report_age_s: float
    fingerprint: bytes
    chunks: tuple[tuple[int, int, float], ...]


@dataclass
class _PendingCommand:
    spec: CommandSpec
    progress: _ProgressState | None = None
    reference_forecast: StepForecast | None = None
    reference_time_s: float | None = None
    invalid_reason: str | None = None


@dataclass(frozen=True)
class _Calibration:
    command_id: str
    completed_at_s: float
    accurate: bool


class CommandRuntimePredictor:
    """
    Track fixed-step commands without admitting resources or retaining GPU state.

    The owner serializes calls and supplies one monotonic clock. A provider converts
    report timestamps into `report_age_s` before calling this process-local module.
    Every command ID identifies one attempt and must not be reused.
    """

    def __init__(self, config: PredictabilityConfig | None = None) -> None:
        if config is None:
            config = PredictabilityConfig()
        if not isinstance(config, PredictabilityConfig):
            raise ValueError("Predictability configuration has an invalid type.")
        self.config = config
        self._pending: dict[str, _PendingCommand] = {}
        self._history: OrderedDict[ProfileKey, deque[_Calibration]] = OrderedDict()
        self._retired: OrderedDict[str, CommandSpec] = OrderedDict()
        self._last_now_s: float | None = None

    @staticmethod
    def _key(spec: CommandSpec) -> ProfileKey:
        return spec.workload, spec.total_steps, spec.model_id

    def _advance_time(self, now_s: float) -> None:
        if not _finite_number(now_s) or now_s < 0:
            raise ValueError("Current time must be finite and nonnegative.")
        if self._last_now_s is not None and now_s < self._last_now_s:
            raise ValueError("Current time must not move backwards.")
        self._last_now_s = now_s
        for key, samples in list(self._history.items()):
            while samples and now_s - samples[0].completed_at_s >= self.config.history_ttl_s:
                samples.popleft()
            if not samples:
                del self._history[key]

    def _calibration(self, spec: CommandSpec) -> tuple[int, bool]:
        key = self._key(spec)
        samples = self._history.get(key)
        if samples is None:
            return 0, True
        self._history.move_to_end(key)
        return len(samples), all(sample.accurate for sample in samples)

    def _is_retired(self, command_id: str) -> bool:
        return command_id in self._retired or any(
            sample.command_id == command_id for samples in self._history.values() for sample in samples
        )

    def _retire(self, pending: _PendingCommand) -> None:
        self._retired[pending.spec.command_id] = pending.spec
        self._retired.move_to_end(pending.spec.command_id)
        while len(self._retired) > self.config.max_pending:
            self._retired.popitem(last=False)

    def _assessment(
        self,
        pending: _PendingCommand | None,
        spec: CommandSpec,
        reason: str,
        *,
        forecast: StepForecast | None = None,
        rate_ratio: float | None = None,
        drift_ratio: float | None = None,
    ) -> PredictabilityAssessment:
        count, _ = self._calibration(spec)
        return PredictabilityAssessment(
            eligible=reason == "predictable",
            reason=reason,
            forecast=forecast,
            calibration_commands=count,
            rate_ratio=rate_ratio,
            drift_ratio=drift_ratio,
            reference_time_s=pending.reference_time_s if pending else None,
        )

    def _progress_state(self, progress: StepProgress) -> _ProgressState | None:
        if (
            not isinstance(progress, StepProgress)
            or type(progress.total_steps) is not int
            or type(progress.completed_steps) is not int
            or not 0 <= progress.completed_steps <= progress.total_steps
            or progress.total_steps < 1
            or not isinstance(progress.chunk_steps, tuple)
            or not isinstance(progress.chunk_wall_s, tuple)
            or len(progress.chunk_steps) != len(progress.chunk_wall_s)
            or any(type(count) is not int or count < 1 or not _finite_number(count) for count in progress.chunk_steps)
            or any(not _finite_number(wall) or wall <= 0 for wall in progress.chunk_wall_s)
            or sum(progress.chunk_steps) > progress.completed_steps
            or not _finite_number(progress.report_age_s)
            or progress.report_age_s < 0
        ):
            return None
        digest = sha256()
        end = progress.completed_steps - sum(progress.chunk_steps)
        chunks = deque(maxlen=max(8, self.config.min_chunks) + self.config.warmup_chunks)
        for count, wall in zip(progress.chunk_steps, progress.chunk_wall_s, strict=True):
            if not _finite_number(wall / count) or wall / count <= 0:
                return None
            digest.update(f"{count}:{float(wall).hex()}\n".encode())
            chunks.append((end, end + count, wall))
            end += count
        return _ProgressState(progress.completed_steps, progress.report_age_s, digest.digest(), tuple(chunks))

    @staticmethod
    def _progress_error(
        previous: _ProgressState | None,
        current: _ProgressState,
        *,
        complete: bool,
    ) -> str | None:
        if previous is None:
            return None
        if current.completed_steps < previous.completed_steps:
            return "progress_regressed"
        if current.completed_steps == previous.completed_steps:
            if current.fingerprint != previous.fingerprint:
                return "progress_changed_without_advance"
            if current.report_age_s < previous.report_age_s:
                return "report_age_regressed"
            return None
        if not current.chunks:
            return None if complete else "progress_history_changed"
        if current.chunks[0][0] < previous.completed_steps:
            boundaries = {end for _, end, _ in current.chunks}
            if previous.completed_steps not in boundaries:
                return "progress_history_changed"
        prior = {(start, end): wall for start, end, wall in previous.chunks}
        for start, end, wall in current.chunks:
            if end <= previous.completed_steps and prior.get((start, end)) != wall:
                return "progress_history_changed"
        return None

    def assess(
        self,
        spec: CommandSpec,
        progress: StepProgress | None,
        *,
        now_s: float,
    ) -> PredictabilityAssessment:
        """
        Assess current progress and freeze the first stable shadow forecast.

        Args:
            spec (CommandSpec): Immutable identity for this execution attempt.
            progress (StepProgress | None): Provider progress with a normalized report age.
            now_s (float): Current time on the owner's monotonic clock.

        Returns:
            PredictabilityAssessment: Current eligibility and the frozen calibration boundary.
        """
        if not isinstance(spec, CommandSpec):
            raise ValueError("Command specification has an invalid type.")
        self._advance_time(now_s)
        pending = self._pending.get(spec.command_id)
        if pending is None:
            if self._is_retired(spec.command_id):
                return self._assessment(None, spec, "command_retired")
            if len(self._pending) >= self.config.max_pending:
                return self._assessment(None, spec, "pending_capacity_exhausted")
            pending = self._pending[spec.command_id] = _PendingCommand(spec)
        elif pending.spec != spec:
            pending.invalid_reason = "command_identity_changed"
        if pending.invalid_reason:
            return self._assessment(pending, spec, pending.invalid_reason)
        if spec.total_steps is None:
            return self._assessment(pending, spec, "unknown_total_steps")
        if progress is None:
            return self._assessment(pending, spec, "progress_unavailable")
        state = self._progress_state(progress)
        if state is None:
            return self._assessment(pending, spec, "malformed_progress")
        if (
            progress.command_id != spec.command_id
            or progress.workload != spec.workload
            or progress.total_steps != spec.total_steps
        ):
            return self._assessment(pending, spec, "progress_identity_mismatch")
        error = self._progress_error(pending.progress, state, complete=progress.completed_steps == spec.total_steps)
        if error:
            pending.invalid_reason = error
            return self._assessment(pending, spec, error)
        pending.progress = state
        counts, durations = progress.chunk_steps, progress.chunk_wall_s
        if sum(counts) == progress.completed_steps:
            counts = counts[self.config.warmup_chunks :]
            durations = durations[self.config.warmup_chunks :]
        window = max(8, self.config.min_chunks)
        counts, durations = counts[-window:], durations[-window:]
        if progress.completed_steps == spec.total_steps:
            forecast = predict_remaining((), (), spec.total_steps, spec.total_steps)
            return self._assessment(pending, spec, "command_complete", forecast=forecast)
        if len(counts) < self.config.min_chunks:
            return self._assessment(pending, spec, "insufficient_chunks")
        rates = [wall / count for count, wall in zip(counts, durations, strict=True)]
        ratio = max(rates) / min(rates)
        half = max(1, len(rates) // 2)
        early, late = median(rates[:half]), median(rates[-half:])
        drift = max(early, late) / min(early, late)
        if not isfinite(ratio) or not isfinite(drift):
            return self._assessment(pending, spec, "malformed_progress")
        if ratio > self.config.max_rate_ratio:
            return self._assessment(pending, spec, "unstable_step_rate", rate_ratio=ratio, drift_ratio=drift)
        if drift > self.config.max_drift_ratio:
            return self._assessment(pending, spec, "step_rate_drift", rate_ratio=ratio, drift_ratio=drift)
        forecast = predict_remaining(
            counts,
            durations,
            progress.completed_steps,
            spec.total_steps,
            progress.report_age_s,
            report_every_steps=max(progress.chunk_steps),
            window_chunks=window,
            min_chunks=self.config.min_chunks,
        )
        if forecast.status == "overrun":
            return self._assessment(pending, spec, "overdue_progress", rate_ratio=ratio, drift_ratio=drift)
        if any(not _finite_number(value) for value in (forecast.estimate_s, forecast.lower_s, forecast.upper_s)):
            return self._assessment(pending, spec, "malformed_progress")
        if pending.reference_forecast is None:
            pending.reference_forecast = forecast
            pending.reference_time_s = now_s
        count, accurate = self._calibration(spec)
        reason = "predictable"
        if not accurate:
            reason = "history_error"
        elif count < self.config.min_completed_commands:
            reason = "insufficient_calibration"
        return self._assessment(pending, spec, reason, forecast=forecast, rate_ratio=ratio, drift_ratio=drift)

    def finish(
        self,
        spec: CommandSpec,
        *,
        succeeded: bool,
        completed_steps: int,
        actual_remaining_s: float | None,
        now_s: float,
    ) -> None:
        """
        Retire one attempt and calibrate its frozen shadow prediction on success.

        `actual_remaining_s` spans from the assessment's first `reference_time_s`
        to actual service completion, including provider tail work. It never refers
        to the latest progress report or the full command duration.

        Args:
            spec (CommandSpec): The same immutable identity used in `assess`.
            succeeded (bool): Whether the provider confirmed successful completion.
            completed_steps (int): The provider's final completed work count.
            actual_remaining_s (float | None): Observed time from the frozen reference boundary.
            now_s (float): Current time on the owner's monotonic clock.
        """
        if not isinstance(spec, CommandSpec):
            raise ValueError("Command specification has an invalid type.")
        if type(succeeded) is not bool or type(completed_steps) is not int or completed_steps < 0:
            raise ValueError("Completion status and completed step count have invalid types or values.")
        if spec.total_steps is not None and completed_steps > spec.total_steps:
            raise ValueError("Completed work cannot exceed the declared total.")
        if actual_remaining_s is not None and (not _finite_number(actual_remaining_s) or actual_remaining_s < 0):
            raise ValueError("Actual remaining time must be finite and nonnegative.")
        self._advance_time(now_s)
        pending = self._pending.get(spec.command_id)
        if pending is None:
            return
        if pending.spec != spec:
            pending.invalid_reason = "command_identity_changed"
            return
        del self._pending[spec.command_id]
        self._retire(pending)
        if (
            not succeeded
            or completed_steps != spec.total_steps
            or actual_remaining_s is None
            or pending.reference_forecast is None
            or pending.invalid_reason is not None
        ):
            return
        predicted = pending.reference_forecast.estimate_s
        tolerance = max(self.config.absolute_error_floor_s, self.config.max_relative_error * actual_remaining_s)
        accurate = predicted is not None and abs(actual_remaining_s - predicted) <= tolerance
        key = self._key(spec)
        if key not in self._history:
            self._history[key] = deque(maxlen=self.config.history_window)
        self._history[key].append(_Calibration(spec.command_id, now_s, accurate))
        self._history.move_to_end(key)
        while len(self._history) > self.config.max_profiles:
            self._history.popitem(last=False)

    def forget(self, command_id: str) -> None:
        """Retire an abandoned command without inventing a completion observation."""
        if not isinstance(command_id, str) or not command_id.strip():
            raise ValueError("Command identity must be a nonempty string.")
        pending = self._pending.pop(command_id, None)
        if pending is not None:
            self._retire(pending)
