"""Forecast nonpreemptive GPU loans from versioned completed timing samples."""

from collections import OrderedDict, deque
from dataclasses import dataclass
from math import isfinite
from statistics import median

from .command_prediction import WorkloadKey


def _duration(value):
    if type(value) not in (float, int) or not isfinite(value) or value < 0:
        raise ValueError("Durations must be finite nonnegative numbers.")
    return value


def additional_wait(save_return_s, borrower_s, restore_s, remaining_gap_s, *, proactive_restore=True):
    """Compare A with an already warm baseline under one serialized, nonpreemptive loan."""
    save_return_s, borrower_s, restore_s, remaining_gap_s = map(
        _duration, (save_return_s, borrower_s, restore_s, remaining_gap_s)
    )
    if proactive_restore:
        return max(0.0, save_return_s + borrower_s + restore_s - remaining_gap_s)
    return max(0.0, save_return_s + borrower_s - remaining_gap_s) + restore_s


@dataclass(frozen=True)
class ReuseSample:
    save_return_s: float
    borrower_s: float
    restore_s: float
    total_gap_s: float

    def __post_init__(self):
        for value in (self.save_return_s, self.borrower_s, self.restore_s, self.total_gap_s):
            _duration(value)


@dataclass(frozen=True)
class ReuseForecast:
    samples: int
    point_wait_s: float
    envelope_wait_s: float
    remaining_gap_s: float
    save_return_s: float
    borrower_s: float
    restore_s: float
    release_recommended: bool
    reason: str

    def __post_init__(self):
        if type(self.samples) is not int or self.samples < 1 or type(self.release_recommended) is not bool:
            raise ValueError("Forecast advice requires a positive sample count and a boolean recommendation.")
        for value in (
            self.point_wait_s,
            self.envelope_wait_s,
            self.remaining_gap_s,
            self.save_return_s,
            self.borrower_s,
            self.restore_s,
        ):
            _duration(value)
        if self.envelope_wait_s < self.point_wait_s:
            raise ValueError("An empirical envelope cannot be smaller than its point estimate.")


class ReuseDelayPredictor:
    """
    Keep bounded history for a complete A/B/protocol identity supplied by the adapter.

    The envelope combines observed cost maxima and the shortest surviving gap. It is
    an empirical guard, not a statistical coverage guarantee or a deadline guarantee.
    Only completed cycles may be observed. Failed or censored runs are not zero samples.
    """

    def __init__(self, *, min_samples=3, window=16, max_profiles=128, ttl_s=3600):
        if any(type(v) is not int or v < 1 for v in (min_samples, window, max_profiles)):
            raise ValueError("History limits must be positive integers.")
        if min_samples > window or _duration(ttl_s) == 0:
            raise ValueError("History must fit the calibration minimum and have a positive lifetime.")
        self.min_samples, self.window, self.max_profiles, self.ttl_s = min_samples, window, max_profiles, ttl_s
        self._history = OrderedDict()
        self._now_s = 0.0

    def _advance(self, now_s):
        _duration(now_s)
        if now_s < self._now_s:
            raise ValueError("Observation time must not move backwards.")
        self._now_s = now_s
        for key, samples in list(self._history.items()):
            while samples and now_s - samples[0][0] >= self.ttl_s:
                samples.popleft()
            if not samples:
                del self._history[key]

    def observe(self, key: WorkloadKey, sample: ReuseSample, *, now_s: float):
        if not isinstance(key, WorkloadKey) or not isinstance(sample, ReuseSample):
            raise ValueError("Completed timing observations require a workload identity and a valid sample.")
        self._advance(now_s)
        self._history.setdefault(key, deque(maxlen=self.window)).append((now_s, sample))
        self._history.move_to_end(key)
        while len(self._history) > self.max_profiles:
            self._history.popitem(last=False)

    def invalidate(self, key: WorkloadKey, *, now_s: float):
        """Abstain after an interrupted or stale profile until it is calibrated again."""
        if not isinstance(key, WorkloadKey):
            raise ValueError("Invalidation requires a versioned workload identity.")
        self._advance(now_s)
        self._history.pop(key, None)

    def predict(
        self,
        key: WorkloadKey,
        *,
        now_s: float,
        gap_elapsed_s=0.0,
        max_extra_wait_s=0.0,
        proactive_restore=True,
    ) -> ReuseForecast | None:
        self._advance(now_s)
        _duration(gap_elapsed_s)
        _duration(max_extra_wait_s)
        if not isinstance(key, WorkloadKey):
            raise ValueError("Prediction requires a versioned workload identity.")
        # Condition on the fact that A has not returned yet, rather than subtracting
        # elapsed time from an unconditional historical mean.
        samples = [sample for _, sample in self._history.get(key, ()) if sample.total_gap_s > gap_elapsed_s]
        if len(samples) < self.min_samples:
            return None
        costs = [[getattr(s, name) for s in samples] for name in ("save_return_s", "borrower_s", "restore_s")]
        gaps = [s.total_gap_s - gap_elapsed_s for s in samples]
        point_costs = [median(values) for values in costs]
        point_gap = median(gaps)
        point = additional_wait(*point_costs, point_gap, proactive_restore=proactive_restore)
        envelope = additional_wait(*(max(values) for values in costs), min(gaps), proactive_restore=proactive_restore)
        useful = min(gaps) > max(costs[0])
        recommended = useful and envelope <= max_extra_wait_s
        return ReuseForecast(
            len(samples),
            point,
            envelope,
            point_gap,
            *point_costs,
            recommended,
            "Observed envelope fits the delay budget." if recommended else "Keep reservation under the delay budget.",
        )
