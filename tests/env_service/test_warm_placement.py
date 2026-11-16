"""Exercise conservative model reuse decisions with incomplete duration forecasts."""

import pytest
from psrl.workers.env_service.warm_placement import (
    DeviceSnapshot,
    DurationInterval,
    PlacementRequest,
    choose_placement,
    estimate_device,
    estimate_remaining,
)


def request(*, rebuild=(4, 6), execution=(2, 3), deadline=None):
    return PlacementRequest(
        "model",
        100,
        DurationInterval(*rebuild) if rebuild else None,
        DurationInterval(*execution) if execution else None,
        deadline,
    )


def hot(*, remaining=(1, 2), queued=(), device_id="hot"):
    return DeviceSnapshot(
        device_id,
        frozenset({"model"}),
        0,
        True,
        DurationInterval(*remaining) if remaining else None,
        tuple(DurationInterval(*duration) if duration else None for duration in queued),
    )


def cold(*, capacity=100, device_id="cold"):
    return DeviceSnapshot(device_id, frozenset(), capacity)


def test_short_wait_reuses_model_but_long_wait_pays_for_rebuild():
    short = choose_placement(request(), [hot(remaining=(1, 2)), cold()])
    long = choose_placement(request(), [hot(remaining=(10, 12)), cold()])
    assert short.action == "wait" and short.device_id == "hot"
    assert long.action == "rebuild" and long.device_id == "cold"
    assert long.selected.waiting == DurationInterval(0, 0)
    assert long.selected.rebuild == DurationInterval(4, 6)
    assert long.selected.execution == DurationInterval(2, 3)
    assert long.selected.completion == DurationInterval(6, 9)


def test_wait_includes_the_active_command_and_every_queued_command():
    estimate = estimate_device(request(), hot(remaining=(3, 5), queued=((4, 6), (7, 8))))
    assert estimate.waiting == DurationInterval(14, 19)
    assert estimate.completion == DurationInterval(16, 22)
    assert choose_placement(request(), [hot(remaining=(3, 5), queued=((4, 6), (7, 8))), cold()]).action == "rebuild"


def test_cold_queue_delay_can_eliminate_the_advantage_of_rebuilding():
    occupied_cold = DeviceSnapshot("cold", frozenset(), 100, True, DurationInterval(10, 12))
    decision = choose_placement(request(), [hot(remaining=(10, 12)), occupied_cold])
    assert decision.action == "wait"
    assert decision.estimates[1].completion == DurationInterval(16, 21)


@pytest.mark.parametrize("remaining, margin", [((5, 8), 0), ((7, 10), 0), ((8, 10), 1)])
def test_overlap_ties_and_insufficient_margin_preserve_affinity(remaining, margin):
    decision = choose_placement(request(), [hot(remaining=remaining), cold()], switch_margin_s=margin)
    assert decision.action == "wait"
    assert decision.device_id == "hot"


@pytest.mark.parametrize("active, queued", [(None, ()), ((1, 2), (None,))])
def test_unknown_active_or_queued_work_never_becomes_zero_wait(active, queued):
    decision = choose_placement(request(), [hot(remaining=active, queued=queued), cold()])
    assert decision.action == "defer" and decision.device_id is None
    assert decision.estimates[0].waiting is None
    assert decision.estimates[0].completion is None
    assert decision.estimates[1].completion == DurationInterval(6, 9)


def test_known_hot_replica_is_preferred_over_unknown_hot_replica():
    decision = choose_placement(request(), [hot(remaining=None), hot(device_id="known"), cold()])
    assert decision.action == "wait" and decision.device_id == "known"


def test_insufficient_cold_memory_cannot_be_used_even_when_hot_queue_is_long():
    decision = choose_placement(request(), [hot(remaining=(100, 120)), cold(capacity=99)])
    assert decision.action == "wait" and decision.device_id == "hot"
    assert not decision.estimates[1].memory_fits
    assert choose_placement(request(), [cold(capacity=99)]).action == "reject"


def test_resident_model_needs_no_additional_cache_capacity_or_rebuild_prediction():
    idle = DeviceSnapshot("idle", frozenset({"model"}), 0)
    decision = choose_placement(request(rebuild=None), [idle])
    assert decision.action == "reuse"
    assert decision.selected.rebuild == DurationInterval(0, 0)
    assert decision.selected.completion == DurationInterval(2, 3)


def test_unknown_command_duration_or_cold_rebuild_defers_admission():
    assert choose_placement(request(execution=None), [hot(), cold()]).action == "defer"
    assert choose_placement(request(rebuild=None), [cold()]).action == "defer"
    assert choose_placement(request(), [cold()]).action == "rebuild"


def test_deadline_distinguishes_impossible_uncertain_and_predicted_safe():
    assert choose_placement(request(deadline=5), [cold()]).action == "reject"
    assert choose_placement(request(deadline=8), [cold()]).action == "defer"
    assert choose_placement(request(deadline=9), [cold()]).action == "rebuild"
    decision = choose_placement(request(deadline=9), [hot(remaining=(10, 12)), cold()])
    assert decision.action == "rebuild"
    assert decision.estimates[0].deadline_state == "miss"


def test_progress_estimate_accounts_for_completed_and_ongoing_steps():
    progress = estimate_remaining([1, 2], completed_steps=2, total_steps=5, elapsed_step_s=0.5)
    assert progress.status == "estimated"
    assert progress.per_step == DurationInterval(0.8, 3)
    assert progress.remaining.lower_s == pytest.approx(1.9)
    assert progress.remaining.upper_s == pytest.approx(8.5)


@pytest.mark.parametrize("elapsed", [3, 3.1, 100])
def test_overdue_ongoing_step_invalidates_forecast_instead_of_clamping_to_zero(elapsed):
    progress = estimate_remaining([1, 2], completed_steps=9, total_steps=10, elapsed_step_s=elapsed)
    assert progress.status == "overrun"
    assert progress.remaining is None
    decision = choose_placement(
        request(), [DeviceSnapshot("hot", frozenset({"model"}), 0, True, progress.remaining), cold()]
    )
    assert decision.action == "defer"


def test_progress_without_history_stays_unknown_but_observed_completion_is_zero():
    missing = estimate_remaining([], completed_steps=0, total_steps=10)
    complete = estimate_remaining([], completed_steps=10, total_steps=10)
    assert missing.remaining is None and missing.status == "insufficient_history"
    assert complete.remaining == DurationInterval(0, 0) and complete.status == "complete"


@pytest.mark.parametrize("lower, upper", [(-1, 2), (3, 2), (0, float("inf")), (float("nan"), 2)])
def test_invalid_interval_is_rejected(lower, upper):
    with pytest.raises(ValueError):
        DurationInterval(lower, upper)


def test_inconsistent_progress_and_device_snapshots_are_rejected():
    with pytest.raises(ValueError, match="ongoing"):
        estimate_remaining([1], completed_steps=2, total_steps=2, elapsed_step_s=0.1)
    with pytest.raises(ValueError, match="counts"):
        estimate_remaining([1], completed_steps=3, total_steps=2)
    with pytest.raises(ValueError, match="positive"):
        estimate_remaining([0], completed_steps=0, total_steps=1)
    with pytest.raises(ValueError, match="idle"):
        DeviceSnapshot("idle", frozenset(), 100, False, DurationInterval(1, 2))
    with pytest.raises(ValueError, match="unique"):
        choose_placement(request(), [cold(), cold()])
