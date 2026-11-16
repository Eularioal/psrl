"""Check progress-age accounting and fixed-step prediction failure boundaries."""

import pytest
from psrl.workers.env_service.step_predictor import predict_remaining


def test_constant_wall_chunks_include_computation_and_progress_publication():
    forecast = predict_remaining([16, 16], [2, 2], 32, 96, report_grace_s=0)
    assert forecast.status == "estimated"
    assert forecast.per_step_estimate_s == pytest.approx(0.125)
    assert forecast.estimate_s == pytest.approx(8)
    assert forecast.lower_s == pytest.approx(6.4)
    assert forecast.upper_s == pytest.approx(10.4)


def test_observation_age_is_measured_against_the_next_chunk_instead_of_one_step():
    forecast = predict_remaining([16, 16], [2, 2], 32, 96, 1, report_grace_s=0)
    assert forecast.status == "estimated"
    assert forecast.estimate_s == pytest.approx(7)
    assert forecast.lower_s == pytest.approx(5.4)
    assert forecast.upper_s == pytest.approx(9.4)


def test_missing_next_report_invalidates_the_forecast_instead_of_predicting_zero():
    forecast = predict_remaining([16, 16], [2, 2], 32, 48, 2.7, report_grace_s=0)
    assert forecast.status == "overrun"
    assert forecast.estimate_s is None
    assert forecast.lower_s is None and forecast.upper_s is None


def test_final_short_chunk_has_a_shorter_report_deadline():
    forecast = predict_remaining([16, 16], [2, 2], 32, 36, 0.7, report_grace_s=0)
    assert forecast.status == "overrun"
    assert forecast.next_report_upper_s == pytest.approx(0.65)


def test_a_late_current_chunk_does_not_erase_work_after_that_chunk():
    forecast = predict_remaining([16, 16], [2, 2], 32, 96, 2.2, report_grace_s=0)
    assert forecast.status == "estimated"
    assert forecast.estimate_s == pytest.approx(6)
    assert forecast.lower_s == pytest.approx(4.8)
    assert forecast.upper_s == pytest.approx(8.2)


def test_partial_observed_chunks_are_normalized_by_their_actual_update_counts():
    forecast = predict_remaining([16, 8], [2, 1], 24, 40, report_grace_s=0)
    assert forecast.estimate_s == pytest.approx(2)
    assert forecast.per_step_estimate_s == pytest.approx(0.125)


def test_recent_window_adapts_without_using_older_slow_chunks():
    forecast = predict_remaining([16] * 4, [10, 10, 2, 2], 64, 96, window_chunks=2, report_grace_s=0)
    assert forecast.estimate_s == pytest.approx(4)
    assert forecast.sample_chunks == 2


def test_median_point_is_not_domimated_by_one_slow_startup_chunk():
    forecast = predict_remaining([16] * 5, [20, 2, 2, 2, 2], 80, 96)
    assert forecast.estimate_s == pytest.approx(2)
    assert forecast.lower_s <= forecast.estimate_s <= forecast.upper_s


def test_missing_history_stays_unknown_and_observed_completion_is_zero():
    missing = predict_remaining([], [], 0, 96)
    complete = predict_remaining([], [], 96, 96, 5)
    assert missing.status == "insufficient_history" and missing.estimate_s is None
    assert complete.status == "complete" and complete.estimate_s == 0


@pytest.mark.parametrize("age", [-1, float("nan"), float("inf")])
def test_invalid_observation_age_is_rejected(age):
    with pytest.raises(ValueError, match="age"):
        predict_remaining([16, 16], [2, 2], 32, 96, age)


@pytest.mark.parametrize(
    "counts, durations, completed, total",
    [
        ([16], [], 16, 96),
        ([0, 16], [2, 2], 32, 96),
        ([16, 16], [2, 0], 32, 96),
        ([16, 16], [2, 2], 16, 96),
        ([16, 16], [2, 2], 97, 96),
        ([True], [1], 1, 2),
    ],
)
def test_inconsistent_progress_is_rejected(counts, durations, completed, total):
    with pytest.raises(ValueError):
        predict_remaining(counts, durations, completed, total)


@pytest.mark.parametrize(
    "options",
    [
        {"report_every_steps": 0},
        {"window_chunks": 1},
        {"lower_factor": 1.1},
        {"upper_factor": 0.9},
        {"report_grace_s": -1},
        {"min_chunks": True},
    ],
)
def test_invalid_model_options_are_rejected(options):
    with pytest.raises(ValueError):
        predict_remaining([16, 16], [2, 2], 32, 96, **options)
