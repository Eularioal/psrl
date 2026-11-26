"""Check loan timing, causal histories and conditional remaining gaps."""

from dataclasses import replace

import pytest
from psrl.workers.env_service.command_prediction import WorkloadKey
from psrl.workers.env_service.reuse_prediction import ReuseDelayPredictor, ReuseSample, additional_wait

KEY = WorkloadKey("code-a", "model-a", "a-phase+b-steps+protocol", "H20", "runtime-a")


def test_restore_protocol_changes_the_wait_even_when_b_fits():
    assert additional_wait(1, 10, 3, 20) == 0, "Proactive restoration can finish before A returns."
    assert additional_wait(1, 10, 3, 20, proactive_restore=False) == 3, "On-demand restoration is exposed."
    assert additional_wait(1, 10, 3, 2) == 12, "An early return must wait for the serialized loan."
    for value in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="Durations"):
            additional_wait(value, 10, 3, 20)


def test_prediction_is_causal_versioned_and_expires():
    predictor = ReuseDelayPredictor(ttl_s=10)
    for i in range(3):
        assert predictor.predict(KEY, now_s=i) is None, "Uncalibrated forecasts must abstain."
        predictor.observe(KEY, ReuseSample(1, 10, 3, 20), now_s=i)
    result = predictor.predict(KEY, now_s=3)
    assert result.release_recommended and result.point_wait_s == 0, "A calibrated fitting loan is eligible."
    assert predictor.predict(replace(KEY, code_version="new-code"), now_s=3) is None, (
        "Changed code must not inherit old execution timing."
    )
    assert predictor.predict(KEY, now_s=12) is None, "Expired history must not authorize lending."
    with pytest.raises(ValueError, match="backwards"):
        predictor.predict(KEY, now_s=11)


def test_elapsed_gap_conditions_on_surviving_sessions():
    predictor = ReuseDelayPredictor(min_samples=2)
    for gap in (2, 3, 20, 30):
        predictor.observe(KEY, ReuseSample(1, 5, 2, gap), now_s=0)
    result = predictor.predict(KEY, now_s=1, gap_elapsed_s=10)
    assert result.samples == 2 and result.remaining_gap_s == 15, "Only still-idle historical sessions apply."
    assert result.release_recommended, "Both surviving gaps fit the loan."
    assert predictor.predict(KEY, now_s=1, gap_elapsed_s=25) is None, "Too few surviving gaps require abstention."


def test_tail_observation_blocks_lending_without_faking_a_guarantee():
    predictor = ReuseDelayPredictor()
    for runtime in (2, 2, 40):
        predictor.observe(KEY, ReuseSample(1, runtime, 3, 20), now_s=0)
    result = predictor.predict(KEY, now_s=1, max_extra_wait_s=1)
    assert result.point_wait_s == 0 and result.envelope_wait_s == 24, "Median alone hides a long borrower."
    assert not result.release_recommended, "Observed overruns must affect the admission decision."


def test_interrupted_profile_requires_new_calibration():
    predictor = ReuseDelayPredictor()
    other = replace(KEY, code_version="other-code")
    for key in (KEY, other):
        for _ in range(3):
            predictor.observe(key, ReuseSample(1, 2, 1, 20), now_s=0)
    predictor.invalidate(KEY, now_s=1)
    assert predictor.predict(KEY, now_s=1) is None, "An interrupted run must not retain optimistic old advice."
    assert predictor.predict(other, now_s=1).release_recommended, "Invalidation is scoped to one workload."
    for i in range(3):
        assert predictor.predict(KEY, now_s=2) is None, "A failed cycle is not a completed timing observation."
        predictor.observe(KEY, ReuseSample(1, 30, 1, 20), now_s=2)
    assert not predictor.predict(KEY, now_s=2).release_recommended, "Recalibration must include the new cost."
