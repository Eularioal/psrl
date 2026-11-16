"""Verify command trust boundaries, shadow calibration and bounded recovery."""

from dataclasses import FrozenInstanceError, replace

import pytest
from psrl.workers.env_service.command_prediction import (
    CommandRuntimePredictor,
    CommandSpec,
    PredictabilityConfig,
    StepProgress,
    WorkloadKey,
)

WORKLOAD = WorkloadKey("code-v1", "weights-v1", "batch=1,len=256,fp16", "gpu-class-a", "runtime-v1")


def command(command_id="command", **changes):
    return replace(CommandSpec(command_id, WORKLOAD, "model", 128), **changes)


def progress(spec, **changes):
    return replace(
        StepProgress(spec.command_id, spec.workload, spec.total_steps, 64, (16,) * 4, (2.0,) * 4, 0), **changes
    )


def complete_shadow(predictor, command_id, now_s, *, multiplier=1.0, workload=WORKLOAD, total_steps=128):
    spec = command(command_id, workload=workload, total_steps=total_steps)
    assessment = predictor.assess(spec, progress(spec), now_s=now_s)
    actual = assessment.forecast.estimate_s * multiplier
    predictor.finish(
        spec, succeeded=True, completed_steps=total_steps, actual_remaining_s=actual, now_s=now_s + actual
    )
    return now_s + actual + 1


def calibrate(predictor, count=3, *, now_s=0, prefix="history", **changes):
    for index in range(count):
        now_s = complete_shadow(predictor, f"{prefix}-{index}", now_s, **changes)
    return now_s


def test_first_three_independent_commands_are_shadow_only_then_auto_is_eligible():
    predictor = CommandRuntimePredictor()
    now_s = 0
    for index in range(3):
        spec = command(f"warm-{index}")
        assessment = predictor.assess(spec, progress(spec), now_s=now_s)
        assert not assessment.eligible, "Uncalibrated commands must remain in shadow mode."
        assert assessment.reason == "insufficient_calibration", "Calibration must explain the fallback."
        assert assessment.calibration_commands == index, "Only independent completed commands may count."
        assert assessment.reference_time_s == now_s, "The first shadow reference must be exposed."
        for offset in (0.1, 0.2, 0.3):
            repeated = predictor.assess(spec, progress(spec, report_age_s=offset), now_s=now_s + offset)
            assert repeated.calibration_commands == index, "Progress polls cannot count as completed commands."
        predictor.finish(spec, succeeded=True, completed_steps=128, actual_remaining_s=8, now_s=now_s + 8)
        now_s += 9
    spec = command("eligible")
    assessment = predictor.assess(spec, progress(spec), now_s=now_s)
    assert assessment.eligible and assessment.reason == "predictable", "Three accurate commands should enable auto."
    assert assessment.forecast.estimate_s == pytest.approx(8), "The forecast must use normalized chunk rates."


def test_later_reports_preserve_the_first_calibration_boundary_and_estimate():
    predictor = CommandRuntimePredictor()
    spec = command()
    first = predictor.assess(spec, progress(spec), now_s=0)
    later = predictor.assess(
        spec, progress(spec, completed_steps=96, chunk_steps=(16,) * 6, chunk_wall_s=(2.0,) * 6), now_s=1
    )
    assert later.forecast.estimate_s == pytest.approx(4), "Current forecasts should follow new progress."
    assert later.reference_time_s == first.reference_time_s == 0, "The shadow calibration boundary must not move."
    predictor.finish(spec, succeeded=True, completed_steps=128, actual_remaining_s=4, now_s=4)
    probe = command("probe")
    assessment = predictor.assess(probe, progress(probe), now_s=5)
    assert assessment.reason == "history_error", (
        "Finishing at the latest estimate must not hide earlier prediction error."
    )


def test_calibration_accepts_a_small_absolute_error_for_short_commands():
    predictor = CommandRuntimePredictor(PredictabilityConfig(min_completed_commands=1))
    spec = command(total_steps=65)
    first = predictor.assess(spec, progress(spec), now_s=0)
    assert first.forecast.estimate_s == pytest.approx(0.125), "One remaining step should have its measured cost."
    predictor.finish(spec, succeeded=True, completed_steps=65, actual_remaining_s=0.3, now_s=0.3)
    probe = command("probe", total_steps=65)
    assessment = predictor.assess(probe, progress(probe), now_s=1)
    assert assessment.eligible, "The absolute error allowance must protect short commands from tiny denominators."


def test_one_bad_completion_blocks_auto_until_it_leaves_the_sliding_window():
    predictor = CommandRuntimePredictor(PredictabilityConfig(history_window=3))
    now_s = calibrate(predictor)
    now_s = complete_shadow(predictor, "bad", now_s, multiplier=2)
    for index in range(2):
        probe = command(f"blocked-{index}")
        assessment = predictor.assess(probe, progress(probe), now_s=now_s)
        assert assessment.reason == "history_error", "An inaccurate retained sample must block auto."
        predictor.forget(probe.command_id)
        now_s = complete_shadow(predictor, f"recovery-{index}", now_s)
    now_s = complete_shadow(predictor, "recovery-final", now_s)
    probe = command("recovered")
    assessment = predictor.assess(probe, progress(probe), now_s=now_s)
    assert assessment.eligible, "Three fresh healthy completions should replace the bad window."
    assert assessment.calibration_commands == 3, "Calibration memory must respect the configured window."


def test_expired_history_returns_to_shadow_instead_of_trusting_old_conditions():
    predictor = CommandRuntimePredictor(PredictabilityConfig(history_ttl_s=30))
    now_s = calibrate(predictor)
    probe = command("probe")
    assessment = predictor.assess(probe, progress(probe), now_s=now_s + 31)
    assert assessment.reason == "insufficient_calibration", "Expired calibration must disable auto."
    assert assessment.calibration_commands == 0, "Expired samples must be removed."
    assert assessment.forecast is not None, "Fresh progress should remain available for shadow learning."


@pytest.mark.parametrize(
    "field,value",
    [
        ("code_version", "code-v2"),
        ("model_version", "weights-v2"),
        ("input_signature", "batch=8,len=256,fp16"),
        ("device_class", "gpu-class-b"),
        ("runtime_version", "runtime-v2"),
    ],
)
def test_workload_changes_do_not_inherit_previous_calibration(field, value):
    predictor = CommandRuntimePredictor()
    now_s = calibrate(predictor)
    spec = command("new-profile", workload=replace(WORKLOAD, **{field: value}))
    assessment = predictor.assess(spec, progress(spec), now_s=now_s)
    assert assessment.calibration_commands == 0, "Changed execution conditions need separate calibration."
    assert not assessment.eligible, "Identity changes cannot reuse timing evidence."


@pytest.mark.parametrize("changes", [{"model_id": "other-model"}, {"total_steps": 256}])
def test_model_identity_and_total_work_are_part_of_the_calibration_key(changes):
    predictor = CommandRuntimePredictor()
    now_s = calibrate(predictor)
    spec = command("other-shape", **changes)
    assessment = predictor.assess(spec, progress(spec), now_s=now_s)
    assert assessment.calibration_commands == 0, "Model identity and total work must partition history."


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"chunk_wall_s": (2.0, 2.0, 2.0, 5.0)}, "unstable_step_rate"),
        ({"chunk_wall_s": (2.0, 2.0, 2.4, 2.8)}, "step_rate_drift"),
        ({"report_age_s": 3.0}, "overdue_progress"),
        ({"chunk_steps": (16, 16, 16), "chunk_wall_s": (2.0, 2.0, 2.0), "completed_steps": 48}, "insufficient_chunks"),
    ],
)
def test_unstable_drifting_or_missing_progress_cannot_enable_prediction(changes, reason):
    predictor = CommandRuntimePredictor()
    spec = command()
    assessment = predictor.assess(spec, progress(spec, **changes), now_s=0)
    assert not assessment.eligible and assessment.reason == reason, "Untrustworthy progress must explain its fallback."
    assert assessment.forecast is None, "Unsafe raw timing must not reach the placement calculator."
    assert assessment.reference_time_s is None, "Unsafe progress cannot become a calibration boundary."


def test_warmup_is_removed_only_when_the_original_first_chunk_is_present():
    predictor = CommandRuntimePredictor()
    spec = command()
    first = predictor.assess(spec, progress(spec, chunk_wall_s=(20.0, 2.0, 2.0, 2.0)), now_s=0)
    assert first.forecast.estimate_s == pytest.approx(8), "Known startup work should be excluded once."
    later = predictor.assess(
        spec, progress(spec, completed_steps=80, chunk_steps=(16,) * 3, chunk_wall_s=(2.0,) * 3), now_s=1
    )
    assert later.forecast.estimate_s == pytest.approx(6), "A rolling window must keep all post-startup samples."


def test_a_short_final_chunk_does_not_change_its_rate_or_report_deadline():
    predictor = CommandRuntimePredictor()
    spec = command(total_steps=68)
    assessment = predictor.assess(spec, progress(spec, report_age_s=0.8), now_s=0)
    assert assessment.reason == "overdue_progress", "The final four steps must use their shorter expected deadline."


def test_identical_progress_can_be_polled_with_increasing_age():
    predictor = CommandRuntimePredictor()
    spec = command()
    first = predictor.assess(spec, progress(spec), now_s=0)
    later = predictor.assess(spec, progress(spec, report_age_s=1), now_s=1)
    assert later.reason == "insufficient_calibration", "Repeated progress polling must remain valid."
    assert later.forecast.estimate_s == pytest.approx(7), "Elapsed time must reduce the current remaining work."
    assert later.reference_time_s == first.reference_time_s, "Polling must not move the calibration boundary."


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"completed_steps": 48, "chunk_steps": (16,) * 3, "chunk_wall_s": (2.0,) * 3}, "progress_regressed"),
        ({"chunk_wall_s": (2.0, 2.0, 2.1, 2.0)}, "progress_changed_without_advance"),
        ({"report_age_s": 0}, "report_age_regressed"),
        (
            {"completed_steps": 80, "chunk_steps": (16,) * 5, "chunk_wall_s": (2.0, 2.1, 2.0, 2.0, 2.0)},
            "progress_history_changed",
        ),
        ({"completed_steps": 80, "chunk_steps": (20,) * 4, "chunk_wall_s": (2.5,) * 4}, "progress_history_changed"),
    ],
)
def test_replayed_or_rewritten_progress_retires_trust_for_that_attempt(changes, reason):
    predictor = CommandRuntimePredictor()
    spec = command()
    predictor.assess(spec, progress(spec, report_age_s=0.1), now_s=0)
    assessment = predictor.assess(spec, progress(spec, **changes), now_s=1)
    assert assessment.reason == reason and assessment.forecast is None, (
        "Replayed or rewritten history must be rejected."
    )
    recovered = predictor.assess(spec, progress(spec, report_age_s=0.2), now_s=2)
    assert recovered.reason == reason, "A compromised attempt cannot regain trust by replaying a previous good report."
    predictor.finish(spec, succeeded=True, completed_steps=128, actual_remaining_s=8, now_s=8)
    probe = command("probe")
    assert predictor.assess(probe, progress(probe), now_s=9).calibration_commands == 0, (
        "A compromised attempt cannot contribute calibration evidence."
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"command_id": "other-command"},
        {"workload": replace(WORKLOAD, model_version="v2")},
        {"total_steps": 256},
    ],
)
def test_report_identity_must_match_its_command(changes):
    predictor = CommandRuntimePredictor()
    spec = command()
    assessment = predictor.assess(spec, progress(spec, **changes), now_s=0)
    assert assessment.reason == "progress_identity_mismatch", "Foreign progress cannot be used for this command."
    assert assessment.forecast is None, "Foreign progress must not produce an estimate."


@pytest.mark.parametrize(
    "changes",
    [
        {"completed_steps": True},
        {"completed_steps": -1},
        {"completed_steps": 129},
        {"chunk_steps": (0, 16, 16, 16)},
        {"chunk_steps": (True, 16, 16, 16)},
        {"chunk_steps": (16, 16)},
        {"chunk_steps": [16, 16, 16, 16]},
        {"chunk_wall_s": (2.0, 2.0, 0.0, 2.0)},
        {"chunk_wall_s": (2.0, 2.0, float("nan"), 2.0)},
        {"chunk_wall_s": (2.0, 2.0, float("inf"), 2.0)},
        {"chunk_wall_s": (2.0, 2.0, "2", 2.0)},
        {"report_age_s": -1},
        {"report_age_s": float("nan")},
        {"report_age_s": True},
    ],
)
def test_malformed_provider_reports_return_an_explicit_unknown(changes):
    predictor = CommandRuntimePredictor()
    spec = command()
    assessment = predictor.assess(spec, progress(spec, **changes), now_s=0)
    assert assessment.reason == "malformed_progress", "Malformed provider input must produce a bounded fallback."
    assert assessment.forecast is None, "Malformed input cannot produce an estimate."


def test_opaque_work_and_missing_progress_stay_unknown():
    predictor = CommandRuntimePredictor()
    opaque = command("opaque", total_steps=None)
    fixed = command("fixed")
    assert predictor.assess(opaque, None, now_s=0).reason == "unknown_total_steps", (
        "Opaque commands require the conservative path."
    )
    assert predictor.assess(fixed, None, now_s=0).reason == "progress_unavailable", (
        "Declared fixed work still requires observations."
    )


def test_same_command_id_cannot_change_its_immutable_specification():
    predictor = CommandRuntimePredictor()
    original = command()
    predictor.assess(original, progress(original), now_s=0)
    changed = replace(original, total_steps=256)
    assessment = predictor.assess(changed, progress(changed), now_s=1)
    assert assessment.reason == "command_identity_changed", "A command ID cannot name a different attempt."
    assert predictor.assess(original, progress(original), now_s=2).reason == "command_identity_changed", (
        "Restoring the old specification must not erase an identity violation."
    )


@pytest.mark.parametrize("succeeded,completed,actual", [(False, 128, 8), (True, 64, 8), (True, 128, None)])
def test_failed_incomplete_or_unmeasured_commands_do_not_train_the_predictor(succeeded, completed, actual):
    predictor = CommandRuntimePredictor()
    spec = command()
    predictor.assess(spec, progress(spec), now_s=0)
    predictor.finish(spec, succeeded=succeeded, completed_steps=completed, actual_remaining_s=actual, now_s=8)
    probe = command("probe")
    assert predictor.assess(probe, progress(probe), now_s=9).calibration_commands == 0, (
        "Only confirmed full measured success may enter timing history."
    )


def test_duplicate_finish_and_reassessment_of_a_retired_id_do_not_double_count():
    predictor = CommandRuntimePredictor()
    spec = command()
    predictor.assess(spec, progress(spec), now_s=0)
    for _ in range(3):
        predictor.finish(spec, succeeded=True, completed_steps=128, actual_remaining_s=8, now_s=8)
    retired = predictor.assess(spec, progress(spec), now_s=9)
    assert retired.reason == "command_retired", "A completed ID must not start a second calibration sample."
    assert retired.calibration_commands == 1, "Completion replay must count exactly once."


def test_forget_releases_pending_capacity_and_preserves_unretired_work():
    predictor = CommandRuntimePredictor(PredictabilityConfig(max_pending=2))
    first, second, third = command("first"), command("second"), command("third")
    predictor.assess(first, progress(first), now_s=0)
    predictor.assess(second, progress(second), now_s=0)
    assert predictor.assess(third, progress(third), now_s=0).reason == "pending_capacity_exhausted", (
        "An overloaded predictor must refuse new state instead of evicting an active command."
    )
    predictor.forget(first.command_id)
    assert predictor.assess(first, progress(first), now_s=0).reason == "command_retired", (
        "Forgetting an attempt must not fabricate a fresh execution."
    )
    assert predictor.assess(third, progress(third), now_s=0).forecast is not None, (
        "Retiring an abandoned attempt must make room for new commands."
    )
    predictor.finish(second, succeeded=True, completed_steps=128, actual_remaining_s=8, now_s=8)
    assert predictor.assess(third, progress(third, report_age_s=1), now_s=8).calibration_commands == 1, (
        "Capacity pressure must preserve the original reference of unretired commands."
    )


def test_profile_and_retired_state_remain_bounded_without_relearning_retained_history_ids():
    predictor = CommandRuntimePredictor(PredictabilityConfig(max_profiles=1, max_pending=1))
    now_s = complete_shadow(predictor, "first", 0)
    now_s = complete_shadow(predictor, "second", now_s)
    first = command("first")
    assert predictor.assess(first, progress(first), now_s=now_s).reason == "command_retired", (
        "History must deduplicate commands after their retirement tombstone is evicted."
    )
    now_s = complete_shadow(predictor, "different", now_s, workload=replace(WORKLOAD, device_class="other"))
    probe = command("original-profile")
    assert predictor.assess(probe, progress(probe), now_s=now_s).calibration_commands == 0, (
        "The profile limit must evict the least recently used calibration profile."
    )
    assert len(predictor._history) <= 1 and len(predictor._retired) <= 1, "All retained profile state must be bounded."


def test_completion_report_is_not_a_zero_time_training_sample():
    predictor = CommandRuntimePredictor()
    spec = command()
    assessment = predictor.assess(spec, progress(spec, completed_steps=128, chunk_steps=(), chunk_wall_s=()), now_s=0)
    assert assessment.reason == "command_complete", "Observed completion should be explicit."
    assert assessment.forecast.estimate_s == 0 and not assessment.eligible, "Completed work cannot need placement."
    assert assessment.reference_time_s is None, "A zero-time completion must not create a shadow reference."
    predictor.finish(spec, succeeded=True, completed_steps=128, actual_remaining_s=0, now_s=0)
    probe = command("probe")
    assert predictor.assess(probe, progress(probe), now_s=0).calibration_commands == 0, (
        "A command first observed after completion provides no prediction accuracy evidence."
    )


def test_terminal_summary_can_omit_chunks_after_valid_intermediate_progress():
    predictor = CommandRuntimePredictor(PredictabilityConfig(min_completed_commands=1))
    spec = command()
    first = predictor.assess(spec, progress(spec), now_s=0)
    terminal = progress(spec, completed_steps=128, chunk_steps=(), chunk_wall_s=())
    final = predictor.assess(spec, terminal, now_s=8)
    assert final.reason == "command_complete", "A terminal summary may omit the intermediate chunk history."
    assert final.reference_time_s == first.reference_time_s, "Terminal progress must preserve the shadow boundary."
    predictor.finish(spec, succeeded=True, completed_steps=128, actual_remaining_s=8, now_s=8)
    probe = command("probe")
    assert predictor.assess(probe, progress(probe), now_s=9).eligible, (
        "A valid terminal summary must preserve independent completion calibration."
    )


def test_terminal_summary_cannot_clear_an_earlier_progress_violation():
    predictor = CommandRuntimePredictor()
    spec = command()
    predictor.assess(spec, progress(spec), now_s=0)
    predictor.assess(spec, progress(spec, chunk_wall_s=(2.1,) * 4), now_s=1)
    terminal = progress(spec, completed_steps=128, chunk_steps=(), chunk_wall_s=())
    final = predictor.assess(spec, terminal, now_s=8)
    assert final.reason == "progress_changed_without_advance", (
        "Terminal progress cannot erase invalid earlier evidence."
    )
    predictor.finish(spec, succeeded=True, completed_steps=128, actual_remaining_s=8, now_s=8)
    probe = command("probe")
    assert predictor.assess(probe, progress(probe), now_s=9).calibration_commands == 0, (
        "An invalid attempt must remain excluded after a terminal report."
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"min_chunks": 0},
        {"min_chunks": True},
        {"warmup_chunks": -1},
        {"min_completed_commands": 9},
        {"history_window": 0},
        {"max_profiles": 0},
        {"max_pending": 0},
        {"max_rate_ratio": 0.9},
        {"max_drift_ratio": float("inf")},
        {"max_relative_error": -1},
        {"absolute_error_floor_s": 0},
        {"history_ttl_s": float("nan")},
    ],
)
def test_invalid_configuration_fails_before_tracking_work(changes):
    with pytest.raises(ValueError):
        PredictabilityConfig(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"command_id": ""},
        {"model_id": ""},
        {"workload": "opaque"},
        {"total_steps": 0},
        {"total_steps": True},
        {"relocatable": 1},
    ],
)
def test_invalid_internal_specifications_fail_loudly(changes):
    with pytest.raises(ValueError):
        command(**changes)


def test_identity_is_immutable_and_the_clock_must_be_monotonic():
    with pytest.raises(FrozenInstanceError):
        WORKLOAD.model_version = "changed"
    with pytest.raises(ValueError):
        replace(WORKLOAD, code_version="")
    predictor = CommandRuntimePredictor()
    spec = command()
    predictor.assess(spec, progress(spec), now_s=1)
    with pytest.raises(ValueError, match="backwards"):
        predictor.assess(spec, progress(spec), now_s=0)
    with pytest.raises(ValueError, match="finite"):
        predictor.assess(spec, progress(spec), now_s=float("nan"))
