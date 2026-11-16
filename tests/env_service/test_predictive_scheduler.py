"""Verify evidence gates and capacity boundaries before warm placement."""

from dataclasses import replace

import pytest
from psrl.workers.env_service.command_prediction import CommandSpec, StepProgress, WorkloadKey
from psrl.workers.env_service.predictive_scheduler import (
    CommandCostProfile,
    DeviceState,
    PlacementConfig,
    PlacementRequest,
    PredictionAwareScheduler,
    QueuedWork,
)

KEY = WorkloadKey("code-v1", "weights-v1", "shape=32;batch=8;dtype=f32;phase=train", "gpu-class", "runtime-v1")


def request(command_id="new", *, steps=100, home=None, relocatable=True, model="model-v1", workspace=10):
    return PlacementRequest(CommandSpec(command_id, KEY, model, steps, relocatable), 40, workspace, home)


def progress(spec, *, age=0, walls=(1.0, 1.0, 1.0, 1.0)):
    return StepProgress(spec.command_id, spec.workload, spec.total_steps, 40, (10,) * 4, walls, age)


def profile(req, *, execution=2.0, rebuild=1.0, observed=0.0, device_class="gpu-class"):
    return CommandCostProfile(
        replace(req.spec.workload, device_class=device_class),
        req.spec.model_id,
        req.spec.total_steps,
        req.model_bytes,
        req.workspace_bytes,
        (rebuild,) * 3,
        (execution,) * 3,
        observed,
    )


def calibrate(scheduler):
    for i in range(3):
        req = request(f"calibration-{i}")
        assessment = scheduler.predictor.assess(req.spec, progress(req.spec), now_s=i * 10)
        scheduler.predictor.finish(
            req.spec,
            succeeded=True,
            completed_steps=100,
            actual_remaining_s=assessment.forecast.estimate_s,
            now_s=i * 10 + 6,
        )


def states(*, walls=(1.0,) * 4, age=0, cold_budget=100, queued=()):
    active = request("active")
    return (
        DeviceState(
            "hot",
            "gpu-class",
            frozenset({"model-v1"}),
            100,
            QueuedWork(active),
            progress(active.spec, walls=walls, age=age),
            queued,
        ),
        DeviceState("cold", "gpu-class", frozenset(), cold_budget),
    )


def plan(scheduler, *, devices=None, req=None, profiles=None, now=30):
    req = request() if req is None else req
    if devices is None:
        devices = states()
    if profiles is None:
        profiles = {device.device_id: profile(req) for device in devices}
    return scheduler.plan(req, devices, profiles, now_s=now)


def test_disabled_by_default_without_fabricated_time_estimates():
    decision = plan(PredictionAwareScheduler())
    assert (decision.device_id, decision.reason, decision.prediction_used) == ("hot", "mode_off", False)
    assert decision.comparison is None


def test_observation_collects_shadow_without_changing_placement():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="observe"))
    decision = plan(scheduler)
    assert decision.device_id == "hot" and decision.reason == "mode_observe"
    assessment = scheduler.predictor.assess(request("active").spec, progress(request("active").spec), now_s=30)
    assert assessment.reference_time_s == 30
    assert assessment.reason == "insufficient_calibration"


def test_auto_requires_independent_calibration():
    decision = plan(PredictionAwareScheduler(PlacementConfig(mode="auto")))
    assert decision.device_id == "hot"
    assert decision.reason == "queue_time_unpredictable"
    assert ("hot", "insufficient_calibration") in decision.assessments


def test_validated_active_command_enables_profitable_cold_copy():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    calibrate(scheduler)
    decision = plan(scheduler)
    assert decision.prediction_used
    assert (decision.device_id, decision.action) == ("cold", "rebuild")
    assert decision.comparison.selected.completion.point_s == 3
    hot = next(value for value in decision.comparison.estimates if value.device_id == "hot")
    assert hot.completion.point_s == 8


@pytest.mark.parametrize(
    "walls,age,reason",
    [
        ((1.0, 1.0, 1.0, 3.0), 0, "unstable_step_rate"),
        ((1.0,) * 4, 10, "overdue_progress"),
        ((1.0, 1.0, 1.3, 1.3), 0, "step_rate_drift"),
    ],
)
def test_bad_live_progress_disables_timing_decision(walls, age, reason):
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    calibrate(scheduler)
    decision = plan(scheduler, devices=states(walls=walls, age=age))
    assert decision.device_id == "hot" and not decision.prediction_used
    assert ("hot", reason) in decision.assessments


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"observed_at_s": -1}, "invalid"),
        ({"observed_at_s": 31}, "profile_expired"),
        ({"observed_at_s": 0}, "profile_expired"),
        ({"total_steps": 101}, "profile_identity_mismatch"),
        ({"workspace_bytes": 11}, "profile_identity_mismatch"),
        ({"execution_samples_s": (1.0, 1.0)}, "profile_insufficient_or_unstable"),
        ({"execution_samples_s": (1.0, 1.0, 5.0)}, "profile_insufficient_or_unstable"),
    ],
)
def test_profile_validation(change, reason):
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto", max_profile_age_s=30))
    calibrate(scheduler)
    req = request()
    if reason == "invalid":
        with pytest.raises(ValueError):
            replace(profile(req), **change)
        return
    profiles = {"hot": replace(profile(req, observed=1), **change), "cold": profile(req, observed=1)}
    decision = plan(scheduler, profiles=profiles)
    assert not decision.prediction_used and decision.reason == reason


def test_missing_cold_rebuild_profile_keeps_affinity():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    calibrate(scheduler)
    req = request()
    profiles = {"hot": profile(req), "cold": replace(profile(req), rebuild_samples_s=())}
    decision = plan(scheduler, profiles=profiles)
    assert decision.device_id == "hot" and not decision.prediction_used


def test_hot_replica_does_not_require_rebuild_samples():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    req = request()
    device = DeviceState("hot", "gpu-class", frozenset({"model-v1"}), 10)
    decision = plan(scheduler, devices=(device,), profiles={"hot": replace(profile(req), rebuild_samples_s=())})
    assert decision.device_id == "hot" and decision.prediction_used


def test_unknown_queued_service_remains_unknown():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    calibrate(scheduler)
    queued = QueuedWork(request("queued"))
    decision = plan(scheduler, devices=states(queued=(queued,)))
    assert decision.device_id == "hot" and decision.reason == "queue_time_unpredictable"
    assert ("hot", "queued:profile_unavailable") in decision.assessments


def test_queued_cold_rebuild_is_counted_before_future_cache_reuse():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    req = request(steps=2)
    queued_req = request("queued", steps=2)
    queued = QueuedWork(queued_req, profile(queued_req, execution=3, rebuild=4))
    devices = (
        DeviceState("hot", "gpu-class", frozenset(), 100, queued=(queued,)),
        DeviceState("cold", "gpu-class", frozenset(), 100),
    )
    decision = plan(scheduler, devices=devices, req=req)
    values = {estimate.device_id: estimate for estimate in decision.comparison.estimates}
    assert values["hot"].waiting.point_s == 7
    assert values["hot"].rebuild.point_s == 0
    assert values["hot"].completion.point_s == 9


def test_future_cache_reservations_prevent_overbooking():
    scheduler = PredictionAwareScheduler()
    queued = QueuedWork(request("queued", model="other-model"))
    devices = (DeviceState("one", "gpu-class", frozenset(), 80, queued=(queued,)),)
    decision = plan(scheduler, devices=devices)
    assert decision.device_id is None and decision.action == "reject"


def test_same_future_model_is_charged_once_but_workspace_always_charged():
    scheduler = PredictionAwareScheduler()
    queued = QueuedWork(request("queued"))
    devices = (DeviceState("one", "gpu-class", frozenset(), 50, queued=(queued,)),)
    assert plan(scheduler, devices=devices).device_id == "one"
    assert plan(scheduler, devices=devices, req=request(workspace=11)).action == "reject"


def test_capacity_is_a_gate_even_after_calibration():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    calibrate(scheduler)
    decision = plan(scheduler, devices=states(cold_budget=49))
    assert decision.device_id == "hot"
    assert all(value.device_id != "cold" for value in decision.comparison.estimates)


def test_nontransferable_state_stays_home():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    decision = plan(scheduler, req=request(home="hot", relocatable=False))
    assert decision.device_id == "hot" and decision.reason == "state_not_relocatable"
    with pytest.raises(ValueError):
        request(relocatable=False)


def test_opaque_command_uses_ordinary_placement():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    decision = plan(scheduler, req=request(steps=None), profiles={})
    assert decision.device_id == "hot" and decision.reason == "unknown_total_steps"


def test_heterogeneous_device_requires_its_own_profile():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    req = request()
    devices = (DeviceState("other", "slower-class", frozenset(), 100),)
    decision = plan(scheduler, devices=devices, profiles={"other": profile(req)})
    assert decision.reason == "profile_identity_mismatch" and not decision.prediction_used
    decision = plan(scheduler, devices=devices, profiles={"other": profile(req, device_class="slower-class")})
    assert decision.prediction_used


def test_bad_calibration_trips_automatic_placement():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    calibrate(scheduler)
    spec = request("bad-history").spec
    scheduler.predictor.assess(spec, progress(spec), now_s=27)
    scheduler.predictor.finish(spec, succeeded=True, completed_steps=100, actual_remaining_s=30, now_s=28)
    decision = plan(scheduler)
    assert not decision.prediction_used
    assert ("hot", "history_error") in decision.assessments


@pytest.mark.parametrize("change", [{"model_bytes": 400}, {"version": "weights-v2"}])
def test_target_cannot_reinterpret_the_identity_of_a_queued_model(change):
    queued_request = request("queued")
    target = request()
    if "model_bytes" in change:
        target = replace(target, model_bytes=change["model_bytes"])
    else:
        key = replace(target.spec.workload, model_version=change["version"])
        target = replace(target, spec=replace(target.spec, workload=key))
    device = DeviceState("one", "gpu-class", frozenset(), 100, queued=(QueuedWork(queued_request),))
    with pytest.raises(ValueError, match="Immutable model identities"):
        plan(PredictionAwareScheduler(), devices=(device,), req=target, profiles={})


@pytest.mark.parametrize("active", [True, False])
def test_active_and_queued_work_cannot_borrow_another_device_class_history(active):
    queued = QueuedWork(request("prior"))
    with pytest.raises(ValueError, match="device class"):
        DeviceState(
            "one",
            "different-class",
            frozenset(),
            100,
            active=queued if active else None,
            queued=() if active else (queued,),
        )


def test_existing_overcommitted_queue_is_excluded_from_new_admission():
    queued = QueuedWork(request("queued"))
    device = DeviceState("one", "gpu-class", frozenset(), 49, queued=(queued,))
    decision = plan(PredictionAwareScheduler(), devices=(device,), profiles={})
    assert decision.action == "reject" and decision.device_id is None
    assert ("one", "prior_queue_exceeds_budget") in decision.assessments


def test_fresh_well_calibrated_prediction_can_still_choose_warm_wait():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
    calibrate(scheduler)
    req = request()
    costs = {"hot": profile(req), "cold": profile(req, rebuild=10)}
    decision = plan(scheduler, profiles=costs)
    assert decision.prediction_used
    assert decision.device_id == "hot" and decision.action == "wait"


def test_interval_mode_requires_separated_empirical_ranges():
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto", decision_mode="interval"))
    calibrate(scheduler)
    decision = plan(scheduler)
    assert decision.prediction_used and decision.device_id == "cold"
    assert not decision.comparison.statistical_guarantee


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "force"},
        {"min_profile_samples": True},
        {"max_profile_age_s": 0},
        {"max_profile_rate_ratio": 0.9},
        {"switch_margin_s": float("nan")},
        {"decision_mode": "guaranteed"},
    ],
)
def test_invalid_configuration_is_not_silently_accepted(kwargs):
    with pytest.raises(ValueError):
        PlacementConfig(**kwargs)
