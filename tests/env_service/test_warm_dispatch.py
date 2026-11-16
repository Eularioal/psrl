"""Check advisory placement, truthful cost accounting and unknown-progress fallbacks."""

from dataclasses import asdict, replace

import pytest
from psrl.workers.env_service.step_predictor import predict_remaining
from psrl.workers.env_service.warm_dispatch import (
    ModelCost,
    QueueSnapshot,
    TimeEstimate,
    WarmPlacementController,
    estimate_from_forecast,
    fits_capacity,
    required_gpu_bytes,
)
from psrl.workers.env_service.warm_placement import DurationInterval


def duration(point, lower=None, upper=None):
    return TimeEstimate(point, DurationInterval(point if lower is None else lower, point if upper is None else upper))


def model_cost(*, rebuild=5, execution=4, workspace=10):
    return ModelCost(
        "model@version",
        100,
        duration(rebuild) if rebuild is not None else None,
        duration(execution) if execution is not None else None,
        workspace,
    )


def hot(*, wait=20, capacity=10, device_id="hot", queued=()):
    return QueueSnapshot(
        device_id,
        frozenset({"model@version"}),
        capacity,
        True,
        duration(wait) if wait is not None else None,
        queued,
    )


def cold(*, capacity=110, queued=()):
    return QueueSnapshot("cold", frozenset(), capacity, queued_service=queued)


def test_point_and_interval_decisions_keep_the_original_uncertainty():
    devices = [replace(hot(), active_remaining=duration(12, 1, 20)), cold()]
    cost = replace(model_cost(), rebuild=duration(4, 2, 6))
    point = WarmPlacementController("point").decide(cost, devices)
    interval = WarmPlacementController("interval").decide(cost, devices)
    assert point.action == "rebuild" and point.device_id == "cold"
    assert interval.action == "wait" and interval.device_id == "hot"
    assert point.selected.completion == duration(8, 6, 10)
    assert point.estimates[0].completion == duration(16, 5, 24)
    assert not asdict(point)["statistical_guarantee"]
    assert not asdict(point.selected.completion)["statistical_guarantee"]


@pytest.mark.parametrize("wait,expected", [(7, "wait"), (7.01, "rebuild")])
def test_point_switch_margin_is_strict_and_not_added_to_execution(wait, expected):
    decision = WarmPlacementController(switch_margin_s=2).decide(model_cost(), [hot(wait=wait), cold()])
    assert decision.action == expected
    assert decision.estimates[1].completion.point_s == 9


def test_every_prior_command_contributes_complete_service_including_its_rebuild():
    device = hot(wait=2, queued=(duration(3, 2, 4), duration(5, 4, 6)))
    decision = WarmPlacementController().decide(model_cost(), [device, cold()])
    assert decision.action == "rebuild"
    assert decision.estimates[0].waiting == duration(10, 8, 12)
    assert decision.estimates[0].completion == duration(14, 12, 16)


@pytest.mark.parametrize("device", [hot(wait=None), hot(queued=(None,))])
def test_unknown_hot_service_falls_back_to_affinity_without_inventing_a_finish(device):
    decision = WarmPlacementController().decide(model_cost(), [device, cold()])
    assert decision.action == "wait" and decision.device_id == "hot"
    assert decision.reason_code == "unknown_hot_completion_affinity_fallback"
    assert decision.selected.waiting is None and decision.selected.completion is None


def test_a_known_hot_replica_is_the_affinity_reference_before_an_unknown_replica():
    decision = WarmPlacementController().decide(model_cost(), [hot(wait=None), hot(wait=1, device_id="known"), cold()])
    assert decision.action == "wait" and decision.device_id == "known"


def test_unknown_execution_is_safe_to_queue_on_the_hot_replica_but_not_a_cold_copy():
    cost = model_cost(execution=None)
    decision = WarmPlacementController().decide(cost, [hot(), cold()])
    assert decision.action == "wait" and decision.selected.completion is None
    missing = WarmPlacementController().decide(cost, [cold()])
    assert missing.action == "defer" and missing.device_id is None


def test_all_policies_share_workspace_and_model_capacity_accounting():
    cost = model_cost()
    assert required_gpu_bytes(cost, hot()) == 10
    assert required_gpu_bytes(cost, cold()) == 110
    assert fits_capacity(cost, hot()) and fits_capacity(cost, cold())
    devices = [hot(capacity=9), cold(capacity=109)]
    assert not fits_capacity(cost, devices[0]) and not fits_capacity(cost, devices[1])
    for strategy in ("affinity", "always_copy", "predictive"):
        decision = WarmPlacementController().decide(cost, devices, strategy=strategy)
        assert decision.action == "reject" and decision.reason_code == "no_declared_capacity"
        assert decision.device_id is None


def test_affinity_needs_workspace_even_when_the_model_is_resident():
    decision = WarmPlacementController().decide(model_cost(), [hot(capacity=9), cold()], strategy="affinity")
    assert decision.action == "rebuild" and decision.device_id == "cold"
    assert not decision.estimates[0].capacity_fits


def test_idle_hot_model_needs_no_rebuild_estimate_or_additional_model_bytes():
    idle = QueueSnapshot("hot", frozenset({"model@version"}), 10)
    decision = WarmPlacementController().decide(model_cost(rebuild=None), [idle])
    assert decision.action == "reuse"
    assert decision.selected.rebuild == duration(0)
    assert decision.selected.required_gpu_bytes == 10


def test_per_device_costs_can_reverse_the_shared_profile_decision():
    cost = model_cost()
    devices = [hot(), cold()]
    default = WarmPlacementController().decide(cost, devices)
    local = WarmPlacementController().decide(cost, devices, device_costs={"cold": model_cost(execution=50)})
    assert default.action == "rebuild"
    assert local.action == "wait"
    assert local.estimates[1].execution.point_s == 50
    assert local.estimates[0].execution.point_s == 4


def test_affinity_and_always_copy_are_distinct_even_without_a_predicted_copy_gain():
    devices = [hot(wait=1), cold()]
    affinity = WarmPlacementController().decide(model_cost(), devices, strategy="affinity")
    copy = WarmPlacementController().decide(model_cost(), devices, strategy="always_copy")
    assert affinity.action == "wait" and affinity.selected.completion.point_s == 5
    assert copy.action == "rebuild" and copy.selected.completion.point_s == 9


@pytest.mark.parametrize("cost,device", [(model_cost(rebuild=None), cold()), (model_cost(), cold(queued=(None,)))])
def test_always_copy_requires_known_cold_service_and_retains_affinity_otherwise(cost, device):
    decision = WarmPlacementController().decide(cost, [hot(), device], strategy="always_copy")
    assert decision.action == "wait" and decision.device_id == "hot"
    assert decision.estimates[1].completion is None


def test_always_copy_does_not_need_a_known_hot_wait_to_choose_a_known_legal_copy():
    decision = WarmPlacementController().decide(model_cost(), [hot(wait=None), cold()], strategy="always_copy")
    assert decision.action == "rebuild" and decision.selected.capacity_fits
    assert decision.selected.completion.point_s == 9


def test_deadline_uses_the_empirical_interval_even_in_point_mode():
    cost = replace(model_cost(), rebuild=duration(5, 3, 7))
    within = WarmPlacementController().decide(cost, [cold()], deadline_s=11)
    uncertain = WarmPlacementController().decide(cost, [cold()], deadline_s=9)
    miss = WarmPlacementController().decide(cost, [cold()], deadline_s=6)
    assert within.action == "rebuild" and within.selected.deadline_state == "within"
    assert uncertain.action == "defer" and uncertain.estimates[0].deadline_state == "uncertain"
    assert miss.action == "reject" and miss.reason_code == "predicted_deadline_miss"


def test_forecast_adapter_preserves_point_interval_and_unknown_status():
    forecast = predict_remaining([16, 16], [2, 2], 32, 96)
    estimate = estimate_from_forecast(forecast)
    assert estimate.point_s == forecast.estimate_s
    assert estimate.interval == DurationInterval(forecast.lower_s, forecast.upper_s)
    assert estimate_from_forecast(predict_remaining([], [], 0, 96)) is None
    assert estimate_from_forecast(predict_remaining([16, 16], [2, 2], 32, 96, 3)) is None
    assert estimate_from_forecast(predict_remaining([], [], 96, 96)) == duration(0)
    with pytest.raises(ValueError):
        estimate_from_forecast(replace(forecast, estimate_s=None))


def test_queue_snapshot_does_not_retain_mutable_input_collections():
    models, queued = ["model@version"], [duration(3)]
    snapshot = QueueSnapshot("hot", models, 10, queued_service=queued)
    models.clear()
    queued.clear()
    assert snapshot.resident_models == frozenset({"model@version"})
    assert snapshot.queued_service == (duration(3),)


@pytest.mark.parametrize("point,interval", [(True, (0, 1)), (-1, (0, 1)), (2, (0, 1)), (float("inf"), (0, 1))])
def test_invalid_points_are_rejected(point, interval):
    with pytest.raises(ValueError):
        TimeEstimate(point, DurationInterval(*interval))


@pytest.mark.parametrize(
    "options",
    [
        {"available_gpu_bytes": True},
        {"available_gpu_bytes": -1},
        {"resident_models": "model"},
        {"resident_models": {1}},
        {"busy": 1},
        {"busy": False, "active_remaining": duration(1)},
        {"queued_service": (3,)},
        {"active_remaining": 3},
        {"queued_service": ""},
    ],
)
def test_invalid_queue_inputs_fail_before_decision(options):
    values = dict(device_id="device", resident_models=frozenset(), available_gpu_bytes=110)
    with pytest.raises(ValueError):
        QueueSnapshot(**(values | options))


@pytest.mark.parametrize(
    "override",
    [replace(model_cost(), model_id="other"), replace(model_cost(), model_bytes=101), model_cost(workspace=11), 3],
)
def test_device_overrides_cannot_change_the_workload_memory_contract(override):
    with pytest.raises(ValueError, match="same model"):
        WarmPlacementController().decide(model_cost(), [hot(), cold()], device_costs={"cold": override})


def test_duplicate_devices_unknown_cost_keys_and_invalid_options_are_rejected():
    controller = WarmPlacementController()
    with pytest.raises(ValueError, match="unique"):
        controller.decide(model_cost(), [cold(), cold()])
    with pytest.raises(ValueError, match="known device"):
        controller.decide(model_cost(), [cold()], device_costs={"typo": model_cost()})
    with pytest.raises(ValueError, match="strategy"):
        controller.decide(model_cost(), [cold()], strategy="invented")
    with pytest.raises(ValueError, match="Switch margin"):
        WarmPlacementController(switch_margin_s=True)
    with pytest.raises(ValueError, match="Mode"):
        WarmPlacementController(mode="invented")
