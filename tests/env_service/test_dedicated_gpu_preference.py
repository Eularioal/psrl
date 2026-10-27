"""Verify device priority through worker eligibility, dispatch and final admission."""

import asyncio
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules, gpu_placements_from_worker_groups, select_worker_nodes
from psrl.workers.env_service.placement import EnvAwareDispatcher
from psrl.workers.env_service.types import EnvDemand, EnvUnit

MODEL = "192.0.2.1"
OTHER = "192.0.2.2"
DEMAND = EnvDemand(cpu=1, memory_mb=128, needs_gpu=True, gpu_memory_mb=1024)


def unit(name, ip, gpu, slots=2, mode="episode"):
    return EnvUnit(name, ip, 4, 1024, (gpu,), slots, 2048, gpu_reservation=mode)


def rules(policy="prefer_other"):
    return NodeRules.build(policy, [MODEL], train_rollout_gpu_indices={MODEL: [0]})


def test_same_node_spare_device_is_preferred_and_allowed_by_cpu_only():
    spare, occupied = unit("spare", MODEL, 1), unit("occupied", MODEL, 0)
    assert rules().tier(spare, DEMAND) == 0, "An independent device must remain preferred."
    assert rules().tier(occupied, DEMAND) == 1, "The model device must remain a fallback."
    assert rules("cpu_only").tier(spare, DEMAND) == 0, "Safe same-node GPUs must be eligible."
    assert rules("cpu_only").tier(occupied, DEMAND) is None, "Model GPUs need explicit sharing permission."
    assert rules("avoid").tier(spare, DEMAND) is None, "The strict node exclusion must remain effective."


def test_missing_device_inventory_does_not_claim_a_protected_gpu_is_free():
    unknown = NodeRules.build("cpu_only", [MODEL])
    assert unknown.tier(unit("unknown", MODEL, 1), DEMAND) is None, "Unknown placement must remain protected."


def test_fallback_worker_nodes_survive_initial_selection():
    selected, _ = select_worker_nodes([("model", MODEL), ("other", OTHER)], rules())
    assert selected == [("other", OTHER), ("model", MODEL)], "Runtime fallback needs a worker on both nodes."


def test_dispatch_preference_precedes_load_and_round_robin():
    ledger = EnvUnitLedger([unit("protected", MODEL, 0), unit("dedicated", OTHER, 0)], rules())
    capacities = ledger.node_capacity()
    capacities[MODEL] = replace(capacities[MODEL], node_load_per_cpu=0.0)
    capacities[OTHER] = replace(capacities[OTHER], node_load_per_cpu=0.5)
    dispatcher = EnvAwareDispatcher([MODEL, OTHER])
    dispatcher.update(capacities)
    assert dispatcher.assign([True], 0, demands=[DEMAND]) == [1], "Device preference must precede CPU load."
    assert dispatcher.assign([True], 0) == [1], "Count-only snapshots must preserve the priority too."


def test_command_pressure_uses_another_gpu_then_returns_to_preferred_capacity():
    ledger = EnvUnitLedger(
        [unit("protected", MODEL, 0, mode="command"), unit("dedicated", OTHER, 0, mode="command")], rules()
    )
    ledger.report_measurement(OTHER, {0: 1800})
    ledger.report_measurement(MODEL, {0: 128})
    dispatcher = EnvAwareDispatcher([MODEL, OTHER])
    dispatcher.update(ledger.node_capacity())
    assert dispatcher.assign([True], 1, demands=[DEMAND]) == [0], "Full memory must outrank an empty host slot."
    first = ledger.try_admit("first", DEMAND)
    assert first.lease.node_ip == MODEL, "Final admission must agree with the available fallback."
    ledger.release(first.lease.lease_id)
    ledger.report_measurement(OTHER, {0: 0})
    second = ledger.try_admit("second", DEMAND)
    assert second.lease.node_ip == OTHER, "Preference must resume when independent capacity returns."


def test_local_admission_prefers_spare_gpu_then_queues_when_both_are_full():
    ledger = EnvUnitLedger(
        [unit("a-model", MODEL, 0, mode="command"), unit("z-spare", MODEL, 1, mode="command")], rules()
    )
    ledger.report_measurement(MODEL, {0: 0, 1: 0})
    assert ledger.try_admit("first", DEMAND, MODEL).lease.gpu_indices == (1,), "The spare GPU must win ties."
    ledger.report_measurement(MODEL, {0: 1900, 1: 1900})
    queued = ledger.try_admit("queued", DEMAND, MODEL).lease
    assert queued.gpu_indices == (1,) and not queued.gpu_active, "Queueing a session must not grant GPU capacity."


def test_group_planning_charges_command_capacity_between_groups():
    ledger = EnvUnitLedger([unit("protected", MODEL, 0), unit("dedicated", OTHER, 0, mode="command")], rules())
    dispatcher = EnvAwareDispatcher([MODEL, OTHER])
    dispatcher.update(ledger.node_capacity())
    choices = dispatcher.assign([True, True], 0, demands=[DEMAND, DEMAND], group_sizes=[2, 1])
    assert choices == [1, 0], "A planned group must consume its advertised capacity for subsequent placement."


def test_admission_filters_unhealthy_nodes_and_does_not_move_existing_sessions():
    ledger = EnvUnitLedger([unit("protected", MODEL, 0), unit("dedicated", OTHER, 0)], rules())
    admitted = ledger.try_admit("session", DEMAND, available_node_ips={MODEL})
    assert admitted.lease.node_ip == MODEL, "Unavailable preferred nodes cannot receive leases."
    retried = ledger.try_admit("session", DEMAND, available_node_ips={OTHER})
    assert retried.status == "wait", "An existing session must not silently change its node."


def test_runtime_inventory_keeps_every_assigned_device_and_rejects_unknown_nodes():
    class Group:
        def execute_all_sync(self, method):
            return [[0, 2], [1]] if method == "get_runtime_gpu_ids" else ["n1", "n2"]

    placements = gpu_placements_from_worker_groups({"rollout_0": Group(), "ps": object()}, {"n1": MODEL, "n2": OTHER})
    assert placements == {"rollout_0": frozenset({(MODEL, 0), (MODEL, 2), (OTHER, 1)})}, "All device IDs must count."
    with pytest.raises(ValueError, match="absent"):
        gpu_placements_from_worker_groups({"rollout_0": Group()}, {"n1": MODEL})


@pytest.mark.asyncio
async def test_router_uses_global_priority_and_returns_the_chosen_nodes_url():
    from psrl.workers.env_service.router import RayEnvRouter

    router = RayEnvRouter(
        {"node_policy": "prefer_other", "units": [asdict(unit("p", MODEL, 0)), asdict(unit("d", OTHER, 0))]},
        [MODEL],
    )
    router._ensure_reaper = lambda: None
    router._node_health = {ip: SimpleNamespace(available=True, failures=0) for ip in (MODEL, OTHER)}
    router._health_task = lambda ip, force=False: None

    async def ready(ip):
        return None, f"http://{ip}/telemetry"

    router._node_task = lambda ip: asyncio.create_task(ready(ip))
    router.set_gpu_placement({MODEL: [0]})
    result = await router.try_admit("global", DEMAND)
    assert result.lease.node_ip == OTHER, "Health polling order must not determine placement."
    assert result.lease.node_service_url == f"http://{OTHER}/telemetry", "Telemetry must follow the chosen node."
    with pytest.raises(RuntimeError, match="leases exist"):
        router.set_gpu_placement({MODEL: [1]})
