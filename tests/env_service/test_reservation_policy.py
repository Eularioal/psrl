"""Exercise mixed reservation admission and conservative capability resolution."""

import asyncio
from dataclasses import replace

import pytest
from psrl.workers.env_service.harbor_resources import demand_from_task
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.reservation_policy import choose_reservation
from psrl.workers.env_service.types import EnvDemand, EnvUnit


def test_automatic_modes_require_a_managed_execution_contract():
    assert choose_reservation("auto", "unknown", True).mode == "episode", "Unknown state must remain reserved."
    assert choose_reservation("auto", "resident_state", True).mode == "episode", "Resident state must remain reserved."
    assert choose_reservation("auto", "independent_commands", True).mode == "command", "Managed commands can return."
    assert choose_reservation("episode", "independent_commands", True).mode == "episode", "Explicit policy wins."
    with pytest.raises(ValueError, match="Unknown GPU execution contract"):
        EnvDemand(gpu_execution_contract="independent")


def test_mixed_pool_keeps_episode_memory_charged_while_command_sessions_queue():
    unit = EnvUnit("u", "node", 8, 128, (0,), 4, 80, gpu_reservation="auto")
    ledger = EnvUnitLedger([unit], NodeRules.build("any"))
    resident = EnvDemand(1, 16, True, 60, gpu_execution_contract="resident_state")
    commands = replace(resident, gpu_execution_contract="independent_commands")
    a = ledger.try_admit("a", resident, "node")
    b = ledger.try_admit("b", commands, "node")
    assert a.status == b.status == "granted", "Host admission can succeed while a command waits for GPU."
    assert (a.lease.gpu_active, b.lease.gpu_active) == (True, False), "Only the resident session charges GPU."
    assert ledger.snapshot()[0].declared_gpu_memory_mb == 60, "Resident allocation cannot disappear."
    assert ledger.try_admit("c", resident, "node").status == "wait", "A second resident session must queue."
    again = ledger.try_admit("a", commands, "node")
    assert again.lease.gpu_reservation == "episode", "An existing lease cannot change its return protocol."
    assert b.lease.gpu_reservation_reason, "A chosen mode needs an auditable reason."
    ledger.release(a.lease.lease_id)
    assert ledger.try_admit("c", resident, "node").status == "granted", "Released capacity must be reusable."


def test_auto_command_uses_existing_physical_return_checks():
    from psrl.workers.env_service.command_gpu import GpuCommandScheduler

    from tests.env_service.test_gpu_commands import Node, Remote

    async def run():
        unit = EnvUnit("u", "node", 8, 32, (0,), 4, 80, gpu_reservation="auto")
        ledger = EnvUnitLedger([unit], NodeRules.build("any"))
        demand = EnvDemand(2, 8, True, 51, gpu_execution_contract="independent_commands")
        a, b = [ledger.try_admit(str(i), demand, "node").lease for i in range(2)]
        node = Node()

        async def get_node(ip):
            return Remote(node)

        scheduler = GpuCommandScheduler(ledger, get_node)
        grant = await scheduler.acquire(a.lease_id, "container-a", "first")
        assert grant["status"] == "granted", "Resolved automatic mode must execute through the scheduler."
        assert (await scheduler.acquire(b.lease_id, "container-b", "second"))["status"] == "wait", (
            "A full device must block another automatic command."
        )
        await scheduler.release(a.lease_id, "first", grant["epoch"])
        assert not ledger._leases[a.lease_id].gpu_active and not node.active, "Return needs physical acknowledgement."
        assert (await scheduler.acquire(b.lease_id, "container-b", "second"))["status"] == "granted", (
            "A waiting command must reuse the returned device."
        )

    asyncio.run(run())


def test_auto_rejects_unsupported_device_lifecycles():
    unit = EnvUnit("u", "node", 8, 32, (0,), 4, 80, gpu_reservation="auto")
    with pytest.raises(ValueError, match="Automatic reservations"):
        replace(unit, shared_rollout_replica_rank=0, model_headroom_mb=10)
    with pytest.raises(ValueError, match="Automatic reservations"):
        replace(unit, gpu_execution_backend="mps", mps_image="image")


def test_cost_advice_can_retain_a_capable_session_but_cannot_override_capability():
    from psrl.workers.env_service.command_prediction import WorkloadKey
    from psrl.workers.env_service.reuse_prediction import ReuseDelayPredictor, ReuseSample

    predictor = ReuseDelayPredictor()
    key = WorkloadKey("code", "model", "input-and-protocol", "device", "runtime")
    for _ in range(3):
        predictor.observe(key, ReuseSample(1, 20, 2, 6), now_s=0)
    advice = predictor.predict(key, now_s=1, max_extra_wait_s=1)
    unit = EnvUnit("u", "node", 8, 128, (0,), 4, 80, gpu_reservation="auto")
    ledger = EnvUnitLedger([unit], NodeRules.build("any"))
    demand = EnvDemand(
        1,
        16,
        True,
        60,
        gpu_execution_contract="independent_commands",
        gpu_release_forecast=advice,
        gpu_timing_required=True,
    )
    uncalibrated = EnvUnitLedger([unit], NodeRules.build("any"))
    missing = replace(demand, gpu_release_forecast=None)
    held = uncalibrated.try_admit("missing-history", missing, "node")
    assert held.lease.gpu_reservation == "episode" and held.lease.gpu_active, (
        "Cost-aware admission must retain capacity when calibration is missing or invalidated."
    )
    assert uncalibrated.try_admit("other", missing, "node").status == "wait", (
        "Abstention must keep the GPU charged in the ledger."
    )
    a = ledger.try_admit("a", demand, "node")
    assert a.lease.gpu_reservation == "episode" and a.lease.gpu_active, "Unprofitable return keeps its GPU charge."
    assert ledger.try_admit("b", demand, "node").status == "wait", "Cost-selected episodes must respect capacity."
    optimistic = replace(advice, release_recommended=True)
    assert (
        choose_reservation("auto", "independent_commands", True, optimistic, require_timing=True).mode == "command"
    ), "A calibrated fitting forecast may enable a capable command session."
    assert choose_reservation("auto", "resident_state", True, optimistic).mode == "episode", (
        "A forecast cannot provide missing state recovery capability."
    )
    assert choose_reservation("command", "independent_commands", True, advice).mode == "command", (
        "Explicit policy must retain its comparison semantics."
    )


def test_host_task_contract_survives_resource_budget_translation(tmp_path):
    (tmp_path / "task.toml").write_text(
        '[environment]\ngpus = 1\n[metadata.env_service]\ngpu_execution_contract = "independent_commands"\n'
    )
    demand = demand_from_task(str(tmp_path))
    assert demand.gpu_execution_contract == "independent_commands", "Task protocol must reach admission."
    assert demand.with_default_gpu_memory(40).gpu_execution_contract == demand.gpu_execution_contract, (
        "Memory defaults cannot discard capability metadata."
    )
