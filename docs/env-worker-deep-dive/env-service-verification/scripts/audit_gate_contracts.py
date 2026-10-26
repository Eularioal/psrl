"""Exercise real gate logic with teaching telemetry on CPU, without Docker or a GPU."""

import asyncio
import json
import time
from unittest.mock import patch

from psrl.workers.env_service import EnvDemand, EnvUnit, EnvUnitLedger, NodeRules
from psrl.workers.gen import session_router


async def gate_b_cases():
    calls = 0

    async def idle():
        nonlocal calls
        calls += 1
        return {0: (0.0, 20000.0)}

    async def busy():
        nonlocal calls
        calls += 1
        return {0: (100.0, 20000.0)}

    with (
        patch.object(session_router, "_GATE_B_MAX_WAIT_SEC", 0.05),
        patch.object(session_router, "_GATE_B_POLL_INTERVAL_SEC", 0.01),
        patch.object(session_router, "_read_gpu_utilization", idle),
    ):
        # Diagram B: Two replies observe an idle card, without reserving its next turn.
        passed = []

        async def reply(sid):
            await session_router._wait_for_gpu_idle("/dev/nvidia0", sid, shared=True)
            passed.append(sid)

        await asyncio.gather(reply("A"), reply("B"))
        assert sorted(passed) == ["A", "B"], "An idle observation is not an exclusive token."
        concurrent = {"replies_released": list(passed), "telemetry_calls": calls}

        # Diagram B: Idle resident memory does not spend the shared-card wait budget.
        calls = 0
        await reply("resident-shared")
        shared_polls = calls
        calls = 0
        await session_router._wait_for_gpu_idle("/dev/nvidia0", "resident-unshared", shared=False)
        unshared_polls = calls
        assert shared_polls == 1 and unshared_polls > 1

        # Diagram B: The original gate deliberately releases a still-busy device at its cap.
        with patch.object(session_router, "_read_gpu_utilization", busy):
            calls = 0
            started = time.monotonic()
            await reply("busy-at-timeout")
            elapsed = time.monotonic() - started
            assert calls > 1 and elapsed >= 0.05
    return {
        "simultaneous_idle_observers": concurrent,
        "resident_memory_polls": {"shared": shared_polls, "unshared": unshared_polls},
        "busy_timeout": {"reply_released": True, "last_util_pct": 100, "elapsed_s": elapsed},
    }


def gate_a_cases():
    gib = 1024
    rules = NodeRules.build("any")

    def demand(size):
        return EnvDemand(cpu=1, memory_mb=1024, needs_gpu=True, gpu_memory_mb=size * gib)

    # Diagram A: Registered demand exceeds a dedicated device budget and must wait.
    unit = EnvUnit("single", "192.0.2.10", 4, 4096, (0,), 4, 80 * gib)
    ledger = EnvUnitLedger([unit], rules)
    a = ledger.try_admit("A", demand(50), unit.node_ip)
    b = ledger.try_admit("B", demand(50), unit.node_ip)
    assert a.status == "granted" and b.status == "wait"
    ledger.release(a.lease.lease_id)
    assert ledger.try_admit("B", demand(50), unit.node_ip).status == "granted"

    # Diagram A: Total telemetry cannot separate an owner's unused reservation from foreign use.
    ledger = EnvUnitLedger([unit], rules)
    ledger.try_admit("A", demand(50), unit.node_ip)
    ledger.report_measurement(unit.node_ip, {0: 40 * gib})
    foreign_case = ledger.try_admit("B", demand(30), unit.node_ip)
    assert foreign_case.status == "granted"
    return {
        "known_demands": {"first": a.status, "second": b.status, "after_release": "granted"},
        "foreign_usage_teaching_counterexample": {
            "capacity_gib": 80,
            "A_declared_gib": 50,
            "A_actual_now_gib": 20,
            "unregistered_actual_gib": 20,
            "measured_total_gib": 40,
            "B_declared_gib": 30,
            "B_admission": foreign_case.status,
            "potential_total_if_registered_demands_reach_peak_gib": 100,
            "scope": "Admission arithmetic only. No real allocation or OOM was attempted.",
        },
    }


async def main():
    result = {
        "scope": "Production functions with teaching telemetry and demands. CPU only, not GPU performance.",
        "gate_a": gate_a_cases(),
        "gate_b": await gate_b_cases(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
