"""Use the production CPU ledger to explain 14 GiB tasks and a 13 GiB remainder."""

import dataclasses
import json

from psrl.workers.env_service import EnvDemand, EnvUnit, EnvUnitLedger, NodeRules


def main():
    # Diagram 3: One physical card is represented by one ordinary shared unit.
    unit = EnvUnit("example", "192.0.2.10", 4, 4096, (0,), 4, 55 * 1024)
    ledger = EnvUnitLedger([unit], NodeRules.build("any"))
    demand = EnvDemand(cpu=1, memory_mb=1024, needs_gpu=True, gpu_memory_mb=14 * 1024)
    # Diagram 2 -> 3: Admission adds exact demands rather than rounding to a unit.
    admissions = [ledger.try_admit(f"large-{i}", demand) for i in range(4)]
    assert [item.status for item in admissions] == ["granted", "granted", "granted", "wait"]
    before = dataclasses.asdict(ledger.snapshot()[0])
    small = ledger.try_admit("small", dataclasses.replace(demand, gpu_memory_mb=10 * 1024))
    assert small.status == "granted"
    after = dataclasses.asdict(ledger.snapshot()[0])
    assert after["declared_gpu_memory_mb"] == 52 * 1024
    # Diagram 5: A released reservation restores budget for the waiting task.
    ledger.release(admissions[0].lease.lease_id)
    ledger.release(small.lease.lease_id)
    retry = ledger.try_admit("large-3", demand)
    assert retry.status == "granted"
    for session in ("large-1", "large-2", "large-3"):
        ledger.release_session(session)
    assert ledger.snapshot()[0].containers_running == 0
    print(
        json.dumps(
            {
                "scope": "Production CPU ledger with teaching values, no GPU or Docker",
                "large_admissions": [item.status for item in admissions],
                "after_three_large": before,
                "smaller_task_admission": small.status,
                "after_smaller_task": after,
                "large_retry_after_release": retry.status,
                "final_active_leases": ledger.snapshot()[0].containers_running,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
