"""Demonstrate configured capacity and episode accounting with simulated GPU telemetry."""

import json

from psrl.workers.env_service.gpu_memory import accounted_gpu_memory_mb
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.types import GRANTED, WAIT, EnvDemand, EnvUnit, validate_units

GIB = 1024


def demonstrate(capacity_gib, *, trusted_ownership=False):
    # Step 1 declares one whole device and its scheduling budget.
    unit = EnvUnit("demo", "demo-node", 16, 65536, (0,), 8, capacity_gib * GIB)
    ledger = EnvUnitLedger([unit], NodeRules.build("any"), clock=lambda: 0)
    demand = EnvDemand(cpu=1, memory_mb=1024, needs_gpu=True, gpu_memory_mb=20 * GIB)
    active = []
    rows = []
    for index in range(4):
        # Steps 2 and 3 simulate each admitted task allocating 14 GiB before the next request.
        measured = len(active) * 14 * GIB
        owned = {lease.lease_id: {0: 14 * GIB} for lease in active} if trusted_ownership else None
        ledger.report_measurement("demo-node", {0: measured}, owned_mb_by_device=owned)
        admission = ledger.try_admit(f"task-{index}", demand, "demo-node")
        rows.append({"request": index + 1, "measured_gib": measured / GIB, "status": admission.status})
        if admission.lease:
            active.append(admission.lease)
    # Step 4 simulates acknowledged cleanup, then checks that a waiter can enter.
    ledger.report_measurement("demo-node", {0: 0})
    for lease in active:
        ledger.release(lease.lease_id)
    resumed = ledger.try_admit("after-cleanup", demand, "demo-node")
    assert resumed.status == GRANTED, "Capacity must become available after acknowledged cleanup."
    ledger.release(resumed.lease.lease_id)
    return rows


def main():
    small, whole = demonstrate(24), demonstrate(96)
    credited = demonstrate(96, trusted_ownership=True)
    assert all(r["status"] == GRANTED for r in credited), "Verified episode ownership must admit the fourth task."
    assert [r["status"] for r in small] == [GRANTED, WAIT, WAIT, WAIT], "The small quota must serialize these tasks."
    assert [r["status"] for r in whole] == [GRANTED, GRANTED, GRANTED, WAIT], (
        "Episode accounting must preserve reservations without ownership credits."
    )
    reservations = {"a": 20 * GIB, "b": 20 * GIB, "c": 20 * GIB}
    owned = dict.fromkeys(reservations, 14 * GIB)
    conservative = accounted_gpu_memory_mb(42 * GIB, reservations) / GIB
    attributed = accounted_gpu_memory_mb(42 * GIB, reservations, owned) / GIB
    assert conservative == 102 and attributed == 60, "Current ownership must avoid double counting realized promises."
    try:
        validate_units(
            [
                EnvUnit("first", "demo-node", 1, 1024, (0,), 1, 24 * GIB),
                EnvUnit("second", "demo-node", 1, 1024, (0,), 1, 72 * GIB),
            ]
        )
    except ValueError:
        duplicate_device_rejected = True
    else:
        raise AssertionError("Two units must not claim the same device.")
    print(
        json.dumps(
            {
                "scope": "CPU teaching example using real ledger code and simulated memory, no GPU benchmark",
                "quota_24_gib": small,
                "capacity_96_gib_without_ownership": whole,
                "capacity_96_gib_with_ownership": credited,
                "three_active_effective_gib_without_ownership": conservative,
                "three_active_effective_gib_with_ownership": attributed,
                "duplicate_device_rejected": duplicate_device_rejected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
