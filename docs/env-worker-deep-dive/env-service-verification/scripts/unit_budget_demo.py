"""Exercise production unit admission using teaching budgets, without GPU or Docker."""

import dataclasses
import json

from psrl.workers.env_service import EnvDemand, EnvUnitLedger, NodeRules
from psrl.workers.env_service.config_check import validate_env_service_config


def main():
    # Step 1: Parse and validate a static resource pool with one ordinary GPU unit.
    config = {
        "node_policy": "any",
        "default_gpu_memory_mb": 20480,
        "units": [
            {
                "unit_id": "example-gpu0",
                "node_ip": "192.0.2.10",
                "gpu_indices": [0],
                "cpu_quota": 8,
                "memory_mb": 65536,
                "gpu_memory_mb": 61440,
                "max_containers": 4,
            }
        ],
    }
    ledger = EnvUnitLedger(
        validate_env_service_config(config),
        NodeRules.build("any"),
        default_gpu_memory_mb=config["default_gpu_memory_mb"],
    )
    # Steps 2 and 3: Three demands fit, but the fourth exceeds the GPU budget.
    demand = EnvDemand(cpu=2, memory_mb=16384, needs_gpu=True, gpu_memory_mb=20480)
    admissions = [ledger.try_admit(f"task-{i}", demand) for i in range(4)]
    assert [item.status for item in admissions] == ["granted", "granted", "granted", "wait"]
    snapshot = dataclasses.asdict(ledger.snapshot()[0])
    oversized = ledger.try_admit("oversized", EnvDemand(needs_gpu=True, gpu_memory_mb=81920))
    assert oversized.status == "reject"
    # Step 5: Releasing one lease makes room without changing the unit definition.
    ledger.release(admissions[0].lease.lease_id)
    retry = ledger.try_admit("task-3", demand)
    assert retry.status == "granted"
    for session in ("task-1", "task-2", "task-3"):
        ledger.release_session(session)
    assert ledger.snapshot()[0].containers_running == 0
    print(
        json.dumps(
            {
                "scope": "Teaching resource values, production CPU ledger, no physical GPU experiment",
                "config": config,
                "task_demand": dataclasses.asdict(demand),
                "admissions": [item.status for item in admissions],
                "after_three": snapshot,
                "oversized": oversized.status,
                "retry_after_release": retry.status,
                "final_active_leases": ledger.snapshot()[0].containers_running,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
