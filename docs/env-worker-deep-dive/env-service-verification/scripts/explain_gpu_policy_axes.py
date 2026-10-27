#!/usr/bin/env python3
"""Exercise production configuration rules on teaching inputs without Docker or CUDA."""

import json

from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.types import EnvDemand, EnvUnit, validate_units


def main():
    model_node = "192.168.1.10"
    demand = EnvDemand(1, 1024, True, 1024)
    # P1/P2: A granted lease inherits its configured unit's mode.
    for mode in ("episode", "command"):
        unit = EnvUnit("env", "192.168.1.11", 4, 8192, (0,), 2, 8192, gpu_reservation=mode)
        ledger = EnvUnitLedger([unit], NodeRules.build(policy="any"), lease_ttl_s=60)
        lease = ledger.try_admit("rollout", demand).lease
        assert lease.gpu_reservation == mode
        assert lease.gpu_active == (mode == "episode")
        print(json.dumps({"scope": "CPU configuration check", "unit_mode": mode, "lease_mode": lease.gpu_reservation}))
    # P3: Node eligibility does not inspect which card the model owns on that node.
    units = [
        EnvUnit("same-card", model_node, 4, 8192, (0,), 1, 8192),
        EnvUnit("different-card", model_node, 4, 8192, (7,), 1, 8192),
        EnvUnit("other-node", "192.168.1.11", 4, 8192, (0,), 1, 8192),
        EnvUnit("bound-lending", model_node, 4, 8192, (0,), 1, 8192, rollout_replica_rank=0),
    ]
    for policy, expected in (("cpu_only", [False, False, True, True]), ("any", [True] * 4)):
        rules = NodeRules.build(policy=policy, train_rollout_node_ips=[model_node])
        allowed = [rules.tier(unit, demand) is not None for unit in units]
        assert allowed == expected
        print(json.dumps({"node_policy": policy, "eligible": dict(zip((u.unit_id for u in units), allowed))}))
    # P4: Environment units may not independently claim the same physical device.
    try:
        validate_units([units[0], units[3]])
    except ValueError as exc:
        print(json.dumps({"overlapping_environment_units_rejected": str(exc)}))
    else:
        raise AssertionError("Overlapping environment units were accepted.")


if __name__ == "__main__":
    main()
