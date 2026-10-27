"""CPU illustration using the real scheduler and a simulated pre-allocation snapshot.

Run from the repository root with PYTHONPATH=. . No Ray, Docker or CUDA is started.
The simulated node keeps external memory at 70 GiB while grants are acknowledged.
New task allocations have not started. This models the observed admission window,
not a GPU benchmark or a guarantee that every real run follows this interleaving.
"""

import asyncio
import json
from types import SimpleNamespace

from psrl.workers.env_service.command_gpu import GpuCommandScheduler
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.types import EnvDemand, EnvUnit


async def main():
    # Diagram 4b: host leases exist before their GPU commands start.
    unit = EnvUnit("demo", "node", 16, 32768, (0,), 4, 90 * 1024, gpu_reservation="command")
    ledger = EnvUnitLedger([unit], NodeRules.build(policy="any"))
    demand = EnvDemand(4, 8192, True, 15 * 1024)
    leases = [ledger.try_admit(str(i), demand, "node").lease for i in range(3)]

    async def snapshot(_indices):
        # Diagram 2: external allocation remains while new tasks have not allocated yet.
        return {
            "devices": {0: {"used_mb": 70 * 1024, "total_mb": 97871}},
            "owned_mb": {},
            "attribution_errors": {},
        }

    async def grant(*_args):
        # Diagram 4b: permission granted does not imply CUDA allocation completed.
        return None

    async def get_node(_ip):
        return SimpleNamespace(
            gpu_command_memory_snapshot=SimpleNamespace(remote=lambda indices, commands: snapshot(indices)),
            grant_gpu_command=SimpleNamespace(remote=grant),
        )

    scheduler = GpuCommandScheduler(ledger, get_node)
    rows = []
    for i, lease in enumerate(leases):
        result = await scheduler.acquire(lease.lease_id, f"container-{i}", f"request-{i}")
        rows.append(
            {
                "request": i + 1,
                "status": result["status"],
                "declared_gib": ledger._declared_gpu_mb(unit) / 1024,
                "effective_gib": ledger._effective_gpu_mb(unit, ledger._clock()) / 1024,
                "external_plus_granted_task_allocations_gib": 70
                + 14 * (sum(row["status"] == "granted" for row in rows) + (result["status"] == "granted")),
            }
        )
    print(json.dumps({"scope": "CPU mechanism illustration; telemetry simulated", "requests": rows}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
