"""CPU-only mechanism check: real file locks and production ledger, no CUDA/Docker."""

import asyncio
import dataclasses
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from psrl.utils.concurrency.gpu_device_gate import GpuDeviceGate
from psrl.utils.concurrency.slot import SlotManager
from psrl.workers.env_service import EnvDemand, EnvUnit, EnvUnitLedger, NodeRules


async def main():
    # Diagram 2/3a: two output pools, the same two logical GPU names.
    with tempfile.TemporaryDirectory(prefix="cross-pool-demo-") as scratch:
        pools = [str(Path(scratch) / f"pool-{i}") for i in range(2)]
        allocated = []
        try:
            with patch.dict(os.environ, {"PSRL_SCIACCEL_GPU_DEVICES": "/dev/nvidia900,/dev/nvidia901"}):
                allocated = await asyncio.wait_for(
                    asyncio.gather(*[GpuDeviceGate(pool).acquire() for pool in pools for _ in range(2)]), timeout=3
                )
            occupancy = dict(Counter(a.device for a in allocated))
            assert sorted(occupancy.values()) == [2, 2]
        finally:
            for a in allocated:
                GpuDeviceGate.release(a)
            for pool in pools:
                shutil.rmtree(SlotManager.slot_lock_dir(pool, "sciaccel_gpu_gate"), ignore_errors=True)
    # Diagram 3b: all four requests go through one ledger, each declares 51 GiB.
    units = [EnvUnit(f"gpu-{i}", "192.0.2.10", 16, 32768, (i,), 4, 80 * 1024) for i in (900, 901)]
    ledger = EnvUnitLedger(units, NodeRules.build("any"))
    demand = EnvDemand(4, 8192, True, 51 * 1024)
    decisions = [ledger.try_admit(f"session-{i}", demand) for i in range(4)]
    assert [x.status for x in decisions] == ["granted", "granted", "wait", "wait"]
    snapshot = [dataclasses.asdict(s) for s in ledger.snapshot()]
    # Diagram 5: returning reservations allows the pending work to proceed.
    for item in decisions[:2]:
        ledger.release(item.lease.lease_id)
    retried = [ledger.try_admit(f"session-{i}", demand) for i in (2, 3)]
    assert all(x.status == "granted" for x in retried)
    for item in retried:
        ledger.release(item.lease.lease_id)
    assert all(s.containers_running == 0 for s in ledger.snapshot())
    print(
        json.dumps(
            {
                "scope": "CPU mechanism check; logical devices only; no CUDA OOM simulated as a real result",
                "off_duplicate_device_owners": occupancy,
                "on_admission": [d.status for d in decisions],
                "on_snapshot": snapshot,
                "on_retry_after_release": [d.status for d in retried],
                "final_leases": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
