"""CPU-only audit of a real OFF overlap and the ON reservation arithmetic.

Reads existing experiment records; starts no training, Docker, Ray, or CUDA.
The arithmetic is an explanation, not an execution of the production ledger.
"""

import argparse
import json
from pathlib import Path


def explain(root):
    records = [json.loads(p.read_text()) for p in (root / "off").glob("pool-*/attempts/*.json")]
    a = next(r for r in records if r["uid"] == "1" and r["attempt"] == 1)
    b = next(r for r in records if r["uid"] == "4" and r["attempt"] == 1)
    assert a["accepted"] and b["accepted"]
    assert a["pressure"]["gpu_uuid_hex"] == b["pressure"]["gpu_uuid_hex"]
    # Diagram 3: the CUDA process freed memory before its enclosing episode ended.
    assert (
        a["pressure"]["finished_at"] < b["pressure"]["allocated_at"] < b["pressure"]["finished_at"] < a["finished_at"]
    )
    origin = a["pressure"]["allocated_at"]
    events = [
        ("A(uid=1) CUDA allocated", origin),
        ("A CUDA freed; episode continues", a["pressure"]["finished_at"]),
        ("B(uid=4) CUDA allocated on same GPU", b["pressure"]["allocated_at"]),
        ("B CUDA freed; episode continues", b["pressure"]["finished_at"]),
        ("A episode finished", a["finished_at"]),
        ("B episode finished", b["finished_at"]),
    ]
    # Diagram 1/2: no capacity-aware admission in the historical static path.
    collisions = [
        r
        for r in records
        if r["pressure"].get("status") == "cuda_oom"
        and origin <= r["pressure"]["finished_at"] <= a["pressure"]["finished_at"]
    ]
    # Diagram 4/5: arithmetic corresponding to _effective_gpu_mb / _fits_now.
    # These GiB numbers are the experiment configuration, not allocated memory.
    budget, declared, measured, incoming = 80, 51, 0, 51
    effective = max(declared, measured)
    idle = json.loads((root / "idle-reservation-example.json").read_text())
    snapshot = idle["snapshot"]
    return {
        "scope": "Existing GPU records + explanatory arithmetic; no new GPU execution",
        "off_a_record": "off/pool-0/attempts/1-01.json",
        "off_b_record": "off/pool-1/attempts/4-01.json",
        "same_gpu_uuid_hex": a["pressure"]["gpu_uuid_hex"],
        "off_timeline_seconds_relative_to_a_allocation": [
            {"event": label, "seconds": round(t - origin, 3)} for label, t in events
        ],
        "off_oom_during_a_cuda": [{"uid": r["uid"], "attempt": r["attempt"]} for r in collisions],
        "on_arithmetic_gib": {
            "budget": budget,
            "declared": declared,
            "physical_measured": measured,
            "incoming": incoming,
            "effective_plus_incoming": effective + incoming,
            "fits": effective + incoming <= budget,
        },
        "on_real_idle_snapshot_time": snapshot["time"],
        "on_real_unit_declared_mib": [u["declared_gpu_memory_mb"] for u in snapshot["units"]],
        "on_real_node_waiters": max(u["sessions_waiting"] for u in snapshot["units"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results/science-fixed-work-20261019",
    )
    print(json.dumps(explain(parser.parse_args().root), indent=2))
