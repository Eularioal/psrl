#!/usr/bin/env python3
"""Model two independent rollouts with session or invocation GPU reservations."""

import json
from dataclasses import dataclass


@dataclass
class Rollout:
    name: str
    phase: int = 0
    ready_at: int = 0
    running_until: int | None = None
    holds_gpu: bool = False
    done: bool = False


def simulate(capacity_gib: int, scope: str, *, persistent: bool = False) -> dict:
    """Use logical seconds, without CUDA, containers, real waits, or model requests."""
    if persistent and scope == "invocation":
        raise ValueError("Persistent GPU state needs a session reservation in this model.")
    if scope not in ("session", "invocation") or capacity_gib < 6:
        raise ValueError("Expected a supported scope and room for one 6 GiB request.")
    rollouts = [Rollout("A"), Rollout("B")]
    events = []
    for now in range(101):
        # G4/G5: Complete work before admitting another owner at this logical instant.
        for rollout in rollouts:
            if rollout.running_until != now:
                continue
            rollout.running_until = None
            rollout.phase += 1
            if rollout.phase == 2:
                rollout.done = True
                rollout.holds_gpu = False
                events.append(f"t={now}: {rollout.name} completes and releases")
            else:
                rollout.ready_at = now + 10
                if scope == "invocation":
                    rollout.holds_gpu = False
                events.append(f"t={now}: {rollout.name} starts 10s CPU/model gap")
        if all(rollout.done for rollout in rollouts):
            assert not any(rollout.holds_gpu for rollout in rollouts)
            return {
                "kind": "teaching_simulation_not_gpu_measurement",
                "capacity_gib": capacity_gib,
                "scope": scope,
                "persistent": persistent,
                "makespan_s": now,
                "events": events,
            }
        # G2/G3: Each independent environment requests 6 GiB before its 5s GPU phase.
        for rollout in rollouts:
            if rollout.done or rollout.running_until is not None or rollout.ready_at > now:
                continue
            reserved = sum(6 for owner in rollouts if owner.holds_gpu)
            if not rollout.holds_gpu and reserved + 6 > capacity_gib:
                continue
            rollout.holds_gpu = True
            rollout.running_until = now + 5
            events.append(f"t={now}: {rollout.name} GPU phase {rollout.phase + 1} starts")
        assert sum(6 for owner in rollouts if owner.holds_gpu) <= capacity_gib
    raise RuntimeError("The bounded teaching simulation did not finish.")


def main() -> None:
    for capacity, scope, expected in (
        (8, "session", 40),
        (8, "invocation", 25),
        (16, "session", 20),
        (16, "invocation", 20),
    ):
        result = simulate(capacity, scope)
        assert result["makespan_s"] == expected
        print(json.dumps(result))
    persistent = simulate(8, "session", persistent=True)
    assert persistent["makespan_s"] == 40
    print(json.dumps(persistent))
    try:
        simulate(8, "invocation", persistent=True)
    except ValueError as exc:
        print(json.dumps({"unsupported_contract_rejected": str(exc)}))
    else:
        raise AssertionError("Persistent state was incorrectly treated as released.")


if __name__ == "__main__":
    main()
