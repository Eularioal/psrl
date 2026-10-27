#!/usr/bin/env python3
"""
Demo of the env service's two headline behaviours, on the real `EnvUnitLedger`.

1. One card, several containers. `gate_a_after.py` hands each GPU episode a whole card, so
   N cards serve N episodes. With units, containers share a card until its device memory
   (or the unit's container ceiling) is full, so the same cards serve more episodes.

2. Train and rollout nodes. A container lands on the node of the worker that starts it, so
   whether containers may share a node with the vLLM engines and the trainer is a policy:
   `any`, `cpu_only`, `avoid` or `prefer_other`.

No GPU, Ray or Docker needed: the ledger is a pure state machine, and "devices" are labels.

Usage:
    PYTHONPATH=. python3 examples/sciaccel_rl/diagnostics/env_service_demo.py
    PYTHONPATH=. python3 examples/sciaccel_rl/diagnostics/env_service_demo.py --episodes 12 --card-gb 80 --task-gb 24

Exit code: 0 when every printed expectation held, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

from psrl.workers.env_service import (
    GRANTED,
    REJECT,
    WAIT,
    EnvDemand,
    EnvUnit,
    EnvUnitLedger,
    NodeRules,
)

ENV_NODE = "10.0.0.1"
TRAIN_NODE = "10.0.0.9"


def one_card_several_containers(episodes: int, card_gb: int, task_gb: int, cards: int) -> bool:
    print("=" * 78)
    print(f"1. {cards} card(s) of {card_gb} GB, each GPU task needs {task_gb} GB, {episodes} episodes arrive at once")
    print("=" * 78)
    fits_per_card = card_gb // task_gb

    # Old behaviour (GpuDeviceGate): one episode per card.
    exclusive = min(episodes, cards)

    # New behaviour: one unit per card, containers share it until memory is full.
    units = [
        EnvUnit(f"card{i}", ENV_NODE, 32.0, 262144, (i,), max_containers=fits_per_card, gpu_memory_mb=card_gb * 1024)
        for i in range(cards)
    ]
    ledger = EnvUnitLedger(units, NodeRules.build("any"), default_gpu_memory_mb=task_gb * 1024)
    demand = EnvDemand(needs_gpu=True)
    outcomes = [ledger.try_admit(f"episode-{i}", demand, ENV_NODE) for i in range(episodes)]
    running = [o for o in outcomes if o.status == GRANTED]
    waiting = [o for o in outcomes if o.status == WAIT]

    print(f"  exclusive cards (GpuDeviceGate): {exclusive} episodes run, {episodes - exclusive} wait")
    print(f"  shared units (env service):      {len(running)} episodes run, {len(waiting)} wait")
    for snap in ledger.snapshot():
        bar = "#" * snap.leases_active + "." * (snap.max_containers - snap.leases_active)
        print(
            f"    {snap.unit_id}: [{bar}] {snap.leases_active}/{snap.max_containers} containers, "
            f"{snap.declared_gpu_memory_mb // 1024}/{snap.gpu_memory_mb // 1024} GB declared"
        )

    expected = min(episodes, cards * fits_per_card)
    ok = len(running) == expected and len(running) >= exclusive
    print(f"  expected {expected} running ({cards} cards x {fits_per_card} per card): {'OK' if ok else 'MISMATCH'}")

    # A card that a neighbour has filled in ways the ledger did not grant must also refuse.
    crowded = EnvUnitLedger(units[:1], NodeRules.build("any"), default_gpu_memory_mb=task_gb * 1024)
    blocked = crowded.try_admit("late", demand, ENV_NODE, measured_gpu_used_mb={0: (card_gb - 4) * 1024.0})
    measured_ok = blocked.status == WAIT
    print(
        f"  a card whose nvidia-smi shows {card_gb - 4} GB already in use (nobody declared it): "
        f"{blocked.status.upper()} {'OK' if measured_ok else 'MISMATCH'}"
    )
    print()
    return ok and measured_ok


def node_policies() -> bool:
    print("=" * 78)
    print(f"2. Node policy. {TRAIN_NODE} hosts the trainer and vLLM. {ENV_NODE} is an env node")
    print("=" * 78)
    units = [
        EnvUnit("train-cpu", TRAIN_NODE, 32.0, 131072, (), 4, 0),
        EnvUnit("train-gpu", TRAIN_NODE, 16.0, 65536, (0,), 2, 80 * 1024),
        EnvUnit("env-cpu", ENV_NODE, 32.0, 131072, (), 1, 0),
        EnvUnit("env-gpu", ENV_NODE, 16.0, 65536, (1,), 1, 80 * 1024),
    ]
    cases = {
        # policy: (cpu episode lands on, gpu episode lands on)
        "any": ("env-cpu", "env-gpu"),
        "cpu_only": ("env-cpu", "env-gpu"),
        "avoid": ("env-cpu", "env-gpu"),
        "prefer_other": ("env-cpu", "env-gpu"),
    }
    all_ok = True
    print(f"  {'policy':<14}{'CPU episode, hint=train node':<34}{'GPU episode, hint=train node'}")
    for policy in cases:
        ledger = EnvUnitLedger(units, NodeRules.build(policy, train_rollout_node_ips=[TRAIN_NODE]))
        cpu = ledger.try_admit("cpu", EnvDemand(), TRAIN_NODE)
        gpu = ledger.try_admit("gpu", EnvDemand(needs_gpu=True), TRAIN_NODE)

        def show(result):
            if result.status == GRANTED:
                return f"granted -> {result.lease.unit_id}"
            return "REJECT" if result.status == REJECT else "WAIT"

        print(f"  {policy:<14}{show(cpu):<34}{show(gpu)}")
        # Under these policies a worker on the train node may host CPU episodes only (cpu_only)
        # or nothing at all (avoid). `any` and `prefer_other` allow both.
        expected_cpu = {"any": GRANTED, "cpu_only": GRANTED, "avoid": REJECT, "prefer_other": GRANTED}[policy]
        expected_gpu = {"any": GRANTED, "cpu_only": REJECT, "avoid": REJECT, "prefer_other": GRANTED}[policy]
        all_ok &= cpu.status == expected_cpu and gpu.status == expected_gpu
    print("  (the hint node is where the agent worker runs, so these are the answers a worker there gets)")

    # With no hint the router weighs every node. Tier 0 (non-train nodes) always beats tier 1
    # (train/rollout nodes), even when the tier-0 unit is a GPU unit hosting a CPU episode.
    rules = NodeRules.build("prefer_other", train_rollout_node_ips=[TRAIN_NODE])
    ledger = EnvUnitLedger(units, rules)
    first = ledger.try_admit("a", EnvDemand())
    second = ledger.try_admit("b", EnvDemand())
    print(f"  prefer_other, no hint: 1st CPU episode -> {first.lease.unit_id}, 2nd -> {second.lease.unit_id}")
    print("    (env-cpu is full after one container, but env-gpu is still on a non-train node, so tier 0 wins)")
    tier_ok = (first.lease.unit_id, second.lease.unit_id) == ("env-cpu", "env-gpu")

    # Spilling onto a train node only happens once the env node has no room at all.
    only_env_cpu = [u for u in units if u.unit_id in ("env-cpu", "train-cpu")]
    ledger = EnvUnitLedger(only_env_cpu, rules)
    first = ledger.try_admit("a", EnvDemand())
    second = ledger.try_admit("b", EnvDemand())
    print(f"  prefer_other, env node has one slot: 1st -> {first.lease.unit_id}, 2nd -> {second.lease.unit_id}")
    print("    (the env node is full, so the 2nd episode spills to a train-node unit instead of waiting)")
    spill_ok = (first.lease.unit_id, second.lease.unit_id) == ("env-cpu", "train-cpu")
    print(f"  {'OK' if tier_ok and spill_ok else 'MISMATCH'}\n")
    return all_ok and tier_ok and spill_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=10, help="Concurrent GPU episodes in part 1.")
    parser.add_argument("--cards", type=int, default=2, help="GPU cards in part 1.")
    parser.add_argument("--card-gb", type=int, default=80, help="Device memory per card, in GB.")
    parser.add_argument("--task-gb", type=int, default=24, help="Device memory one GPU task is charged, in GB.")
    args = parser.parse_args()

    results = [
        one_card_several_containers(args.episodes, args.card_gb, args.task_gb, args.cards),
        node_policies(),
    ]
    print("RESULT:", "all expectations held" if all(results) else "SOME EXPECTATIONS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
