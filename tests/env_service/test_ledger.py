"""
Tests for `EnvUnitLedger`: one card hosting several containers, resource-keyed admission,
node policy, measured occupancy and TTL reclaim.

Plain asserts and no fixtures, so these run under pytest and as `python test_ledger.py`.
Numbers in comments are worked out by hand so a failure is easy to read.
"""

from psrl.workers.env_service import (
    GRANTED,
    REJECT,
    WAIT,
    EnvDemand,
    EnvUnit,
    EnvUnitLedger,
    NodeRules,
    validate_units,
)

NODE_A = "10.0.0.1"
NODE_B = "10.0.0.2"
TRAIN = "10.0.0.9"

GPU_DEMAND = EnvDemand(cpu=1.0, needs_gpu=True)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _ledger(units, rules=None, clock=None, **kwargs):
    return EnvUnitLedger(
        units,
        rules or NodeRules.build("any"),
        clock=clock or FakeClock(),
        id_factory=lambda: "x",
        **kwargs,
    )


def _gpu_unit(unit_id="g0", node=NODE_A, indices=(0,), max_containers=10, gpu_mb=80000):
    return EnvUnit(unit_id, node, 16.0, 65536, indices, max_containers, gpu_mb)


def _cpu_unit(unit_id="c0", node=NODE_A, cpu=16.0, mem=65536, max_containers=8):
    return EnvUnit(unit_id, node, cpu, mem, (), max_containers, 0)


# --------------------------------------------------------------------------- one card, several containers


def test_one_card_hosts_several_containers_until_device_memory_is_full():
    # 80000 MB card, 24000 MB per task: 3 fit (72000), the 4th would need 96000.
    ledger = _ledger([_gpu_unit()], default_gpu_memory_mb=24000)
    results = [ledger.try_admit(f"s{i}", GPU_DEMAND, NODE_A) for i in range(4)]
    assert [r.status for r in results] == [GRANTED, GRANTED, GRANTED, WAIT]
    # All three share the same card, which is the point of the unit model.
    assert {r.lease.gpu_indices for r in results[:3]} == {(0,)}


def test_container_ceiling_binds_before_device_memory():
    # Memory would allow 3, but max_containers=2 is the tighter limit.
    ledger = _ledger([_gpu_unit(max_containers=2)], default_gpu_memory_mb=24000)
    statuses = [ledger.try_admit(f"s{i}", GPU_DEMAND, NODE_A).status for i in range(3)]
    assert statuses == [GRANTED, GRANTED, WAIT]


def test_a_released_slot_admits_the_next_waiter():
    ledger = _ledger([_gpu_unit(max_containers=1)])
    first = ledger.try_admit("s0", GPU_DEMAND, NODE_A)
    assert ledger.try_admit("s1", GPU_DEMAND, NODE_A).status == WAIT
    assert ledger.release(first.lease.lease_id) is True
    assert ledger.try_admit("s1", GPU_DEMAND, NODE_A).status == GRANTED


def test_cpu_episode_on_a_gpu_unit_does_not_charge_device_memory():
    ledger = _ledger([_gpu_unit(max_containers=10)], default_gpu_memory_mb=80000)
    # One GPU task takes the whole card, yet CPU-only episodes still fit on the same unit.
    assert ledger.try_admit("gpu", GPU_DEMAND, NODE_A).status == GRANTED
    assert ledger.try_admit("cpu", EnvDemand(cpu=1.0), NODE_A).status == GRANTED
    assert ledger.try_admit("gpu2", GPU_DEMAND, NODE_A).status == WAIT


def test_cpu_episodes_do_not_drift_onto_a_gpu_unit_just_because_it_looks_emptier():
    # The CPU unit is half busy (load 3/4) while the GPU unit is empty (load 1/3). Picking on
    # load alone would put the CPU episode on the GPU unit, where it burns one of that unit's
    # three container slots and can leave a GPU episode waiting for a card that is free.
    cpu = _cpu_unit("c0", max_containers=4)
    gpu = _gpu_unit("g0", max_containers=3)
    ledger = _ledger([cpu, gpu])
    for i in range(2):
        assert ledger.try_admit(f"busy{i}", EnvDemand(), NODE_A).lease.unit_id == "c0"
    assert ledger.try_admit("next", EnvDemand(), NODE_A).lease.unit_id == "c0"
    # The GPU unit's slots are all still free for GPU episodes.
    snap = {s.unit_id: s for s in ledger.snapshot()}
    assert snap["g0"].containers_running == 0


def test_a_cpu_episode_still_falls_back_to_a_gpu_unit_when_no_cpu_unit_has_room():
    cpu = _cpu_unit("c0", max_containers=1)
    gpu = _gpu_unit("g0", max_containers=3)
    ledger = _ledger([cpu, gpu])
    assert ledger.try_admit("a", EnvDemand(), NODE_A).lease.unit_id == "c0"
    assert ledger.try_admit("b", EnvDemand(), NODE_A).lease.unit_id == "g0"  # last resort, not a refusal


def test_a_lease_is_marked_shared_only_when_its_unit_can_host_several_containers():
    solo = _ledger([_gpu_unit("solo", max_containers=1)]).try_admit("s", GPU_DEMAND, NODE_A).lease
    many = _ledger([_gpu_unit("many", max_containers=3)]).try_admit("s", GPU_DEMAND, NODE_A).lease
    cpu = _ledger([_cpu_unit()]).try_admit("s", EnvDemand(), NODE_A).lease
    assert (solo.shared, many.shared, cpu.shared) == (False, True, False)


def test_multi_device_unit_hands_every_container_all_of_its_devices():
    unit = _gpu_unit(indices=(2, 3), gpu_mb=160000)
    ledger = _ledger([unit], default_gpu_memory_mb=24000)
    lease = ledger.try_admit("s0", GPU_DEMAND, NODE_A).lease
    assert lease.gpu_indices == (2, 3)


# --------------------------------------------------------------------------- idempotence and arithmetic


def test_admitting_the_same_session_twice_returns_the_same_lease():
    ledger = _ledger([_gpu_unit(max_containers=1)])
    a = ledger.try_admit("s0", GPU_DEMAND, NODE_A)
    b = ledger.try_admit("s0", GPU_DEMAND, NODE_A)
    assert a.lease.lease_id == b.lease.lease_id
    assert len(ledger.snapshot()[0].sessions) == 1


def test_release_is_idempotent():
    ledger = _ledger([_cpu_unit()])
    lease = ledger.try_admit("s0", EnvDemand(), NODE_A).lease
    assert ledger.release(lease.lease_id) is True
    assert ledger.release(lease.lease_id) is False
    assert ledger.release_session("s0") is False


def test_selection_is_resource_weighted_not_a_container_count():
    # Unit A holds one heavy container (14 of 16 CPUs). Unit B holds three light ones.
    # A count-based balancer would pick A (1 < 3). The resource-weighted load after placing
    # one more CPU is A: max(2/8, 15/16)=0.9375, B: max(4/8, 4/16)=0.5, so B must win.
    a = _cpu_unit("a", cpu=16.0)
    b = _cpu_unit("b", cpu=16.0)
    ledger = _ledger([a, b])
    heavy = ledger.try_admit("heavy", EnvDemand(cpu=14.0), NODE_A)
    assert heavy.lease.unit_id == "a"  # both empty, tie broken by unit id
    # Once `a` is heavy, every light episode lands on `b` even as `b`'s container count grows.
    for i in range(3):
        assert ledger.try_admit(f"light{i}", EnvDemand(cpu=1.0), NODE_A).lease.unit_id == "b"
    assert ledger.try_admit("next", EnvDemand(cpu=1.0), NODE_A).lease.unit_id == "b"


def test_host_memory_is_summed_across_a_units_containers():
    ledger = _ledger([_cpu_unit(mem=10000)])
    assert ledger.try_admit("a", EnvDemand(memory_mb=6000), NODE_A).status == GRANTED
    assert ledger.try_admit("b", EnvDemand(memory_mb=6000), NODE_A).status == WAIT  # 12000 > 10000
    assert ledger.try_admit("c", EnvDemand(memory_mb=4000), NODE_A).status == GRANTED  # exactly 10000


def test_node_ip_restricts_admission_to_that_nodes_units():
    ledger = _ledger([_cpu_unit("a", node=NODE_A), _cpu_unit("b", node=NODE_B)])
    assert ledger.try_admit("s0", EnvDemand(), NODE_B).lease.unit_id == "b"
    assert ledger.try_admit("s1", EnvDemand(), NODE_A).lease.unit_id == "a"


# --------------------------------------------------------------------------- reject versus wait


def test_a_gpu_demand_with_no_gpu_unit_is_rejected_not_queued_forever():
    ledger = _ledger([_cpu_unit()])
    result = ledger.try_admit("s0", GPU_DEMAND, NODE_A)
    assert result.status == REJECT
    assert "GPU" in result.reason


def test_a_demand_larger_than_an_empty_unit_is_rejected():
    ledger = _ledger([_gpu_unit(gpu_mb=80000)])
    result = ledger.try_admit("s0", EnvDemand(needs_gpu=True, gpu_memory_mb=100000), NODE_A)
    assert result.status == REJECT
    assert "exceeds" in result.reason


def test_unknown_node_is_rejected():
    ledger = _ledger([_cpu_unit(node=NODE_A)])
    assert ledger.try_admit("s0", EnvDemand(), NODE_B).status == REJECT


# --------------------------------------------------------------------------- measured occupancy


def test_measured_memory_larger_than_declared_wins():
    # 80000 MB card. Nothing declared, but a neighbour is using 70000 MB. A 24000 MB task
    # would make 94000 > 80000, so it must wait even though the ledger holds no lease.
    ledger = _ledger([_gpu_unit()], default_gpu_memory_mb=24000)
    result = ledger.try_admit("s0", GPU_DEMAND, NODE_A, measured_gpu_used_mb={0: 70000.0})
    assert result.status == WAIT


def test_measured_memory_smaller_than_declared_does_not_free_capacity():
    # A loading container declared 24000 but has only touched 1000 so far. Believing the
    # 1000 would over-admit once it finishes loading, so the declared figure holds.
    ledger = _ledger([_gpu_unit(gpu_mb=48000)], default_gpu_memory_mb=24000)
    assert ledger.try_admit("a", GPU_DEMAND, NODE_A, {0: 1000.0}).status == GRANTED
    assert ledger.try_admit("b", GPU_DEMAND, NODE_A, {0: 1000.0}).status == GRANTED  # 48000 <= 48000
    assert ledger.try_admit("c", GPU_DEMAND, NODE_A, {0: 1000.0}).status == WAIT


def test_a_stale_measurement_is_ignored():
    clock = FakeClock()
    ledger = _ledger([_gpu_unit()], clock=clock, default_gpu_memory_mb=24000, measurement_ttl_s=60.0)
    assert ledger.try_admit("a", GPU_DEMAND, NODE_A, {0: 79000.0}).status == WAIT
    clock.now = 61.0  # the reading is now older than its TTL and must not block the unit
    assert ledger.try_admit("a", GPU_DEMAND, NODE_A).status == GRANTED


def test_a_measurement_missing_one_of_the_units_devices_is_ignored():
    ledger = _ledger([_gpu_unit(indices=(0, 1), gpu_mb=160000)], default_gpu_memory_mb=24000)
    # Only device 0 reported, so the unit-wide figure is unknown rather than a partial sum.
    assert ledger.try_admit("a", GPU_DEMAND, NODE_A, {0: 150000.0}).status == GRANTED


# --------------------------------------------------------------------------- node policy


def test_cpu_only_policy_keeps_gpu_episodes_off_train_nodes():
    units = [_gpu_unit("g-train", node=TRAIN), _cpu_unit("c-train", node=TRAIN), _gpu_unit("g-env", node=NODE_A)]
    rules = NodeRules.build("cpu_only", train_rollout_node_ips=[TRAIN])
    ledger = _ledger(units, rules)
    assert ledger.try_admit("gpu", GPU_DEMAND, TRAIN).status == REJECT
    assert ledger.try_admit("cpu", EnvDemand(), TRAIN).status == GRANTED
    assert ledger.try_admit("gpu", GPU_DEMAND, NODE_A).status == GRANTED


def test_avoid_policy_keeps_every_container_off_train_nodes():
    rules = NodeRules.build("avoid", train_rollout_node_ips=[TRAIN])
    ledger = _ledger([_cpu_unit("c-train", node=TRAIN), _cpu_unit("c-env", node=NODE_A)], rules)
    assert ledger.try_admit("s0", EnvDemand(), TRAIN).status == REJECT
    assert ledger.try_admit("s1", EnvDemand(), NODE_A).status == GRANTED


def test_prefer_other_spills_onto_train_nodes_only_when_other_nodes_are_full():
    rules = NodeRules.build("prefer_other", train_rollout_node_ips=[TRAIN])
    units = [_cpu_unit("c-env", node=NODE_A, max_containers=1), _cpu_unit("c-train", node=TRAIN, max_containers=2)]
    ledger = _ledger(units, rules)
    # No node hint: the router weighs both. The env node is tier 0 so it goes first.
    assert ledger.try_admit("s0", EnvDemand()).lease.unit_id == "c-env"
    # The env node is now full, so the fallback tier takes over instead of waiting.
    assert ledger.try_admit("s1", EnvDemand()).lease.unit_id == "c-train"


def test_any_policy_ignores_the_train_rollout_list():
    rules = NodeRules.build("any", train_rollout_node_ips=[TRAIN])
    ledger = _ledger([_gpu_unit("g-train", node=TRAIN)], rules)
    assert ledger.try_admit("s0", GPU_DEMAND, TRAIN).status == GRANTED


def test_deny_list_excludes_a_node_even_under_any_policy():
    rules = NodeRules.build("any", deny_node_ips=[NODE_A])
    ledger = _ledger([_cpu_unit("a", node=NODE_A)], rules)
    assert ledger.try_admit("s0", EnvDemand(), NODE_A).status == REJECT


# --------------------------------------------------------------------------- TTL reclaim and telemetry


def test_a_lease_without_heartbeats_is_reclaimed_and_queued_for_node_cleanup():
    clock = FakeClock()
    ledger = _ledger([_gpu_unit(max_containers=1)], clock=clock, lease_ttl_s=100.0)
    lease = ledger.try_admit("dead", GPU_DEMAND, NODE_A).lease
    clock.now = 101.0
    reaped = ledger.reap()
    assert [r.lease_id for r in reaped] == [lease.lease_id]
    # Capacity is free again immediately...
    assert ledger.try_admit("next", GPU_DEMAND, NODE_A).status == GRANTED
    # ...and the node learns exactly once which containers to stop.
    assert [r.lease_id for r in ledger.pop_reaped(NODE_A)] == [lease.lease_id]
    assert ledger.pop_reaped(NODE_A) == []


def test_heartbeats_keep_a_lease_alive():
    clock = FakeClock()
    ledger = _ledger([_cpu_unit()], clock=clock, lease_ttl_s=100.0)
    lease = ledger.try_admit("s0", EnvDemand(), NODE_A).lease
    for t in (60.0, 120.0, 180.0):
        clock.now = t
        assert ledger.heartbeat(lease.lease_id) is True
        assert ledger.reap() == []


def test_heartbeat_for_a_reclaimed_lease_reports_false():
    clock = FakeClock()
    ledger = _ledger([_cpu_unit()], clock=clock, lease_ttl_s=10.0)
    lease = ledger.try_admit("s0", EnvDemand(), NODE_A).lease
    clock.now = 11.0
    ledger.reap()
    assert ledger.heartbeat(lease.lease_id) is False


def test_snapshot_reports_declared_and_measured_memory_and_waiters():
    ledger = _ledger([_gpu_unit(max_containers=1)], default_gpu_memory_mb=24000)
    ledger.try_admit("a", GPU_DEMAND, NODE_A, {0: 5000.0})
    ledger.try_admit("b", GPU_DEMAND, NODE_A, {0: 5000.0})  # waits
    snap = ledger.snapshot()[0]
    assert snap.containers_running == 1
    assert snap.declared_gpu_memory_mb == 24000
    assert snap.measured_gpu_memory_mb == 5000.0
    assert snap.sessions_waiting == 1
    ledger.cancel_wait("b")
    assert ledger.snapshot()[0].sessions_waiting == 0


def test_node_capacity_is_bounded_by_both_containers_and_device_memory():
    # 3 container slots but only 2 default-sized tasks (2 * 24000 = 48000 <= 50000 < 72000).
    ledger = _ledger([_gpu_unit(max_containers=3, gpu_mb=50000)], default_gpu_memory_mb=24000)
    cap = ledger.node_capacity()[NODE_A]
    assert (cap.gpu_free, cap.cpu_free, cap.gpu_eligible, cap.cpu_eligible) == (2, 3, True, True)


def test_node_capacity_marks_a_train_node_gpu_ineligible_under_cpu_only():
    rules = NodeRules.build("cpu_only", train_rollout_node_ips=[TRAIN])
    ledger = _ledger([_gpu_unit("g", node=TRAIN)], rules)
    cap = ledger.node_capacity()[TRAIN]
    assert cap.gpu_eligible is False and cap.gpu_free == 0
    assert cap.cpu_eligible is True


# --------------------------------------------------------------------------- invariants under random load


def test_random_operations_never_oversubscribe_a_unit_under_prefer_other():
    _random_operations_hold_the_invariants("prefer_other")


def test_random_operations_never_oversubscribe_or_break_cpu_only_node_policy():
    _random_operations_hold_the_invariants("cpu_only")


def _random_operations_hold_the_invariants(policy):
    """
    Drive the ledger with thousands of random admits, releases, heartbeats, reaps and
    measurements, and after every step re-derive each unit's usage from the lease table.

    The invariants are the ones a bug would break silently: a unit never exceeds its
    container ceiling, CPU, host memory or device memory, the session index and lease table
    agree, a GPU demand never lands on a CPU unit, and under `cpu_only` no GPU lease ever
    lands on a train node.
    """
    import random

    rng = random.Random(20260412)
    units = [
        _gpu_unit("g0", NODE_A, (0,), max_containers=3, gpu_mb=80000),
        _gpu_unit("g1", NODE_A, (1, 2), max_containers=4, gpu_mb=160000),
        _cpu_unit("c0", NODE_A, cpu=8.0, mem=16000, max_containers=5),
        _gpu_unit("g2", NODE_B, (0,), max_containers=2, gpu_mb=48000),
        _cpu_unit("c1", NODE_B, cpu=4.0, mem=8000, max_containers=3),
    ]
    rules = NodeRules.build(policy, train_rollout_node_ips=[NODE_B])
    clock = FakeClock()
    ledger = _ledger(units, rules, clock=clock, default_gpu_memory_mb=24000, lease_ttl_s=50.0)
    by_id = {u.unit_id: u for u in units}
    live: list[str] = []  # lease ids we still hold
    granted = rejected = waited = 0

    def check():
        usage = {u.unit_id: [0, 0.0, 0, 0] for u in units}  # containers, cpu, mem, gpu_mb
        for lease in ledger._leases.values():
            slot = usage[lease.unit_id]
            slot[0] += 1
            slot[1] += lease.demand.cpu
            slot[2] += lease.demand.memory_mb
            slot[3] += lease.demand.gpu_memory_mb if lease.demand.needs_gpu else 0
            assert ledger._by_session[lease.session_id] == lease.lease_id
        assert len(ledger._by_session) == len(ledger._leases)
        for unit_id, (n, cpu, mem, gpu_mb) in usage.items():
            unit = by_id[unit_id]
            assert n <= unit.max_containers, f"{unit_id}: {n} containers > {unit.max_containers}"
            assert cpu <= unit.cpu_quota + 1e-6, f"{unit_id}: cpu {cpu} > {unit.cpu_quota}"
            assert mem <= unit.memory_mb, f"{unit_id}: mem {mem} > {unit.memory_mb}"
            assert gpu_mb <= unit.gpu_memory_mb, f"{unit_id}: gpu {gpu_mb} > {unit.gpu_memory_mb}"

    for step in range(3000):
        action = rng.random()
        if action < 0.55:
            demand = EnvDemand(
                cpu=rng.choice([0.5, 1.0, 2.0, 4.0]),
                memory_mb=rng.choice([0, 1000, 4000, 9000]),
                needs_gpu=rng.random() < 0.5,
                gpu_memory_mb=rng.choice([0, 10000, 24000, 70000]),
            )
            node = rng.choice([NODE_A, NODE_B, None])
            measured = {0: rng.uniform(0, 90000), 1: rng.uniform(0, 90000), 2: rng.uniform(0, 90000)}
            result = ledger.try_admit(f"s{step}", demand, node, measured if rng.random() < 0.3 else None)
            if result.status == GRANTED:
                granted += 1
                live.append(result.lease.lease_id)
                if node is not None:
                    assert result.lease.node_ip == node
                unit = by_id[result.lease.unit_id]
                assert unit.has_gpu or not demand.needs_gpu, "a GPU demand landed on a CPU unit"
                if policy == "cpu_only" and demand.needs_gpu:
                    assert unit.node_ip != NODE_B, "cpu_only put a GPU lease on a train node"
            elif result.status == REJECT:
                rejected += 1
            else:
                waited += 1
        elif action < 0.80 and live:
            ledger.release(live.pop(rng.randrange(len(live))))
        elif action < 0.90 and live:
            ledger.heartbeat(rng.choice(live))
        else:
            clock.now += rng.uniform(0, 30)
            live = [lease_id for lease_id in live if lease_id in ledger._leases or ledger.heartbeat(lease_id)]
            ledger.reap()
            live = [lease_id for lease_id in live if lease_id in ledger._leases]
        check()

    # The run must have exercised every outcome, or the invariants were never really tested.
    assert granted > 100 and waited > 50 and rejected > 5, (granted, waited, rejected)


# --------------------------------------------------------------------------- unit validation


def test_two_units_may_not_claim_the_same_device():
    try:
        validate_units([_gpu_unit("a", indices=(0, 1)), _gpu_unit("b", indices=(1,))])
    except ValueError as exc:
        assert "never overlap" in str(exc)
    else:
        raise AssertionError("overlapping devices were accepted")


def test_same_device_index_on_different_nodes_is_fine():
    validate_units([_gpu_unit("a", node=NODE_A, indices=(0,)), _gpu_unit("b", node=NODE_B, indices=(0,))])


def test_invalid_unit_declarations_are_rejected():
    bad = [
        lambda: EnvUnit("", NODE_A, 1, 1),
        lambda: EnvUnit("u", NODE_A, 0, 1),
        lambda: EnvUnit("u", NODE_A, 1, 1, max_containers=0),
        lambda: EnvUnit("u", NODE_A, 1, 1, gpu_indices=(0,), gpu_memory_mb=0),
        lambda: EnvUnit("u", NODE_A, 1, 1, gpu_indices=(), gpu_memory_mb=10),
        lambda: EnvUnit("u", NODE_A, 1, 1, gpu_indices=(0, 0), gpu_memory_mb=10),
    ]
    for make in bad:
        try:
            make()
        except ValueError:
            continue
        raise AssertionError("an invalid unit was accepted")


if __name__ == "__main__":
    import sys

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
