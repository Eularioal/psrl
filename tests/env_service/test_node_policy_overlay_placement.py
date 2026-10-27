"""
Tests for the Ray-free helpers: node policy, Compose overlays with runtime-resolved driver
libraries, and unit-aware dispatch.

Run with `PYTHONPATH=. python3 tests/env_service/test_node_policy_overlay_placement.py`
or under pytest.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from psrl.workers.env_service import (
    EnvDemand,
    EnvLease,
    EnvUnit,
    NodeCapacity,
    NodePolicy,
    NodeRules,
    lint_units,
    parse_policy,
    select_worker_nodes,
)
from psrl.workers.env_service.node_policy import node_ips_from_worker_groups, restrict_to_unit_nodes
from psrl.workers.env_service.overlay import (
    CONTROL_DEVICES,
    LEASE_LABEL,
    UNIT_LABEL,
    DriverLibraryError,
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.placement import EnvAwareDispatcher, needs_gpu_from_extra_info
from psrl.workers.env_service.probe import GpuDevice, parse_gpu_csv

TRAIN = "10.0.0.9"
ENV = "10.0.0.1"
OTHER = "10.0.0.2"


# --------------------------------------------------------------------------- node policy


def test_parse_policy_accepts_names_and_lists_choices_on_a_typo():
    assert parse_policy("CPU_ONLY") is NodePolicy.CPU_ONLY
    assert parse_policy(NodePolicy.AVOID) is NodePolicy.AVOID
    try:
        parse_policy("nope")
    except ValueError as exc:
        assert "cpu_only" in str(exc) and "prefer_other" in str(exc)
    else:
        raise AssertionError("a bad policy name was accepted")


def test_select_worker_nodes_avoid_drops_train_nodes():
    rules = NodeRules.build("avoid", train_rollout_node_ips=[TRAIN])
    selected, note = select_worker_nodes([("n1", TRAIN), ("n2", ENV), ("n3", OTHER)], rules)
    assert [ip for _, ip in selected] == [ENV, OTHER]
    assert "removed 1" in note


def test_select_worker_nodes_avoid_fails_loudly_when_nothing_is_left():
    rules = NodeRules.build("avoid", train_rollout_node_ips=[TRAIN])
    try:
        select_worker_nodes([("n1", TRAIN)], rules)
    except ValueError as exc:
        assert "leaves no node" in str(exc)
    else:
        raise AssertionError("avoid with only train nodes should raise")


def test_select_worker_nodes_prefer_other_falls_back_instead_of_failing():
    rules = NodeRules.build("prefer_other", train_rollout_node_ips=[TRAIN])
    selected, note = select_worker_nodes([("n1", TRAIN)], rules)
    assert [ip for _, ip in selected] == [TRAIN]
    assert "falling back" in note
    selected, _ = select_worker_nodes([("n1", TRAIN), ("n2", ENV)], rules)
    assert [ip for _, ip in selected] == [ENV]


def test_select_worker_nodes_cpu_only_and_any_keep_every_node():
    for policy in ("cpu_only", "any"):
        rules = NodeRules.build(policy, train_rollout_node_ips=[TRAIN])
        selected, note = select_worker_nodes([("n1", TRAIN), ("n2", ENV)], rules)
        assert len(selected) == 2 and note == ""


def test_select_worker_nodes_applies_the_deny_list():
    rules = NodeRules.build("any", deny_node_ips=[OTHER])
    selected, _ = select_worker_nodes([("n1", ENV), ("n2", OTHER)], rules)
    assert [ip for _, ip in selected] == [ENV]


def test_restrict_to_unit_nodes_drops_nodes_that_declare_no_unit():
    # A worker on a node with no unit would have every episode refused and refilled forever.
    kept, note = restrict_to_unit_nodes([("n1", ENV), ("n2", OTHER), ("n3", TRAIN)], [ENV, TRAIN])
    assert [ip for _, ip in kept] == [ENV, TRAIN]
    assert OTHER in note


def test_restrict_to_unit_nodes_is_silent_when_every_node_has_a_unit():
    kept, note = restrict_to_unit_nodes([("n1", ENV)], [ENV, OTHER])
    assert kept == [("n1", ENV)] and note == ""


def test_restrict_to_unit_nodes_fails_loudly_when_no_node_has_a_unit():
    try:
        restrict_to_unit_nodes([("n1", ENV)], [OTHER])
    except ValueError as exc:
        assert "declares an env unit" in str(exc) and OTHER in str(exc)
    else:
        raise AssertionError("no unit-bearing node should raise")


def test_the_two_node_filters_compose_in_the_order_the_trainer_applies_them():
    # The trainer runs the policy filter first, then the unit filter. Under `avoid` the train
    # node goes first, so a unit declared only there leaves nothing and must raise.
    rules = NodeRules.build("avoid", train_rollout_node_ips=[TRAIN])
    kept, _ = select_worker_nodes([("n1", TRAIN), ("n2", ENV)], rules)
    try:
        restrict_to_unit_nodes(kept, [TRAIN])
    except ValueError:
        return
    raise AssertionError("a unit only on a train node must not survive policy=avoid")


def test_lint_flags_a_gpu_unit_that_sits_on_a_train_node():
    gpu_on_train = EnvUnit("g", TRAIN, 8, 1000, (0,), 2, 80000)
    cpu_on_train = EnvUnit("c", TRAIN, 8, 1000)
    rules = NodeRules.build("cpu_only", train_rollout_node_ips=[TRAIN])
    warnings = lint_units([gpu_on_train, cpu_on_train], rules)
    assert len(warnings) == 1 and "'g'" in warnings[0]
    assert lint_units([gpu_on_train], NodeRules.build("any", train_rollout_node_ips=[TRAIN]))  # still warned


def test_node_ips_from_worker_groups_reads_placement_metadata():
    class Pool:
        def __init__(self, pgs):
            self.pgs = pgs

    class Group:
        def __init__(self, pgs):
            self.resource_pool = Pool(pgs)

    tables = {"pg1": {"bundles_to_node_id": {0: "nodeA", 1: "nodeA"}}, "pg2": {"bundles_to_node_id": {0: "nodeB"}}}
    groups = {
        "actor": Group(["pg1"]),
        "rollout_0": Group(["pg2"]),
        "ps": Group(["pg2"]),  # CPU parameter-server pool, must be ignored
        "empty": Group([]),
    }
    ips = node_ips_from_worker_groups(
        groups,
        placement_group_table=lambda pg: tables[pg],
        node_id_to_ip={"nodeA": "10.0.0.1", "nodeB": "10.0.0.2"},
    )
    assert ips == frozenset({"10.0.0.1", "10.0.0.2"})
    only_actor = node_ips_from_worker_groups(
        {"actor": groups["actor"], "ps": groups["ps"]},
        placement_group_table=lambda pg: tables[pg],
        node_id_to_ip={"nodeA": "10.0.0.1", "nodeB": "10.0.0.2"},
    )
    assert only_actor == frozenset({"10.0.0.1"})  # nodeB came only from the ignored PS pool


# --------------------------------------------------------------------------- driver libraries and overlay

_PROC_VERSION = (
    "NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.247.01  Wed Oct 30 10:15:02 UTC 2024\n"
    "GCC version:  gcc version 11.4.0\n"
)


def test_kernel_driver_version_is_read_from_proc():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "version"
        path.write_text(_PROC_VERSION)
        assert read_kernel_driver_version(str(path)) == "535.247.01"
        assert read_kernel_driver_version(str(Path(tmp) / "missing")) is None
        path.write_text("garbage")
        assert read_kernel_driver_version(str(path)) is None


def test_driver_libs_are_found_by_exact_version_never_by_the_libcuda_so_1_symlink():
    # B1: on the real fleet libcuda.so.1 points at 570 while the kernel module is 535. The
    # resolver must pick the 535 file by name and must not mount libcuda.so.1 itself.
    present = {
        "/usr/lib64/libcuda.so.1",  # the misleading symlink (-> 570), must be ignored
        "/usr/lib64/libcuda.so.570.133.20",
        "/usr/lib64/libcuda.so.535.247.01",
        "/usr/lib64/libnvidia-ml.so.535.247.01",
        "/usr/lib64/libnvidia-ptxjitcompiler.so.535.247.01",
    }
    libs = resolve_driver_libs("535.247.01", exists=present.__contains__)
    hosts = [host for host, _ in libs.mounts]
    assert "/usr/lib64/libcuda.so.535.247.01" in hosts
    assert not any(h.endswith("libcuda.so.1") for h in hosts)
    assert not any("570" in h for h in hosts)
    containers = {c for _, c in libs.mounts}
    assert "/usr/lib64/libcuda.so.1" in containers  # visible under the expected name inside the container
    assert libs.missing_optional == ()


def test_ptx_jit_compiler_is_mounted_when_present_and_reported_when_not():
    # B2: libnvidia-ptxjitcompiler.so.1 was missing from the code-side mount list.
    base = {"/usr/lib64/libcuda.so.535.1", "/usr/lib64/libnvidia-ml.so.535.1"}
    with_ptx = base | {"/usr/lib64/libnvidia-ptxjitcompiler.so.535.1"}
    libs = resolve_driver_libs("535.1", exists=with_ptx.__contains__)
    assert any(c.endswith("libnvidia-ptxjitcompiler.so.1") for _, c in libs.mounts)
    libs = resolve_driver_libs("535.1", exists=base.__contains__)
    assert libs.missing_optional == ("libnvidia-ptxjitcompiler",)


def test_a_missing_required_driver_library_fails_with_an_actionable_message():
    try:
        resolve_driver_libs("535.1", exists=lambda p: False)
    except DriverLibraryError as exc:
        assert "libcuda" in str(exc) and "535.1" in str(exc) and "readlink" in str(exc)
    else:
        raise AssertionError("missing libcuda should raise")


def test_libraries_are_searched_across_several_directories():
    present = {"/usr/lib/x86_64-linux-gnu/libcuda.so.1.2", "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1.2"}
    libs = resolve_driver_libs("1.2", exists=present.__contains__)
    assert all(host.startswith("/usr/lib/x86_64-linux-gnu/") for host, _ in libs.mounts)


def _lease(gpu_indices=(), unit_id="u1", lease_id="u1-1-x"):
    return EnvLease(lease_id, "sess", unit_id, ENV, gpu_indices, EnvDemand(), 0.0, 0.0)


def _libs():
    present = {"/usr/lib64/libcuda.so.9", "/usr/lib64/libnvidia-ml.so.9", "/usr/lib64/libnvidia-ptxjitcompiler.so.9"}
    return resolve_driver_libs("9", exists=present.__contains__)


def test_overlay_mounts_exactly_the_leases_devices_not_a_hardcoded_nvidia0():
    # B3: the static override always mounted /dev/nvidia0.
    overlay = build_lease_overlay(
        _lease((3, 4)),
        _libs(),
        gpu_devices={i: GpuDevice(i, i, f"GPU-{i}") for i in range(8)},
        device_exists=lambda p: True,
    )
    main = overlay["services"]["main"]
    assert "/dev/nvidia3:/dev/nvidia3" in main["devices"]
    assert "/dev/nvidia4:/dev/nvidia4" in main["devices"]
    assert "/dev/nvidia0:/dev/nvidia0" not in main["devices"]
    for control in CONTROL_DEVICES:
        assert f"{control}:{control}" in main["devices"]


def test_overlay_skips_control_devices_that_do_not_exist_on_the_host():
    overlay = build_lease_overlay(
        _lease((0,)),
        _libs(),
        gpu_devices={i: GpuDevice(i, i, f"GPU-{i}") for i in range(8)},
        device_exists=lambda p: p == "/usr/bin/nvidia-smi",
    )
    main = overlay["services"]["main"]
    assert main["devices"] == ["/dev/nvidia0:/dev/nvidia0"]
    assert "/usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro" in main["volumes"]


def test_overlay_stamps_lease_and_unit_labels_for_reaping():
    # B6: Harbor sets no labels on local containers, so the lease label is what lets the
    # reaper find an orphan after a crash.
    lease = _lease((0,), unit_id="g7", lease_id="g7-4-abc")
    overlay = build_lease_overlay(
        lease, _libs(), gpu_devices={i: GpuDevice(i, i, f"GPU-{i}") for i in range(8)}, device_exists=lambda p: True
    )
    labels = overlay["services"]["main"]["labels"]
    assert f"{LEASE_LABEL}=g7-4-abc" in labels and f"{UNIT_LABEL}=g7" in labels


def test_cpu_lease_overlay_carries_labels_only():
    main = build_lease_overlay(_lease(()), None)["services"]["main"]
    assert set(main) == {"labels"}


def test_gpu_lease_without_driver_libs_is_a_programming_error():
    try:
        build_lease_overlay(_lease((0,)), None)
    except ValueError:
        return
    raise AssertionError("a GPU lease without driver libs was accepted")


def test_overlay_resolves_nvml_index_to_device_minor():
    overlay = build_lease_overlay(
        _lease((1,)),
        _libs(),
        gpu_devices={1: GpuDevice(1, 3, "GPU-assigned")},
        device_exists=lambda path: False,
    )
    assert overlay["services"]["main"]["devices"] == ["/dev/nvidia3:/dev/nvidia3"], "Wrong physical GPU mounted."


def test_write_overlay_round_trips_through_yaml_and_is_a_real_file():
    overlay = build_lease_overlay(
        _lease((1,)),
        _libs(),
        gpu_devices={i: GpuDevice(i, i, f"GPU-{i}") for i in range(8)},
        device_exists=lambda p: True,
    )
    path = write_overlay(overlay)
    try:
        assert path.exists() and path.suffix == ".yaml"
        assert yaml.safe_load(path.read_text()) == overlay
    finally:
        path.unlink(missing_ok=True)
    assert not os.path.exists(path)


# --------------------------------------------------------------------------- telemetry parsing


def test_parse_gpu_csv_skips_malformed_lines():
    text = "0, 97, 18000\n1, 0, 300\nthis is not csv\n2, abc, 5\n"
    assert parse_gpu_csv(text) == {0: (97.0, 18000.0), 1: (0.0, 300.0)}
    assert parse_gpu_csv("") == {}


# --------------------------------------------------------------------------- unit-aware dispatch


def test_needs_gpu_from_extra_info_handles_dicts_json_and_junk():
    assert needs_gpu_from_extra_info({"gpus": 1}) is True
    assert needs_gpu_from_extra_info({"gpus": 0}) is False
    assert needs_gpu_from_extra_info('{"gpus": 2}') is True
    assert needs_gpu_from_extra_info("not json") is False
    assert needs_gpu_from_extra_info(None) is False
    assert needs_gpu_from_extra_info({"gpus": "x"}) is False
    assert needs_gpu_from_extra_info({}) is False


class _Clock:
    now = 0.0

    def __call__(self):
        return self.now


def _dispatcher(nodes, clock=None):
    return EnvAwareDispatcher(nodes, clock=clock or _Clock())


def test_without_a_snapshot_dispatch_requires_health_information():
    d = _dispatcher(["a", "b", "c"])
    with pytest.raises(RuntimeError, match="fresh"):
        d.assign([False] * 4, rr_start=1)


def test_a_stale_snapshot_does_not_override_known_node_constraints():
    clock = _Clock()
    d = _dispatcher(["a", "b"], clock)
    d.update({"a": NodeCapacity(0, 0, True, False), "b": NodeCapacity(5, 5, True, True)})
    clock.now = 100.0  # older than max_snapshot_age_s
    with pytest.raises(RuntimeError, match="fresh"):
        d.assign([True], rr_start=0)


def test_gpu_groups_skip_workers_on_nodes_without_gpu_capacity():
    d = _dispatcher(["cpu-node", "gpu-node"])
    d.update(
        {
            "cpu-node": NodeCapacity(cpu_free=4, gpu_free=0, cpu_eligible=True, gpu_eligible=False),
            "gpu-node": NodeCapacity(cpu_free=4, gpu_free=2, cpu_eligible=True, gpu_eligible=True),
        }
    )
    # Plain round robin from 0 would give the first GPU group to cpu-node.
    assert d.assign([True, True], rr_start=0) == [1, 1]


def test_a_batch_spreads_across_nodes_instead_of_piling_on_one():
    d = _dispatcher(["n1", "n2"])
    d.update(
        {
            "n1": NodeCapacity(cpu_free=1, gpu_free=1, cpu_eligible=True, gpu_eligible=True),
            "n2": NodeCapacity(cpu_free=1, gpu_free=1, cpu_eligible=True, gpu_eligible=True),
        }
    )
    assert sorted(d.assign([True, True], rr_start=0)) == [0, 1]


def test_when_every_node_is_full_a_group_goes_to_a_node_that_could_host_it_later():
    d = _dispatcher(["never", "later"])
    d.update(
        {
            "never": NodeCapacity(cpu_free=0, gpu_free=0, cpu_eligible=True, gpu_eligible=False),
            "later": NodeCapacity(cpu_free=0, gpu_free=0, cpu_eligible=True, gpu_eligible=True),
        }
    )
    # Gate A will wait on `later`, whereas on `never` it would be rejected outright.
    assert d.assign([True], rr_start=0) == [1]


def test_when_no_node_is_eligible_dispatch_reports_unavailability():
    d = _dispatcher(["a", "b"])
    d.update({"a": NodeCapacity(), "b": NodeCapacity()})
    with pytest.raises(RuntimeError, match="eligible"):
        d.assign([True], rr_start=1)


def test_siblings_share_a_group_so_slots_are_counted_per_group_not_per_child():
    d = _dispatcher(["n1", "n2"])
    d.update(
        {
            "n1": NodeCapacity(cpu_free=1, gpu_free=1, cpu_eligible=True, gpu_eligible=True),
            "n2": NodeCapacity(cpu_free=9, gpu_free=9, cpu_eligible=True, gpu_eligible=True),
        }
    )
    # One group takes one slot, so the second group must spill to n2 once n1's single slot is used.
    assert d.assign([True, True], rr_start=0) == [0, 1]


def test_dispatcher_rejects_an_empty_worker_list():
    try:
        EnvAwareDispatcher([])
    except ValueError:
        return
    raise AssertionError("an empty worker list was accepted")


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
