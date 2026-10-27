"""
Tests for startup validation of `psrl.env_service` and for turning a task into a demand.

Run with `PYTHONPATH=. python3 tests/env_service/test_config_and_demand.py` or under pytest.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from psrl.workers.env_service.config_check import validate_env_service_config  # noqa: E402
from psrl.workers.env_service.harbor_resources import demand_from_task  # noqa: E402

GOOD_GPU = {
    "unit_id": "g0",
    "node_ip": "10.0.0.1",
    "cpu_quota": 16,
    "memory_mb": 65536,
    "gpu_indices": [0],
    "max_containers": 3,
    "gpu_memory_mb": 80000,
}


def _expect_error(cfg, *needles):
    try:
        validate_env_service_config(cfg)
    except ValueError as exc:
        for needle in needles:
            assert needle in str(exc), f"{needle!r} not in {exc!s}"
        return
    raise AssertionError(f"config was accepted but should fail on {needles}: {cfg}")


# --------------------------------------------------------------------------- config validation


def test_a_valid_config_returns_its_parsed_units():
    units = validate_env_service_config({"units": [GOOD_GPU], "node_policy": "avoid"})
    assert [u.unit_id for u in units] == ["g0"] and units[0].gpu_indices == (0,)


def test_an_enabled_service_with_no_units_is_an_error():
    _expect_error({"units": []}, "at least one", "units")
    _expect_error({}, "at least one")


def test_a_bad_policy_name_lists_the_choices():
    _expect_error({"units": [GOOD_GPU], "node_policy": "sometimes"}, "cpu_only", "prefer_other")


def test_a_malformed_unit_names_its_position_and_the_missing_key():
    broken = {k: v for k, v in GOOD_GPU.items() if k != "node_ip"}
    _expect_error({"units": [GOOD_GPU, broken]}, "units[1]", "node_ip")


def test_overlapping_devices_across_units_are_rejected():
    twin = {**GOOD_GPU, "unit_id": "g1"}
    _expect_error({"units": [GOOD_GPU, twin]}, "never overlap")


def test_a_heartbeat_slower_than_half_the_ttl_is_rejected():
    # With beat=100 and ttl=150 a live lease can expire between two beats.
    _expect_error({"units": [GOOD_GPU], "slot_ttl_s": 150, "heartbeat_interval_s": 100}, "half")
    validate_env_service_config({"units": [GOOD_GPU], "slot_ttl_s": 200, "heartbeat_interval_s": 100})  # boundary ok


def test_non_positive_timing_values_are_rejected():
    _expect_error({"units": [GOOD_GPU], "slot_ttl_s": 0}, "positive")
    _expect_error({"units": [GOOD_GPU], "poll_interval_s": 5, "max_poll_interval_s": 1}, "poll_interval_s")


def test_a_gpu_unit_smaller_than_one_default_task_is_rejected():
    small = {**GOOD_GPU, "gpu_memory_mb": 10000}
    _expect_error({"units": [small], "default_gpu_memory_mb": 24000}, "could never host")


def test_a_non_positive_default_gpu_memory_is_rejected():
    _expect_error({"units": [GOOD_GPU], "default_gpu_memory_mb": 0}, "default_gpu_memory_mb")


def test_the_shipped_defaults_are_internally_consistent():
    import yaml

    shipped = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "psrl/trainer/config/psrl/env_service.yaml").read_text()
    )
    assert shipped["enable"] is False, "the service must ship disabled so existing recipes are unaffected"
    # Everything except the empty units list must already satisfy the validator.
    validate_env_service_config({**shipped, "units": [GOOD_GPU]})


# --------------------------------------------------------------------------- demand from task.toml


def _task_dir(toml_text):
    tmp = tempfile.mkdtemp()
    if toml_text is not None:
        (Path(tmp) / "task.toml").write_text(toml_text)
    return tmp


def test_demand_reads_cpus_memory_and_gpus_from_task_toml():
    demand = demand_from_task(_task_dir("[environment]\ncpus = 4\nmemory_mb = 8192\ngpus = 1\n"))
    assert (demand.cpu, demand.memory_mb, demand.needs_gpu) == (4.25, 8320, True)
    assert demand.gpu_memory_mb == 0  # Harbor declares none, so the router default applies


def test_demand_for_a_cpu_task_never_charges_device_memory():
    demand = demand_from_task(_task_dir("[environment]\ncpus = 2\nmemory_mb = 4096\n"), gpu_memory_mb_override=24000)
    assert demand.needs_gpu is False and demand.gpu_memory_mb == 0


def test_an_explicit_device_memory_override_is_used_for_gpu_tasks():
    demand = demand_from_task(_task_dir("[environment]\ngpus = 1\n"), gpu_memory_mb_override=12000)
    assert demand.needs_gpu is True and demand.gpu_memory_mb == 12000


def test_the_dataset_rows_gpu_claim_is_never_silently_dropped():
    # task.toml says no GPU, but the dataset row says it needs one: trust the stricter claim.
    assert demand_from_task(_task_dir("[environment]\ngpus = 0\n"), needs_gpu_hint=True).needs_gpu is True


def test_a_missing_task_toml_degrades_to_one_cpu_instead_of_failing_the_episode():
    demand = demand_from_task(_task_dir(None), needs_gpu_hint=True)
    assert (demand.cpu, demand.memory_mb, demand.needs_gpu) == (1.25, 2176, True)


def test_a_malformed_task_toml_degrades_the_same_way():
    demand = demand_from_task(_task_dir("this is = = not toml ["))
    assert (demand.cpu, demand.memory_mb, demand.needs_gpu) == (1.25, 2176, False)


def test_junk_resource_values_fall_back_to_defaults():
    demand = demand_from_task(_task_dir('[environment]\ncpus = "many"\nmemory_mb = "lots"\ngpus = "x"\n'))
    assert (demand.cpu, demand.memory_mb, demand.needs_gpu) == (1.25, 2176, False)


def test_a_zero_cpu_declaration_still_charges_one_cpu():
    assert demand_from_task(_task_dir("[environment]\ncpus = 0\n")).cpu == 1.25


if __name__ == "__main__":
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
