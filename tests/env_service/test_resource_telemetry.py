"""Separate physical use from reservations and retain cleanup until acknowledged."""

import time
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.env_service import resource_telemetry
from psrl.workers.env_service.placement import EnvAwareDispatcher
from psrl.workers.env_service.router import EnvRouterCore
from psrl.workers.env_service.types import EnvDemand


def core():
    return EnvRouterCore(
        {
            "node_policy": "any",
            "units": [
                {"unit_id": ip, "node_ip": ip, "cpu_quota": 4, "memory_mb": 1024, "max_containers": 2}
                for ip in ("a", "b")
            ],
        }
    )


@pytest.mark.asyncio
async def test_physical_count_and_partial_measurement(monkeypatch):
    rows = [{"Id": str(i), "Labels": {"psrl.env_unit": "a"}, "State": "running"} for i in range(4)]
    rows.append({"Id": "other", "Labels": {"psrl.env_unit": "other"}, "State": "running"})
    monkeypatch.setattr(resource_telemetry, "list_unit_containers", AsyncMock(return_value=rows))
    stats = {
        "memory_stats": {"usage": 10 * 1024**2},
        "cpu_stats": {"cpu_usage": {"total_usage": 200}, "system_cpu_usage": 2000, "online_cpus": 4},
        "precpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 1000},
    }
    reader = AsyncMock(return_value=stats)
    monkeypatch.setattr(resource_telemetry, "container_stats", reader)
    unit = (await resource_telemetry.read_node_resources(["a"]))["units"]["a"]
    assert unit["containers_running"] == 4
    assert unit["measured_memory_mb"] == 40
    assert unit["measured_cpu_cores"] == pytest.approx(1.6)
    reader.side_effect = [stats, TimeoutError("missing"), stats, stats]
    unit = (await resource_telemetry.read_node_resources(["a"]))["units"]["a"]
    assert unit["containers_running"] == 4
    assert unit["measured_memory_mb"] is None
    assert "TimeoutError" in unit["error"]


@pytest.mark.asyncio
async def test_stale_measurements_are_unknown_and_do_not_change_reservations():
    router = core()
    lease = router._ledger.try_admit("one", EnvDemand(cpu=1, memory_mb=100), "a").lease
    router._resource_samples["a"] = (
        time.monotonic(),
        {
            "cpu_load_1m": 2,
            "cpu_count": 4,
            "units": {"a": {"containers_running": 4, "measured_memory_mb": 40}},
        },
    )
    snapshot = (await router.snapshot())[0]
    assert (snapshot.leases_active, snapshot.containers_running, snapshot.memory_used_mb) == (1, 4, 100)
    assert (await router.node_capacity())["a"].node_load_per_cpu == 0.5
    router._resource_samples["a"] = (time.monotonic() - 100, router._resource_samples["a"][1])
    snapshot = (await router.snapshot())[0]
    assert snapshot.leases_active == 1 and snapshot.containers_running is None
    assert (await router.node_capacity())["a"].node_load_per_cpu is None
    assert router._ledger.retire(lease.lease_id)
    assert router._ledger.pending_reaped("a") == router._ledger.pending_reaped("a")
    router._ledger.acknowledge_reaped("b", lease.lease_id)
    assert len(router._ledger.pending_reaped("a")) == 1
    router._ledger.acknowledge_reaped("a", lease.lease_id)
    assert not router._ledger.pending_reaped("a")


@pytest.mark.asyncio
async def test_load_only_breaks_ties_between_fitting_healthy_nodes():
    router = core()
    capacities = await router.node_capacity()
    capacities["a"] = replace(capacities["a"], node_load_per_cpu=2)
    capacities["b"] = replace(capacities["b"], node_load_per_cpu=0.1)
    dispatcher = EnvAwareDispatcher(["a", "b"])
    dispatcher.update(capacities)
    demand = EnvDemand(cpu=1, memory_mb=100)
    assert dispatcher.assign([False], 0, demands=[demand]) == [1]
    capacities["b"] = replace(capacities["b"], available=False)
    dispatcher.update(capacities)
    assert dispatcher.assign([False], 0, demands=[demand]) == [0]
