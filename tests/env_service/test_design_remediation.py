"""Exercise routing ownership, device locality, container budgets and node failures."""

import asyncio
import importlib.util
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from psrl.workers.env_service import EnvDemand, EnvUnit, EnvUnitLedger, NodeRules
from psrl.workers.env_service.command_gpu import GpuCommandScheduler
from psrl.workers.env_service.harbor_resources import demand_from_task
from psrl.workers.env_service.placement import EnvAwareDispatcher, EnvPlacementUnavailable
from psrl.workers.env_service.router import RayEnvRouter

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "design_audit", REPO / "docs/env-worker-deep-dive/env-service-verification/scripts/audit_remaining_env_design.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.asyncio
async def test_real_migration_restores_routing_after_abort_and_drain_failure():
    results = await audit.migration_cases()
    for result in results["cases"].values():
        assert result["transitions"] == [False, True], "Each migration must release its own pause."
    assert results["cases"]["abort"]["error"] == "RuntimeError", "The command failure must remain visible."
    assert results["cases"]["drain"]["error"] == "TimeoutError", "The drain failure must remain visible."


@pytest.mark.asyncio
async def test_pause_preserves_existing_and_concurrent_persistent_pause():
    harness = audit.MigrationHarness("none")
    harness.running = False
    async with harness._pauses.pause():
        pass
    assert not harness.running, "A preexisting pause belongs to its original owner."
    await harness._pauses.set_running(True)
    async with harness._pauses.pause():
        await harness._pauses.set_running(False)
    assert not harness.running, "A concurrent sleep must survive migration cleanup."
    await harness._pauses.set_running(True)
    async with harness._pauses.pause():
        async with harness._pauses.pause():
            await harness._pauses.set_running(True)
        assert not harness.running, "One remaining owner must keep routing paused."
    assert harness.running, "The last owner must apply the latest persistent state."


@pytest.mark.asyncio
async def test_repeated_cancellation_joins_resume_without_leaking_pause():
    harness = audit.MigrationHarness("none")
    entered, resuming, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def write(running):
        if running:
            resuming.set()
            await release.wait()
        await harness._set_routing_loop_running(running)

    harness._pauses._write_running = write

    async def migrate():
        async with harness._pauses.pause():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(migrate())
    await entered.wait()
    task.cancel()
    await resuming.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), "Cancellation must join the bounded resume operation."
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert harness.running, "Cancellation must not leak a routing pause."


@pytest.mark.asyncio
async def test_pause_acknowledgement_failure_restores_previous_state():
    harness = audit.MigrationHarness("none")

    async def write(running):
        await harness._set_routing_loop_running(running)
        if not running:
            raise TimeoutError("Lost pause acknowledgement.")

    harness._pauses._write_running = write
    with pytest.raises(TimeoutError):
        async with harness._pauses.pause():
            pytest.fail("A failed pause must not execute the migration body.")
    assert harness.running, "An applied pause with a lost reply must still be released."


def two_cards(mode="episode"):
    unit = EnvUnit("gpu", "node", 8, 8192, (0, 1), 4, 80, gpu_reservation=mode, gpu_memory_per_device_mb=(40, 40))
    return EnvUnitLedger([unit], NodeRules.build("any"))


def test_single_device_peak_and_asymmetric_pressure_cannot_use_aggregate_headroom():
    ledger = two_cards()
    assert ledger.try_admit("too-wide", EnvDemand(needs_gpu=True, gpu_memory_mb=60)).status == "reject", (
        "A 60 MB peak cannot fit either 40 MB device."
    )
    demand = EnvDemand(needs_gpu=True, gpu_memory_mb=20, gpu_memory_per_device_mb=(10, 10))
    assert ledger.try_admit("split", demand, "node", {0: 35, 1: 0}).status == "wait", (
        "Device zero cannot borrow device one's idle memory."
    )
    assert ledger.try_admit("split", demand, "node", {0: 0, 1: 0}).status == "granted", (
        "An explicit partition fits when both devices have room."
    )


@pytest.mark.asyncio
async def test_command_grant_checks_each_physical_device_before_enabling_access():
    ledger = two_cards("command")
    demand = EnvDemand(needs_gpu=True, gpu_memory_mb=20, gpu_memory_per_device_mb=(10, 10))
    lease = ledger.try_admit("command", demand, "node").lease
    memory = {
        "devices": {0: {"used_mb": 35, "total_mb": 40}, 1: {"used_mb": 0, "total_mb": 40}},
        "owned_mb": {},
        "attribution_errors": {},
    }
    node = SimpleNamespace(
        gpu_command_memory_snapshot=SimpleNamespace(remote=AsyncMock(return_value=memory)),
        grant_gpu_command=SimpleNamespace(remote=AsyncMock()),
    )
    scheduler = GpuCommandScheduler(ledger, AsyncMock(return_value=node))
    assert (await scheduler.acquire(lease.lease_id, "container", "request"))["status"] == "wait", (
        "A hot device must block access even when aggregate memory fits."
    )
    node.grant_gpu_command.remote.assert_not_awaited()
    memory["devices"][0]["used_mb"] = 0
    assert (await scheduler.acquire(lease.lease_id, "container", "request"))["status"] == "granted", (
        "The same queued request must progress after device pressure clears."
    )
    node.grant_gpu_command.remote.assert_awaited_once()


def test_a_gpu_only_verifier_requires_a_gpu_lease(tmp_path):
    (tmp_path / "task.toml").write_text(
        '[environment]\ngpus=0\n[verifier]\nenvironment_mode="separate"\n[verifier.environment]\ngpus=1\n'
    )
    assert demand_from_task(str(tmp_path)).needs_gpu, "Verifier devices are part of the episode's resource demand."


def test_harbor_config_roundtrip_preserves_final_limits_for_main_and_verifier(tmp_path, monkeypatch):
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.trial.config import EnvironmentConfig
    from psrl.workers.env_service.container_budget import ContainerBudget
    from psrl.workers.env_service.docker import ManagedDockerEnvironment

    source = tmp_path / "compose.yaml"
    source.write_text("services:\n  main: {}\n  network: {}\n")
    budget = ContainerBudget(2, 4096, 1, environments=2)
    config = EnvironmentConfig(kwargs={"container_budget": asdict(budget)})
    restored = EnvironmentConfig.model_validate_json(config.model_dump_json())
    verifier = restored.model_copy(update={"extra_docker_compose": []})
    monkeypatch.setattr(DockerEnvironment, "__init__", lambda *args, **kwargs: None)
    monkeypatch.setattr(DockerEnvironment, "_docker_compose_paths", property(lambda self: [source]))
    for environment in (restored, verifier):
        env = ManagedDockerEnvironment(**environment.kwargs)
        try:
            services = yaml.safe_load(env._docker_compose_paths[-1].read_text())["services"]
            assert services["main"]["mem_limit"] == "4096M", "Serialization must preserve main limits."
            assert services["network"]["cpus"] == 0.25, "Verifier sidecars must inherit the admitted budget."
        finally:
            env._ownership_dir.cleanup()


def test_container_set_budget_covers_separate_verifier_and_all_sidecars(tmp_path):
    (tmp_path / "task.toml").write_text(
        "[environment]\ncpus=2\nmemory_mb=4096\n"
        '[verifier]\nenvironment_mode="separate"\n'
        "[verifier.environment]\ncpus=4\nmemory_mb=8192\n"
    )
    (tmp_path / "docker-compose.yaml").write_text("services:\n  main: {}\n  database: {}\n")
    demand = demand_from_task(str(tmp_path))
    assert (demand.cpu, demand.memory_mb) == (9, 16896), "Both complete environment sets must be reserved."
    limits = demand.container_budget.service_limits({"main", "database", "network"})
    assert limits["main"]["mem_limit"] == "8192M", "The verifier's larger main budget must fit."
    assert limits["database"]["cpus"] == 0.25, "Auxiliary services must have explicit limits."
    assert limits["network"]["deploy"]["replicas"] == 1, "Replicas cannot bypass the container budget."
    with pytest.raises(ValueError, match="exceed"):
        demand.container_budget.service_limits({"main", "database", "network", "extra"})


def test_dispatch_accounts_for_group_size_host_memory_and_shared_cpu_gpu_slots():
    ledger = EnvUnitLedger(
        [EnvUnit("small", "a", 8, 100, max_containers=8), EnvUnit("large", "b", 8, 1000, max_containers=8)],
        NodeRules.build("any"),
    )
    dispatcher = EnvAwareDispatcher(["a", "b"])
    dispatcher.update(ledger.node_capacity())
    demands = [EnvDemand(cpu=1, memory_mb=60)] * 2
    assert dispatcher.assign([False, False], 0, demands=demands, group_sizes=[2, 2]) == [1, 1], (
        "Two siblings need 120 MB, so the 100 MB node must not receive the group."
    )
    ledger._blocked_units.add("large")
    dispatcher.update(ledger.node_capacity())
    with pytest.raises(EnvPlacementUnavailable):
        dispatcher.assign([False], 0, demands=[EnvDemand(memory_mb=200)], group_sizes=[1])


def make_health_router(monkeypatch):
    router = RayEnvRouter(
        {
            "node_policy": "any",
            "health": {"interval_s": 0.001, "timeout_s": 0.1},
            "units": [{"unit_id": "cpu", "node_ip": "node", "cpu_quota": 2, "memory_mb": 4096}],
        }
    )
    service = SimpleNamespace(
        ready=SimpleNamespace(remote=AsyncMock(return_value="http://node")),
        prepare=SimpleNamespace(remote=AsyncMock(return_value={})),
        cleanup=SimpleNamespace(remote=AsyncMock(return_value=0)),
        close=SimpleNamespace(remote=AsyncMock()),
    )
    monkeypatch.setattr(router, "_create_node", lambda ip: service)
    monkeypatch.setattr(router, "_ensure_reaper", lambda: None)
    monkeypatch.setattr("psrl.workers.env_service.router.ray.kill", lambda *args, **kwargs: None)
    return router, service


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [False, True])
async def test_dead_node_replacement_requires_all_live_leases_to_end(monkeypatch, active):
    import ray

    router, service = make_health_router(monkeypatch)
    await router._check_node("node")
    lease = router._ledger.try_admit("owner", EnvDemand(cpu=1), "node").lease if active else None
    service.prepare.remote.side_effect = ray.exceptions.RayActorError()
    await router._check_node("node")
    assert ("node" in router._node_services) == active
    assert router._node_health["node"].failures == 1
    if lease:
        router._ledger.release(lease.lease_id)
    await router.shutdown_nodes()


@pytest.mark.asyncio
async def test_shutdown_cannot_discard_unacknowledged_physical_cleanup(monkeypatch):
    router, service = make_health_router(monkeypatch)
    await router._check_node("node")
    lease = router._ledger.try_admit("owner", EnvDemand(cpu=1), "node").lease
    await router.release(lease.lease_id)
    with pytest.raises(RuntimeError, match="physical cleanup"):
        await router.shutdown_nodes()
    assert not router._nodes_closing
    service.close.remote.assert_not_awaited()
    router._ledger.acknowledge_reaped("node", lease.lease_id)
    await router.shutdown_nodes()


@pytest.mark.asyncio
async def test_pending_gpu_cleanup_does_not_recreate_stateful_node(monkeypatch):
    import ray

    router, service = make_health_router(monkeypatch)
    await router._check_node("node")
    router._ledger = EnvUnitLedger([EnvUnit("gpu", "node", 2, 4096, (0,), 1, 40)], NodeRules.build("any"))
    lease = router._ledger.try_admit("gpu-owner", EnvDemand(needs_gpu=True, gpu_memory_mb=10), "node").lease
    router._ledger.retire(lease.lease_id)
    service.prepare.remote.side_effect = ray.exceptions.RayActorError()
    await router._check_node("node")
    assert "node" in router._node_services
    assert not router._node_health["node"].available
    router._ledger.acknowledge_reaped("node", lease.lease_id)
    await router.shutdown_nodes()


@pytest.mark.asyncio
async def test_resource_query_failure_does_not_mark_a_healthy_node_failed(monkeypatch):
    router, service = make_health_router(monkeypatch)
    service.resource_snapshot = SimpleNamespace(remote=AsyncMock(side_effect=TimeoutError("Missing stats")))
    await router._check_node("node")
    for _ in range(10):
        if "node" in router._resource_errors:
            break
        await asyncio.sleep(0)
    assert "TimeoutError" in router._resource_errors["node"]
    assert router._node_health["node"].available
    assert router._node_health["node"].failures == 0
    await router.shutdown_nodes()


@pytest.mark.asyncio
async def test_runtime_node_failure_recovers_only_after_successful_probe_and_cleanup(monkeypatch):
    router, service = make_health_router(monkeypatch)
    await router._check_node("node")
    service.prepare.remote.side_effect = RuntimeError("Docker is unavailable.")
    await router._check_node("node")
    assert not (await router.node_capacity())["node"].available, "A failed node must leave placement."
    service.prepare.remote.side_effect = None
    service.cleanup.remote.return_value = 1
    await router._check_node("node")
    assert not router._node_health["node"].available, "Healthy Docker does not prove orphan cleanup."
    service.cleanup.remote.return_value = 0
    await router._check_node("node")
    assert router._node_health["node"].available, "Recovery needs both liveness and cleanup."
    await router.shutdown_nodes()


@pytest.mark.asyncio
async def test_admission_does_not_overwrite_new_node_telemetry_with_older_client_reading(monkeypatch):
    router, service = make_health_router(monkeypatch)
    router._ledger = EnvUnitLedger([EnvUnit("gpu", "node", 2, 4096, (0,), 1, 40)], NodeRules.build("any"))
    service.prepare.remote.return_value = {0: {"used_mb": 35, "total_mb": 40}}
    demand = EnvDemand(needs_gpu=True, gpu_memory_mb=10)
    result = await router.try_admit("episode", demand, "node", {0: 0})
    assert result.status == "wait", "Old client telemetry must not erase fresh node pressure."
    await router.shutdown_nodes()


@pytest.mark.asyncio
async def test_unhealthy_node_exhausts_retry_budget_without_spinning(monkeypatch):
    router, service = make_health_router(monkeypatch)
    service.prepare.remote.side_effect = RuntimeError("Docker is unavailable.")
    for _ in range(3):
        task = router._health_task("node", force=True)
        await task
    attempts = service.prepare.remote.await_count
    assert router._health_task("node", force=True) is None, "Recovery must have a finite retry budget."
    result = await router.try_admit("episode", EnvDemand(), "node")
    assert result.status == "reject", "Exhausted infrastructure retries must fail visibly."
    assert service.prepare.remote.await_count == attempts, "Admission must not restart exhausted recovery."
    await router.shutdown_nodes()


@pytest.mark.asyncio
async def test_production_cleanup_guard_survives_health_integration():
    cases = await audit.router_expiry_cases()
    assert {key: value["admission"] for key, value in cases.items()} == {
        "pending": "wait",
        "confirmed": "granted",
        "new_reap_after_cleanup": "wait",
    }, "No admission may cross an unacknowledged physical cleanup boundary."
