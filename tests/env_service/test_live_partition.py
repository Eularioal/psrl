"""Exercise live partition ownership, hysteresis and version restoration."""

import asyncio
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.gen.gpu_handoff import gpu_control

from tests.env_service.test_gpu_commands import Remote
from tests.env_service.test_gpu_sharing import Model, shared_setup

spec = importlib.util.spec_from_file_location(
    "live_partition", Path(__file__).parents[2] / "psrl/workers/gen/rollout_coordination/env_partition.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

CONFIG = dict(
    interval_s=2,
    min_replicas=1,
    request_capacity=16,
    shrink_fraction=0.65,
    restore_fraction=0.9,
    shrink_window_s=6,
    restore_window_s=4,
    cooldown_s=60,
    transition_timeout_s=2,
    sleep_memory_mb=1024,
    stats_max_age_s=10,
)


def test_sustained_demand_and_restore_hysteresis():
    policy = module.PartitionPolicy(CONFIG)
    for t in (0, 2, 4):
        assert policy.observe(t, 2, 0, 8, 0, True) is None
    assert policy.observe(6, 2, 0, 8, 0, True) == "borrow"
    policy.last_change = 6
    assert policy.observe(8, 1, 1, 0, 0, False) is None
    assert policy.observe(10, 1, 1, 16, 2, True) is None
    assert policy.observe(12, 1, 1, 16, 2, True) is None
    assert policy.observe(14, 1, 1, 16, 2, True) == "restore"


def test_gaps_reset_low_load_evidence_and_never_borrow_last_replica():
    policy = module.PartitionPolicy(CONFIG)
    assert policy.observe(0, 2, 0, 0, 0, True) is None
    assert policy.observe(10, 2, 0, 0, 0, True) is None
    assert policy.observe(12, 1, 1, 0, 0, True) is None
    assert policy.low_since is None


class PartitionModel(Model):
    def __init__(self):
        super().__init__()
        self.trace = []
        self.version = 0

    async def wait_for_requests_to_drain(self):
        self.trace.append("drain")

    @gpu_control
    async def pause_generation(self, **kwargs):
        self.trace.append("pause")

    async def close_grpc_generate_admission(self):
        self.trace.append("close")

    @gpu_control
    async def nixl_sleep(self, level):
        self.trace.append("sleep")
        self.sleeping = True

    @gpu_control
    async def nixl_wake_up(self):
        self.trace.append("wake")
        self.sleeping = False

    @gpu_control
    async def pause_for_sync(self):
        self._gpu_sync_active = True

    @gpu_control
    async def pull_model_for_sync(self, version):
        self.version = version + 1
        self.trace.append("pull")
        return self.version

    @gpu_control
    async def resume_after_sync(self):
        self.trace.append("resume")
        self._gpu_sync_active = False


class Coordinator(module.EnvPartitionMixin):
    def __init__(self):
        self.model = PartitionModel()
        self.server_handles = {"a": Remote(self.model)}
        self._env_gpu_transition_lock = asyncio.Lock()
        self._env_gpu_reservations = {}
        self._env_partition_units = {"a": "unit-a"}
        self._env_partition_states = {"a": "resident"}
        self._env_partition_config = CONFIG
        self._env_partition_events = []
        self._env_partition_policy = module.PartitionPolicy(CONFIG)
        self.router = SimpleNamespace(
            set_partition_unit_state=AsyncMock(),
            verify_partition_sleep=AsyncMock(
                return_value={"non_environment_mb": 128, "owned_mb": {}, "devices": {2: {"used_mb": 128}}}
            ),
        )
        self._env_partition_router = Remote(self.router)
        self._gateway_post_json = AsyncMock(return_value={"updated": 1})
        self._publish_weight_version_updates = AsyncMock()
        self.set_rollout_instance_model_version = lambda item, version: self.versions.update({item: version})
        self._wait_for_replica_syncs = AsyncMock()
        self.instance_to_version_after_sync = {}
        self.versions = {}
        self.ps_model_version = 4
        self.global_paused = False

    def _expand_replica_instance_ids(self, ids):
        return ids

    def _routing_pause_controller(self):
        return self

    @asynccontextmanager
    async def pause(self):
        self.global_paused = True
        try:
            yield
        finally:
            self.global_paused = False


@pytest.mark.asyncio
async def test_borrow_waits_for_physical_return_without_global_routing_pause():
    c = Coordinator()
    await c.model.acquire_env_command("old")
    transition = asyncio.create_task(c._partition_transition("a", "borrow"))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not transition.done()
    assert not c.global_paused
    assert c._env_gpu_reservations == {"a": "partition"}
    assert await c.model.acquire_env_command("new") is None
    await c.model.release_env_command("old")
    await transition
    c.router.verify_partition_sleep.assert_awaited_once()
    assert c._env_partition_states["a"] == "borrowed"
    assert await c.model.acquire_env_command("env") == "sleeping"
    restore = asyncio.create_task(c._partition_transition("a", "restore"))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not restore.done()
    assert "wake" not in c.model.trace
    assert not c.global_paused
    await c.model.release_env_command("env")
    await restore
    assert c.model.trace[-3:] == ["wake", "pull", "resume"]
    assert c.versions[("a", 0)] == 5
    assert c.instance_to_version_after_sync[("a", 0)] == 5
    assert not c._env_gpu_reservations
    assert c._env_partition_states["a"] == "resident"


@pytest.mark.asyncio
async def test_failed_physical_release_quarantines_both_admission_paths():
    c = Coordinator()
    c.router.verify_partition_sleep.side_effect = RuntimeError("GPU still occupied.")
    with pytest.raises(RuntimeError, match="occupied"):
        await c._partition_transition("a", "borrow")
    assert c._env_partition_states["a"] == "quarantined"
    assert c._env_gpu_reservations["a"] == "partition"
    assert await c.model.acquire_env_command("new") is None
    assert c.router.set_partition_unit_state.await_args.args == ("unit-a", "quarantined")


@pytest.mark.asyncio
async def test_router_closes_commands_during_transition_and_preserves_host_unit():
    ledger, scheduler, node, (lease, _, _), model = shared_setup()
    model.sleeping = True
    scheduler.borrowed_command_cap = 4
    await scheduler.set_partition_state("u", "draining")
    assert (await scheduler.acquire(lease.lease_id, "c", "r"))["status"] == "wait"
    assert not node.active
    assert scheduler.partition_snapshot()["u"]["waiting"] == 1
    await scheduler.set_partition_state("u", "borrowed")
    grant = await scheduler.acquire(lease.lease_id, "c", "r")
    assert grant["status"] == "granted"
    assert lease.unit_id == "u"
    await scheduler.release(lease.lease_id, "r", grant["epoch"])


@pytest.mark.asyncio
async def test_sync_overlap_does_not_override_partition_transition():
    from tests.env_service.test_sync_overlap_experiment import Model as SyncModel

    model = SyncModel()
    await model.acquire_env_command("a")
    await model.set_env_partition_state("restoring")
    sync = asyncio.create_task(model.pause_for_sync())
    await asyncio.sleep(0)
    assert not sync.done()
    await model.release_env_command("a")
    await asyncio.wait_for(sync, 1)


@pytest.mark.asyncio
async def test_sync_selection_is_serialized_and_excludes_borrowed_replicas():
    path = (
        Path(__file__).parents[2] / "psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py"
    )
    sync_spec = importlib.util.spec_from_file_location("partition_sync", path)
    sync_module = importlib.util.module_from_spec(sync_spec)
    sync_spec.loader.exec_module(sync_module)
    c = SimpleNamespace(
        _env_gpu_transition_lock=asyncio.Lock(),
        _env_gpu_reservations={"a": "partition"},
        _sync_with_ps_locked=AsyncMock(),
    )
    await c._env_gpu_transition_lock.acquire()
    sync = asyncio.create_task(sync_module.SyncAndMigrateMixin.sync_with_ps(c, [("a", 0), ("b", 0)]))
    await asyncio.sleep(0)
    c._sync_with_ps_locked.assert_not_awaited()
    c._env_gpu_transition_lock.release()
    await sync
    c._sync_with_ps_locked.assert_awaited_once_with([("b", 0)], False, True)


@pytest.mark.asyncio
async def test_borrowed_card_is_preferred_over_resident_surplus():
    from dataclasses import replace

    from tests.env_service.test_command_placement import setup

    ledger, scheduler, _, (lease, _, _), _ = setup()
    models = [Model(), Model()]
    for rank, unit_id in enumerate(("u0", "u1")):
        ledger._units[unit_id] = replace(
            ledger._units[unit_id], shared_rollout_replica_rank=rank, model_headroom_mb=10
        )
        scheduler.shared_servers[unit_id] = Remote(models[rank])
    models[1].sleeping = True
    scheduler.borrowed_command_cap = 2
    scheduler.placement_policy = "first_fit"
    await scheduler.set_partition_state("u1", "borrowed")
    grant = await scheduler.acquire(lease.lease_id, "c", "r")
    assert grant["unit_id"] == "u1"
    assert not models[0]._gpu_control.shared_commands
    assert lease.unit_id == "u0"
    await scheduler.release(lease.lease_id, "r", grant["epoch"])


@pytest.mark.asyncio
async def test_overlap_keeps_old_commands_and_admits_during_request_drain():
    c = Coordinator()
    c._env_partition_config = dict(CONFIG, handover_overlap=True)
    draining, finish_drain = asyncio.Event(), asyncio.Event()

    async def drain():
        draining.set()
        await finish_drain.wait()

    c.model.wait_for_requests_to_drain = drain
    await c.model.acquire_env_command("old")
    transition = asyncio.create_task(c._partition_transition("a", "borrow"))
    await draining.wait()
    assert await c.model.acquire_env_command("during-drain") == "resident"
    finish_drain.set()
    await asyncio.wait_for(transition, 1)
    assert c.model._gpu_control.shared_commands == {"old": "resident", "during-drain": "resident"}
    assert c._env_partition_states["a"] == "borrowed"
    sleep = next(e for e in c.model._gpu_control.control_events if e["method"] == "nixl_sleep")
    assert len(sleep["shared_at_entry"]) == 2
    restore = asyncio.create_task(c._partition_transition("a", "restore"))
    await asyncio.sleep(0)
    assert not restore.done()
    assert await c.model.acquire_env_command("late") is None
    await c.model.release_env_command("old")
    await c.model.release_env_command("during-drain")
    await asyncio.wait_for(restore, 1)


@pytest.mark.asyncio
async def test_overlap_blocks_new_commands_until_physical_sleep_verified():
    c = Coordinator()
    c._env_partition_config = dict(CONFIG, handover_overlap=True)
    entered, release = asyncio.Event(), asyncio.Event()

    async def verify(*args):
        entered.set()
        await release.wait()
        return c.router.verify_partition_sleep.return_value

    c.router.verify_partition_sleep.side_effect = verify
    await c.model.acquire_env_command("old")
    transition = asyncio.create_task(c._partition_transition("a", "borrow"))
    await entered.wait()
    assert await c.model.acquire_env_command("new") is None
    assert c._env_partition_states["a"] == "draining"
    release.set()
    await transition
    assert await c.model.acquire_env_command("new") == "sleeping"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "used,owned,errors,passes",
    [(9000, 8500, {}, True), (9000, 7000, {}, False), (9000, 8500, {"x": "unknown"}, False), (9000, 10000, {}, False)],
)
async def test_sleep_verification_deducts_only_measured_environment_memory(used, owned, errors, passes):
    from psrl.workers.env_service.router import EnvRouterCore

    lock = asyncio.Lock()
    memory = {"devices": {2: {"used_mb": used}}, "owned_mb": {"lease": owned}, "attribution_errors": errors}
    node = SimpleNamespace(gpu_command_memory_snapshot=AsyncMock(return_value=memory))
    request = SimpleNamespace(lease_id="lease", container_id="container", epoch=1, unit_id="u", state="active")
    fake = SimpleNamespace(
        _ledger=SimpleNamespace(_units={"u": SimpleNamespace(node_ip="node", gpu_indices=(2,))}),
        _gpu_commands=SimpleNamespace(_lock=lambda _: lock, requests={"r": request}),
        _gpu_command_node=AsyncMock(return_value=Remote(node)),
    )
    if passes:
        result = await EnvRouterCore.verify_partition_sleep(fake, "u", 1024)
        assert result["non_environment_mb"] == 500
    else:
        with pytest.raises(RuntimeError):
            await EnvRouterCore.verify_partition_sleep(fake, "u", 1024)
    node.gpu_command_memory_snapshot.assert_awaited_once_with((2,), [("lease", "container", 1)])


@pytest.mark.asyncio
async def test_live_idle_load_can_accumulate_without_iteration_statistics():
    c = Coordinator()
    c._env_partition_states = {"a": "resident", "b": "resident"}
    c._env_partition_config = dict(CONFIG, shrink_window_s=0.001)
    c._env_partition_policy = module.PartitionPolicy(c._env_partition_config)
    c.instance_to_model_version = {("a", 0): 4, ("b", 0): 4}
    c.replica_sync_tasks = {}
    c._gateway_get_json = AsyncMock(return_value={"pending_request_num": 0})
    c.router.partition_snapshot = AsyncMock(return_value={"u": {"waiting": 1, "shared_active": 0}})
    loads = SimpleNamespace(env_partition_load=AsyncMock(return_value=0))
    c.server_handles = {"a": Remote(loads), "b": Remote(loads)}
    c._partition_transition = AsyncMock()
    await c._env_partition_tick()
    await asyncio.sleep(0.002)
    await c._env_partition_tick()
    c._partition_transition.assert_awaited_once_with("a", "borrow")
    loads.env_partition_load.side_effect = TimeoutError("unavailable")
    await c._env_partition_tick()
    assert c._env_partition_policy.low_since is None
    assert c._env_partition_events[-1]["skipped"] == "live_load_unavailable"
