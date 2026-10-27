"""Exercise GPU handoff ordering, fencing, stale requests and physical cleanup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.env_service.lending import GpuLendingBinding, GpuPhaseClient
from psrl.workers.env_service.lending_node import returned_to_baseline
from psrl.workers.env_service.router import EnvRouterCore
from psrl.workers.env_service.types import EnvDemand
from psrl.workers.gen.gpu_handoff import GpuControlGate, GpuHandoffMixin, gpu_control


class Remote:
    """Preserve async call boundaries without starting a Ray cluster."""

    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        return SimpleNamespace(remote=getattr(self.target, name))


class Engine(GpuHandoffMixin):
    """Supply model operations to the real server handoff mixin."""

    def __init__(self):
        self._gpu_control = GpuControlGate()
        self._gpu_sync_active = False
        self._gpu_weights_valid = True
        self._last_env_owner = None
        self._env_handoff_ready = False
        self.sleeping = False
        self.wakes = 0
        self.pulls = 0
        self.engine = SimpleNamespace(is_sleeping=AsyncMock(side_effect=lambda: self.sleeping))
        self.grpc_servicer = SimpleNamespace(pause_generation_admission=lambda: None)
        self.close_grpc_generate_admission = AsyncMock()
        self.pause_generation = AsyncMock()

    @gpu_control
    async def nixl_sleep(self, level):
        assert level == 1
        self.sleeping = True

    @gpu_control
    async def nixl_wake_up(self):
        self.sleeping = False
        self.wakes += 1

    @gpu_control
    async def pull_model_for_sync(self, version):
        self.pulls += 1
        self._gpu_weights_valid = True

    @gpu_control
    async def resume_after_sync(self):
        self._gpu_sync_active = False


class Node:
    """Model independently observable process ownership and container freezing."""

    def __init__(self, engine):
        self.engine = engine
        self.paused = True
        self.busy = False
        self.removed = False
        self.snapshot = {0: {"uuid": "GPU-a", "used_mb": 200, "total_mb": 96000, "pids": [10]}}

    async def gpu_lending_snapshot(self, indices):
        assert self.engine.sleeping
        return self.snapshot

    async def resume_gpu_lease(self, owner):
        assert self.engine.sleeping
        self.paused = False

    async def freeze_gpu_lease(self, owner, baseline, tolerance, timeout):
        self.paused = True
        if self.busy:
            raise TimeoutError("Background CUDA process still owns memory.")

    async def remove_gpu_lease(self, owner):
        self.paused = True
        self.busy = False
        self.removed = True

    async def confirm_gpu_return(self, indices, baseline, tolerance):
        assert self.removed and not self.busy


def setup():
    config = {
        "node_policy": "cpu_only",
        "units": [
            {
                "unit_id": "borrow",
                "node_ip": "node",
                "cpu_quota": 4,
                "memory_mb": 4096,
                "gpu_indices": [0],
                "gpu_memory_mb": 48000,
                "max_containers": 1,
                "rollout_replica_rank": 0,
            }
        ],
    }
    router = EnvRouterCore(config, ["node"])
    engine = Engine()
    node = Node(engine)
    binding = GpuLendingBinding(Remote(engine), Remote(node), router._ledger._units["borrow"], {})
    router._gpu_bindings["borrow"] = binding
    router._ledger._blocked_units.clear()
    return router, engine, node, binding


@pytest.mark.asyncio
async def test_repeated_docker_model_docker_cycles_preserve_lease_and_fence_wakes():
    router, engine, node, binding = setup()
    admission = await router.try_admit("episode", EnvDemand(needs_gpu=True), "node", {0: 90000})
    lease = admission.lease
    assert lease.gpu_lending
    await router.set_gpu_phase(lease.lease_id, "env", 0)
    client = GpuPhaseClient(Remote(router), lease.lease_id)
    for _ in range(3):
        assert engine.sleeping and not node.paused
        competing_wake = asyncio.create_task(engine.nixl_wake_up())
        await asyncio.sleep(0)
        assert not competing_wake.done()
        await client.set_phase("model")
        await asyncio.wait_for(competing_wake, 1)
        assert node.paused and not engine.sleeping
        assert (await router.snapshot())[0].leases_active == 1
        await client.set_phase("env")
    await router.release(lease.lease_id)
    assert node.removed and not engine.sleeping
    assert binding.owner is None and (await router.snapshot())[0].leases_active == 0


@pytest.mark.asyncio
async def test_background_cuda_blocks_wake_and_capacity_until_cleanup():
    router, engine, node, binding = setup()
    lease = (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).lease
    await router.set_gpu_phase(lease.lease_id, "env", 0)
    node.busy = True
    remove = node.remove_gpu_lease
    node.remove_gpu_lease = AsyncMock(side_effect=RuntimeError("Docker unavailable."))
    with pytest.raises(TimeoutError):
        await router.set_gpu_phase(lease.lease_id, "model", 1)
    assert binding.phase == "quarantined" and engine.sleeping and engine.wakes == 0
    assert (await router.try_admit("other", EnvDemand(needs_gpu=True), "node")).status == "reject"
    node.remove_gpu_lease = remove
    await router.release(lease.lease_id)
    assert node.removed and engine.wakes == 1


@pytest.mark.asyncio
async def test_stale_phase_and_unknown_owner_cannot_reopen_container():
    router, engine, node, binding = setup()
    lease = (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).lease
    await router.set_gpu_phase(lease.lease_id, "env", 0)
    await router.set_gpu_phase(lease.lease_id, "model", 1)
    with pytest.raises(RuntimeError, match="Stale"):
        await router.set_gpu_phase(lease.lease_id, "env", 0)
    with pytest.raises(RuntimeError, match="borrower"):
        await asyncio.wait_for(engine.return_env_gpu("other"), 1)
    assert node.paused and binding.phase == "model"
    await router.release(lease.lease_id)


@pytest.mark.asyncio
async def test_phase_budget_ends_a_live_heartbeating_episode():
    router, engine, node, binding = setup()
    lease = (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).lease
    await router.set_gpu_phase(lease.lease_id, "env", 0)
    assert await router.heartbeat(lease.lease_id)
    binding.deadline = 0
    await router.reclaim_gpu_phases("node")
    assert node.removed and not engine.sleeping
    with pytest.raises(RuntimeError, match="live"):
        await router.set_gpu_phase(lease.lease_id, "model", 1)


@pytest.mark.asyncio
async def test_cancelled_sleep_keeps_gpu_fenced():
    engine = Engine()
    entered = asyncio.Event()

    async def stalled_sleep(level):
        entered.set()
        await asyncio.Event().wait()

    engine.nixl_sleep = stalled_sleep
    operation = asyncio.create_task(engine.prepare_env_gpu("lease"))
    await entered.wait()
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    with pytest.raises(RuntimeError, match="did not complete"):
        await engine.prepare_env_gpu("lease")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(engine.nixl_wake_up(), 0.01)


@pytest.mark.asyncio
async def test_discarded_weights_are_loaded_before_level_one_backup():
    engine = Engine()
    engine.sleeping = True
    engine._gpu_weights_valid = False
    await engine.prepare_env_gpu("lease")
    assert engine.pulls == 1 and engine.sleeping
    engine.pause_generation.assert_awaited_once_with(mode="wait", clear_cache=False)
    await engine.return_env_gpu("lease")
    await engine.return_env_gpu("lease")
    assert not engine.sleeping


@pytest.mark.asyncio
async def test_scaling_cannot_sleep_the_only_instance_during_a_model_turn():
    router, engine, node, binding = setup()
    lease = (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).lease
    await router.set_gpu_phase(lease.lease_id, "env", 0)
    await router.set_gpu_phase(lease.lease_id, "model", 1)
    scaling_sleep = asyncio.create_task(engine.nixl_sleep(level=1))
    await asyncio.sleep(0)
    assert not scaling_sleep.done() and not engine.sleeping
    await router.release(lease.lease_id)
    await asyncio.wait_for(scaling_sleep, 1)
    assert engine.sleeping and node.removed


@pytest.mark.asyncio
async def test_real_science_agent_hook_brackets_each_model_request(monkeypatch):
    pytest.importorskip("harbor")
    from examples.sciaccel_rl.agent import TruncatingTerminus2
    from harbor.agents.terminus_2 import Terminus2

    router, engine, node, binding = setup()
    lease = (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).lease
    await router.set_gpu_phase(lease.lease_id, "env", 0)
    agent = object.__new__(TruncatingTerminus2)
    agent._gpu_phase_client = GpuPhaseClient(Remote(router), lease.lease_id)

    async def query(*args, **kwargs):
        assert node.paused and not engine.sleeping
        return "next command"

    monkeypatch.setattr(Terminus2, "_query_llm", query)
    for _ in range(2):
        assert await agent._query_llm() == "next command"
        assert not node.paused and engine.sleeping
    await router.release(lease.lease_id)


@pytest.mark.asyncio
async def test_unbound_borrowing_units_cannot_admit_work():
    router, engine, node, binding = setup()
    router._gpu_bindings.clear()
    router._ledger._blocked_units.add("borrow")
    assert (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).status == "wait"


def test_lending_configuration_rejects_multiple_episode_owners():
    from psrl.workers.env_service.config_check import validate_env_service_config

    router, engine, node, binding = setup()
    from dataclasses import asdict

    unit = asdict(binding.unit)
    unit["max_containers"] = 2
    with pytest.raises(ValueError, match="one container slot"):
        validate_env_service_config({"units": [unit]})


@pytest.mark.asyncio
async def test_late_container_start_blocks_cleanup_and_wake():
    router, engine, node, binding = setup()
    lease = (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).lease
    await router.set_gpu_phase(lease.lease_id, "env", 0)
    await router.begin_gpu_container(lease.lease_id, "start-token")
    with pytest.raises(RuntimeError, match="in flight"):
        await router.release(lease.lease_id)
    assert engine.sleeping and binding.phase == "quarantined"
    await router.end_gpu_container(lease.lease_id, "start-token")
    await router.release(lease.lease_id)
    assert node.removed and not engine.sleeping
    with pytest.raises(RuntimeError, match="live"):
        await router.begin_gpu_container(lease.lease_id, "late-start")


@pytest.mark.asyncio
async def test_container_start_cannot_race_a_return_already_in_progress():
    router, engine, node, binding = setup()
    lease = (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).lease
    await router.set_gpu_phase(lease.lease_id, "env", 0)
    checking = asyncio.Event()
    proceed = asyncio.Event()
    freeze = node.freeze_gpu_lease

    async def slow_freeze(*args):
        checking.set()
        await proceed.wait()
        await freeze(*args)

    node.freeze_gpu_lease = slow_freeze
    returning = asyncio.create_task(router.set_gpu_phase(lease.lease_id, "model", 1))
    await checking.wait()
    with pytest.raises(RuntimeError, match="environment GPU phase"):
        await router.begin_gpu_container(lease.lease_id, "racing-start")
    proceed.set()
    await returning
    assert not engine.sleeping and node.paused
    await router.release(lease.lease_id)


@pytest.mark.asyncio
async def test_queued_phase_request_cannot_revive_a_released_lease():
    router, engine, node, binding = setup()
    lease = (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).lease
    await binding.lock.acquire()
    request = asyncio.create_task(router.set_gpu_phase(lease.lease_id, "env", 0))
    await asyncio.sleep(0)
    router._ledger.release(lease.lease_id)
    binding.lock.release()
    with pytest.raises(RuntimeError, match="outlived"):
        await request
    assert binding.owner is None and not engine.sleeping


@pytest.mark.asyncio
async def test_ambiguous_container_start_remains_quarantined():
    router, engine, node, binding = setup()
    lease = (await router.try_admit("episode", EnvDemand(needs_gpu=True), "node")).lease
    await router.set_gpu_phase(lease.lease_id, "env", 0)
    await router.begin_gpu_container(lease.lease_id, "start-token")
    await router.fail_gpu_container(lease.lease_id, "start-token")
    with pytest.raises(RuntimeError, match="in flight"):
        await router.release(lease.lease_id)
    assert engine.sleeping and binding.phase == "quarantined"


@pytest.mark.asyncio
async def test_node_freezes_only_after_gpu_process_exits(monkeypatch):
    import json

    from psrl.workers.env_service import lending_node

    state = {"paused": False, "busy": True, "unpauses": 0}

    async def command(*args):
        if args[:2] == ("docker", "ps"):
            assert "label=psrl.env_lease=lease" in args
            return "owned-container\n"
        if args[:2] == ("docker", "inspect"):
            return json.dumps([{"Id": "owned-container", "State": {"Running": True, "Paused": state["paused"]}}])
        if args[:2] == ("docker", "pause"):
            state["paused"] = True
        elif args[:2] == ("docker", "unpause"):
            state.update(paused=False, busy=False, unpauses=state["unpauses"] + 1)
        else:
            raise AssertionError(args)
        return ""

    monkeypatch.setattr(lending_node, "run_command", command)
    monkeypatch.setattr(
        lending_node.NvmlProbe,
        "read_gpu_snapshot",
        lambda self, indices: {
            0: {
                "uuid": "GPU-a",
                "used_mb": 500 if state["busy"] else 200,
                "total_mb": 96000,
                "pids": [10, 20] if state["busy"] else [10],
            }
        },
    )
    node = lending_node.GpuLendingNodeMixin()
    baseline = {0: {"uuid": "GPU-a", "used_mb": 200, "pids": [10]}}
    await node.freeze_gpu_lease("lease", baseline, 16, 1)
    assert state == {"paused": True, "busy": False, "unpauses": 1}


@pytest.mark.asyncio
async def test_node_cleanup_waits_for_inflight_freeze_and_rejects_delayed_resume():
    from psrl.workers.env_service.lending_node import GpuLendingNodeMixin

    node = GpuLendingNodeMixin()
    entered = asyncio.Event()
    finish_freeze = asyncio.Event()
    order = []

    async def freeze(*args):
        entered.set()
        await finish_freeze.wait()
        order.append("freeze_finished")

    async def remove(*args):
        order.append("removed")

    node._freeze_gpu_lease = freeze
    node._remove_gpu_lease = remove
    node._resume_gpu_lease = AsyncMock()
    freezing = asyncio.create_task(node.freeze_gpu_lease("lease", {}, 16, 1))
    await entered.wait()
    removing = asyncio.create_task(node.remove_gpu_lease("lease"))
    await asyncio.sleep(0)
    assert not removing.done()
    finish_freeze.set()
    await asyncio.gather(freezing, removing)
    assert order == ["freeze_finished", "removed"]
    with pytest.raises(RuntimeError, match="returned GPU lease"):
        await node.resume_gpu_lease("lease")
    with pytest.raises(RuntimeError, match="returned GPU lease"):
        await node.freeze_gpu_lease("lease", {}, 16, 1)
    node._resume_gpu_lease.assert_not_awaited()


@pytest.mark.asyncio
async def test_production_sync_reloads_discarded_weights_at_same_version():
    import ast
    import logging
    from contextlib import asynccontextmanager, nullcontext
    from pathlib import Path
    from types import MethodType

    source = Path("psrl/workers/gen/vllm_async_server.py").read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PSRL_vLLMHttpServer")
    method = next(
        node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "pull_model_for_sync"
    )

    @asynccontextmanager
    async def transfer_context(*args):
        yield

    namespace = {
        "gpu_control": gpu_control,
        "shared_pull_model_context_async": transfer_context,
        "log_dual_events": lambda *args, **kwargs: nullcontext(),
        "psrl_logger": logging.getLogger(__name__),
        "EventType": SimpleNamespace(PULL="pull"),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "production_sync", "exec"), namespace)
    engine = Engine()
    engine._gpu_weights_valid = False
    engine.curr_rollout_instance_model_version = [7]
    engine.get_instance_num = lambda: 1
    engine.base_worker_id = "replica"
    engine.stat_collector = None
    engine.kv_set_current_version = lambda version: None
    engine.gen_interface = SimpleNamespace(
        ps_manager_handle=SimpleNamespace(
            get_rollout_instance_model_version=SimpleNamespace(remote=AsyncMock(return_value=7))
        )
    )

    async def pull():
        engine.pulls += 1
        engine._gpu_weights_valid = True

    engine.pull_model = pull
    sync = MethodType(namespace["pull_model_for_sync"], engine)
    assert await sync(7) == 7
    assert engine.pulls == 1
    assert await sync(7) == 7
    assert engine.pulls == 1


@pytest.mark.parametrize("change", ["identity", "process", "memory", "missing"])
def test_physical_return_needs_identity_processes_and_memory(change):
    baseline = {0: {"uuid": "GPU-a", "used_mb": 200, "pids": [10]}}
    snapshot = {0: dict(baseline[0])}
    assert returned_to_baseline(snapshot, baseline, 16)
    if change == "identity":
        snapshot[0]["uuid"] = "GPU-b"
    elif change == "process":
        snapshot[0]["pids"] = [10, 20]
    elif change == "memory":
        snapshot[0]["used_mb"] = 500
    else:
        snapshot.clear()
    assert not returned_to_baseline(snapshot, baseline, 16)


@pytest.mark.asyncio
async def test_coordinator_sleep_does_not_pause_routing_for_a_leased_instance():
    import ast
    from pathlib import Path
    from types import MethodType

    tree = ast.parse(Path("psrl/workers/gen/rollout_coordination/coordinator.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RolloutCoordinator")
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "sleep_instances_for_control"
    )
    namespace = {"asyncio": asyncio}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "production_coordinator", "exec"), namespace)
    coordinator = SimpleNamespace(
        _env_gpu_transition_lock=asyncio.Lock(),
        _env_gpu_reservations={"borrowed": "lease"},
        _set_routing_loop_running=AsyncMock(),
        _do_sleep_instance=AsyncMock(),
    )
    sleep = MethodType(namespace["sleep_instances_for_control"], coordinator)
    assert await sleep([("borrowed", 0)]) == {"slept": []}
    coordinator._set_routing_loop_running.assert_not_called()
    assert await sleep([("borrowed", 0), ("other", 0)]) == {"slept": [("other", 0)]}
    coordinator._do_sleep_instance.assert_awaited_once_with("other")
    assert coordinator._set_routing_loop_running.await_args.args == (True,)
