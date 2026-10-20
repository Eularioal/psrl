"""Check command reservations, physical acknowledgements and stale request fencing."""

import asyncio
from types import SimpleNamespace

import pytest
from psrl.workers.env_service.command_gpu import GpuCommandScheduler
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.types import EnvDemand, EnvUnit


class Node:
    """Model independently acknowledged GPU permission changes."""

    def __init__(self):
        self.active = set()
        self.busy = False
        self.gate = None
        self.return_gate = None
        self.return_started = asyncio.Event()
        self.snapshots = 0

    async def gpu_lending_snapshot(self, indices):
        self.snapshots += 1
        return {0: {"used_mb": len(self.active) * 50, "total_mb": 95, "uuid": "gpu", "pids": []}}

    async def grant_gpu_command(self, lease_id, container_id, indices, epoch):
        if self.gate is not None:
            await self.gate.wait()
        self.active.add(lease_id)

    async def revoke_gpu_command(self, lease_id, container_id, indices, epoch, telemetry_indices):
        self.return_started.set()
        if self.return_gate is not None:
            await self.return_gate.wait()
        if self.busy:
            raise RuntimeError("Live CUDA process remains.")
        self.active.discard(lease_id)
        return await self.gpu_lending_snapshot(indices)

    async def remove_gpu_lease(self, lease_id):
        self.active.discard(lease_id)


class Remote:
    def __init__(self, node):
        self.node = node

    def __getattr__(self, name):
        return SimpleNamespace(remote=getattr(self.node, name))


def setup():
    clock = [0.0]
    unit = EnvUnit("u", "node", 8, 32, (0,), 4, 80, gpu_reservation="command")
    ledger = EnvUnitLedger([unit], NodeRules.build(policy="any"), clock=lambda: clock[0], lease_ttl_s=10)
    demand = EnvDemand(2, 8, True, 51)
    leases = [ledger.try_admit(str(i), demand, "node").lease for i in range(3)]
    node = Node()

    async def get_node(ip):
        return Remote(node)

    return ledger, GpuCommandScheduler(ledger, get_node), node, leases, clock


def test_cpu_work_survives_gpu_return_and_fifo_reacquisition():
    async def run():
        ledger, scheduler, node, (a, b, c), _ = setup()
        assert ledger.snapshot()[0].memory_used_mb == 24, "Host memory remains reserved."
        assert ledger.snapshot()[0].declared_gpu_memory_mb == 0, "Idle GPU phases must not charge memory."
        first = await scheduler.acquire(a.lease_id, "a", "a1")
        assert first["status"] == "granted", "First command must start."
        assert (await scheduler.acquire(b.lease_id, "b", "b1"))["status"] == "wait", "B must wait."
        await scheduler.release(a.lease_id, "a1", first["epoch"])
        assert a.lease_id in ledger._leases and not ledger._leases[a.lease_id].gpu_active, (
            "Returning GPU must preserve the host lease."
        )
        assert (await scheduler.acquire(a.lease_id, "a", "a2"))["status"] == "wait", "A must rejoin behind B."
        second = await scheduler.acquire(b.lease_id, "b", "b1")
        assert second["status"] == "granted", "B must receive the released GPU."
        await scheduler.release(b.lease_id, "b1", second["epoch"])
        resumed = await scheduler.acquire(a.lease_id, "a", "a2")
        assert resumed["epoch"] > first["epoch"], "Reacquisition must use a newer epoch."
        with pytest.raises(RuntimeError, match="stale"):
            await scheduler.release(a.lease_id, "a2", first["epoch"])
        assert ledger._leases[a.lease_id].gpu_active, "A stale release must not free the active phase."
        await scheduler.release(a.lease_id, "a1", first["epoch"])
        assert ledger._leases[a.lease_id].gpu_active, "An old completed release must not affect a newer command."
        await scheduler.release(a.lease_id, "a2", resumed["epoch"])
        assert not node.active, "Every command must return physical access."

    asyncio.run(run())


def test_unconfirmed_return_keeps_budget_and_blocks_other_commands():
    async def run():
        ledger, scheduler, node, (a, b, _), _ = setup()
        result = await scheduler.acquire(a.lease_id, "a", "a1")
        node.busy = True
        with pytest.raises(RuntimeError, match="CUDA"):
            await scheduler.release(a.lease_id, "a1", result["epoch"])
        assert ledger.snapshot()[0].declared_gpu_memory_mb == 51, "Uncertain CUDA ownership must remain charged."
        with pytest.raises(RuntimeError, match="cleanup"):
            await scheduler.acquire(b.lease_id, "b", "b1")
        await scheduler.finish(a.lease_id, a.unit_id)
        ledger.release(a.lease_id)
        node.busy = False
        assert (await scheduler.acquire(b.lease_id, "b", "b1"))["status"] == "granted", (
            "Cleanup must recover capacity."
        )

    asyncio.run(run())


def test_cancel_before_acquire_rejects_late_delivery():
    async def run():
        _, scheduler, node, (a, _, _), _ = setup()
        await scheduler.cancel(a.lease_id, "late")
        with pytest.raises(RuntimeError):
            await scheduler.acquire(a.lease_id, "a", "late")
        assert not node.active, "A cancelled request must never activate devices."

    asyncio.run(run())


def test_expiry_during_activation_keeps_unit_blocked_until_cleanup():
    async def run():
        ledger, scheduler, node, (a, _, _), clock = setup()
        node.gate = asyncio.Event()
        task = asyncio.create_task(scheduler.acquire(a.lease_id, "a", "a1"))
        await asyncio.sleep(0)
        clock[0] = 11
        expired = ledger.reap()
        node.gate.set()
        with pytest.raises(RuntimeError, match="expired"):
            await task
        assert "u" in ledger._blocked_units, "Expiry must retain a physical cleanup fence."
        for lease in expired:
            await scheduler.finish(lease.lease_id, lease.unit_id)
        assert not node.active and "u" not in ledger._blocked_units, "All expired containers must be removed."

    asyncio.run(run())


def test_device_controller_mount_root_in_nested_execution_node():
    from pathlib import Path

    from psrl.workers.env_service.command_gpu_node import DeviceCgroupFence

    root = Path("/sys/fs/cgroup/devices")
    nested = "42 1 0:39 /kubepods/pod/node /sys/fs/cgroup/devices rw - cgroup cgroup rw,devices"
    host = "42 1 0:39 / /sys/fs/cgroup/devices rw - cgroup cgroup rw,devices"
    assert DeviceCgroupFence.mount_root(root, nested) == Path("/kubepods/pod/node")
    assert DeviceCgroupFence.mount_root(root, host) == Path("/")
    with pytest.raises(RuntimeError, match="mount root"):
        DeviceCgroupFence.mount_root(root, "42 1 0:39 / /sys/fs/cgroup rw - cgroup2 cgroup rw")


def test_cancelled_queue_head_does_not_block_next_request():
    async def run():
        ledger, scheduler, _, (a, b, c), _ = setup()
        active = await scheduler.acquire(a.lease_id, "a", "a")
        assert (await scheduler.acquire(b.lease_id, "b", "b"))["status"] == "wait"
        assert (await scheduler.acquire(c.lease_id, "c", "c"))["status"] == "wait"
        await scheduler.cancel(b.lease_id, "b")
        await scheduler.release(a.lease_id, "a", active["epoch"])
        assert (await scheduler.acquire(c.lease_id, "c", "c"))["status"] == "granted"
        assert ledger.snapshot()[0].gpu_commands_waiting == 0

    asyncio.run(run())


def test_command_overlay_denies_gpu_access_at_container_creation():
    from psrl.workers.env_service.overlay import DriverLibs, build_lease_overlay
    from psrl.workers.env_service.probe import GpuDevice

    _, _, _, (lease, _, _), _ = setup()
    overlay = build_lease_overlay(
        lease,
        DriverLibs("535", ()),
        device_exists=lambda path: False,
        gpu_devices={0: GpuDevice(index=0, minor=7, uuid="gpu")},
    )["services"]["main"]
    assert overlay["devices"] == ["/dev/nvidia7:/dev/nvidia7:m"]


def test_command_units_reject_borrowed_rollout_devices():
    with pytest.raises(ValueError, match="dedicated"):
        EnvUnit("u", "node", 8, 32, (0,), 1, 80, rollout_replica_rank=0, gpu_reservation="command")


def test_security_options_are_written_once_after_harbor_overlays(tmp_path, monkeypatch):
    import yaml
    from harbor.environments.docker.docker import DockerEnvironment
    from psrl.workers.env_service.docker import ManagedDockerEnvironment

    source = tmp_path / "lease.yaml"
    source.write_text("services:\n  main:\n    labels: [psrl.env_lease=owned]\n  sidecar: {}\n")
    monkeypatch.setattr(DockerEnvironment, "_docker_compose_paths", property(lambda self: [source]))
    env = object.__new__(ManagedDockerEnvironment)
    env._lease_overlay = yaml.safe_load(source.read_text())
    env._actor_id = "actor"
    env._gpu_commands = {"lease_id": "owned"}
    env._ownership_path = tmp_path / "ownership.yaml"
    paths = env._docker_compose_paths
    documents = [yaml.safe_load(path.read_text()) for path in paths]
    options = [x for d in documents for x in d["services"]["main"].get("security_opt", [])]
    assert options == ["no-new-privileges:true"], "Repeated security options break Docker Compose validation."
    assert documents[-1]["services"]["sidecar"]["labels"]["psrl.env_lease"] == "owned"


def test_return_is_owned_without_blocking_queue_or_releasing_budget():
    async def run():
        ledger, scheduler, node, (a, b, _), _ = setup()
        node.return_gate = asyncio.Event()
        grant = await scheduler.acquire(a.lease_id, "a", "a1")
        result = await scheduler.begin_return(a.lease_id, "a1", grant["epoch"])
        assert result["state"] == "returning"
        await node.return_started.wait()
        snapshots = node.snapshots
        assert (await asyncio.wait_for(scheduler.acquire(b.lease_id, "b", "b1"), 0.2))["status"] == "wait"
        for _ in range(10):
            assert (await scheduler.acquire(b.lease_id, "b", "b1"))["status"] == "wait"
        assert node.snapshots == snapshots, "A full ledger must reject polling without competing for device I/O."
        assert ledger.snapshot()[0].declared_gpu_memory_mb == 51
        assert (await scheduler.acquire(a.lease_id, "a", "a2"))["status"] == "wait"
        waiter = asyncio.create_task(scheduler.release(a.lease_id, "a1", grant["epoch"]))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not scheduler.returns["a1"].cancelled(), "Client cancellation must not cancel service cleanup."
        node.return_gate.set()
        await scheduler.release(a.lease_id, "a1", grant["epoch"])
        assert (await scheduler.acquire(b.lease_id, "b", "b1"))["status"] == "granted"
        assert not ledger._leases[a.lease_id].gpu_active and ledger._leases[b.lease_id].gpu_active

    asyncio.run(run())


@pytest.mark.parametrize("expire", [False, True])
def test_cleanup_joins_return_before_retiring_state(expire):
    async def run():
        ledger, scheduler, node, (a, _, _), clock = setup()
        node.return_gate = asyncio.Event()
        grant = await scheduler.acquire(a.lease_id, "a", "a1")
        await scheduler.begin_return(a.lease_id, "a1", grant["epoch"])
        await node.return_started.wait()
        if expire:
            clock[0] = 11
            ledger.reap()
        finishing = asyncio.create_task(scheduler.finish(a.lease_id, a.unit_id))
        await asyncio.sleep(0)
        assert not finishing.done(), "Cleanup must join the physical return before deleting its state."
        with pytest.raises(RuntimeError, match="closing|live"):
            await scheduler.acquire(a.lease_id, "a", "a2")
        node.return_gate.set()
        await finishing
        assert not node.active and not scheduler.returns and not scheduler.requests

    asyncio.run(run())


def test_return_failure_is_queryable_and_timeout_keeps_quota():
    async def run():
        ledger, scheduler, node, (a, b, _), _ = setup()
        node.return_gate = asyncio.Event()
        scheduler.return_timeout_s = 0.01
        grant = await scheduler.acquire(a.lease_id, "a", "a1")
        await scheduler.begin_return(a.lease_id, "a1", grant["epoch"])
        with pytest.raises(RuntimeError, match="TimeoutError"):
            await scheduler.release(a.lease_id, "a1", grant["epoch"])
        status = scheduler.return_status(a.lease_id, "a1", grant["epoch"])
        assert status["state"] == "quarantined" and status["returned_at"] is None
        assert ledger.snapshot()[0].declared_gpu_memory_mb == 51
        with pytest.raises(RuntimeError, match="TimeoutError"):
            await scheduler.wait_returns(a.lease_id)
        with pytest.raises(RuntimeError, match="cleanup"):
            await scheduler.acquire(b.lease_id, "b", "b1")
        await scheduler.finish(a.lease_id, a.unit_id)
        assert not node.active

    asyncio.run(run())


def test_invalid_return_telemetry_cannot_clear_declared_capacity():
    from unittest.mock import AsyncMock

    async def run():
        ledger, scheduler, node, (a, _, _), _ = setup()
        grant = await scheduler.acquire(a.lease_id, "a", "a1")
        node.revoke_gpu_command = AsyncMock(return_value={0: {}})
        with pytest.raises(RuntimeError, match="KeyError"):
            await scheduler.release(a.lease_id, "a1", grant["epoch"])
        assert ledger.snapshot()[0].declared_gpu_memory_mb == 51
        assert a.unit_id in ledger._blocked_units
        await scheduler.finish(a.lease_id, a.unit_id)

    asyncio.run(run())


@pytest.mark.parametrize("fail", [False, True])
def test_environment_can_plan_before_return_but_joins_before_next_command(monkeypatch, fail):
    from unittest.mock import AsyncMock

    from harbor.environments.docker.docker import DockerEnvironment
    from psrl.workers.env_service.docker import ManagedDockerEnvironment

    async def run():
        ledger, scheduler, node, (a, _, _), _ = setup()
        node.return_gate = asyncio.Event()
        node.busy = fail
        router = SimpleNamespace(
            acquire_gpu_command=SimpleNamespace(remote=scheduler.acquire),
            release_gpu_command=SimpleNamespace(remote=scheduler.release),
            begin_gpu_command_return=SimpleNamespace(remote=scheduler.begin_return),
            cancel_gpu_command=SimpleNamespace(remote=scheduler.cancel),
        )
        monkeypatch.setattr("ray.get_actor", lambda name: router)
        execute = AsyncMock(return_value=SimpleNamespace(stdout="stable", return_code=0))
        stop = AsyncMock()
        monkeypatch.setattr(DockerEnvironment, "exec", execute)
        monkeypatch.setattr(DockerEnvironment, "stop", stop)
        env = object.__new__(ManagedDockerEnvironment)
        env._gpu_commands = {"router_name": "test", "lease_id": a.lease_id}
        env._gpu_command_lock = asyncio.Lock()
        env._pending_gpu_return = None
        env._run_docker_compose_command = AsyncMock(return_value=SimpleNamespace(stdout="a"))
        result = await asyncio.wait_for(env.exec_gpu("command1", overlap_return=True), 0.2)
        assert result.stdout == "stable" and ledger.snapshot()[0].declared_gpu_memory_mb == 51
        model = AsyncMock(return_value="next plan")
        assert await model() == "next plan"
        following = asyncio.create_task(env.exec_gpu("command2"))
        await asyncio.sleep(0)
        assert execute.await_count == 1 and not following.done()
        node.return_gate.set()
        if fail:
            with pytest.raises(RuntimeError, match="CUDA"):
                await following
            with pytest.raises(RuntimeError, match="CUDA"):
                await env.stop(delete=True)
            assert execute.await_count == 1, "A deferred failure must prevent the next command."
        else:
            await following
            await env.stop(delete=True)
            assert execute.await_count == 2
        stop.assert_awaited_once_with(delete=True)
        await scheduler.finish(a.lease_id, a.unit_id)

    asyncio.run(run())
