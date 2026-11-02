"""Verify independent job admission and retention of uncertain physical resources."""

import asyncio
from types import SimpleNamespace

import pytest
from psrl.workers.env_service.gpu_jobs import GpuJobManager
from psrl.workers.env_service.parallel_gpu import MpsDomain
from psrl.workers.env_service.types import EnvUnit


def domain(monkeypatch, mode="command"):
    unit = SimpleNamespace(gpu_indices=(0,), gpu_memory_mb=2100, gpu_reservation=mode)
    node = SimpleNamespace(_gpu_probe=lambda: None)
    manager = MpsDomain(node, unit)
    manager.leases["lease"] = SimpleNamespace(demand=SimpleNamespace(gpu_memory_mb=2048), gpu_closing=False)
    monkeypatch.setattr(
        "psrl.workers.env_service.parallel_gpu.read_command_memory",
        lambda *args: {"devices": {0: {"used_mb": 50, "total_mb": 8192}}, "owned_mb": {}},
    )
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["command", "episode"])
async def test_two_fit_third_waits_until_physical_return(monkeypatch, mode):
    manager = domain(monkeypatch, mode)
    rows = [{"job_id": str(i), "lease_id": "lease", "memory_mb": 1024, "state": "queued", "wait": 1} for i in range(3)]
    manager.jobs.update((r["job_id"], r) for r in rows)
    try:
        await manager.admit(rows[0])
        await manager.admit(rows[1])
        waiting = asyncio.create_task(manager.admit(rows[2]))
        await asyncio.sleep(0.06)
        assert not waiting.done()
        rows[0]["state"] = "returning"
        await asyncio.sleep(0.06)
        assert not waiting.done()
        rows[0]["state"] = "completed"
        await asyncio.wait_for(waiting, 0.5)
        assert rows[2]["state"] == "starting"
    finally:
        for row in rows:
            row["state"] = "completed"
        await manager.close()


@pytest.mark.asyncio
async def test_uncertain_cleanup_blocks_domain(monkeypatch):
    manager = domain(monkeypatch)

    async def launch(row):
        raise RuntimeError("Injected transport failure.")

    async def cleanup(row):
        raise RuntimeError("Injected physical return failure.")

    manager.launch, manager.cleanup = launch, cleanup
    row = {"job_id": "a", "lease_id": "lease", "memory_mb": 1024, "state": "queued", "wait": 1, "timeout": 1}
    manager.jobs["a"] = row
    await manager.execute(row)
    assert row["state"] == "quarantined" and row["returned"] is None
    with pytest.raises(RuntimeError, match="unavailable"):
        manager.submit("lease", "container", "b", "true", 1024, 1, 1, None, {})
    row["state"] = "completed"
    await manager.close()


@pytest.mark.asyncio
async def test_job_manager_can_cancel_one_parallel_job():
    entered = []
    release = asyncio.Event()

    async def execute(job):
        entered.append(job.job_id)
        await release.wait()

    manager = GpuJobManager(execute, concurrency=2)
    a, b = [manager.start("true")["job_id"] for _ in range(2)]
    await asyncio.sleep(0)
    assert len(entered) == 2
    assert (await manager.cancel(a))["state"] == "cancelled"
    assert not manager.jobs[b].task.done()
    release.set()
    assert (await manager.wait(b, released=True))["state"] == "completed"


def test_mps_rejects_ambiguous_device_configuration():
    with pytest.raises(ValueError):
        EnvUnit(
            "unit",
            "node",
            1,
            1024,
            gpu_indices=(0, 1),
            gpu_memory_mb=2048,
            gpu_memory_per_device_mb=(1024, 1024),
            gpu_execution_backend="mps",
            mps_image="image",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("termination_fails", [False, True])
async def test_cuda_teardown_runs_thawed_and_kill_requires_ack(monkeypatch, termination_fails):
    manager = domain(monkeypatch)
    state = {"frozen": False, "killed": False, "closed": False}

    class Fence:
        def set_access(self, allowed):
            assert not allowed

        def freeze(self):
            state["frozen"] = True

        def thaw(self):
            state["frozen"] = False

        def processes(self):
            return set() if state["killed"] else {17}

        def kill(self):
            assert state["frozen"]
            state["killed"] = True

        def close(self):
            state["closed"] = True

    def bind(*args):
        assert state["frozen"]
        return SimpleNamespace(client=manager.controller.client)

    def terminate(*args):
        assert not state["frozen"]
        if termination_fails:
            raise RuntimeError("Injected CUDA termination failure.")

    from unittest.mock import AsyncMock

    from psrl.workers.env_service.mps import MpsClient

    manager.controller = SimpleNamespace(bind=bind, terminate=terminate, client=MpsClient(17, 1, 12, "GPU-test", 100))
    manager.control = lambda text: "12" if text == "get_server_list" else "17 1 12 GPU-test 100 command"
    manager.node.gpu_lending_snapshot = AsyncMock(return_value={0: {"pids": []}})
    row = {"job_id": "job", "fence": Fence()}
    if termination_fails:
        with pytest.raises(RuntimeError, match="termination failure"):
            await manager.cleanup(row)
        assert not state["killed"] and not state["closed"] and not state["frozen"]
    else:
        await manager.cleanup(row)
        assert state["killed"] and state["closed"] and not state["frozen"]
    await manager.close()


def test_mps_service_overhead_is_reserved():
    from psrl.workers.env_service.ledger import EnvUnitLedger
    from psrl.workers.env_service.node_policy import NodeRules
    from psrl.workers.env_service.types import EnvDemand

    unit = EnvUnit(
        "unit",
        "node",
        5,
        4096,
        gpu_indices=(0,),
        max_containers=3,
        gpu_memory_mb=4096,
        gpu_reservation="command",
        gpu_execution_backend="mps",
        mps_image="image",
    )
    ledger = EnvUnitLedger([unit], NodeRules.build("any"))
    demand = EnvDemand(2, 1024, True, 1024)
    assert ledger.try_admit("a", demand).lease is not None
    assert ledger.try_admit("b", demand).lease is not None
    assert ledger.try_admit("c", demand).lease is None
    snap = ledger.snapshot()[0]
    assert snap.cpu_used == 5 and snap.memory_used_mb == 2560


@pytest.mark.asyncio
async def test_cancel_during_launch_waits_for_owned_startup(monkeypatch):
    manager = domain(monkeypatch)
    entered, settle = asyncio.Event(), asyncio.Event()
    events = []

    async def launch(row):
        entered.set()
        await settle.wait()
        events.append("startup_settled")
        raise asyncio.CancelledError()

    async def cleanup(row):
        assert events == ["startup_settled"]
        events.append("cleanup")

    manager.launch, manager.cleanup = launch, cleanup
    row = {"job_id": "job", "lease_id": "lease", "memory_mb": 1024, "state": "queued", "wait": 1, "timeout": 1}
    manager.jobs["job"] = row
    task = asyncio.create_task(manager.execute(row))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not events
    settle.set()
    await task
    assert events == ["startup_settled", "cleanup"] and row["state"] == "cancelled"
    await manager.close()


@pytest.mark.asyncio
async def test_expired_mps_lease_cannot_delete_before_cuda_return(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from psrl.workers.env_service import node

    lease = SimpleNamespace(lease_id="lease", unit_id="unit", gpu_execution_backend="mps")
    router = SimpleNamespace(
        reclaim_gpu_phases=SimpleNamespace(remote=AsyncMock()),
        pending_reaped=SimpleNamespace(remote=AsyncMock(return_value=[lease])),
    )
    service = node.EnvNodeService(router, "node")
    service.release_mps_lease = AsyncMock(side_effect=RuntimeError("CUDA teardown unconfirmed."))
    remove = Mock()
    monkeypatch.setattr(node, "force_remove_containers_by_label", remove)
    assert await service.cleanup() == 1
    remove.assert_not_called()
