"""Check independent host charges and local GPU invocation placement through cleanup."""

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from psrl.workers.env_service.command_gpu import GpuCommandScheduler
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.types import EnvDemand, EnvUnit


class Node:
    def __init__(self):
        self.active = {}
        self.external = {0: 0, 1: 0}
        self.fail_return = False
        self.return_gate = None
        self.return_started = asyncio.Event()
        self.grants = []

    async def gpu_lending_snapshot(self, indices):
        return {
            i: {
                "used_mb": self.external[i] + 20 * sum(v == i for v in self.active.values()),
                "total_mb": 100,
                "uuid": f"GPU-{i}",
                "pids": [],
            }
            for i in indices
        }

    async def gpu_command_memory_snapshot(self, indices, commands):
        return {
            "devices": await self.gpu_lending_snapshot(indices),
            "owned_mb": {lease: 20 for lease, _, _ in commands if lease in self.active},
            "attribution_errors": {},
        }

    async def grant_gpu_command(self, lease_id, container_id, indices, epoch, *, candidate_indices=()):
        assert indices[0] in candidate_indices, "Selected card must be mapped into this container."
        self.active[lease_id] = indices[0]
        self.grants.append((lease_id, indices, epoch))

    async def revoke_gpu_command(self, lease_id, container_id, indices, epoch, telemetry_indices):
        self.return_started.set()
        if self.return_gate:
            await self.return_gate.wait()
        if self.fail_return:
            raise RuntimeError("Unreturned GPU process.")
        assert self.active[lease_id] == indices[0], "Return must target the actual selected card."
        del self.active[lease_id]
        return await self.gpu_lending_snapshot(telemetry_indices)

    async def remove_gpu_lease(self, lease_id):
        self.active.pop(lease_id, None)


def setup():
    units = [
        EnvUnit(f"u{i}", "node", 4, 16, (i,), 2, 90, gpu_reservation="command", command_gpu_pool="local")
        for i in range(2)
    ]
    clock = [0.0]
    ledger = EnvUnitLedger(units, NodeRules.build(policy="any"), clock=lambda: clock[0], lease_ttl_s=10)
    leases = [ledger.try_admit(str(i), EnvDemand(2, 8, True, 70), "node").lease for i in range(3)]
    node = Node()

    async def get_node(ip):
        assert ip == "node", "Commands must stay on their session's host."
        return SimpleNamespace(
            **{
                name: SimpleNamespace(remote=getattr(node, name))
                for name in (
                    "gpu_lending_snapshot",
                    "gpu_command_memory_snapshot",
                    "grant_gpu_command",
                    "revoke_gpu_command",
                    "remove_gpu_lease",
                )
            }
        )

    return ledger, GpuCommandScheduler(ledger, get_node), node, leases, clock


@pytest.mark.asyncio
async def test_next_command_moves_without_changing_host_charges_or_workspace_binding():
    ledger, scheduler, node, (a, b, _), _ = setup()
    before = [(s.cpu_used, s.memory_used_mb, s.leases_active) for s in ledger.snapshot()]
    first = await scheduler.acquire(a.lease_id, "a", "a1")
    assert first["unit_id"] == "u0", "Idle ties should retain the original device."
    await scheduler.release(a.lease_id, "a1", first["epoch"])
    node.external[1] = 95
    blocked = await scheduler.acquire(b.lease_id, "b", "b1")
    assert blocked["unit_id"] == "u0", "Fresh pressure must prevent placement on GPU1."
    node.external[1] = 0
    moved = await scheduler.acquire(a.lease_id, "a", "a2")
    assert moved["unit_id"] == "u1" and moved["gpu_uuids"] == ("GPU-1",), "A must use the idle candidate."
    assert (await scheduler.acquire(a.lease_id, "a", "a2")) == moved, "Retries must retain the granted placement."
    assert a.unit_id == ledger._leases[a.lease_id].unit_id == "u0", "Host ownership must remain unchanged."
    after = ledger.snapshot()
    assert [(s.cpu_used, s.memory_used_mb, s.leases_active) for s in after] == before, "Host resources charge once."
    assert [s.declared_gpu_memory_mb for s in after] == [70, 70], "Each active invocation charges its selected GPU."
    assert [s.gpu_commands_active for s in after] == [1, 1], "Snapshot must locate active commands correctly."
    await scheduler.release(a.lease_id, "a1", first["epoch"])
    assert ledger._gpu_allocations[a.lease_id] == "u1", "Late old returns must not release the new allocation."
    with pytest.raises(RuntimeError, match="stale"):
        await scheduler.release(a.lease_id, "a2", first["epoch"])
    await scheduler.release(a.lease_id, "a2", moved["epoch"])
    assert [s.declared_gpu_memory_mb for s in ledger.snapshot()] == [70, 0], "Only the returned device is freed."
    await scheduler.release(b.lease_id, "b1", blocked["epoch"])


@pytest.mark.asyncio
async def test_pool_serializes_admission_and_waits_for_physical_return():
    ledger, scheduler, node, (a, b, c), _ = setup()
    results = await asyncio.gather(*(scheduler.acquire(x.lease_id, x.session_id, x.session_id) for x in (a, b, c)))
    assert [r["status"] for r in results].count("granted") == 2, "The third oversized reservation must queue."
    assert sum(s.declared_gpu_memory_mb for s in ledger.snapshot()) == 140, "Pool locking must prevent overcommit."
    node.return_gate = asyncio.Event()
    await scheduler.begin_return(a.lease_id, a.session_id, results[0]["epoch"])
    await node.return_started.wait()
    assert (await scheduler.acquire(a.lease_id, "a", "next"))["status"] == "wait", "No migration during return."
    assert (await scheduler.acquire(c.lease_id, c.session_id, c.session_id))["status"] == "wait", (
        "Return retains budget."
    )
    node.return_gate.set()
    await scheduler.release(a.lease_id, a.session_id, results[0]["epoch"])
    assert (await scheduler.acquire(a.lease_id, "a", "next"))["status"] == "wait", "Pool FIFO must retain older C."
    assert (await scheduler.acquire(c.lease_id, c.session_id, c.session_id))["status"] == "granted", "C must advance."


@pytest.mark.asyncio
@pytest.mark.parametrize("expire", [False, True])
async def test_moved_command_failure_or_expiry_cleans_actual_gpu(expire):
    ledger, scheduler, node, (a, _, _), clock = setup()
    node.external[0] = 95
    result = await scheduler.acquire(a.lease_id, "a", "moved")
    assert result["unit_id"] == "u1", "The test must exercise a non-home allocation."
    if expire:
        clock[0] = 11
        ledger.reap()
        assert {"u0", "u1"} <= ledger._blocked_units, "Expiry must retain host and selected GPU cleanup barriers."
    else:
        node.fail_return = True
        with pytest.raises(RuntimeError, match="Unreturned"):
            await scheduler.release(a.lease_id, "moved", result["epoch"])
        assert "u1" in ledger._blocked_units, "Failed return must quarantine the selected device."
        assert ledger.snapshot()[1].declared_gpu_memory_mb == 70, "Unconfirmed memory must remain charged."
    await scheduler.finish(a.lease_id, a.unit_id)
    assert a.lease_id not in ledger._gpu_allocations and a.lease_id not in node.active, (
        "Cleanup must remove both owners."
    )
    assert "u1" not in ledger._gpu_cleanup_pending.get(a.lease_id, set()), "Selected GPU barrier must be acknowledged."


def test_candidates_exclude_other_hosts_pools_and_protected_devices():
    base = EnvUnit("home", "a", 4, 16, (0,), 1, 90, gpu_reservation="command", command_gpu_pool="p")
    variants = [
        base,
        EnvUnit.from_mapping({**asdict(base), "unit_id": "remote", "node_ip": "b"}),
        EnvUnit.from_mapping({**asdict(base), "unit_id": "protected", "gpu_indices": [1]}),
        EnvUnit.from_mapping({**asdict(base), "unit_id": "other-pool", "gpu_indices": [2], "command_gpu_pool": "q"}),
    ]
    rules = NodeRules.build(policy="any", train_rollout_gpu_indices={"a": [1]})
    ledger = EnvUnitLedger(variants, rules)
    lease = ledger.try_admit("a", EnvDemand(1, 1, True, 10), "a").lease
    assert lease.gpu_candidate_unit_ids == ("home",), "A local pool must never absorb remote or protected devices."


@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_reservation": "episode"},
        {"shared_rollout_replica_rank": 0},
        {"gpu_indices": (0, 1)},
        {"gpu_execution_backend": "mps"},
    ],
)
def test_pool_configuration_rejects_unsupported_backends(changes):
    with pytest.raises(ValueError, match="Command GPU pools"):
        EnvUnit.from_mapping(
            {
                "unit_id": "u",
                "node_ip": "n",
                "cpu_quota": 4,
                "memory_mb": 16,
                "gpu_indices": (0,),
                "gpu_memory_mb": 90,
                "gpu_reservation": "command",
                "command_gpu_pool": "p",
                **changes,
            }
        )


def test_pool_overlay_maps_candidates_denied_by_default():
    from psrl.workers.env_service.overlay import DriverLibs, build_lease_overlay
    from psrl.workers.env_service.probe import GpuDevice

    _, _, _, (lease, _, _), _ = setup()
    main = build_lease_overlay(
        lease,
        DriverLibs("535", ()),
        device_exists=lambda _: False,
        gpu_devices={0: GpuDevice(0, 7, "a"), 1: GpuDevice(1, 3, "b")},
    )["services"]["main"]
    assert main["devices"] == ["/dev/nvidia7:/dev/nvidia7:m", "/dev/nvidia3:/dev/nvidia3:m"], (
        "The container must discover pool candidates without receiving GPU access."
    )


@pytest.mark.asyncio
async def test_node_rejects_wrong_return_device_and_unreturned_activation():
    from psrl.workers.env_service.command_gpu_node import GpuCommandNodeMixin

    node = GpuCommandNodeMixin()
    state = {
        "lock": asyncio.Lock(),
        "retired": False,
        "command_epoch": 2,
        "command_container": "c",
        "command_indices": (1,),
        "command_returned": False,
    }
    node._gpu_lease_state = lambda _: state
    with pytest.raises(RuntimeError, match="allocated devices"):
        await node.revoke_gpu_command("lease", "c", (0,), 2, (0, 1))
    with pytest.raises(RuntimeError, match="not physically returned"):
        await node.grant_gpu_command("lease", "c", (0,), 3, candidate_indices=(0, 1))
    assert state["command_epoch"] == 2, "Rejected activation must not overwrite the live ownership epoch."


@pytest.mark.asyncio
async def test_executor_injects_granted_uuid_over_caller_device_selection():
    from psrl.workers.env_service.gpu_execution import EpisodeGpuReservation, GpuExecutionBackend
    from psrl.workers.env_service.managed_execution import ManagedGpuExecution

    class Policy(EpisodeGpuReservation):
        mode = "command"

        async def acquire(self, container_id, record, tag):
            record["gpu_uuids"] = ("GPU-selected",)

    class Environment(ManagedGpuExecution):
        async def _gpu_container_id(self):
            return "container"

        async def _exec_unreserved(self, command, **kwargs):
            return kwargs["env"]

    environment = Environment()
    environment._init_gpu_execution(None)
    environment._execution_backend = GpuExecutionBackend(Policy("r", "lease"))
    result = await environment.exec_gpu("probe", env={"CUDA_VISIBLE_DEVICES": "wrong", "KEEP": "yes"})
    assert result == {"CUDA_VISIBLE_DEVICES": "GPU-selected", "KEEP": "yes"}, "The grant owns device selection."
