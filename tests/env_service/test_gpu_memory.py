"""
Exercise pending allocations, verified ownership and conservative admission.
"""

import asyncio
from itertools import product
from types import SimpleNamespace

import pytest
from psrl.workers.env_service.command_gpu import GpuCommandScheduler
from psrl.workers.env_service.gpu_memory import accounted_gpu_memory_mb
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.types import EnvDemand, EnvUnit


def test_accounting_preserves_external_usage_and_each_tasks_unallocated_budget():
    for external, a, b in product((0, 7, 70), (0, 3, 14, 20), (0, 2, 15, 30)):
        actual = accounted_gpu_memory_mb(external + a + b, {"a": 15, "b": 15}, {"a": a, "b": b})
        assert actual == external + max(15, a) + max(15, b), (
            "Overuse by one task cannot satisfy another task's pending reservation."
        )


def test_unknown_ownership_does_not_discount_reservations():
    assert accounted_gpu_memory_mb(70, {"a": 15}) == 85, "External usage and pending memory must add."
    assert accounted_gpu_memory_mb(84, {"a": 15}) == 99, "Unknown ownership cannot be assumed to be ours."
    assert accounted_gpu_memory_mb(84, {"a": 15}, {"a": 14}) == 85, "Verified memory must not be charged twice."


@pytest.mark.parametrize("measured,owned", [(14, {"a": 15}), (70, {"foreign": 10}), (float("nan"), {})])
def test_invalid_credits_fail_closed(measured, owned):
    with pytest.raises(ValueError):
        accounted_gpu_memory_mb(measured, {"a": 15}, owned)


class DelayedAllocationNode:
    """
    Separate permission grants from eventual CUDA allocation and physical return.
    """

    def __init__(self, external):
        self.external = external
        self.allocated = {}

    async def gpu_lending_snapshot(self, indices):
        return {0: {"used_mb": self.external + sum(self.allocated.values()), "total_mb": 95}}

    async def gpu_command_memory_snapshot(self, indices, commands):
        return {
            "devices": await self.gpu_lending_snapshot(indices),
            "owned_mb": {key: self.allocated.get(key, 0) for key, _, _ in commands},
            "attribution_errors": {},
        }

    async def grant_gpu_command(self, lease_id, container_id, indices, epoch):
        self.allocated[lease_id] = 0

    async def revoke_gpu_command(self, lease_id, container_id, indices, epoch, telemetry_indices):
        self.allocated.pop(lease_id)
        return await self.gpu_lending_snapshot(telemetry_indices)


def scenario(external):
    unit = EnvUnit("u", "node", 4, 4, (0,), 4, 90, gpu_reservation="command")
    ledger = EnvUnitLedger([unit], NodeRules.build(policy="any"))
    leases = [ledger.try_admit(str(i), EnvDemand(1, 1, True, 15), "node").lease for i in range(4)]
    node = DelayedAllocationNode(external)

    async def get_node(ip):
        return SimpleNamespace(
            **{
                name: SimpleNamespace(remote=getattr(node, name))
                for name in ("gpu_command_memory_snapshot", "grant_gpu_command", "revoke_gpu_command")
            }
        )

    return GpuCommandScheduler(ledger, get_node), node, leases


def test_resident_memory_blocks_followers_before_allocation_and_after_refill():
    async def run():
        scheduler, node, leases = scenario(70)
        a, b, c, _ = leases
        first = await scheduler.acquire(a.lease_id, "a", "a")
        assert first["status"] == "granted", "One command must fit alongside resident memory."
        for lease, name in ((b, "b"), (c, "c")):
            assert (await scheduler.acquire(lease.lease_id, name, name))["status"] == "wait", (
                "An acknowledged permission must reserve memory before CUDA allocates."
            )
        node.allocated[a.lease_id] = 14
        assert (await scheduler.acquire(b.lease_id, "b", "b"))["status"] == "wait", (
            "Allocation must preserve the remaining reservation."
        )
        await scheduler.release(a.lease_id, "a", first["epoch"])
        assert (await scheduler.acquire(b.lease_id, "b", "b"))["status"] == "granted", (
            "Physical return must admit the queue head."
        )
        assert (await scheduler.acquire(c.lease_id, "c", "c"))["status"] == "wait", (
            "Queue draining must not create another over-admission window."
        )

    asyncio.run(run())


def test_verified_low_pressure_allocations_keep_four_way_concurrency():
    async def run():
        scheduler, node, leases = scenario(0)
        for i, lease in enumerate(leases):
            assert (await scheduler.acquire(lease.lease_id, str(i), str(i)))["status"] == "granted", (
                "Verified allocated memory must not unnecessarily serialize tasks."
            )
            node.allocated[lease.lease_id] = 14
        assert len(node.allocated) == 4, "All four commands must remain concurrently allocated."

    asyncio.run(run())


def test_episode_admission_also_reserves_memory_on_top_of_external_usage():
    unit = EnvUnit("u", "node", 4, 4, (0,), 4, 90)
    ledger = EnvUnitLedger([unit], NodeRules.build(policy="any"))
    demand = EnvDemand(1, 1, True, 15)
    assert ledger.try_admit("a", demand, "node", {0: 70}).status == "granted", "The first episode must fit."
    assert ledger.try_admit("b", demand, "node", {0: 70}).status == "wait", (
        "Episode reservations must also account for pending allocations."
    )


@pytest.mark.asyncio
async def test_node_discards_credit_if_command_epoch_changes_during_observation(monkeypatch):
    from psrl.workers.env_service import command_gpu_node
    from psrl.workers.env_service.lending_node import GpuLendingNodeMixin

    class LocalNode(command_gpu_node.GpuCommandNodeMixin, GpuLendingNodeMixin):
        def _gpu_probe(self):
            return None

    node = LocalNode()
    state = node._gpu_lease_state("lease")
    state.update(command_epoch=1, command_container="container", command_telemetry=object())

    def observe(*_args):
        state["command_epoch"] = 2
        return {"owned_mb": {"lease": 14}, "devices": {}, "attribution_errors": {}}

    monkeypatch.setattr(command_gpu_node, "read_command_memory", observe)
    report = await node.gpu_command_memory_snapshot((0,), [("lease", "container", 1)])
    assert report["owned_mb"] == {}, "Credit from an earlier command must not offset a new reservation."
    with pytest.raises(RuntimeError, match="stale"):
        await node.gpu_command_memory_snapshot((0,), [("lease", "container", 1)])
