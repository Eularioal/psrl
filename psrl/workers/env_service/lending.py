"""Coordinate acknowledged environment and inference phases for one GPU unit."""

import asyncio
import time


class GpuLendingBinding:
    """Own one replica's handoffs and retain ownership after uncertain failures."""

    def __init__(self, server, node, unit, config, coordinator=None):
        self.coordinator = coordinator
        self.server = server
        self.node = node
        self.unit = unit
        self.timeout = float(config.get("transition_timeout_s", 60))
        self.return_tolerance_mb = float(config.get("return_tolerance_mb", 16))
        self.margin_mb = float(config.get("safety_margin_mb", 1024))
        self.max_env_phase_s = float(config.get("max_env_phase_s", 300))
        self.lock = asyncio.Lock()
        self.owner = None
        self.phase = "model"
        self.sequence = -1
        self.baseline = None
        self.deadline = None
        self.starting = set()

    async def set_phase(self, lease, phase: str, sequence: int, *, is_live=None):
        """Apply monotonically numbered phase requests from the episode owner."""
        if phase not in ("env", "model"):
            raise ValueError(f"Unknown GPU phase {phase!r}.")
        async with self.lock:
            if is_live is not None and not is_live():
                raise RuntimeError("GPU phase request outlived its lease.")
            if self.phase == "quarantined":
                raise RuntimeError("GPU unit is quarantined until physical cleanup succeeds.")
            if self.owner not in (None, lease.lease_id):
                raise RuntimeError("GPU unit belongs to another lease.")
            if sequence <= self.sequence:
                if sequence == self.sequence and phase == self.phase:
                    return
                raise RuntimeError("Stale GPU phase request.")
            self.owner = lease.lease_id
            self.sequence = sequence
            try:
                await asyncio.wait_for(self._transition(lease, phase), self.timeout)
                self.phase = phase
            except BaseException:
                self.phase = "quarantined"
                raise

    async def _transition(self, lease, phase):
        if phase == "env" and self.phase != "env":
            await self._enter_env(lease)
        elif phase == "model" and self.phase != "model":
            if self.starting:
                raise RuntimeError("A GPU container is still starting.")
            await self.node.freeze_gpu_lease.remote(
                self.owner, self.baseline, self.return_tolerance_mb, self.timeout * 0.8
            )
            await self.server.return_env_gpu.remote(self.owner)
            self.deadline = None

    async def _enter_env(self, lease):
        while True:
            if self.coordinator is None:
                prepared = await self.server.prepare_env_gpu.remote(self.owner)
            else:
                prepared = await self.coordinator.prepare_env_gpu.remote(self.unit.rollout_replica_rank, self.owner)
            if prepared:
                break
            await asyncio.sleep(0.05)
        self.baseline = await self.node.gpu_lending_snapshot.remote(self.unit.gpu_indices)
        for reading in self.baseline.values():
            # Every device must fit the full demand because task sharding is not declared.
            if reading["used_mb"] + lease.demand.gpu_memory_mb + self.margin_mb > reading["total_mb"]:
                raise RuntimeError("Insufficient physical GPU memory after rollout sleep.")
        await self.node.resume_gpu_lease.remote(self.owner)
        self.deadline = time.monotonic() + self.max_env_phase_s

    async def finish(self, lease_id: str):
        """Remove owned containers and confirm return before releasing the unit."""
        async with self.lock:
            if self.owner is None:
                return
            if self.owner != lease_id:
                raise RuntimeError("Cannot finish another GPU borrower's lease.")
            try:
                await asyncio.wait_for(self._finish(lease_id), self.timeout)
                self.owner = None
                self.phase = "model"
                self.sequence = -1
                self.baseline = None
                self.deadline = None
            except BaseException:
                self.phase = "quarantined"
                raise

    async def _finish(self, lease_id):
        if self.starting:
            raise RuntimeError("Cannot return GPU resources while a container start is in flight.")
        await self.node.remove_gpu_lease.remote(lease_id)
        if self.phase != "model":
            # Missing sleep acknowledgement keeps the unit fenced after cleanup.
            await self.node.confirm_gpu_return.remote(self.unit.gpu_indices, self.baseline, self.return_tolerance_mb)
            await self.server.return_env_gpu.remote(lease_id)
        if self.coordinator is None:
            await self.server.finish_env_gpu.remote(lease_id)
        else:
            await self.coordinator.finish_env_gpu.remote(self.unit.rollout_replica_rank, lease_id)


class GpuPhaseClient:
    """Switch the lease between command batches and model queries."""

    def __init__(self, router, lease_id: str):
        self.router = router
        self.lease_id = lease_id
        self.sequence = 0

    async def set_phase(self, phase: str):
        self.sequence += 1
        await self.router.set_gpu_phase.remote(self.lease_id, phase, self.sequence)
