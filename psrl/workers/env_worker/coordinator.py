"""Route sandbox requests without owning container or placement mechanics."""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import random
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from psrl.workers.env_worker.sandbox import ExecResult, SandboxSpec

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# NOTE(claude): `_ROUND_ROBIN_COUNTER` is shared by coordinators in one process, so
# heterogeneous pools index each eligible subset independently.
_ROUND_ROBIN_COUNTER = itertools.count()
_SELECT_POLL_INTERVAL_S = 0.5
_CLEANUP_TIMEOUT_S = 30.0


@dataclass
class WorkerSlot:
    """Capacity bookkeeping for one registered `EnvWorker`."""

    worker_id: int
    handle: Any
    cpu_slots: int
    gpu_slots: int
    node_ip: str
    used_cpu: int = 0
    used_gpu: int = 0
    cpu_budget: float = float("inf")
    memory_budget: int = 2**63
    cpu_reserved: float = 0.0
    memory_reserved: int = 0

    def fits(self, spec: SandboxSpec) -> bool:
        """Require container, CPU, RAM and GPU capacity together."""
        return (
            self.has_cpu_capacity
            and self.has_gpu_capacity(spec.gpus)
            and self.cpu_reserved + (spec.cpus or 0) + spec.auxiliary_cpus <= self.cpu_budget
            and self.memory_reserved + memory_bytes(spec.memory) + memory_bytes(spec.auxiliary_memory)
            <= self.memory_budget
        )

    @property
    def has_cpu_capacity(self) -> bool:
        return self.used_cpu < self.cpu_slots

    def has_gpu_capacity(self, requested: int) -> bool:
        return self.used_gpu + requested <= self.gpu_slots


def select_by_method(
    method: str,
    candidates: list[WorkerSlot],
    spec: SandboxSpec,
) -> WorkerSlot | None:
    """
    Choose a worker for one sandbox request.

    Args:
        method (str): Routing method, one of least_loaded, round_robin, random.
        candidates (list[WorkerSlot]): Registered workers with current load.
        spec (SandboxSpec): Sandbox being placed. Its `gpus` field filters candidates.

    Returns:
        WorkerSlot | None: The chosen worker, or None when nothing can host the
            request right now, which tells the caller to queue.

    Raises:
        ValueError: If `method` is not a known routing method.
    """
    if method not in ("least_loaded", "round_robin", "random"):
        raise ValueError(f"Unknown routing method {method!r}.")

    eligible = [slot for slot in candidates if slot.fits(spec)]
    if not eligible:
        return None

    if method == "least_loaded":
        return min(eligible, key=lambda slot: (slot.used_cpu, slot.worker_id))
    if method == "round_robin":
        return eligible[next(_ROUND_ROBIN_COUNTER) % len(eligible)]

    return random.choice(eligible)


def memory_bytes(value: str | None) -> int:
    """Parse Docker memory sizes into bytes for admission accounting."""
    if value is None:
        return 0
    text = str(value).strip().lower().removesuffix("b")
    units = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    factor = units.get(text[-1:], 1)
    return int(float(text[:-1] if factor != 1 else text) * factor)


@dataclass
class SandboxHandle:
    """Client reference to a remotely owned sandbox reservation."""

    sandbox_id: str
    worker: Any
    coordinator: Any = None
    worker_id: int = -1
    gpus: int = 0
    heartbeat_interval_s: float = 30.0
    _destroyed: bool = field(default=False, repr=False)
    _heartbeat_task: Any = field(default=None, repr=False)

    def start_heartbeat(self) -> None:
        """Keep the reservation alive while its owning episode is running."""
        self._heartbeat_task = asyncio.create_task(self._beat())

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_s)
            try:
                if not await self.coordinator.heartbeat.remote(self.sandbox_id):
                    return
            except Exception:
                psrl_logger.warning("Sandbox heartbeat failed.", exc_info=True)

    async def exec(self, command: str, timeout_s: float, no_output_timeout_s: float | None = None) -> ExecResult:
        """Run one command in this sandbox's persistent shell."""
        return await self.worker.exec.remote(self.sandbox_id, command, timeout_s, no_output_timeout_s)

    async def read_file(self, path: str) -> bytes:
        """Read one bounded file from this sandbox."""
        return await self.worker.read_file.remote(self.sandbox_id, path)

    async def write_file(self, path: str, data: bytes) -> None:
        """Write a file through the owning worker."""
        await self.worker.write_file.remote(self.sandbox_id, path, data)

    async def evaluate(self, spec: SandboxSpec, command: str, timeout_s: float) -> ExecResult:
        """Run the trusted verifier on an isolated submission snapshot."""
        return await self.worker.evaluate.remote(self.sandbox_id, spec, command, timeout_s)

    async def destroy(self) -> None:
        """Stop heartbeats and release only after remote cleanup succeeds."""
        if self._destroyed:
            return
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
        if self.coordinator is not None:
            await self.coordinator.release.remote(self.sandbox_id)
        else:
            await self.worker.destroy_sandbox.remote(self.sandbox_id)
        self._destroyed = True


async def acquire_sandbox(coordinator: Any, spec: SandboxSpec, timeout_s: float) -> SandboxHandle:
    """Poll outside the coordinator so waiters never occupy its RPC capacity."""
    request_id = f"s-{uuid.uuid4().hex}"
    handle = None

    async def admit():
        nonlocal handle
        while handle is None:
            handle = await coordinator.reserve.remote(spec, request_id)
            if handle is None:
                await asyncio.sleep(0.25)
        handle.coordinator = coordinator
        handle.start_heartbeat()
        await handle.worker.create_sandbox.remote(spec, request_id)
        return handle

    try:
        return await asyncio.wait_for(admit(), timeout_s)
    except BaseException:
        try:
            if handle is not None:
                await handle.destroy()
            else:
                await coordinator.release.remote(request_id)
        except Exception:
            psrl_logger.exception("Sandbox acquisition cleanup failed. The watchdog will retry.")
        raise


@dataclass
class Reservation:
    """One resource hold, retained until its physical containers are removed."""

    worker_id: int
    spec: SandboxSpec
    last_seen: float
    closing: bool = False


class EnvWorkerCoordinator:
    """Reserve resources and reclaim abandoned containers on their owning nodes."""

    def __init__(self, routing_method: str = "least_loaded"):
        self.routing_method = routing_method
        self._slots: dict[int, WorkerSlot] = {}
        self._reservations: dict[str, Reservation] = {}
        self._cancelled: dict[str, float] = {}
        self._cleanup_locks: dict[str, asyncio.Lock] = {}
        self._watchdog: asyncio.Task | None = None

    async def register_worker(
        self,
        worker_handle: Any,
        worker_id: int,
        cpu_slots: int,
        gpu_slots: int,
        node_ip: str,
        cpu_budget: float = float("inf"),
        memory_budget: int = 2**63,
    ) -> None:
        """Register one worker with explicit resource budgets."""
        self._slots[worker_id] = WorkerSlot(
            worker_id,
            worker_handle,
            cpu_slots,
            gpu_slots,
            node_ip,
            cpu_budget=cpu_budget,
            memory_budget=memory_budget,
        )
        if self._watchdog is None:
            self._watchdog = asyncio.create_task(self._reap_loop())

    async def reserve(self, spec: SandboxSpec, request_id: str) -> SandboxHandle | None:
        """Make one immediate admission attempt. Never wait inside an RPC."""
        if request_id in self._cancelled:
            raise RuntimeError("Sandbox reservation was cancelled.")
        if not self._slots:
            raise RuntimeError("No env worker is registered.")
        existing = self._reservations.get(request_id)
        if existing is not None:
            if existing.closing:
                raise RuntimeError("Sandbox reservation is closing.")
            return self._handle(request_id, existing)
        if not any(
            slot.cpu_slots > 0
            and spec.gpus <= slot.gpu_slots
            and (spec.cpus or 0) + spec.auxiliary_cpus <= slot.cpu_budget
            and memory_bytes(spec.memory) + memory_bytes(spec.auxiliary_memory) <= slot.memory_budget
            for slot in self._slots.values()
        ):
            raise ValueError("Sandbox demand exceeds every worker's resource budget.")
        slot = select_by_method(self.routing_method, list(self._slots.values()), spec)
        if slot is None:
            return None
        slot.used_cpu += 1
        slot.used_gpu += spec.gpus
        slot.cpu_reserved += (spec.cpus or 0) + spec.auxiliary_cpus
        slot.memory_reserved += memory_bytes(spec.memory) + memory_bytes(spec.auxiliary_memory)
        reservation = Reservation(slot.worker_id, spec, time.monotonic())
        self._reservations[request_id] = reservation
        return self._handle(request_id, reservation)

    def _handle(self, request_id: str, reservation: Reservation) -> SandboxHandle:
        return SandboxHandle(
            request_id,
            self._slots[reservation.worker_id].handle,
            worker_id=reservation.worker_id,
            gpus=reservation.spec.gpus,
            heartbeat_interval_s=min(30.0, reservation.spec.idle_timeout_s / 3),
        )

    async def heartbeat(self, request_id: str) -> bool:
        """Refresh only a live reservation."""
        reservation = self._reservations.get(request_id)
        if reservation is None or reservation.closing:
            return False
        reservation.last_seen = time.monotonic()
        return True

    async def release(self, request_id: str) -> None:
        """Delete on the owning worker before returning resource capacity."""
        self._cancelled[request_id] = time.monotonic()
        async with self._cleanup_locks.setdefault(request_id, asyncio.Lock()):
            reservation = self._reservations.get(request_id)
            if reservation is None:
                return
            reservation.closing = True
            slot = self._slots[reservation.worker_id]
            await asyncio.wait_for(slot.handle.destroy_sandbox.remote(request_id), _CLEANUP_TIMEOUT_S)
            slot.used_cpu -= 1
            slot.used_gpu -= reservation.spec.gpus
            slot.cpu_reserved -= (reservation.spec.cpus or 0) + reservation.spec.auxiliary_cpus
            slot.memory_reserved -= memory_bytes(reservation.spec.memory) + memory_bytes(
                reservation.spec.auxiliary_memory
            )
            del self._reservations[request_id]

    async def reap_expired(self) -> list[str]:
        """Retry failed cleanup and reclaim owners that stopped heartbeating."""
        now = time.monotonic()
        reaped = []
        semaphore = asyncio.Semaphore(8)

        async def reap(request_id):
            async with semaphore:
                try:
                    await self.release(request_id)
                    reaped.append(request_id)
                except Exception:
                    psrl_logger.exception("Sandbox cleanup failed. Capacity remains reserved.")

        await asyncio.gather(
            *(
                reap(request_id)
                for request_id, reservation in list(self._reservations.items())
                if reservation.closing or now - reservation.last_seen > reservation.spec.idle_timeout_s
            )
        )
        for request_id, cancelled_at in list(self._cancelled.items()):
            if now - cancelled_at > 3600 and request_id not in self._reservations:
                self._cancelled.pop(request_id, None)
                self._cleanup_locks.pop(request_id, None)
        return reaped

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            await self.reap_expired()

    async def shutdown(self) -> None:
        """Stop admission and remove all owned containers before actor shutdown."""
        if self._watchdog is not None:
            self._watchdog.cancel()
            with suppress(asyncio.CancelledError):
                await self._watchdog
        for request_id in list(self._reservations):
            await self.release(request_id)

    async def stats(self) -> dict[str, Any]:
        """Report slot occupancy and actual CPU and RAM reservations."""
        return {
            "routing_method": self.routing_method,
            "workers": [
                {
                    name: getattr(slot, name)
                    for name in (
                        "worker_id",
                        "node_ip",
                        "cpu_slots",
                        "used_cpu",
                        "gpu_slots",
                        "used_gpu",
                        "cpu_budget",
                        "memory_budget",
                        "cpu_reserved",
                        "memory_reserved",
                    )
                }
                for slot in self._slots.values()
            ],
        }
