"""
Agent-loop-side client for the EnvRouter: Gate A's wait, heartbeat and release.

Gate A runs before `Job.create`, so waiting costs nothing: no container exists yet, and
the caller may wait as long as it takes. That is the opposite of Gate B, which holds a
reply inside a live HTTP request and so must be bounded (design document section 6.3).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from psrl.workers.env_service.probe import read_gpu_telemetry
from psrl.workers.env_service.types import GRANTED, REJECT, EnvDemand, EnvLease

psrl_logger = logging.getLogger("psrl.env_service.client")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

TelemetryReader = Callable[[], Awaitable[dict[int, tuple[float, float]]]]


class EnvAdmissionRejected(RuntimeError):
    """No unit could ever host the demand, so retrying is pointless."""


class EnvAdmissionTimeout(TimeoutError):
    """The caller's own `max_wait_s` ran out while waiting for capacity."""


class EnvLeaseHandle:
    """
    A granted lease plus the background task that keeps it alive.

    Args:
        client (EnvServiceClient): The issuing client.
        lease (EnvLease): The granted lease.
    """

    def __init__(self, client: EnvServiceClient, lease: EnvLease):
        self._client = client
        self.lease = lease
        self._heartbeat_task: asyncio.Task | None = None
        self._released = False

    def start_heartbeat(self) -> None:
        """Begin refreshing the TTL. Must be called on the loop that will run the episode."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.get_running_loop().create_task(self._beat())

    async def _beat(self) -> None:
        interval = self._client.heartbeat_interval_s
        while True:
            await asyncio.sleep(interval)
            try:
                alive = await self._client.router.heartbeat.remote(self.lease.lease_id)
            except Exception as exc:  # noqa: BLE001
                psrl_logger.debug("Heartbeat for %s failed: %s.", self.lease.lease_id, exc)
                continue
            if not alive:
                # The router already reclaimed it. Keep running: killing a live episode
                # over bookkeeping would destroy up to hours of work.
                psrl_logger.warning("Lease %s is unknown to the router (reclaimed?).", self.lease.lease_id)

    async def release(self) -> None:
        """Stop heartbeating and give the capacity back. Idempotent and never raises."""
        if self._released:
            return
        self._released = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        try:
            await self._client.router.release.remote(self.lease.lease_id)
        except Exception as exc:  # noqa: BLE001
            # The TTL reaper is the backstop for a lost release.
            psrl_logger.warning("Release of %s failed (%s); TTL reclaim will free it.", self.lease.lease_id, exc)


class EnvServiceClient:
    """
    Talks to the EnvRouter on behalf of one agent loop worker.

    Args:
        router: The router actor handle (or any object with the same awaitable methods).
        node_ip (str): This worker's node. Admission is restricted to units on it, because
            Harbor starts containers on the node of the process that calls it.
        default_gpu_memory_mb (int): Device memory charged to a GPU task that declares none.
        poll_interval_s (float): First retry delay while waiting for capacity.
        max_poll_interval_s (float): Ceiling for the backed-off retry delay.
        heartbeat_interval_s (float): Heartbeat period. Keep well under the router's TTL.
        telemetry_reader (TelemetryReader): Reads this node's GPUs. Replaced in tests.
    """

    def __init__(
        self,
        router: Any,
        node_ip: str,
        *,
        default_gpu_memory_mb: int = 24000,
        poll_interval_s: float = 1.0,
        max_poll_interval_s: float = 5.0,
        heartbeat_interval_s: float = 60.0,
        telemetry_reader: TelemetryReader = read_gpu_telemetry,
    ):
        self.router = router
        self.node_ip = node_ip
        self.default_gpu_memory_mb = int(default_gpu_memory_mb)
        self.poll_interval_s = float(poll_interval_s)
        self.max_poll_interval_s = float(max_poll_interval_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self._telemetry_reader = telemetry_reader

    @classmethod
    def from_config(cls, env_cfg: Mapping | None) -> EnvServiceClient | None:
        """
        Build a client from `psrl.env_service`, or return `None` when it is disabled.

        Returns:
            EnvServiceClient | None: `None` means "use the legacy path", which is how an
            existing recipe that never enabled the service keeps its behaviour.

        Raises:
            RuntimeError: When the service is enabled but the router actor is not running.
        """
        if not env_cfg or not env_cfg.get("enable", False):
            return None
        import ray

        from psrl.workers.env_service.router import DEFAULT_ROUTER_NAME

        name = str(env_cfg.get("router_name", DEFAULT_ROUTER_NAME))
        try:
            router = ray.get_actor(name)
        except ValueError as exc:
            raise RuntimeError(
                f"psrl.env_service.enable=True but no actor named {name!r} is running. The trainer creates it "
                f"at startup; a worker started without it cannot admit episodes."
            ) from exc
        return cls(
            router,
            ray.util.get_node_ip_address(),
            default_gpu_memory_mb=int(env_cfg.get("default_gpu_memory_mb", 24000)),
            poll_interval_s=float(env_cfg.get("poll_interval_s", 1.0)),
            max_poll_interval_s=float(env_cfg.get("max_poll_interval_s", 5.0)),
            heartbeat_interval_s=float(env_cfg.get("heartbeat_interval_s", 60.0)),
        )

    async def acquire(
        self,
        session_id: str,
        demand: EnvDemand,
        *,
        max_wait_s: float | None = None,
    ) -> EnvLeaseHandle:
        """
        Gate A: wait until a unit on this node can host the episode, then hold it.

        Args:
            session_id (str): The episode's session id.
            demand (EnvDemand): What the episode needs.
            max_wait_s (float | None): Give up after this long. `None` waits indefinitely,
                which is correct for Gate A because nothing is held while waiting.

        Returns:
            EnvLeaseHandle: The lease, with its heartbeat started.

        Raises:
            EnvAdmissionRejected: When no unit could ever host the demand.
            EnvAdmissionTimeout: When `max_wait_s` ran out.
        """
        await self.reap_local()
        loop = asyncio.get_running_loop()
        deadline = None if max_wait_s is None else loop.time() + max_wait_s
        delay = self.poll_interval_s
        try:
            while True:
                measured = await self._measure() if demand.needs_gpu else None
                admission = await self.router.try_admit.remote(session_id, demand, self.node_ip, measured)
                if admission.status == GRANTED:
                    handle = EnvLeaseHandle(self, admission.lease)
                    handle.start_heartbeat()
                    return handle
                if admission.status == REJECT:
                    raise EnvAdmissionRejected(admission.reason)
                if deadline is not None and loop.time() >= deadline:
                    raise EnvAdmissionTimeout(f"no capacity for session {session_id!r} within {max_wait_s}s")
                # Jitter keeps many waiting workers from polling the router in lockstep.
                await asyncio.sleep(delay * (0.75 + 0.5 * random.random()))
                delay = min(delay * 1.5, self.max_poll_interval_s)
        except BaseException:
            # Covers a cancelled wait too: a session that gave up must stop counting as waiting.
            try:
                await self.router.cancel_wait.remote(session_id)
            except Exception:  # noqa: BLE001
                pass
            raise

    async def _measure(self) -> dict[int, float] | None:
        """Per-device memory in use on this node, or `None` when it cannot be read."""
        telemetry = await self._telemetry_reader()
        if not telemetry:
            return None
        return {index: mem_mb for index, (_util, mem_mb) in telemetry.items()}

    async def reap_local(self) -> int:
        """
        Stop containers of leases the router reclaimed on this node.

        The router frees the capacity, but it runs on another machine, so the stop happens
        here, by the `psrl.env_lease` label, on the next admission from this node.

        Returns:
            int: How many reclaimed leases were cleaned up.
        """
        try:
            expired = await self.router.pop_reaped.remote(self.node_ip)
        except Exception as exc:  # noqa: BLE001
            psrl_logger.debug("pop_reaped failed: %s.", exc)
            return 0
        if not expired:
            return 0
        from psrl.utils.common.docker_utils import CLEANUP_EXECUTOR, force_remove_containers_by_label
        from psrl.workers.env_service.overlay import LEASE_LABEL

        loop = asyncio.get_running_loop()
        for lease in expired:
            psrl_logger.warning("Stopping containers of reclaimed lease %s.", lease.lease_id)
            # The dedicated cleanup pool, not the loop's default executor: a `docker rm` on a
            # loaded daemon blocks for tens of seconds and must not starve episode I/O.
            await loop.run_in_executor(CLEANUP_EXECUTOR, force_remove_containers_by_label, LEASE_LABEL, lease.lease_id)
        return len(expired)
