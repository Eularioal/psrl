"""Run node-local GPU telemetry and lease cleanup independently of episode owners."""

import asyncio
import logging
import uuid

from aiohttp import web

from psrl.utils.common.docker_utils import CLEANUP_EXECUTOR, force_remove_containers_by_label
from psrl.workers.env_service.command_gpu_node import GpuCommandNodeMixin
from psrl.workers.env_service.docker_api import ping_docker
from psrl.workers.env_service.lending_node import GpuLendingNodeMixin
from psrl.workers.env_service.overlay import LEASE_LABEL
from psrl.workers.env_service.probe import read_gpu_telemetry
from psrl.workers.env_service.resource_telemetry import read_node_resources

psrl_logger = logging.getLogger(__file__)


class EnvNodeService(GpuCommandNodeMixin, GpuLendingNodeMixin):
    """Serve one node for the lifetime of the owning router actor."""

    def __init__(self, router, node_ip: str, gpu_identity_backend="procfs"):
        self.router = router
        self.node_ip = node_ip
        self._identity_mode = gpu_identity_backend
        self._pending = {}
        self._lock = asyncio.Lock()
        self._runner = None
        self._reaper = None
        self._url = None

    async def ready(self) -> str:
        """Start telemetry and cleanup before admitting containers on this node."""
        if self._runner is None:
            path = f"/{uuid.uuid4().hex}/gpu"
            app = web.Application()
            app.router.add_get(path, self._telemetry)
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, self.node_ip, 0)
            await site.start()
            port = self._runner.addresses[0][1]
            self._url = f"http://{self.node_ip}:{port}{path}"
            self._reaper = asyncio.create_task(self._reap_forever())
        return self._url

    async def _telemetry(self, request: web.Request) -> web.Response:
        return web.json_response(await read_gpu_telemetry())

    async def prepare(self, indices):
        """Initialize this service's live queries without reserving or authorizing a GPU."""
        await ping_docker()
        if indices:
            await asyncio.to_thread(self._gpu_probe().read_gpu_devices)
            return await self.gpu_lending_snapshot(indices)
        return {}

    async def resource_snapshot(self, unit_ids, timeout_s=4):
        """Serve observability independently of lease cleanup and device authorization."""
        return await read_node_resources(unit_ids, timeout_s=timeout_s)

    async def close(self):
        """Stop background work before releasing this service's driver connection."""
        if self._reaper is not None:
            self._reaper.cancel()
            await asyncio.gather(self._reaper, return_exceptions=True)
            self._reaper = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        for state in getattr(self, "_gpu_lease_states", {}).values():
            if state.get("command_telemetry"):
                await state["command_telemetry"].stop()
        if hasattr(self, "_nvml_probe"):
            await asyncio.to_thread(self._nvml_probe.close)
        self._identity_closed = True
        if hasattr(self, "_identity_lock"):
            async with self._identity_lock:
                if hasattr(self, "_identity_source"):
                    await asyncio.to_thread(self._identity_source.close)

    async def cleanup(self) -> int:
        """Retain failed removals for retry and return the remaining lease count."""
        async with self._lock:
            await self.router.reclaim_gpu_phases.remote(self.node_ip)
            expired = await self.router.pending_reaped.remote(self.node_ip)
            self._pending.update((lease.lease_id, lease) for lease in expired)
            for lease_id in tuple(self._pending):
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        CLEANUP_EXECUTOR, force_remove_containers_by_label, LEASE_LABEL, lease_id
                    )
                    process = await asyncio.create_subprocess_exec(
                        "docker",
                        "ps",
                        "-aq",
                        "--filter",
                        f"label={LEASE_LABEL}={lease_id}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                        raise
                    if process.returncode == 0 and not stdout.strip():
                        lease = self._pending[lease_id]
                        if lease.gpu_lending:
                            await self.router.finish_gpu_lease.remote(lease_id, lease.unit_id)
                        if lease.gpu_reservation == "command":
                            await self.router.finish_gpu_command_lease.remote(lease_id, lease.unit_id)
                        await self.router.acknowledge_reaped.remote(self.node_ip, lease_id)
                        del self._pending[lease_id]
                except Exception as exc:
                    psrl_logger.warning(f"Lease cleanup failed for {lease_id!r}: {exc!r}.")
            return len(self._pending)

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(1)
            try:
                await self.cleanup()
            except Exception as exc:
                psrl_logger.warning(f"Node cleanup pass failed: {exc!r}.")
