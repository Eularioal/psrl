"""Run node-local GPU telemetry and lease cleanup independently of episode owners."""

import asyncio
import logging
import uuid

from aiohttp import web

from psrl.utils.common.docker_utils import CLEANUP_EXECUTOR, force_remove_containers_by_label
from psrl.workers.env_service.lending_node import GpuLendingNodeMixin
from psrl.workers.env_service.overlay import LEASE_LABEL
from psrl.workers.env_service.probe import read_gpu_telemetry

psrl_logger = logging.getLogger(__file__)


class EnvNodeService(GpuLendingNodeMixin):
    """Serve one node for the lifetime of the owning router actor."""

    def __init__(self, router, node_ip: str):
        self.router = router
        self.node_ip = node_ip
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

    async def cleanup(self) -> int:
        """Retain failed removals for retry and return the remaining lease count."""
        async with self._lock:
            await self.router.reclaim_gpu_phases.remote(self.node_ip)
            expired = await self.router.pop_reaped.remote(self.node_ip)
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
