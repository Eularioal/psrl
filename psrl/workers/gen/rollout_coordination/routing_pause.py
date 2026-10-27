"""Compose temporary routing pauses with persistent coordinator state changes."""

import asyncio
from contextlib import asynccontextmanager


class RoutingPauseController:
    """Keep routing paused until every scoped owner has released its hold."""

    def __init__(self, read_status, write_running, timeout_s=30.0):
        self._read_status = read_status
        self._write_running = write_running
        self._timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._holds = set()
        self._running = None

    async def set_running(self, running):
        """Update the persistent state without overriding temporary pause owners."""
        async with self._lock:
            self._running = running
            await self._write_running(running and not self._holds)

    async def _acquire(self, owner):
        async with self._lock:
            if self._running is None:
                status = await self._read_status()
                self._running = not status["paused"]
            self._holds.add(owner)
            await self._write_running(False)

    async def _release(self, owner):
        async with asyncio.timeout(self._timeout_s):
            async with self._lock:
                if owner not in self._holds:
                    return
                self._holds.remove(owner)
                await self._write_running(self._running and not self._holds)

    @asynccontextmanager
    async def pause(self):
        """Restore only this scope's hold, including on failure or cancellation."""
        owner = object()
        try:
            async with asyncio.timeout(self._timeout_s):
                await self._acquire(owner)
            yield
        finally:
            # A second cancellation must not abandon an in-flight resume request.
            cleanup = asyncio.create_task(self._release(owner))
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError
