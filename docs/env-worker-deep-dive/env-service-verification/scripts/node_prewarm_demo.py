"""Simulate node startup overlap on the CPU without Ray, Docker or GPU allocation."""

import asyncio
import json
import time


class TeachingNode:
    """Share one pending startup between prewarming and the first request."""

    def __init__(self):
        self.pending = None
        self.starts = 0

    async def _start(self):
        # Diagram 3 simulates process startup, endpoint setup and driver initialization.
        self.starts += 1
        await asyncio.sleep(0.06)

    async def ready(self):
        if self.pending is None:
            self.pending = asyncio.create_task(self._start())
        await self.pending

    async def close(self):
        # Diagram 6 retains ownership of any unfinished startup during cleanup.
        if self.pending is not None:
            if not self.pending.done():
                self.pending.cancel()
            await asyncio.gather(self.pending, return_exceptions=True)


async def scenario(mode):
    node = TeachingNode()
    pending = None
    started = time.perf_counter()
    try:
        # Diagram 1 dispatches preparation before model initialization in overlap mode.
        if mode == "serial_prewarm":
            await node.ready()
        elif mode == "overlap_prewarm":
            pending = asyncio.create_task(node.ready())
        # Diagram 2 simulates independent model preparation.
        await asyncio.sleep(0.2)
        first_request = time.perf_counter()
        # Diagram 4 joins readiness only at the first consumer.
        await node.ready()
        service_wait = time.perf_counter() - first_request
        # Diagram 5 represents identical task work in all three cases.
        await asyncio.sleep(0.04)
        if pending is not None:
            await pending
        assert node.starts == 1, "Prewarming and admission must share one startup."
        return {
            "mode": mode,
            "total_s": round(time.perf_counter() - started, 4),
            "first_request_service_wait_s": round(service_wait, 4),
            "node_starts": node.starts,
        }
    finally:
        if pending is not None:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await node.close()


async def main():
    results = []
    for mode in ("lazy", "serial_prewarm", "overlap_prewarm"):
        results.append(await scenario(mode))
    print(json.dumps({"scope": "CPU teaching simulation, not PSRL performance", "results": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
