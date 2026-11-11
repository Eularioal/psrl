"""Simulate training startup and four rollout environment requests on CPU."""

import asyncio
from dataclasses import dataclass, field


@dataclass
class ModelRouter:
    """Model one shared ledger with a fixed environment count limit."""

    capacity: int
    active: dict[str, str] = field(default_factory=dict)
    peak: int = 0
    admitted: int = 0

    def __post_init__(self):
        self.slots = asyncio.Semaphore(self.capacity)

    async def acquire(self, session_id):
        # Diagram F5 sends a request to F2, represented by a local object here.
        print(f"REQUEST session={session_id} active={len(self.active)}")
        await self.slots.acquire()
        self.admitted += 1
        lease_id = f"lease-{self.admitted}"
        self.active[lease_id] = session_id
        self.peak = max(self.peak, len(self.active))
        print(f"GRANTED session={session_id} lease={lease_id} active={len(self.active)}")
        return lease_id

    def release(self, lease_id):
        session_id = self.active.pop(lease_id)
        self.slots.release()
        print(f"RELEASE session={session_id} active={len(self.active)}")


async def rollout(worker, question, sample, router):
    # Diagram F4 represents one rollout coroutine, not an operating system process.
    session_id = f"run-demo/{question}/{sample}"
    lease_id = await router.acquire(session_id)
    try:
        # Diagram F6 simulates model and environment waiting without real services.
        print(f"ENV_START worker={worker} session={session_id}")
        await asyncio.sleep(0.01)
    finally:
        print(f"ENV_CLEANED session={session_id}")
        router.release(lease_id)


async def main():
    # Diagram F1 creates one router before F3 distributes question groups.
    router = ModelRouter(capacity=2)
    tasks = [
        asyncio.create_task(rollout(worker, question, sample, router))
        for worker, question in (("worker-A", "Q1"), ("worker-B", "Q2"))
        for sample in (1, 2)
    ]
    print(f"STARTUP routers=1 rollout_coroutines={len(tasks)} capacity={router.capacity}")
    await asyncio.gather(*tasks)
    assert router.admitted == 4, "All four environment requests must finish."
    assert router.peak == 2, "Only two environments may hold slots together."
    assert not router.active, "Cleanup must leave no active leases."
    print(f"DONE admitted={router.admitted} peak={router.peak} active={len(router.active)}")


if __name__ == "__main__":
    asyncio.run(main())
