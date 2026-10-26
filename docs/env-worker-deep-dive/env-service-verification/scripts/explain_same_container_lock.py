"""Illustrate concurrent submission versus serialized execution without using GPUs."""

import asyncio
import json


async def illustrate(serialized):
    """Model the scope of one environment lock, without modeling device admission."""
    lock = asyncio.Lock()
    state = {"submitted": 0, "prepared": 0, "active": 0, "peak": 0, "completed": 0}
    all_prepared = asyncio.Event()

    async def execute():
        # Diagram C3 models overlapping processes, not GPU kernel parallelism.
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        state["completed"] += 1

    async def command():
        # Diagram C2 lets preparation finish independently of the execution lock.
        state["submitted"] += 1
        await asyncio.sleep(0)
        state["prepared"] += 1
        if state["prepared"] == 4:
            all_prepared.set()
        await all_prepared.wait()
        if serialized:
            async with lock:
                await execute()
        else:
            await execute()

    await asyncio.gather(*(command() for _ in range(4)))
    return state


async def main():
    for name, serialized in (("off_model", False), ("same_container_on_model", True)):
        print(json.dumps({"case": name, **await illustrate(serialized)}))


if __name__ == "__main__":
    asyncio.run(main())
