"""Bound cooperative cleanup while leaving physical release as a separate check."""

import asyncio
import shutil
import time


def _consume(task):
    if not task.cancelled():
        task.exception()


async def await_cleanup(awaitable, deadline, *, timeout_s=None):
    """Stop waiting at the cleanup deadline even if cancellation is not acknowledged."""
    remaining = deadline - time.monotonic()
    if timeout_s is not None:
        remaining = min(remaining, timeout_s)
    if remaining <= 0:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        elif asyncio.isfuture(awaitable):
            awaitable.add_done_callback(_consume)
            awaitable.cancel()
        raise TimeoutError("The cleanup deadline expired.")
    task = asyncio.ensure_future(awaitable)
    task.add_done_callback(_consume)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
    except BaseException:
        task.cancel()
        raise


async def cancel_and_join(tasks, *, deadline, timeout_s=10):
    """Request cancellation and report unfinished clients without claiming GPU release."""
    tasks = tuple(dict.fromkeys(task for task in tasks if task is not asyncio.current_task() and not task.done()))
    for task in tasks:
        task.cancel()
    if tasks:
        joined = asyncio.gather(*tasks, return_exceptions=True)
        joined.add_done_callback(_consume)
        remaining = min(timeout_s, max(0, deadline - time.monotonic()))
        if remaining > 0:
            try:
                await asyncio.wait_for(asyncio.shield(joined), timeout=remaining)
            except TimeoutError:
                pass
    pending = sum(not task.done() for task in tasks)
    return {"cancel_requested": len(tasks), "joined": pending == 0, "pending": pending}


def archive_runtime(runtime_root, root):
    """Copy evidence without deleting runtime files before the caller confirms success."""
    for index in range(2):
        source = runtime_root / f"service-{index}"
        if source.is_dir():
            target = root / f"service-{index}"
            target.mkdir(exist_ok=True)
            for path in source.iterdir():
                if path.is_file():
                    shutil.copy2(path, target / path.name)
