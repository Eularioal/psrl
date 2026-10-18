"""Run synchronous MLGym in a cancellable process with an async sandbox RPC bridge."""

from __future__ import annotations

import asyncio
import multiprocessing
import tempfile
import threading
from contextlib import suppress
from typing import Any


class RemoteHandle:
    """Forward only sandbox operations to the episode's parent process."""

    def __init__(self, connection: Any, sandbox_id: str):
        self.connection = connection
        self.sandbox_id = sandbox_id

    async def _call(self, method: str, *args: Any) -> Any:
        self.connection.send(("call", method, args))
        success, value = self.connection.recv()
        if not success:
            kind, message = value
            if kind == "invalid_submission":
                raise ValueError(message)
            raise RuntimeError(message)
        return value

    async def exec(self, command: str, timeout_s: float, no_output_timeout_s=None):
        return await self._call("exec", command, timeout_s, no_output_timeout_s)

    async def write_file(self, path: str, data: bytes) -> None:
        await self._call("write_file", path, data)

    async def evaluate(self, spec, command: str, timeout_s: float):
        return await self._call("evaluate", spec, command, timeout_s)


def _run_child(connection: Any, payload: dict, runner: Any) -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        payload["event_loop"] = loop
        payload["sandbox_handle"] = RemoteHandle(connection, payload.pop("sandbox_id"))
        connection.send(("result", runner(payload)))
    except BaseException as error:
        with suppress(BrokenPipeError, EOFError):
            connection.send(("error", f"Runner failed: {type(error).__name__}: {error}"))
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        connection.close()


async def run_in_process(payload: dict, handle: Any, runner: Any = None) -> dict:
    """Join or kill the runner before returning, including on timeout or cancellation."""
    if runner is None:
        from examples.airs_bench.runner import run_mlgym_agent

        runner = run_mlgym_agent
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    temporary = tempfile.TemporaryDirectory(prefix="airs-episode-")
    process = context.Process(
        target=_run_child,
        args=(child, {**payload, "sandbox_id": handle.sandbox_id, "episode_dir": temporary.name}, runner),
        daemon=True,
    )
    try:
        process.start()
        child.close()
        while True:
            while not parent.poll():
                if not process.is_alive():
                    raise RuntimeError(f"MLGym process exited without a result: {process.exitcode!r}.")
                await asyncio.sleep(0.02)
            message = await asyncio.to_thread(parent.recv)
            if message[0] == "result":
                return message[1]
            if message[0] == "error":
                raise RuntimeError(message[1])
            _, method, args = message
            if method not in {"exec", "write_file", "evaluate"}:
                raise RuntimeError(f"Unknown sandbox RPC method {method!r}.")
            try:
                result = await getattr(handle, method)(*args)
                reply = (True, result)
            except Exception as error:
                kind = "invalid_submission" if method == "evaluate" and isinstance(error, ValueError) else "runtime"
                reply = (False, (kind, f"Sandbox operation failed: {type(error).__name__}: {error}"))
            await asyncio.to_thread(parent.send, reply)
    finally:
        if process.is_alive():
            process.terminate()
        if process.pid is not None:
            await asyncio.to_thread(process.join, 2)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 2)
        parent.close()
        child.close()
        process.close()
        temporary.cleanup()
