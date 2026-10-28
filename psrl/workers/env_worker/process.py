"""Run bounded Docker client processes without abandoning them on cancellation."""

import asyncio
import codecs
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessResult:
    """Keep stdout, stderr and exit status distinct for stateless commands."""

    stdout: str
    stderr: str
    return_code: int


async def run_process(argv, timeout_s, output_limit=8 * 1024 * 1024, output_callback=None):
    """Bound combined output and terminate the client on timeout or cancellation."""
    process = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    remaining = output_limit

    async def read(stream, name):
        nonlocal remaining
        chunks = []
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := await stream.read(65536):
            remaining -= len(chunk)
            if remaining < 0:
                raise RuntimeError("Docker command output exceeded its byte limit.")
            chunks.append(chunk)
            if output_callback:
                await output_callback(decoder.decode(chunk), name)
        if output_callback:
            await output_callback(decoder.decode(b"", final=True), name)
        return b"".join(chunks).decode(errors="replace")

    readers = [
        asyncio.create_task(read(stream, name))
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr"))
    ]
    try:
        async with asyncio.timeout(timeout_s):
            stdout, stderr = await asyncio.gather(*readers)
            code = await process.wait()
        return ProcessResult(stdout, stderr, code)
    finally:
        for task in readers:
            if not task.done():
                task.cancel()
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await asyncio.gather(*readers, return_exceptions=True)
        await process.wait()
