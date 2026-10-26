"""Compare PATH lookup with a resolved executable without contacting the Docker daemon."""

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path


async def run(output):
    resolved = shutil.which("docker")
    if not resolved:
        raise RuntimeError("Docker client is unavailable.")
    result = {"resolved": resolved, "path": os.environ.get("PATH"), "attempts": []}
    async with asyncio.timeout(60):
        for pair in range(8):
            order = ["docker", resolved] if pair % 2 == 0 else [resolved, "docker"]
            for executable in order:
                started = time.monotonic()
                process = await asyncio.create_subprocess_exec(
                    executable, "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                spawned = time.monotonic()
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                finished = time.monotonic()
                result["attempts"].append(
                    {
                        "pair": pair,
                        "executable": executable,
                        "spawn_s": spawned - started,
                        "total_s": finished - started,
                        "exit_code": process.returncode,
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode(),
                    }
                )
                assert process.returncode == 0, "Docker client version query must succeed."
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
