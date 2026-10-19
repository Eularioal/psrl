"""
Gate B on real hardware: how long does a reply wait behind a neighbour on the same card?

Run once with PYTHONPATH on the pre-change tree and once on the new tree.
"""

import asyncio
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PSRL_GATE_B_MAX_WAIT_SEC"] = "8"
os.environ["PSRL_GATE_B_POLL_INTERVAL_SEC"] = "1"

from psrl.workers.gen import session_router  # noqa: E402

HERE = Path(__file__).resolve().parent
HAS_SHARED = "shared" in inspect.signature(session_router._wait_for_gpu_idle).parameters


def smi(index):
    out = subprocess.run(
        ["nvidia-smi", "-i", str(index), "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    util, mem = (float(x) for x in out.split(","))
    return util, mem


async def timed(device, **kwargs):
    t0 = time.monotonic()
    await session_router._wait_for_gpu_idle(device, "verify", **kwargs)
    return round(time.monotonic() - t0, 1)


async def start(script, *args, card):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(card))
    return await asyncio.create_subprocess_exec(
        sys.executable, str(HERE / script), *map(str, args), env=env, stdout=asyncio.subprocess.PIPE
    )


async def main():
    out = {"tree_has_shared_param": HAS_SHARED, "cap_s": 8}

    # Neighbour holds weights on card 0 and does no compute.
    holder = await start("gpu_episode.py", 22, -1, "resident", card=0)
    await asyncio.wait_for(holder.stdout.readline(), timeout=120)
    await asyncio.sleep(2)
    out["card0_neighbour_idle_util_mem"] = smi(0)
    out["resident_neighbour__unshared_wait_s"] = await timed("/dev/nvidia0")
    if HAS_SHARED:
        out["resident_neighbour__shared_wait_s"] = await timed("/dev/nvidia0", shared=True)
    holder.kill()
    await holder.wait()
    await asyncio.sleep(2)

    # Neighbour really computes for 5s on card 1 (shorter than the cap); time only after it reports READY.
    busy = await start("gpu_busy.py", 5, card=1)
    await asyncio.wait_for(busy.stdout.readline(), timeout=120)
    await asyncio.sleep(1.5)
    out["card1_neighbour_busy_util_mem"] = smi(1)
    t0 = time.monotonic()
    if HAS_SHARED:
        out["busy_neighbour__shared_wait_s"] = await timed("/dev/nvidia1", shared=True)
    else:
        out["busy_neighbour__unshared_wait_s"] = await timed("/dev/nvidia1")
    out["busy_neighbour__seen_after_s"] = round(time.monotonic() - t0, 1)
    await busy.wait()
    print("RESULT " + json.dumps(out))


asyncio.run(main())
