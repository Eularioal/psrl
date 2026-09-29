#!/usr/bin/env python3
"""
AFTER: same scenario as `gate_a_before.py`, but with Gate A applied.

This is the "after" half of the before/after pair. Unlike the "before" script
(which hand-rolls the old, gate-less behavior), this script imports and
exercises the *real, deployed* fix directly:

    from psrl.utils.concurrency.gpu_device_gate import GpuDeviceGate, AllocatedGpu

`GpuDeviceGate` is the module wired into `../runner.py`'s `run_harbor_episode`
(Gate A): before this change, every GPU episode always mounted the literal
static device from `gpu-compose-override.yaml`; now each episode calls
`GpuDeviceGate.acquire()` first, which hands out one *physical* device index
exclusively (backed by `SlotManager`'s `fcntl` file locks -- see
`psrl/utils/concurrency/gpu_device_gate.py`) and blocks until one is free,
instead of handing every caller the same string.

Two things this script checks, both against the *real* module (not a mirror):

1. **Single-process concurrency** (asyncio tasks in this one interpreter):
   same shape as the "before" script -- N concurrent episodes, sweep-line
   overlap analysis -- but now each episode calls `GpuDeviceGate.acquire()` /
   `.release()` around its work, and no two overlap on the same device.

2. **Cross-process exclusivity** (the actual bug scenario): SciAccel's real
   deployment runs multiple independent `AgentLoopWorker` *processes* on one
   node, which never share Python memory, so an in-memory semaphore/dict
   cannot coordinate them -- this is exactly why Gate A is built on
   `SlotManager`'s `fcntl` locks rather than an `asyncio.Semaphore`. This
   script spawns real OS subprocesses (`multiprocessing.Process`, one per
   simulated worker) that each independently import `GpuDeviceGate` and race
   for the same device pool, to prove the exclusivity holds across process
   boundaries too, not just within one asyncio event loop.

Usage:
    python3 examples/sciaccel_rl/diagnostics/gate_a_after.py \\
        --max-concurrent-episodes 4 --num-episodes 6 --hold-seconds 1 --num-gpus 3

    # Prove cross-process exclusivity with real OS processes:
    python3 examples/sciaccel_rl/diagnostics/gate_a_after.py --cross-process \\
        --num-workers 4 --num-episodes 12 --hold-seconds 0.3 --num-gpus 3

    # On a real GPU node, drop --num-gpus / PSRL_SCIACCEL_GPU_DEVICES entirely
    # and GpuDeviceGate will glob the real /dev/nvidia[0-9]* devices instead.

Exit code: 0 if zero overlap was observed on every device (the expected,
"fix works" result). 1 if any device saw >1 simultaneous holder (would mean
Gate A itself has a bug -- report this).
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# This file lives at <repo_root>/examples/sciaccel_rl/diagnostics/gate_a_after.py;
# <repo_root> is what needs to be on sys.path for `import psrl...` to resolve,
# same as how the rest of examples/sciaccel_rl/ resolves that import.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psrl.utils.concurrency.gpu_device_gate import (  # noqa: E402
    GpuDeviceGate,
    discover_gpu_devices,
)


@dataclass
class Interval:
    episode: int
    device: str
    start: float
    end: float = field(default=-1.0)


def overlap_report(intervals: list[Interval]) -> dict[str, int]:
    """Sweep-line peak-concurrency check per device. Returns {device: peak_holders}."""
    by_device: dict[str, list[Interval]] = {}
    for iv in intervals:
        by_device.setdefault(iv.device, []).append(iv)

    peaks: dict[str, int] = {}
    print("Overlap analysis per device:")
    for device, ivs in sorted(by_device.items()):
        events = []
        for iv in ivs:
            events.append((iv.start, 1))
            events.append((iv.end, -1))
        events.sort()
        peak = cur = 0
        for _, delta in events:
            cur += delta
            peak = max(peak, cur)
        tag = "ok, exclusive" if peak <= 1 else "CONTENTION -- Gate A failed to keep this device exclusive"
        print(f"  device={device!r}: peak simultaneous holders = {peak}  -> {tag}")
        peaks[device] = peak
    return peaks


# --- Mode 1: single-process asyncio concurrency, real GpuDeviceGate ---


async def gated_episode(
    episode: int, gate: GpuDeviceGate, hold_seconds: float, intervals: list[Interval], t0: float
) -> None:
    allocated = await gate.acquire()
    assert allocated is not None, "no GPU devices discovered -- set PSRL_SCIACCEL_GPU_DEVICES for this demo"
    start = time.monotonic()
    print(f"  t={start - t0:6.2f}  BEGIN  episode={episode:<3} device={allocated.device}  (Gate A slot {allocated.index})")
    interval = Interval(episode=episode, device=allocated.device, start=start)
    intervals.append(interval)
    try:
        await asyncio.sleep(hold_seconds)
    finally:
        interval.end = time.monotonic()
        print(f"  t={interval.end - t0:6.2f}  END    episode={episode:<3} device={allocated.device}")
        GpuDeviceGate.release(allocated)


async def run_single_process(args: argparse.Namespace, jobs_dir: str) -> int:
    devices = discover_gpu_devices()
    print(f"Discovered {len(devices)} GPU device(s) for this demo: {devices}")
    print(
        f"Scenario: max_concurrent_episodes={args.max_concurrent_episodes}, "
        f"num_episodes={args.num_episodes}"
    )
    print("(Gate A active: each GPU episode reserves one physical device exclusively via GpuDeviceGate.)")

    gate = GpuDeviceGate(jobs_dir=jobs_dir)
    semaphore = asyncio.Semaphore(args.max_concurrent_episodes)
    intervals: list[Interval] = []
    t0 = time.monotonic()

    async def gated(ep: int) -> None:
        async with semaphore:
            await gated_episode(ep, gate, args.hold_seconds, intervals, t0)

    print("Timeline (t = seconds since first BEGIN):")
    await asyncio.gather(*(gated(ep) for ep in range(args.num_episodes)))

    peaks = overlap_report(intervals)
    worst = max(peaks.values(), default=0)
    if worst <= 1:
        print(f"RESULT: zero contention across {len(devices)} device(s) and {args.num_episodes} episodes.")
        return 0
    print(f"RESULT: Gate A failed -- peak {worst} simultaneous holders on one device. This is a bug, report it.")
    return 1


# --- Mode 2: real OS subprocesses, proving fcntl-backed cross-process exclusivity ---


def _worker_process(
    worker_id: int, episodes_per_worker: int, hold_seconds: float, jobs_dir: str, repo_root: str, result_queue: multiprocessing.Queue
) -> None:
    """Runs in its own OS process -- mirrors one independent AgentLoopWorker process."""
    # Re-import inside the child: multiprocessing's "spawn" start method does not
    # inherit the parent's already-imported modules, so this exercises a fully
    # independent `import psrl...` in each process, same as real deployment.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from psrl.utils.concurrency.gpu_device_gate import GpuDeviceGate as _Gate

    async def _run() -> list[tuple[str, float, float]]:
        gate = _Gate(jobs_dir=jobs_dir)
        rows: list[tuple[str, float, float]] = []
        for _ in range(episodes_per_worker):
            allocated = await gate.acquire()
            assert allocated is not None
            start = time.monotonic()
            await asyncio.sleep(hold_seconds)
            end = time.monotonic()
            rows.append((allocated.device, start, end))
            _Gate.release(allocated)
        return rows

    rows = asyncio.run(_run())
    for device, start, end in rows:
        result_queue.put((worker_id, device, start, end))


def run_cross_process(args: argparse.Namespace, jobs_dir: str) -> int:
    devices = discover_gpu_devices()
    print(f"Discovered {len(devices)} GPU device(s) for this demo: {devices}")
    print(
        f"Scenario: {args.num_workers} independent OS processes, "
        f"{args.num_episodes} episodes/process, jobs_dir={jobs_dir}"
    )
    print(
        "(Gate A active, cross-process: each worker process independently imports GpuDeviceGate "
        "and contends for the same fcntl-locked slot pool.)"
    )

    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker_process,
            args=(w, args.num_episodes, args.hold_seconds, jobs_dir, str(REPO_ROOT), result_queue),
        )
        for w in range(args.num_workers)
    ]
    t0 = time.monotonic()
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    intervals: list[Interval] = []
    while not result_queue.empty():
        worker_id, device, start, end = result_queue.get()
        intervals.append(Interval(episode=worker_id, device=device, start=start - t0, end=end - t0))
    intervals.sort(key=lambda iv: iv.start)
    print("Timeline (t = seconds since first process start):")
    for iv in intervals:
        print(f"  t={iv.start:6.2f}..{iv.end:6.2f}  worker={iv.episode:<3} device={iv.device}")

    peaks = overlap_report(intervals)
    worst = max(peaks.values(), default=0)
    total_episodes = len(intervals)
    if worst <= 1:
        print(
            f"RESULT: zero cross-process contention across {args.num_workers} OS processes, "
            f"{total_episodes} total episode(s), {len(devices)} device(s)."
        )
        return 0
    print(f"RESULT: Gate A failed across processes -- peak {worst} simultaneous holders on one device.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-concurrent-episodes", type=int, default=4, help="Single-process mode only.")
    parser.add_argument("--num-episodes", type=int, default=6, help="Total episodes (single-process) or per-worker (cross-process).")
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=int, default=3, help="Fake device count when no real GPU is present.")
    parser.add_argument("--cross-process", action="store_true", help="Use real OS subprocesses instead of asyncio tasks.")
    parser.add_argument("--num-workers", type=int, default=4, help="Cross-process mode only: number of worker processes.")
    parser.add_argument("--jobs-dir", default="/tmp/sciaccel_gate_a_after_demo", help="SlotManager lock directory (shared across processes).")
    args = parser.parse_args()

    if args.num_gpus > 0:
        os.environ.setdefault("PSRL_SCIACCEL_GPU_DEVICES", ",".join(f"/dev/nvidia{i}" for i in range(args.num_gpus)))
    os.makedirs(args.jobs_dir, exist_ok=True)

    if args.cross_process:
        return run_cross_process(args, args.jobs_dir)
    return asyncio.run(run_single_process(args, args.jobs_dir))


if __name__ == "__main__":
    sys.exit(main())
