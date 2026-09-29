#!/usr/bin/env python3
"""
BEFORE: reproduce the SciAccel GPU device contention bug (no Gate A).

This is the "before" half of a before/after pair; the "after" half is
`gate_a_after.py` in this same directory, which imports the real fix
(`psrl.utils.concurrency.gpu_device_gate.GpuDeviceGate`, wired into
`../runner.py`'s `run_harbor_episode`) and shows the identical scenario with
zero contention. Run both with the same `--num-episodes`/`--max-concurrent-episodes`
to compare directly.

What "before" means concretely: every GPU-needing Harbor episode used to mount
the literal, hardcoded device string written in `../config/gpu-compose-override.yaml`
(`/dev/nvidia0:/dev/nvidia0`, see that file's `services.main.devices`), because
`runner.py` (before Gate A was added) always did:

    extra_compose_paths.append(Path(config.harbor.gpu_compose_override))

-- the exact same file path, regardless of how many GPU episodes were running
at once or how many physical GPUs the host actually has. The only admission
control was `asyncio.Semaphore(max_concurrent_episodes)` in `../agent_loop.py`'s
`_acquire_episode_slot` -- it counts *how many* episodes run concurrently,
never *which device* each one gets.

This script reproduces that: it reads the real static override file to get
the real hardcoded device string, then runs several concurrent "episodes"
(real `asyncio` tasks, optionally real `docker run` containers with
`--use-docker`) that all request that same static device, and does a
sweep-line overlap analysis on their real start/end timestamps to show that
multiple episodes hold the identical device at the same wall-clock time.

Usage:
    python3 examples/sciaccel_rl/diagnostics/gate_a_before.py \\
        --max-concurrent-episodes 4 --num-episodes 6 --hold-seconds 1

    # On a real GPU training node, to also prove real containers overlap:
    python3 examples/sciaccel_rl/diagnostics/gate_a_before.py --use-docker \\
        --max-concurrent-episodes 4 --num-episodes 6 --hold-seconds 3

Exit code: 1 if contention was reproduced (peak simultaneous holders > 1 for
some device) -- this is the *expected* result of this "before" script, kept
non-zero so it is easy to assert against in a shell pipeline. 0 if no overlap
was observed (raise --num-episodes / lower --max-concurrent-episodes to try
harder to reproduce it).
"""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

SCIACCEL_DIR = Path(__file__).resolve().parents[1]
STATIC_OVERRIDE = SCIACCEL_DIR / "config" / "gpu-compose-override.yaml"


def read_static_device(override_path: Path) -> str:
    """Extract the first `/dev/nvidiaN` device string from the real override file."""
    text = override_path.read_text()
    match = re.search(r"-\s*(/dev/nvidia\d+):/dev/nvidia\d+", text)
    if not match:
        raise RuntimeError(f"Could not find a /dev/nvidiaN device entry in {override_path}")
    return match.group(1)


@dataclass
class Interval:
    episode: int
    device: str
    start: float
    end: float = field(default=-1.0)


async def run_episode_in_process(episode: int, device: str, hold_seconds: float, intervals: list[Interval], t0: float) -> None:
    start = time.monotonic()
    print(f"  t={start - t0:6.2f}  BEGIN  episode={episode:<3} device={device}")
    interval = Interval(episode=episode, device=device, start=start)
    intervals.append(interval)
    await asyncio.sleep(hold_seconds)
    interval.end = time.monotonic()
    print(f"  t={interval.end - t0:6.2f}  END    episode={episode:<3} device={device}")


async def run_episode_in_docker(episode: int, device: str, hold_seconds: float, intervals: list[Interval], t0: float) -> None:
    """Real `docker run --device <device>` holding the device for `hold_seconds`."""
    name = f"sciaccel-gate-a-before-{episode}-{uuid.uuid4().hex[:8]}"
    start = time.monotonic()
    print(f"  t={start - t0:6.2f}  BEGIN  episode={episode:<3} device={device}  (docker container {name})")
    interval = Interval(episode=episode, device=device, start=start)
    intervals.append(interval)
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--device",
        device,
        "busybox",
        "sleep",
        str(hold_seconds),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    await proc.wait()
    interval.end = time.monotonic()
    print(f"  t={interval.end - t0:6.2f}  END    episode={episode:<3} device={device}")


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
        tag = "CONTENTION -- multiple episodes held this device at once" if peak > 1 else "ok, no overlap"
        print(f"  device={device!r}: peak simultaneous holders = {peak}  -> {tag}")
        peaks[device] = peak
    return peaks


async def main_async(args: argparse.Namespace) -> int:
    device = read_static_device(STATIC_OVERRIDE)
    print(f"Static override device (read from {STATIC_OVERRIDE}): {device}")
    print(
        f"Scenario: max_concurrent_episodes={args.max_concurrent_episodes}, "
        f"num_episodes={args.num_episodes}, use_docker={args.use_docker}"
    )
    print("(No Gate A: every GPU episode requests the identical static device string.)")

    semaphore = asyncio.Semaphore(args.max_concurrent_episodes)
    intervals: list[Interval] = []
    t0 = time.monotonic()
    runner = run_episode_in_docker if args.use_docker else run_episode_in_process

    async def gated(ep: int) -> None:
        async with semaphore:
            await runner(ep, device, args.hold_seconds, intervals, t0)

    print("Timeline (t = seconds since first BEGIN):")
    await asyncio.gather(*(gated(ep) for ep in range(args.num_episodes)))

    peaks = overlap_report(intervals)
    worst = max(peaks.values(), default=0)
    if worst > 1:
        print(
            f"RESULT: reproduced GPU device contention. Up to {worst} episodes were simultaneously "
            f"assigned the same physical device string ({device}) at the same time."
        )
        return 1
    print(
        "RESULT: no overlap observed with this concurrency level -- try raising --num-episodes "
        "or --max-concurrent-episodes to reproduce it."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-concurrent-episodes", type=int, default=4)
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Use real `docker run --device` containers instead of in-process asyncio sleeps.",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
