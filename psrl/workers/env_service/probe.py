"""
Read per-device GPU telemetry with `nvidia-smi`.

The router cannot read a remote node's devices, so each agent loop worker reads its own
node and reports the result with its admission request. That keeps measured occupancy a
field on an existing message instead of a new mechanism (design document section 6.6).
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass

psrl_logger = logging.getLogger("psrl.env_service.probe")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class GpuDevice:
    """Resolve a telemetry index to its physical UUID and character device."""

    index: int
    minor: int
    uuid: str

    @property
    def path(self) -> str:
        """Return the host character device path."""
        return f"/dev/nvidia{self.minor}"


def read_gpu_devices() -> dict[int, GpuDevice]:
    """Read the host GPU inventory without assuming index equals minor number."""
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=30, check=True)
    devices = {}
    for index, element in enumerate(ET.fromstring(result.stdout).findall("gpu")):
        devices[index] = GpuDevice(index, int(element.findtext("minor_number")), element.findtext("uuid"))
    return devices


def parse_gpu_csv(text: str) -> dict[int, tuple[float, float]]:
    """
    Parse `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits`.

    Args:
        text (str): The command output.

    Returns:
        dict[int, tuple[float, float]]: `{index: (utilization_pct, memory_used_mb)}`. Malformed
        lines are skipped rather than failing the whole read.
    """
    data: dict[int, tuple[float, float]] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            data[int(parts[0])] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue
    return data


async def read_gpu_telemetry(timeout_s: float = 5.0) -> dict[int, tuple[float, float]]:
    """
    Run `nvidia-smi` and return `{index: (utilization_pct, memory_used_mb)}`.

    Returns `{}` when `nvidia-smi` is absent, errors, or times out, which callers treat as
    "no measurement" so a dev box with no GPU degrades to declared-only accounting. A timed
    out child is killed and reaped, so a wedged driver cannot leak one process per poll.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        psrl_logger.debug("nvidia-smi unavailable (%s); no GPU measurement.", exc)
        return {}
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        psrl_logger.debug("nvidia-smi timed out after %.1fs; no GPU measurement.", timeout_s)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return {}
    return parse_gpu_csv(stdout.decode(errors="replace"))
