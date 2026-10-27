"""Read current GPU identities, ownership and utilization on the executing node."""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import threading
from contextlib import ExitStack, contextmanager
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


@contextmanager
def nvml_session():
    """Pair driver initialization and shutdown even when a safety query fails."""
    import pynvml

    pynvml.nvmlInit()
    try:
        yield pynvml
    finally:
        pynvml.nvmlShutdown()


class NvmlProbe:
    """Own a lazy driver connection and serialize live queries against its teardown."""

    def __init__(self):
        self._lock = threading.Lock()
        self._context = ExitStack()
        self._nvml = None

    @contextmanager
    def _session(self):
        with self._lock:
            if self._nvml is None:
                self._nvml = self._context.enter_context(nvml_session())
                atexit.register(self.close)
            yield self._nvml

    def close(self):
        """Wait for in-flight queries before releasing the owner's driver reference."""
        with self._lock:
            self._nvml = None
            try:
                self._context.close()
            finally:
                atexit.unregister(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def read_gpu_devices(self) -> dict[int, GpuDevice]:
        """Read current physical identities without caching device handles or mappings."""
        with self._session() as nvml:
            devices = {}
            for index in range(nvml.nvmlDeviceGetCount()):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                devices[index] = GpuDevice(
                    index, nvml.nvmlDeviceGetMinorNumber(handle), nvml.nvmlDeviceGetUUID(handle)
                )
            return devices

    def read_gpu_snapshot(self, indices):
        """Read live memory and compute ownership, propagating incomplete telemetry errors."""
        if not indices:
            raise ValueError("GPU handoff telemetry requires device indices.")
        with self._session() as nvml:
            readings = {}
            for index in indices:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                # Version 2 excludes driver reserved memory from application usage, matching `nvidia-smi`.
                memory = nvml.nvmlDeviceGetMemoryInfo(handle, version=nvml.nvmlMemory_v2)
                processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                process_memory = {}
                for process in processes:
                    value = process.usedGpuMemory
                    process_memory[process.pid] = None if value is None or value >= 2**64 - 1 else value / 1024**2
                readings[index] = {
                    "uuid": nvml.nvmlDeviceGetUUID(handle),
                    "used_mb": memory.used / 1024**2,
                    "total_mb": memory.total / 1024**2,
                    "pids": [process.pid for process in processes],
                    "memory_mb_by_pid": process_memory,
                }
            return readings

    def read_process_snapshot(self, indices, since_us):
        """Read process metrics without substituting device totals for missing samples."""
        with self._session() as nvml:
            readings = {}
            for index in indices:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                memory = {}
                for process in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    value = process.usedGpuMemory
                    memory[process.pid] = None if value is None or value >= 2**64 - 1 else value / 1024**2
                status = "available"
                try:
                    samples = nvml.nvmlDeviceGetProcessUtilization(handle, since_us)
                except nvml.NVMLError_NotFound:
                    status, samples = "no_samples", []
                except nvml.NVMLError_NotSupported:
                    status, samples = "unsupported", []
                readings[index] = {
                    "uuid": nvml.nvmlDeviceGetUUID(handle),
                    "memory_mb_by_pid": memory,
                    "utilization_status": status,
                    "samples": [{"pid": s.pid, "timestamp_us": s.timeStamp, "sm_util": s.smUtil} for s in samples],
                }
            return readings


def read_gpu_devices() -> dict[int, GpuDevice]:
    """Read inventory for a caller without a long-lived node service."""
    with NvmlProbe() as probe:
        return probe.read_gpu_devices()


def read_gpu_snapshot(indices):
    """Read a live snapshot and release the short-lived caller's driver reference."""
    with NvmlProbe() as probe:
        return probe.read_gpu_snapshot(indices)


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
        psrl_logger.debug(f"nvidia-smi unavailable ({exc!r}). No GPU measurement.")
        return {}
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        psrl_logger.debug(f"nvidia-smi timed out after {timeout_s:.1f}s. No GPU measurement.")
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return {}
    return parse_gpu_csv(stdout.decode(errors="replace"))
