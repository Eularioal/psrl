"""Attribute host GPU samples to a container's live process identities."""

import asyncio
import os
import threading
import time
from pathlib import Path


def process_identities(pids, root=Path("/proc"), membership=None):
    """Pair kernel birth ticks with an absolute timestamp for stale sample filtering."""
    boot = time.time() - time.clock_gettime(time.CLOCK_BOOTTIME)
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    identities = {}
    for pid in pids:
        try:
            proc = root / str(pid)
            if membership is not None:
                groups = [line.split(":", 2) for line in (proc / "cgroup").read_text().splitlines()]
                paths = [Path(path) for _, controllers, path in groups if "devices" in controllers.split(",")]
                if not any(path.is_relative_to(membership) for path in paths):
                    continue
            ticks = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
            identities[pid] = (ticks, int((boot + ticks / ticks_per_second) * 1e6))
        except FileNotFoundError:
            continue
    return identities


class TaskGpuTelemetry:
    """Keep only owned samples and report missing support instead of fabricated zeros."""

    def __init__(self, probe, fence, indices, *, interval_sec=0.2):
        self.probe, self.fence, self.indices = probe, fence, indices
        self.interval_sec = interval_sec
        self.started_us = int(time.time() * 1e6)
        self.since_us = self.started_us
        self.devices = {}
        self.errors = []
        self.stop_event = asyncio.Event()
        self.task = None
        self._lock = threading.Lock()
        self.proc_root = Path(os.environ.get("PSRL_GPU_HOST_PROC", "/proc"))

    def start(self):
        self.task = asyncio.create_task(self._run())

    def _owners(self, pids):
        return process_identities(pids, self.proc_root, self.fence.membership)

    def check_proc_root(self):
        # Linux reserves this inode for the initial PID namespace, matching NVML host PIDs.
        if (self.proc_root / "1/ns/pid").stat().st_ino != 0xEFFFFFFC:
            raise RuntimeError("GPU attribution needs host PID procfs. Mount it read-only and set PSRL_GPU_HOST_PROC.")

    def sample(self):
        self.check_proc_root()
        started_us = int(time.time() * 1e6)
        readings = self.probe.read_process_snapshot(self.indices, self.since_us)
        pids = {pid for reading in readings.values() for pid in reading["memory_mb_by_pid"]}
        pids.update(sample["pid"] for reading in readings.values() for sample in reading["samples"])
        before = self._owners(pids)
        after = self._owners(pids)
        owners = {
            pid: identity
            for pid, identity in before.items()
            if after.get(pid, (None,))[0] == identity[0] and identity[1] <= started_us
        }
        with self._lock:
            self.consume(readings, owners)

    def consume(self, readings, owners):
        for index, reading in readings.items():
            device = self.devices.setdefault(
                index,
                {
                    "uuid": reading["uuid"],
                    "peak_memory_mb": None,
                    "peak_sm_util": None,
                    "activity_samples": 0,
                    "observed_samples": 0,
                    "utilization_status": "no_samples",
                },
            )
            if device["uuid"] != reading["uuid"]:
                raise RuntimeError("GPU identity changed during task telemetry.")
            device["utilization_status"] = reading["utilization_status"]
            values = [value for pid, value in reading["memory_mb_by_pid"].items() if pid in owners]
            if all(value is not None for value in values):
                device["peak_memory_mb"] = max(device["peak_memory_mb"] or 0, sum(values))
            timestamps = {}
            for sample in reading["samples"]:
                timestamp = sample["timestamp_us"]
                if timestamp <= self.since_us:
                    continue
                birth = owners.get(sample["pid"])
                if birth is None or timestamp < max(self.started_us, birth[1]):
                    continue
                timestamps[timestamp] = max(timestamps.get(timestamp, 0), sample["sm_util"])
            for utilization in timestamps.values():
                device["observed_samples"] += 1
                device["activity_samples"] += int(utilization >= 10)
                device["peak_sm_util"] = max(device["peak_sm_util"] or 0, utilization)
        timestamps = [s["timestamp_us"] for r in readings.values() for s in r["samples"]]
        if timestamps:
            self.since_us = max(self.since_us, max(timestamps))

    async def _sample(self):
        try:
            await asyncio.wait_for(asyncio.to_thread(self.sample), timeout=5)
        except Exception as exc:
            self.stop_event.set()
            if len(self.errors) < 10:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    async def _run(self):
        while not self.stop_event.is_set():
            await self._sample()
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.interval_sec)
            except TimeoutError:
                pass

    async def stop(self):
        self.stop_event.set()
        if self.task:
            await asyncio.shield(self.task)
        return self.summary()

    def summary(self):
        with self._lock:
            return {
                "source": "host_nvml_container_cgroup",
                "started_us": self.started_us,
                "devices": {i: dict(value) for i, value in self.devices.items()},
                "errors": list(self.errors),
            }


def activity_verified(report, *, min_samples=3, min_utilization=10):
    """Require owned process activity, never a task supplied whole-device CSV."""
    return bool(
        report
        and report.get("source") == "host_nvml_container_cgroup"
        and not report["errors"]
        and any(
            device["observed_samples"] >= min_samples and (device["peak_sm_util"] or 0) >= min_utilization
            for device in report["devices"].values()
        )
    )
