"""Attribute host GPU samples to a container's live process identities."""

import asyncio
import threading
import time

from psrl.workers.env_service.process_identity import ProcfsIdentity


def read_command_memory(probe, tasks, indices):
    """
    Read current device memory and disjoint owner credits with verified process identities.

    Missing process attribution or process memory retains the full reservation. Historical
    peaks and utilization samples never count as current allocated memory.
    """
    eligible, errors = {}, {}
    for lease_id, task in tasks.items():
        try:
            task.check_identity_support()
            eligible[lease_id] = task
        except (OSError, RuntimeError) as exc:
            errors[lease_id] = str(exc)
    initial = probe.read_gpu_snapshot(indices)
    pids = {pid for reading in initial.values() for pid in reading["pids"]}
    before = {}
    for lease_id, task in eligible.items():
        try:
            before[lease_id] = task._owners(pids)
        except (OSError, RuntimeError) as exc:
            errors[lease_id] = str(exc)
    started_us = int(time.time() * 1e6)
    readings = probe.read_gpu_snapshot(indices) if before else initial
    owners, claimed = {}, set()
    for lease_id, identities in before.items():
        try:
            after = eligible[lease_id]._owners(pids)
        except (OSError, RuntimeError) as exc:
            errors[lease_id] = str(exc)
            continue
        stable = {
            pid
            for pid, identity in identities.items()
            if after.get(pid, (None,))[0] == identity[0] and identity[1] <= started_us
        }
        if stable & claimed:
            raise RuntimeError("GPU memory ownership overlaps between command leases.")
        claimed.update(stable)
        owners[lease_id] = stable
    credits = {lease_id: 0.0 for lease_id in owners}
    for index, reading in readings.items():
        if reading["uuid"] != initial[index]["uuid"]:
            raise RuntimeError("GPU identity changed during memory attribution.")
        amounts = {}
        for lease_id, owned in owners.items():
            values = (
                [reading["memory_mb_by_pid"].get(pid) for pid in owned] if index in eligible[lease_id].indices else []
            )
            amounts[lease_id] = sum(value for value in values if value is not None)
            if any(value is None for value in values):
                errors[lease_id] = "Some owned process memory is unavailable. Its reservation remains charged."
        if sum(amounts.values()) > reading["used_mb"]:
            errors[f"device:{index}"] = "Process memory exceeds the device sample. Ownership credit is withheld."
            continue
        for lease_id, amount in amounts.items():
            credits[lease_id] += amount
    return {"devices": readings, "owned_mb": credits, "attribution_errors": errors}


class TaskGpuTelemetry:
    """Keep only owned samples and report missing support instead of fabricated zeros."""

    def __init__(self, probe, fence, indices, *, interval_sec=0.2, identity_backend=None):
        self.probe, self.fence, self.indices = probe, fence, indices
        self.interval_sec = interval_sec
        self.started_us = int(time.time() * 1e6)
        self.since_us = self.started_us
        self.devices = {}
        self.errors = []
        self.stop_event = asyncio.Event()
        self.task = None
        self._lock = threading.Lock()
        self.identity = identity_backend or ProcfsIdentity()
        self.owner = self.identity.bind(fence) if fence is not None else None

    def start(self):
        self.task = asyncio.create_task(self._run())

    def _owners(self, pids):
        if self.owner is None:
            raise RuntimeError("Task GPU telemetry needs a bound container identity.")
        return self.owner.owners(pids)

    def check_identity_support(self):
        self.identity.check_support()

    def close_identity(self):
        if self.owner is not None:
            self.owner.close()

    def sample(self):
        self.check_identity_support()
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
                "identity_backend": self.identity.name,
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
