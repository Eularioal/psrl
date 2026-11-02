"""Attribute a coherent GPU memory sample to live episode container sets."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from psrl.workers.env_service.command_gpu_node import DeviceCgroupFence
from psrl.workers.env_service.docker_api import inspect_container, list_unit_containers
from psrl.workers.env_service.overlay import LEASE_LABEL
from psrl.workers.env_service.task_telemetry import TaskGpuTelemetry, read_command_memory


def container_identity(container):
    """Resolve a restricted container's device cgroup without changing its permissions."""
    config = container["HostConfig"]
    capabilities = {name.removeprefix("CAP_") for name in config.get("CapAdd") or ()}
    if config.get("Privileged") or capabilities & {"ALL", "SYS_ADMIN", "BPF", "PERFMON"}:
        raise RuntimeError("Episode memory attribution requires an isolated container.")
    root = Path("/sys/fs/cgroup/devices")
    pid = container["State"]["Pid"]
    groups = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    relative = next(line.split(":", 2)[2] for line in groups if "devices" in line.split(":")[1].split(","))
    membership = Path(relative)
    mount_root = DeviceCgroupFence.mount_root(root, Path("/proc/self/mountinfo").read_text())
    path = (root / membership.relative_to(mount_root)).resolve(strict=True)
    names = {container["Id"], f"docker-{container['Id']}.scope"}
    if not path.is_relative_to(root.resolve()) or not names.intersection(path.parts):
        raise RuntimeError("Episode container cgroup identity could not be verified.")
    return SimpleNamespace(path=path, membership=membership)


def signature(container):
    """Detect container restarts and lease replacement across a memory query."""
    state = container["State"]
    return (container["Id"], state["Pid"], state["StartedAt"], container["Config"].get("Labels", {}).get(LEASE_LABEL))


class EpisodeMemory:
    """Keep identity registrations while sampling fresh ownership for every admission."""

    def __init__(self, probe, identity):
        self.probe, self.identity = probe, identity
        self.bindings = {}
        self.lock = asyncio.Lock()

    def drop(self, container_id):
        item = self.bindings.pop(container_id, None)
        if item:
            item["task"].close_identity()

    async def close(self):
        async with self.lock:
            for container_id in list(self.bindings):
                self.drop(container_id)

    async def snapshot(self, indices, leases):
        async with self.lock:
            return await self._snapshot(indices, leases)

    async def _snapshot(self, indices, leases):
        expected = {lease_id: (unit_id, tuple(devices)) for lease_id, unit_id, devices in leases}
        rows = await list_unit_containers()
        selected = {
            row["Id"]: row
            for row in rows
            if row.get("Labels", {}).get(LEASE_LABEL) in expected and row.get("State") in ("running", "paused")
        }
        for container_id, item in list(self.bindings.items()):
            if container_id not in selected or item["lease_id"] not in expected:
                self.drop(container_id)
        tasks, errors = {}, {}
        for container_id, row in selected.items():
            lease_id = row["Labels"][LEASE_LABEL]
            unit_id, devices = expected[lease_id]
            try:
                container = await inspect_container(container_id)
                labels = container["Config"].get("Labels", {})
                if labels.get(LEASE_LABEL) != lease_id or labels.get("psrl.env_unit") != unit_id:
                    raise RuntimeError("Episode container labels changed during discovery.")
                if not container["State"]["Running"]:
                    raise RuntimeError("Episode container stopped during discovery.")
                key = signature(container)
                item = self.bindings.get(container_id)
                if item and (item["signature"] != key or item["task"].indices != devices):
                    self.drop(container_id)
                    item = None
                if item is None:
                    fence = await asyncio.to_thread(container_identity, container)
                    task = TaskGpuTelemetry(self.probe, fence, devices, identity_backend=self.identity)
                    item = {"lease_id": lease_id, "signature": key, "task": task}
                    self.bindings[container_id] = item
                tasks[container_id] = item["task"]
            except Exception as exc:
                self.drop(container_id)
                errors[container_id] = str(exc)
        query = asyncio.create_task(asyncio.to_thread(read_command_memory, self.probe, tasks, indices))
        try:
            result = await asyncio.shield(query)
        except BaseException:
            await asyncio.gather(query, return_exceptions=True)
            raise
        owned = {}
        for container_id in tasks:
            item = self.bindings[container_id]
            try:
                current = await inspect_container(container_id)
                if not current["State"]["Running"] or signature(current) != item["signature"]:
                    raise RuntimeError("Episode container changed during the memory query.")
            except Exception as exc:
                errors[container_id] = str(exc)
                self.drop(container_id)
                continue
            values = result["owned_mb_by_device"].get(container_id, {})
            credit = owned.setdefault(item["lease_id"], {})
            for index, amount in values.items():
                credit[index] = credit.get(index, 0.0) + amount
        result["owned_mb_by_device"] = owned
        result["owned_mb"] = {lease: sum(values.values()) for lease, values in owned.items()}
        result["attribution_errors"].update(errors)
        return result
