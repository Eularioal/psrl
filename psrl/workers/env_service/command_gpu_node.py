"""Fence GPU device access while persistent task containers continue CPU work."""

import asyncio
import os
import stat
import time
from pathlib import Path

from psrl.workers.env_service.docker_api import inspect_container
from psrl.workers.env_service.overlay import CONTROL_DEVICES, LEASE_LABEL
from psrl.workers.env_service.task_telemetry import TaskGpuTelemetry


class DeviceCgroupFence:
    """Control a Docker container's device whitelist on a cgroup v1 host."""

    def __init__(self, container, allowed_paths, root=Path("/sys/fs/cgroup/devices")):
        config = container["HostConfig"]
        if config.get("Privileged") or config.get("DeviceRequests") or config.get("DeviceCgroupRules"):
            raise RuntimeError("GPU command containers must use explicit, restricted device mappings.")
        if set(config.get("CapAdd") or ()) & {"ALL", "SYS_ADMIN"}:
            raise RuntimeError("GPU command containers cannot administer device cgroups.")
        self.devices = []
        paths = {d["PathOnHost"] for d in config.get("Devices") or ()}
        if paths != set(allowed_paths):
            raise RuntimeError("Container GPU devices do not match its lease.")
        if any(d["CgroupPermissions"] != "m" for d in config.get("Devices") or ()):
            raise RuntimeError("GPU command containers must start with device access denied.")
        for path in paths:
            info = os.stat(path)
            if not stat.S_ISCHR(info.st_mode):
                raise RuntimeError(f"GPU path {path!r} is not a character device.")
            self.devices.append((os.major(info.st_rdev), os.minor(info.st_rdev)))
        pid = container["State"]["Pid"]
        memberships = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        relative = next(
            (line.split(":", 2)[2] for line in memberships if "devices" in line.split(":")[1].split(",")), None
        )
        if relative is None:
            raise RuntimeError("GPU command fencing requires the cgroup v1 devices controller.")
        mount_root = self.mount_root(root, Path("/proc/self/mountinfo").read_text())
        membership = Path(relative)
        self.membership = membership
        if not membership.is_relative_to(mount_root):
            raise RuntimeError("Container device cgroup is outside the visible controller mount.")
        self.path = (root / membership.relative_to(mount_root)).resolve()
        if not self.path.is_relative_to(root.resolve()) or container["Id"] not in self.path.as_posix():
            raise RuntimeError("Container device cgroup identity could not be verified.")

    @staticmethod
    def mount_root(root, mountinfo):
        """Resolve the controller's mount root inside a containerized execution node."""
        for line in mountinfo.splitlines():
            before, after = line.split(" - ", 1)
            fields, filesystem = before.split(), after.split()
            if filesystem[0] == "cgroup" and "devices" in filesystem[2].split(","):
                if Path(fields[4]) == root:
                    return Path(fields[3])
        raise RuntimeError("Cannot resolve the devices controller mount root.")

    def set_access(self, enabled):
        """Change only the GPU devices assigned to this container."""
        operation = "devices.allow" if enabled else "devices.deny"
        for major, minor in self.devices:
            (self.path / operation).write_text(f"c {major}:{minor} rwm\n")
        permissions = (self.path / "devices.list").read_text().splitlines()
        for major, minor in self.devices:
            readable = writable = False
            for line in permissions:
                kind, numbers, rights = line.split()
                m, n = numbers.split(":")
                if kind in ("a", "c") and m in ("*", str(major)) and n in ("*", str(minor)):
                    readable |= "r" in rights
                    writable |= "w" in rights
            if enabled and not (readable and writable) or not enabled and (readable or writable):
                raise RuntimeError("GPU device permissions did not reach the requested state.")

    def processes(self):
        """Include descendants so a task cannot hide CUDA work in a nested cgroup."""
        return {int(pid) for path in self.path.rglob("cgroup.procs") for pid in path.read_text().split()}

    def gpu_handles(self, pids):
        """Deny new opens before checking existing descriptors and device mappings."""
        found = []
        for pid in pids:
            proc = Path(f"/proc/{pid}")
            try:
                for fd in (proc / "fd").iterdir():
                    try:
                        info = fd.stat()
                    except FileNotFoundError:
                        continue
                    if stat.S_ISCHR(info.st_mode) and (os.major(info.st_rdev), os.minor(info.st_rdev)) in self.devices:
                        found.append(pid)
                        break
                if "/dev/nvidia" in (proc / "maps").read_text():
                    found.append(pid)
            except FileNotFoundError:
                continue
        return found


class GpuCommandNodeMixin:
    """Apply command epochs to owned containers without freezing their CPU processes."""

    async def check_gpu_command_support(self):
        if not Path("/sys/fs/cgroup/devices/devices.list").exists():
            raise RuntimeError("GPU command reservations require a cgroup v1 devices controller.")
        return True

    async def check_gpu_attribution_support(self, indices):
        telemetry = TaskGpuTelemetry(self._gpu_probe(), None, indices)
        telemetry.check_proc_root()
        readings = await asyncio.wait_for(
            asyncio.to_thread(self._gpu_probe().read_process_snapshot, indices, 0), timeout=5
        )
        if any(reading["utilization_status"] == "unsupported" for reading in readings.values()):
            raise RuntimeError("This GPU does not support process utilization attribution.")
        return True

    async def gpu_job_telemetry(self, lease_id, container_id, epoch):
        state = self._gpu_lease_state(lease_id)
        if state.get("command_epoch") != epoch or state.get("command_container") != container_id:
            raise RuntimeError("GPU telemetry request has a stale owner or epoch.")
        telemetry = state.get("command_telemetry")
        return telemetry.summary() if telemetry else None

    async def _command_container(self, lease_id, container_id, indices):
        container = await inspect_container(container_id)
        if container["Config"].get("Labels", {}).get(LEASE_LABEL) != lease_id:
            raise RuntimeError("GPU command container does not belong to this lease.")
        if not container["State"]["Running"] or container["State"]["Paused"]:
            raise RuntimeError("GPU command container must be running and unpaused.")
        inventory = await asyncio.wait_for(asyncio.to_thread(self._gpu_probe().read_gpu_devices), timeout=10)
        paths = [inventory[i].path for i in indices]
        paths.extend(p for p in CONTROL_DEVICES if Path(p).exists())
        return await asyncio.to_thread(DeviceCgroupFence, container, paths)

    async def grant_gpu_command(self, lease_id, container_id, indices, epoch):
        state = self._gpu_lease_state(lease_id)
        async with state["lock"]:
            if state["retired"] or epoch <= state.get("command_epoch", 0):
                raise RuntimeError("GPU command activation has a stale lease or epoch.")
            state["command_epoch"] = epoch
            state["command_container"] = container_id
            state["command_returned"] = False
            fence = await self._command_container(lease_id, container_id, indices)
            await asyncio.to_thread(fence.set_access, True)
            telemetry = TaskGpuTelemetry(self._gpu_probe(), fence, indices)
            state["command_telemetry"] = telemetry
            telemetry.start()

    async def revoke_gpu_command(self, lease_id, container_id, indices, epoch, telemetry_indices):
        """Block future opens and require all existing GPU ownership to disappear."""
        state = self._gpu_lease_state(lease_id)
        async with state["lock"]:
            if state.get("command_epoch") != epoch or state.get("command_container") != container_id:
                raise RuntimeError("GPU command return has a stale epoch or container.")
            fence = await self._command_container(lease_id, container_id, indices)
            await asyncio.to_thread(fence.set_access, False)
            telemetry = state.get("command_telemetry")
            report = await telemetry.stop() if telemetry else None
            deadline = time.monotonic() + 10
            while True:
                pids = await asyncio.to_thread(fence.processes)
                handles = await asyncio.to_thread(fence.gpu_handles, pids)
                readings = await self.gpu_lending_snapshot(telemetry_indices)
                computing = pids & {pid for reading in readings.values() for pid in reading["pids"]}
                if not handles and not computing:
                    state["command_returned"] = True
                    for reading in readings.values():
                        reading["task_telemetry"] = report
                    return readings
                if time.monotonic() >= deadline:
                    raise RuntimeError("GPU command left device handles or CUDA processes after completion.")
                await asyncio.sleep(0.1)
