"""Isolate command devices while retaining the container's CPU and memory accounting."""

import os
import signal
import time
from pathlib import Path

from psrl.workers.env_service.command_gpu_node import DeviceCgroupFence


class JobFence:
    """Own sibling device and freezer groups for one admitted command."""

    def __init__(self, name, container_fence):
        self.path = Path("/sys/fs/cgroup/devices") / name
        self.freezer = Path("/sys/fs/cgroup/freezer") / name
        self.devices = container_fence.devices
        self.path.mkdir()
        try:
            self.freezer.mkdir()
            (self.path / "devices.deny").write_text("a")
            for rule in (container_fence.path / "devices.list").read_text().splitlines():
                (self.path / "devices.allow").write_text(rule)
            self.set_access(False)
        except BaseException:
            self.close()
            raise

    set_access = DeviceCgroupFence.set_access
    processes = DeviceCgroupFence.processes
    gpu_handles = DeviceCgroupFence.gpu_handles

    def attach(self, pid):
        (self.freezer / "cgroup.procs").write_text(str(pid))
        (self.path / "cgroup.procs").write_text(str(pid))

    def freeze(self):
        (self.freezer / "freezer.state").write_text("FROZEN")
        deadline = time.monotonic() + 5
        while (self.freezer / "freezer.state").read_text().strip() != "FROZEN":
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU job processes did not freeze.")
            time.sleep(0.01)

    def kill(self):
        for pid in self.processes():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        (self.freezer / "freezer.state").write_text("THAWED")

    def thaw(self):
        (self.freezer / "freezer.state").write_text("THAWED")

    def close(self):
        for path in (self.path, self.freezer):
            if path.exists():
                path.rmdir()


def container_pid(fence, namespace, inner_pid):
    """Resolve only a process in the verified container's device group."""
    matches = []
    for pid in fence.processes():
        proc = Path("/proc") / str(pid)
        try:
            if (proc / "ns/pid").stat().st_ino != namespace:
                continue
            status = (proc / "status").read_text().splitlines()
            ids = next(line.split()[1:] for line in status if line.startswith("NSpid:"))
            if int(ids[-1]) == inner_pid:
                matches.append(pid)
        except FileNotFoundError:
            continue
    if len(matches) != 1:
        raise RuntimeError("GPU job process is not uniquely owned by its container.")
    return matches[0]
