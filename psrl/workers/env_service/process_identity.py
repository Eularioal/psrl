"""Select an explicit process identity source for task GPU telemetry."""

import os
import time
from pathlib import Path


def process_identities(pids, root=Path("/proc"), membership=None):
    """Pair kernel birth ticks with wall time for stale sample filtering."""
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


class ProcfsIdentity:
    """Resolve owned host PIDs using a verified host procfs mount."""

    name = "procfs"

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("PSRL_GPU_HOST_PROC", "/proc"))

    def check_support(self):
        if (self.root / "1/ns/pid").stat().st_ino != 0xEFFFFFFC:
            raise RuntimeError("GPU attribution needs host PID procfs. Mount it read-only and set PSRL_GPU_HOST_PROC.")

    def bind(self, fence):
        return ProcfsOwner(self, fence.membership)

    def close(self):
        pass


class ProcfsOwner:
    def __init__(self, backend, membership):
        self.backend, self.membership = backend, membership

    def owners(self, pids):
        return process_identities(pids, self.backend.root, self.membership)

    def close(self):
        pass


def validate_identity_backend(name):
    if name not in {"procfs", "ebpf"}:
        raise ValueError("GPU identity backend must be procfs or ebpf.")
    return name


def build_identity_backend(name):
    validate_identity_backend(name)
    if name == "procfs":
        return ProcfsIdentity()
    from psrl.workers.env_service.bpf_identity import BpfIdentity

    return BpfIdentity()
