"""
Build Docker Compose resources for a lease on the current host.

Resolve versioned CUDA and NVML libraries against the loaded kernel driver instead of
following unversioned symlinks. Include the PTX compiler when available. Resolve each
GPU telemetry index through the host inventory before mounting its character device.

Filesystem probes and GPU inventory can be injected for tests without a GPU.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from psrl.workers.env_service.probe import GpuDevice, read_gpu_devices
from psrl.workers.env_service.types import EnvLease

psrl_logger = logging.getLogger("psrl.env_service.overlay")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

LEASE_LABEL = "psrl.env_lease"
UNIT_LABEL = "psrl.env_unit"

# Host-wide control devices. They do not identify one physical GPU, so every container
# may bind-mount them regardless of which card it was assigned.
CONTROL_DEVICES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools")

DEFAULT_LIB_DIRS = (
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib",
)

# Where the libraries appear inside the container, matching the static override.
_CONTAINER_LIB_DIR = "/usr/lib64"
_CONTAINER_LD_LIBRARY_PATH = "/usr/lib64:/usr/local/cuda/lib64"

_KERNEL_VERSION_RE = re.compile(r"Kernel Module\s+([0-9][0-9.]*)")


class DriverLibraryError(RuntimeError):
    """A driver library that CUDA cannot start without was not found on the host."""


@dataclass(frozen=True)
class DriverLibs:
    """
    Driver libraries resolved for one kernel driver version.

    Attributes:
        version (str): The kernel module version the libraries match.
        mounts (tuple[tuple[str, str], ...]): `(host_path, container_path)` pairs.
        missing_optional (tuple[str, ...]): Optional libraries that were not found.
    """

    version: str
    mounts: tuple[tuple[str, str], ...]
    missing_optional: tuple[str, ...] = ()


def read_kernel_driver_version(proc_path: str = "/proc/driver/nvidia/version") -> str | None:
    """
    Read the loaded kernel module's driver version, e.g. `535.247.01`.

    This is the version that every user-space driver library must match. Returns `None`
    on a host with no NVIDIA driver loaded.
    """
    try:
        text = Path(proc_path).read_text()
    except OSError:
        return None
    match = _KERNEL_VERSION_RE.search(text)
    return match.group(1) if match else None


def _find_versioned(name: str, version: str, lib_dirs: Sequence[str], exists: Callable[[str], bool]) -> str | None:
    for directory in lib_dirs:
        candidate = os.path.join(directory, f"{name}.so.{version}")
        if exists(candidate):
            return candidate
    return None


def resolve_driver_libs(
    version: str,
    lib_dirs: Sequence[str] = DEFAULT_LIB_DIRS,
    exists: Callable[[str], bool] = os.path.exists,
) -> DriverLibs:
    """
    Find the driver libraries whose file name carries exactly `version`.

    Looking the file up by its full versioned name, instead of following the unversioned
    `libcuda.so.1` symlink, is what makes this immune to a symlink set that points at
    different driver versions (B1).

    Raises:
        DriverLibraryError: When `libcuda` or `libnvidia-ml` for `version` is missing.
    """
    mounts: list[tuple[str, str]] = []
    missing: list[str] = []

    cuda = _find_versioned("libcuda", version, lib_dirs, exists)
    nvml = _find_versioned("libnvidia-ml", version, lib_dirs, exists)
    if cuda is None or nvml is None:
        absent = [n for n, p in (("libcuda", cuda), ("libnvidia-ml", nvml)) if p is None]
        raise DriverLibraryError(
            f"No {' / '.join(absent)}.so.{version} under {list(lib_dirs)}. The user-space driver must match the "
            f"loaded kernel module {version}; check `cat /proc/driver/nvidia/version` against "
            f"`readlink -f /usr/lib64/libcuda.so.1`."
        )
    mounts.append((cuda, f"{_CONTAINER_LIB_DIR}/libcuda.so.1"))
    mounts.append((cuda, f"{_CONTAINER_LIB_DIR}/libcuda.so"))
    mounts.append((nvml, f"{_CONTAINER_LIB_DIR}/libnvidia-ml.so.1"))

    # B2: PTX JIT needs this inside the container. It is optional only so a host that
    # lacks it degrades to precompiled kernels instead of refusing every episode.
    ptx = _find_versioned("libnvidia-ptxjitcompiler", version, lib_dirs, exists)
    if ptx is None:
        missing.append("libnvidia-ptxjitcompiler")
        psrl_logger.warning(
            "libnvidia-ptxjitcompiler.so.%s not found under %s; PTX JIT will fail inside containers.",
            version,
            list(lib_dirs),
        )
    else:
        mounts.append((ptx, f"{_CONTAINER_LIB_DIR}/libnvidia-ptxjitcompiler.so.1"))
    return DriverLibs(version=version, mounts=tuple(mounts), missing_optional=tuple(missing))


def build_lease_overlay(
    lease: EnvLease,
    driver_libs: DriverLibs | None,
    *,
    device_exists: Callable[[str], bool] = os.path.exists,
    nvidia_smi_path: str = "/usr/bin/nvidia-smi",
    extra_labels: Sequence[str] = (),
    gpu_devices: dict[int, GpuDevice] | None = None,
) -> dict:
    """
    Build the Compose override for one lease's `main` service.

    The devices come from the lease, so each GPU episode mounts exactly its unit's cards
    (B3) and a unit's containers all see the same cards, which is what "one card, several
    containers" means at the Docker level. Every container is stamped with the lease and
    unit as labels, because Harbor sets no labels on local containers (B6) and the lease
    label is what lets TTL reclaim find an orphan.

    Args:
        lease (EnvLease): The granted lease.
        driver_libs (DriverLibs | None): Resolved libraries. Required for a GPU lease.
        device_exists (Callable[[str], bool]): Probe for control device nodes.
        nvidia_smi_path (str): Host path of `nvidia-smi`, mounted when present.
        extra_labels (Sequence[str]): Further `key=value` labels.

    Returns:
        dict: A mapping ready for `yaml.dump`, shaped `{"services": {"main": {...}}}`.

    Raises:
        ValueError: For a GPU lease given no `driver_libs`.
    """
    main: dict = {
        "labels": [
            f"{LEASE_LABEL}={lease.lease_id}",
            f"{UNIT_LABEL}={lease.unit_id}",
            *extra_labels,
        ]
    }
    if lease.gpu_indices:
        if driver_libs is None:
            raise ValueError("A GPU lease needs resolved driver libraries.")
        inventory = read_gpu_devices() if gpu_devices is None else gpu_devices
        devices = [f"{inventory[i].path}:{inventory[i].path}" for i in lease.gpu_indices]
        devices += [f"{d}:{d}" for d in CONTROL_DEVICES if device_exists(d)]
        volumes: list[str] = []
        if device_exists(nvidia_smi_path):
            volumes.append(f"{nvidia_smi_path}:/usr/bin/nvidia-smi:ro")
        volumes += [f"{host}:{container}:ro" for host, container in driver_libs.mounts]
        main["devices"] = devices
        main["volumes"] = volumes
        main["environment"] = [f"LD_LIBRARY_PATH={_CONTAINER_LD_LIBRARY_PATH}"]
    return {"services": {"main": main}}


def write_overlay(overlay: dict) -> Path:
    """
    Write an overlay to a fresh temp file and return its path.

    Uses `NamedTemporaryFile(delete=False)`, which creates the file atomically, instead of
    `tempfile.mktemp`, which only invents a name and leaves a race. The caller deletes it
    once the episode's containers are gone.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="env-service-lease-", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(overlay, handle, sort_keys=False)
        return Path(handle.name)
