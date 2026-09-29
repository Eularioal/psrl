"""
Gate A for the SciAccel (Harbor) path: exclusive per-device GPU reservation.

Background (see docs/psrl/psrl-env-service-envworker-design.md section 15.3(1)
and section 16 for the full writeup): every GPU-needing SciAccel episode used
to mount the exact same hardcoded ``/dev/nvidia0`` from
``examples/sciaccel_rl/config/gpu-compose-override.yaml``, and nothing in
``runner.py``/``agent_loop.py`` tracked *which physical device* was already in
use. The only admission control was ``asyncio.Semaphore(max_concurrent_episodes)``
in ``agent_loop.py``'s ``_acquire_episode_slot`` -- it counts "how many
episodes", never "which device", so two concurrent GPU episodes could (and, on
a multi-GPU host, would) be handed the identical device string at once.

This module is Gate A for that path: it reserves one physical GPU device index
per GPU-needing episode before the container is created, and releases it after
the episode's containers are torn down. It mirrors the *mechanism* the old
EnvWorker already has for its own (non-Harbor) path --
``EnvWorker._allocate_gpu_indices``/``_busy_gpu_indices``
(psrl/psrl/workers/env_worker/worker.py:171-187), an in-memory busy-set inside
one Ray actor -- but implemented with ``SlotManager``'s ``fcntl`` file locks
instead of an in-memory set, because Harbor episodes for one training job can
run from *independent OS processes* on the same node (multiple
``AgentLoopWorker`` processes), which never share Python memory with each
other.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

from psrl.utils.concurrency.slot import SlotManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Real physical GPU character devices, e.g. /dev/nvidia0, /dev/nvidia1, ...
# Deliberately excludes /dev/nvidiactl, /dev/nvidia-uvm*: those are host-wide
# control devices, safe to share across every container regardless of which
# physical GPU it uses.
_DEVICE_GLOB = "/dev/nvidia[0-9]*"
_DEVICE_PREFIX = "/dev/nvidia"

# Lets this be exercised on a machine with no real NVIDIA devices (dev boxes,
# CI, the comparison scripts under docs/psrl/scripts/) without touching the
# GPU-reservation logic itself.
_ENV_DEVICE_OVERRIDE = "PSRL_SCIACCEL_GPU_DEVICES"

# Cross-process lock pool name. All episodes on one host share this pool, so
# concurrent AgentLoopWorker *processes* (not just asyncio tasks) contend for
# the same set of physical devices.
_DEFAULT_SLOT_PREFIX = "sciaccel_gpu_gate"


def is_exclusive_gpu_device(host_path: str) -> bool:
    """True for a bare-index device like ``/dev/nvidia0``/``/dev/nvidia12``.

    False for host-wide control devices (``/dev/nvidiactl``,
    ``/dev/nvidia-uvm``, ``/dev/nvidia-uvm-tools``): those do not represent one
    physical GPU and are safe to bind-mount into every container unchanged.
    """
    if not host_path.startswith(_DEVICE_PREFIX):
        return False
    return host_path[len(_DEVICE_PREFIX) :].isdigit()


def discover_gpu_devices() -> list[str]:
    """
    List physical GPU device paths available for exclusive allocation.

    Reads ``PSRL_SCIACCEL_GPU_DEVICES`` (comma-separated ``/dev/nvidiaN``
    paths) first, so this is testable off a real GPU host -- e.g. the
    before/after comparison scripts under docs/psrl/scripts/ set this to a
    handful of fake paths and never touch a real device node. Falls back to a
    glob of ``/dev/nvidia[0-9]*`` on the actual machine.

    Returns:
        list[str]: Device paths, sorted by numeric index (nvidia0, nvidia1, ...).
    """
    override = os.getenv(_ENV_DEVICE_OVERRIDE, "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]

    def _index(path: str) -> int:
        digits = path[len(_DEVICE_PREFIX) :]
        return int(digits) if digits.isdigit() else 0

    return sorted((p for p in glob.glob(_DEVICE_GLOB) if is_exclusive_gpu_device(p)), key=_index)


@dataclass
class AllocatedGpu:
    """One exclusively-held physical GPU device, returned by `GpuDeviceGate.acquire`."""

    device: str
    index: int
    _slot: tuple[int, int]


class GpuDeviceGate:
    """
    Gate A: reserve/release one physical GPU device per GPU-needing episode.

    Backed by `SlotManager`, so the pool of "how many devices, who is holding
    which slot right now" lives in `fcntl`-locked files under the system temp
    directory rather than in this process's memory -- every process that
    constructs a `GpuDeviceGate` with the same `jobs_dir` contends for the same
    physical devices.
    """

    def __init__(self, jobs_dir: str, slot_prefix: str = _DEFAULT_SLOT_PREFIX):
        self.jobs_dir = jobs_dir
        self.slot_prefix = slot_prefix

    async def acquire(self) -> AllocatedGpu | None:
        """
        Block until one physical GPU device is free, then reserve it.

        Returns:
            AllocatedGpu on success. `None` only when this host has no
            discoverable GPU devices at all (caller should decide whether to
            fail the episode or fall back to the old unsafe static mapping;
            it must not silently claim isolation it cannot provide).
        """
        devices = discover_gpu_devices()
        if not devices:
            logger.warning(
                "GpuDeviceGate found no GPU devices (glob=%r, env override=%r); "
                "cannot provide per-device exclusivity.",
                _DEVICE_GLOB,
                _ENV_DEVICE_OVERRIDE,
            )
            return None

        slot = await SlotManager.acquire(len(devices), self.jobs_dir, prefix=self.slot_prefix)
        if slot is None:
            # discover_gpu_devices() returned non-empty, so max_slots > 0;
            # SlotManager.acquire only returns None when max_slots <= 0.
            # Defensive: should be unreachable.
            return None

        _fd, slot_index = slot
        device = devices[slot_index]
        logger.info("Gate A: reserved %s (slot %d/%d).", device, slot_index, len(devices))
        return AllocatedGpu(device=device, index=slot_index, _slot=slot)

    @staticmethod
    def release(allocated: AllocatedGpu | None) -> None:
        """Release a device reserved by `acquire`. Safe to call with `None`."""
        if allocated is None:
            return
        SlotManager.release(allocated._slot)
        logger.info("Gate A: released %s.", allocated.device)


def build_gpu_override(base_override_path: str, device: str) -> str:
    """
    Build a per-episode Compose override that mounts exactly `device`.

    Starts from the static override file (its driver-library mounts and
    environment stay correct for any device index), and rewrites only
    `services.main.devices`: the one exclusive `/dev/nvidiaN` entry becomes
    the reserved `device`, while shared control devices (`nvidiactl`,
    `nvidia-uvm*`) pass through unchanged, because those are host-wide and
    safe for every container regardless of which GPU it was assigned.

    Args:
        base_override_path: Path to the static override file, e.g.
            `examples/sciaccel_rl/config/gpu-compose-override.yaml`.
        device: The reserved device path, e.g. `/dev/nvidia1`.

    Returns:
        str: Path to a freshly written temp YAML file. The caller owns
        deleting it once the episode's containers have been torn down.
    """
    import tempfile
    from pathlib import Path

    import yaml

    base = yaml.safe_load(Path(base_override_path).read_text()) or {}
    main = dict(base.get("services", {}).get("main", {}))

    rewritten = []
    for entry in main.get("devices", []):
        host_path = str(entry).split(":", 1)[0]
        rewritten.append(f"{device}:{device}" if is_exclusive_gpu_device(host_path) else entry)
    main["devices"] = rewritten or [f"{device}:{device}"]

    override_file = Path(tempfile.mktemp(suffix=".yaml", prefix="sciaccel-gate-a-"))
    override_file.write_text(yaml.dump({"services": {"main": main}}))
    return str(override_file)
