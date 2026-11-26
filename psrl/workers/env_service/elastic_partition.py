"""Coordinate bounded environment admission and drained inference replica lending."""

import asyncio
import math
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass
class PartitionDevice:
    """Declare a single device's explicit compute slots and environment memory budget."""

    name: str
    env_slots: int
    env_memory_mb: int
    inference: bool = False
    state: str = "environment"
    env_active: int = 0
    env_used_mb: int = 0
    infer_active: int = 0


class ElasticPartition:
    """Route through one owner and expose donated devices only after sleep acknowledgement.

    Callers must route every inference request through `inference`, and complete
    physical GPU command return inside `environment`. Callbacks own model control
    and physical validation. This controller never migrates a live CUDA process.
    """

    def __init__(
        self,
        devices,
        sleep,
        wake,
        *,
        dynamic=False,
        resident_sharing=False,
        inference_limit=4,
        transition_timeout_s=60,
    ):
        if not devices or len({d.name for d in devices}) != len(devices):
            raise ValueError("Devices must be nonempty and uniquely identified.")
        if (
            not isinstance(inference_limit, int)
            or inference_limit < 1
            or not math.isfinite(transition_timeout_s)
            or transition_timeout_s <= 0
        ):
            raise ValueError("Inference capacity and transition timeout must be positive.")
        if not any(d.inference for d in devices) or not any(not d.inference for d in devices):
            raise ValueError("Separate initial inference and environment devices are required.")
        for device in devices:
            if (
                not isinstance(device.env_slots, int)
                or device.env_slots <= 0
                or not math.isfinite(device.env_memory_mb)
                or device.env_memory_mb <= 0
            ):
                raise ValueError("Environment capacities must be positive.")
            device.state = "inference" if device.inference else "environment"
        if dynamic and resident_sharing:
            raise ValueError("Choose resident sharing or replica lending for a controller.")
        self.devices = devices
        self.sleep = sleep
        self.wake = wake
        self.dynamic = dynamic
        self.resident_sharing = resident_sharing
        self.inference_limit = inference_limit
        self.timeout = transition_timeout_s
        self.condition = asyncio.Condition()
        self.transition = asyncio.Lock()
        self.waiters = []
        self.infer_waiters = 0
        self.events = []

    def _event(self, action, device):
        self.events.append({"at": time.monotonic(), "action": action, "device": device.name})

    def _fits(self, device, memory_mb):
        return (
            (device.state == "environment" or (self.resident_sharing and device.state == "inference"))
            and device.env_active < device.env_slots
            and device.env_used_mb + memory_mb <= device.env_memory_mb
        )

    @asynccontextmanager
    async def environment(self, memory_mb):
        """Use stable first fit on dedicated cards before considering acknowledged donations."""
        if (
            not math.isfinite(memory_mb)
            or memory_mb <= 0
            or not any(
                memory_mb <= d.env_memory_mb and (not d.inference or self.dynamic or self.resident_sharing)
                for d in self.devices
            )
        ):
            raise ValueError("Environment request cannot fit the configured pool.")
        ticket = object()
        selected = None
        async with self.condition:
            self.waiters.append((ticket, memory_mb))
            self.condition.notify_all()
            try:
                while selected is None:
                    if self.waiters[0][0] is ticket:
                        selected = next(
                            (d for d in sorted(self.devices, key=lambda d: d.inference) if self._fits(d, memory_mb)),
                            None,
                        )
                    if selected is None:
                        await self.condition.wait()
                self.waiters.pop(0)
                selected.env_active += 1
                selected.env_used_mb += memory_mb
                self._event("env_admit", selected)
                self.condition.notify_all()
            finally:
                self.waiters[:] = [item for item in self.waiters if item[0] is not ticket]
                self.condition.notify_all()
        try:
            yield selected.name
        except BaseException:
            async with self.condition:
                selected.state = "quarantined"
                self._event("env_failed", selected)
            raise
        finally:
            async with self.condition:
                selected.env_active -= 1
                selected.env_used_mb -= memory_mb
                self._event("env_return", selected)
                self.condition.notify_all()

    @asynccontextmanager
    async def inference(self):
        """Count requests from routing through completion so draining has a precise boundary."""
        async with self.condition:
            self.infer_waiters += 1
            self.condition.notify_all()
            try:
                while True:
                    eligible = [
                        d for d in self.devices if d.state == "inference" and d.infer_active < self.inference_limit
                    ]
                    if eligible:
                        device = min(eligible, key=lambda d: (d.infer_active, d.name))
                        device.infer_active += 1
                        break
                    await self.condition.wait()
            finally:
                self.infer_waiters -= 1
        try:
            yield device.name
        except BaseException:
            async with self.condition:
                device.state = "quarantined"
                self._event("inference_failed", device)
            raise
        finally:
            async with self.condition:
                device.infer_active -= 1
                self.condition.notify_all()

    async def reconcile(self):
        """Change at most one replica per observation, retaining a live inference survivor."""
        if not self.dynamic:
            return
        async with self.transition:
            async with self.condition:
                replicas = [d for d in self.devices if d.state == "inference"]
                borrowed = [d for d in self.devices if d.inference and d.state == "environment"]
                active = sum(d.infer_active for d in self.devices)
                restore = self.infer_waiters > 0 or not self.waiters
                if borrowed and restore:
                    device = borrowed[0]
                    device.state = "restoring"
                    self._event("restore_start", device)
                    action = "wake"
                elif (
                    self.waiters
                    and len(replicas) > 1
                    and not borrowed
                    and active + self.infer_waiters <= (len(replicas) - 1) * self.inference_limit
                    and not any(self._fits(d, self.waiters[0][1]) for d in self.devices)
                ):
                    compatible = [d for d in replicas if d.env_memory_mb >= self.waiters[0][1]]
                    if not compatible:
                        return
                    device = min(compatible, key=lambda d: (d.infer_active, d.name))
                    device.state = "draining"
                    self._event("drain_start", device)
                    action = "sleep"
                else:
                    return
            try:
                async with asyncio.timeout(self.timeout):
                    async with self.condition:
                        await self.condition.wait_for(
                            lambda: device.infer_active == 0 if action == "sleep" else device.env_active == 0
                        )
                        if device.state == "quarantined":
                            raise RuntimeError("A failed command prevents model restoration.")
                    await (self.sleep(device.name) if action == "sleep" else self.wake(device.name))
                async with self.condition:
                    device.state = "environment" if action == "sleep" else "inference"
                    self._event(action + "_ack", device)
                    self.condition.notify_all()
            except BaseException:
                async with self.condition:
                    device.state = "quarantined"
                    self._event("transition_failed", device)
                    self.condition.notify_all()
                raise

    async def run(self):
        """Observe queues periodically, leaving command execution outside the transition lock."""
        while True:
            await self.reconcile()
            await asyncio.sleep(0.1)
