"""Serialize model control with temporary environment ownership of the GPU."""

import asyncio
from contextlib import asynccontextmanager
from functools import wraps


class GpuControlGate:
    """Keep model mutations out of a borrowed GPU until its owner returns it."""

    def __init__(self):
        self.condition = asyncio.Condition()
        self.borrower = None
        self.reservation = None
        self.task = None
        self.depth = 0

    @asynccontextmanager
    async def operation(self, borrower=None, sleeping=False):
        """Serialize operations, allowing nested calls by the same asyncio task."""
        task = asyncio.current_task()
        if self.task is task:
            self.depth += 1
        else:
            async with self.condition:
                await self.condition.wait_for(
                    lambda: (
                        self.task is None
                        and (self.borrower is None or self.borrower == borrower)
                        and (not sleeping or self.reservation is None)
                    )
                )
                self.task = task
                self.depth = 1
        try:
            yield
        finally:
            async with self.condition:
                self.depth -= 1
                if self.depth == 0:
                    self.task = None
                    self.condition.notify_all()


def gpu_control(method):
    """Apply the server's control gate to a complete GPU mutation."""

    @wraps(method)
    async def guarded(self, *args, **kwargs):
        async with self._gpu_control.operation(sleeping=method.__name__ in ("sleep", "nixl_sleep")):
            return await method(self, *args, **kwargs)

    return guarded


class GpuHandoffMixin:
    """Expose acknowledged GPU handoff while fencing every ordinary control path."""

    def validate_gpu_lending(self):
        """Reject engines that cannot release their GPU allocations."""
        if not self.config.enable_sleep_mode:
            raise ValueError("GPU lending requires enable_sleep_mode=True.")

    async def prepare_env_gpu(self, owner: str) -> bool:
        """Sleep the complete replica, retaining the fence on ambiguous failures."""
        if not owner:
            raise ValueError("A GPU borrower must have an identity.")
        async with self._gpu_control.operation(owner):
            if self._gpu_control.borrower == owner:
                if not self._env_handoff_ready:
                    raise RuntimeError("GPU handoff did not complete. Physical cleanup is required.")
                return True
            if self._gpu_sync_active:
                return False
            if self._gpu_control.reservation not in (None, owner):
                raise RuntimeError("GPU reservation belongs to another lease.")
            self._gpu_control.reservation = owner
            self._gpu_control.borrower = owner
            self._env_handoff_ready = False
            self.grpc_servicer.pause_generation_admission()
            await self.close_grpc_generate_admission()
            await self.pause_generation(mode="wait", clear_cache=False)
            # Level two has no recoverable weights, so materialize them before taking a level one backup.
            if await self.engine.is_sleeping():
                await self.nixl_wake_up()
            if not self._gpu_weights_valid:
                await self.pull_model_for_sync(0)
            await self.nixl_sleep(level=1)
            self._env_handoff_ready = True
            return True

    async def return_env_gpu(self, owner: str) -> None:
        """Restore inference after the node has acknowledged physical GPU return."""
        if self._gpu_control.borrower is None and self._last_env_owner == owner:
            return
        if self._gpu_control.borrower != owner:
            raise RuntimeError("GPU return does not match the current borrower.")
        async with self._gpu_control.operation(owner):
            if self._gpu_control.borrower != owner:
                raise RuntimeError("GPU return does not match the current borrower.")
            await self.nixl_wake_up()
            await self.resume_after_sync()
            self._gpu_control.borrower = None
            self._last_env_owner = owner
            self._env_handoff_ready = False

    async def finish_env_gpu(self, owner: str) -> None:
        """Release the episode reservation only after its containers are removed."""
        async with self._gpu_control.operation(owner):
            if self._gpu_control.reservation not in (None, owner):
                raise RuntimeError("GPU reservation belongs to another lease.")
            if self._gpu_control.borrower == owner:
                await self.return_env_gpu(owner)
            self._gpu_control.reservation = None
