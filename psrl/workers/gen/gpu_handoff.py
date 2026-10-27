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
        self.shared_commands = {}
        self.pending_operations = 0

    @asynccontextmanager
    async def operation(self, borrower=None, sleeping=False):
        """Serialize operations, allowing nested calls by the same asyncio task."""
        task = asyncio.current_task()
        if self.task is task:
            self.depth += 1
        else:
            async with self.condition:
                self.pending_operations += 1
                try:
                    await self.condition.wait_for(
                        lambda: (
                            self.task is None
                            and not self.shared_commands
                            and (self.borrower is None or self.borrower == borrower)
                            and (not sleeping or self.reservation is None)
                        )
                    )
                    self.task = task
                    self.depth = 1
                finally:
                    self.pending_operations -= 1
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

    async def validate_env_sharing(self, headroom_mb, margin_mb):
        """Require measured inference workspace plus an explicit deployment margin."""
        profiles = await self.collective_rpc(method="env_sharing_profile")
        if not profiles or any(p["transient_mb"] is None for p in profiles):
            raise ValueError("Resident sharing requires vLLM automatic memory profiling.")
        required = max(p["transient_mb"] for p in profiles) + margin_mb
        if headroom_mb < required:
            raise ValueError(f"Model headroom {headroom_mb} MiB is below profiled requirement {required} MiB.")
        return {"profiles": profiles, "required_headroom_mb": required}

    async def acquire_env_command(self, owner, reuse_sleeping=True):
        """Fence control without pausing inference or waking an already sleeping engine."""
        gate = self._gpu_control
        async with gate.condition:
            if owner in gate.shared_commands:
                return gate.shared_commands[owner]
            if gate.task is not None or gate.pending_operations or gate.borrower or gate.reservation:
                return None
            # Hold the condition until the snapshot and reservation are committed together.
            sleeping = await self.engine.is_sleeping()
            if (sleeping and not reuse_sleeping) or (not sleeping and self._gpu_sync_active):
                return None
            if not sleeping and not self._gpu_weights_valid:
                return None
            mode = "sleeping" if sleeping else "resident"
            gate.shared_commands[owner] = mode
            return mode

    async def release_env_command(self, owner):
        """Release only after the node has acknowledged physical command cleanup."""
        async with self._gpu_control.condition:
            self._gpu_control.shared_commands.pop(owner, None)
            self._gpu_control.condition.notify_all()

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
