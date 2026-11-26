"""Execute GPU work through one lifecycle with explicit reservation policies."""

import asyncio
import logging
import math
import time
import uuid
from contextlib import contextmanager

psrl_logger = logging.getLogger(__name__)


class GpuOperationTimeout(TimeoutError):
    """Retain the failed execution phase and request evidence across cleanup."""

    def __init__(self, phase, record, cause):
        self.phase = phase
        self.record = record
        self.cause_message = str(cause)
        super().__init__(f"GPU operation timed out during {phase} for {record['request_id']}: {self.cause_message}")

    def __reduce__(self):
        return type(self), (self.phase, self.record, self.cause_message)


@contextmanager
def execution_phase(record, phase, *, deadline=None):
    """Keep phase timing and the original failure even if cancellation cleanup fails."""
    started = time.monotonic()
    stage = {"phase": phase, "started_at": time.time()}
    if deadline is not None:
        stage["remaining_budget_s"] = max(0, deadline - asyncio.get_running_loop().time())
    record.setdefault("phases", []).append(stage)
    try:
        yield
        stage["status"] = "complete"
    except BaseException as error:
        stage.update(
            status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        record.setdefault("failure", {"phase": phase, "type": type(error).__name__, "message": str(error)})
        if isinstance(error, TimeoutError) and not isinstance(error, GpuOperationTimeout):
            raise GpuOperationTimeout(phase, record, error) from error
        raise
    finally:
        stage.update(finished_at=time.time(), elapsed_s=time.monotonic() - started)
        if stage["status"] != "complete":
            psrl_logger.warning(f"GPU request {record['request_id']!r} phase result: {stage!r}.")


class EpisodeGpuReservation:
    """Keep the GPU capacity already admitted by the session ledger."""

    mode = "episode"

    def __init__(self, router_name, lease_id):
        if not router_name or not lease_id:
            raise ValueError("GPU execution requires a router and a lease owner.")
        self.router_name = router_name
        self.lease_id = lease_id

    async def acquire(self, container_id, record, tag):
        record["admitted_at"] = time.time()

    async def finish(self, record, overlap):
        record["execution_finished_at"] = time.time()
        return False

    async def wait_return(self, record):
        return None

    async def cancel(self, record):
        return None


class CommandGpuReservation(EpisodeGpuReservation):
    """Borrow per invocation and retain capacity until physical return is acknowledged."""

    mode = "command"

    @property
    def router(self):
        import ray

        return ray.get_actor(self.router_name)

    async def acquire(self, container_id, record, tag):
        while True:
            admission = await self.router.acquire_gpu_command.remote(
                self.lease_id, container_id, record["request_id"], tag
            )
            if admission["status"] == "granted":
                record.update(epoch=admission["epoch"], admitted_at=time.time())
                record.update(
                    {
                        key: admission[key]
                        for key in ("unit_id", "node_ip", "gpu_indices", "gpu_uuids")
                        if key in admission
                    }
                )
                return
            await asyncio.sleep(0.1)

    async def finish(self, record, overlap):
        if overlap:
            status = await self.router.begin_gpu_command_return.remote(
                self.lease_id, record["request_id"], record["epoch"]
            )
            if status["error"]:
                raise RuntimeError(f"GPU command return failed: {status['error']}")
            record["return_accepted_at"] = time.time()
            return True
        await self.wait_return(record)
        return False

    async def wait_return(self, record):
        status = await self.router.release_gpu_command.remote(self.lease_id, record["request_id"], record["epoch"])
        record.update(released_at=status["returned_at"], capacity_released=True)

    async def cancel(self, record):
        await self.router.cancel_gpu_command.remote(self.lease_id, record["request_id"])


def reservation_policy(binding):
    """Reject unsupported modes instead of silently running an unreserved job."""
    policies = {"episode": EpisodeGpuReservation, "command": CommandGpuReservation}
    mode = binding["mode"]
    if mode not in policies:
        raise ValueError(f"Unknown GPU reservation mode {mode!r}.")
    return policies[mode](binding["router_name"], binding["lease_id"])


class GpuExecutionBackend:
    """Share execution, deadlines and cancellation without changing reservation scope."""

    def __init__(self, policy):
        self.policy = policy
        self.lock = asyncio.Lock()
        self.pending_return = None
        self.last_record = None

    async def _wait_return(self):
        if self.pending_return is not None:
            with execution_phase(self.pending_return, "physical_return"):
                await self.policy.wait_return(self.pending_return)
            self.pending_return = None

    async def wait_return(self):
        async with self.lock:
            await self._wait_return()

    async def execute(
        self,
        run,
        container_id,
        *,
        timeout_sec,
        wait_timeout_sec,
        overlap_return=False,
        job=None,
        cancel=None,
        before_cancel=None,
    ):
        if not all(math.isfinite(t) and t > 0 for t in (timeout_sec, wait_timeout_sec)):
            raise ValueError("GPU execution and queue timeouts must be positive and finite.")
        # Include local serialization and deferred return in the same admission deadline.
        deadline = asyncio.get_running_loop().time() + wait_timeout_sec
        record = {
            "request_id": job.job_id if job else uuid.uuid4().hex,
            "requested_at": time.time(),
            "reservation_mode": self.policy.mode,
            "capacity_released": False,
            "wait_timeout_s": wait_timeout_sec,
            "execution_timeout_s": timeout_sec,
        }
        if job:
            job.reservation = record
        with execution_phase(record, "local_queue", deadline=deadline):
            async with asyncio.timeout_at(deadline):
                await self.lock.acquire()
        try:
            self.last_record = record
            with execution_phase(record, "previous_return", deadline=deadline):
                async with asyncio.timeout_at(deadline):
                    await self._wait_return()
            with execution_phase(record, "container_lookup", deadline=deadline):
                async with asyncio.timeout_at(deadline):
                    owner = await container_id()
                    record["container_id"] = owner
            try:
                with execution_phase(record, "gpu_admission", deadline=deadline):
                    async with asyncio.timeout_at(deadline):
                        await self.policy.acquire(owner, record, job.tag if job else "")
                if job:
                    job.transition("running")
                with execution_phase(record, "command_execution"):
                    async with asyncio.timeout(timeout_sec):
                        output = await run()
                record["command_finished_at"] = time.time()
                record["return_requested_at"] = time.time()
                if job:
                    job.output = output
                    job.transition("returning")
                    job.output_ready.set()
                with execution_phase(record, "return_request"):
                    if await self.policy.finish(record, overlap_return):
                        self.pending_return = record
                return output
            except BaseException:
                # Stop owned processes before the reservation policy can return capacity.
                async def cleanup():
                    with execution_phase(record, "cancellation_cleanup"):
                        if before_cancel:
                            await before_cancel()
                        if cancel:
                            await cancel()
                        await self.policy.cancel(record)

                cleaning = asyncio.create_task(cleanup())
                try:
                    await asyncio.shield(cleaning)
                except asyncio.CancelledError:
                    await cleaning
                    raise
                raise
        finally:
            self.lock.release()
