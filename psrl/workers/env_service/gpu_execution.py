"""Execute GPU work through one lifecycle with explicit reservation policies."""

import asyncio
import math
import time
import uuid


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
            await self.policy.wait_return(self.pending_return)
            self.pending_return = None

    async def wait_return(self):
        async with self.lock:
            await self._wait_return()

    async def execute(
        self, run, container_id, *, timeout_sec, wait_timeout_sec, overlap_return=False, job=None, cancel=None
    ):
        if not all(math.isfinite(t) and t > 0 for t in (timeout_sec, wait_timeout_sec)):
            raise ValueError("GPU execution and queue timeouts must be positive and finite.")
        # Include local serialization and deferred return in the same admission deadline.
        deadline = asyncio.get_running_loop().time() + wait_timeout_sec
        async with asyncio.timeout_at(deadline):
            await self.lock.acquire()
        try:
            record = {
                "request_id": job.job_id if job else uuid.uuid4().hex,
                "requested_at": time.time(),
                "reservation_mode": self.policy.mode,
                "capacity_released": False,
            }
            self.last_record = record
            if job:
                job.reservation = record
            async with asyncio.timeout_at(deadline):
                await self._wait_return()
                owner = await container_id()
            try:
                async with asyncio.timeout_at(deadline):
                    await self.policy.acquire(owner, record, job.tag if job else "")
                if job:
                    job.transition("running")
                async with asyncio.timeout(timeout_sec):
                    output = await run()
                record["command_finished_at"] = time.time()
                record["return_requested_at"] = time.time()
                if job:
                    job.output = output
                    job.transition("returning")
                    job.output_ready.set()
                if await self.policy.finish(record, overlap_return):
                    self.pending_return = record
                return output
            except BaseException:
                # Stop owned processes before the reservation policy can return capacity.
                async def cleanup():
                    try:
                        if cancel:
                            await cancel()
                    finally:
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
