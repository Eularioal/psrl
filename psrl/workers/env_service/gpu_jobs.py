"""Track bounded GPU jobs independently of terminal observations and model turns."""

import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class GpuJob:
    """Retain output readiness separately from acknowledged resource release."""

    job_id: str
    command: str
    tag: str
    timeout_sec: float
    wait_timeout_sec: float
    cwd: str | None
    env: dict[str, str]
    state: str = "queued"
    output: object = None
    error: str | None = None
    events: dict = field(default_factory=dict)
    output_ready: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    queued_mono: float = field(default_factory=time.monotonic)

    def transition(self, state):
        self.state = state
        self.events[state] = time.time()

    def snapshot(self):
        output = self.output
        return {
            "job_id": self.job_id,
            "tag": self.tag,
            "state": self.state,
            "output_ready": self.output_ready.is_set(),
            "exit_code": getattr(output, "return_code", None),
            "stdout": (getattr(output, "stdout", None) or "")[-16000:],
            "stderr": (getattr(output, "stderr", None) or "")[-16000:],
            "error": self.error,
            "events": dict(self.events),
        }


class GpuJobManager:
    """Own jobs through physical return using an executor supplied by the backend."""

    def __init__(self, execute, *, max_jobs=128):
        self.execute = execute
        self.max_jobs = max_jobs
        self.jobs = {}
        self.lock = asyncio.Lock()
        self.closed = False

    def start(self, command, *, tag="", timeout_sec=300, wait_timeout_sec=300, cwd=None, env=None):
        if self.closed or len(self.jobs) >= self.max_jobs:
            raise RuntimeError("GPU job owner is closed or its episode job limit was reached.")
        if not isinstance(command, str) or not command.strip() or len(command) > 65536:
            raise ValueError("GPU command must contain between 1 and 65536 characters.")
        if not all(math.isfinite(t) and 0 < t <= 3600 for t in (timeout_sec, wait_timeout_sec)):
            raise ValueError("GPU execution and queue timeouts must be finite and at most 3600 seconds.")
        if not isinstance(tag, str) or len(tag) > 128:
            raise ValueError("GPU job tag must be a string of at most 128 characters.")
        if cwd is not None and (not isinstance(cwd, str) or not cwd.startswith("/")):
            raise ValueError("GPU job cwd must be an absolute container path.")
        if env is not None and (
            not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items())
        ):
            raise ValueError("GPU job environment must map strings to strings.")
        job = GpuJob(uuid.uuid4().hex, command, tag, timeout_sec, wait_timeout_sec, cwd, dict(env or {}))
        job.transition("queued")
        self.jobs[job.job_id] = job
        job.task = asyncio.create_task(self._run(job))
        return job.snapshot()

    async def _run(self, job):
        try:
            # Same-container GPU jobs serialize, while unrelated CPU and model work can progress.
            async with asyncio.timeout(job.wait_timeout_sec):
                await self.lock.acquire()
            try:
                job.wait_timeout_sec -= time.monotonic() - job.queued_mono
                if job.wait_timeout_sec <= 0:
                    raise TimeoutError("GPU job queue deadline expired.")
                await self.execute(job)
                job.transition("completed")
            finally:
                self.lock.release()
        except asyncio.CancelledError:
            job.transition("cancelled")
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.transition("failed")
        finally:
            job.output_ready.set()

    def status(self, job_id):
        return self.jobs[job_id].snapshot()

    async def wait(self, job_id, *, timeout_sec=30, released=False):
        if not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 3600:
            raise ValueError("GPU job observation timeout must be finite and at most 3600 seconds.")
        job = self.jobs[job_id]
        try:
            async with asyncio.timeout(timeout_sec):
                if released:
                    await asyncio.shield(asyncio.gather(job.task, return_exceptions=True))
                else:
                    await job.output_ready.wait()
        except TimeoutError:
            pass
        return job.snapshot()

    async def cancel(self, job_id):
        job = self.jobs[job_id]
        if not job.task.done() and not job.task.cancelling():
            job.task.cancel()
        await asyncio.shield(asyncio.gather(job.task, return_exceptions=True))
        if job.task.cancelled():
            job.transition("cancelled")
            job.output_ready.set()
        return job.snapshot()

    async def close(self):
        self.closed = True
        for job in self.jobs.values():
            if not job.task.done() and not job.task.cancelling():
                job.task.cancel()
        await asyncio.gather(*(job.task for job in self.jobs.values()), return_exceptions=True)
        errors = [job.error for job in self.jobs.values() if job.error]
        if errors:
            raise RuntimeError(f"GPU jobs failed: {errors!r}.")
