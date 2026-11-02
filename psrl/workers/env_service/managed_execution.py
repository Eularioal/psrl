"""Share managed GPU jobs across Docker lifecycle providers."""

import asyncio
import json
import shlex
import uuid
from pathlib import Path

from psrl.workers.env_service import gpu_job_process
from psrl.workers.env_service.gpu_execution import GpuExecutionBackend, reservation_policy
from psrl.workers.env_service.gpu_jobs import GpuJob, GpuJobManager


class ManagedGpuExecution:
    """Use provider hooks for command transport and container identity."""

    def _init_gpu_execution(self, binding):
        if binding is not None and not {"mode", "router_name", "lease_id"} <= binding.keys():
            raise ValueError("GPU execution requires mode, router name and lease identity.")
        self._gpu_stopping = False
        self._gpu_execution = binding
        self._execution_backend = GpuExecutionBackend(reservation_policy(binding)) if binding else None
        self._gpu_commands = binding if binding and binding["mode"] == "command" else None
        self._gpu_command_lock = asyncio.Lock()
        self._pending_gpu_return = None
        self.last_gpu_command = None
        self._mps = binding if binding and binding.get("backend") == "mps" else None
        self._gpu_jobs = GpuJobManager(self._execute_gpu_job, concurrency=128 if self._mps else 1)
        self._prepared_commands = set()

    def _require_gpu_execution(self):
        if getattr(self, "_gpu_stopping", False):
            raise RuntimeError("GPU execution is stopping.")

    def _backend(self):
        """Adapt the legacy command binding while keeping one executor per environment."""
        backend = getattr(self, "_execution_backend", None)
        if backend is None:
            binding = getattr(self, "_gpu_execution", None)
            legacy = getattr(self, "_gpu_commands", None)
            if binding is None and legacy:
                binding = {**legacy, "mode": "command"}
            if binding:
                backend = self._execution_backend = GpuExecutionBackend(reservation_policy(binding))
        return backend

    async def wait_for_gpu_return(self):
        """Settle managed execution without releasing an episode's session reservation."""
        backend = self._backend()
        if backend:
            await backend.wait_return()

    async def exec_gpu(
        self,
        command,
        *,
        timeout_sec=300,
        wait_timeout_sec=300,
        overlap_return=False,
        _job=None,
        _cancel_process=None,
        **kwargs,
    ):
        """Execute through a common backend with either session or command GPU ownership."""
        self._require_gpu_execution()
        if getattr(self, "_mps", None) and _job is None:
            job_id = self.start_gpu_job(
                command,
                timeout_sec=timeout_sec,
                wait_timeout_sec=wait_timeout_sec,
                cwd=kwargs.get("cwd"),
                env=kwargs.get("env"),
                gpu_memory_mb=kwargs.get("gpu_memory_mb"),
            )["job_id"]
            job = self._gpu_jobs.jobs[job_id]
            try:
                await asyncio.shield(job.task)
            except BaseException:
                await self.cancel_gpu_job(job_id)
                raise
            if job.error or job.state != "completed":
                raise RuntimeError(job.error or "MPS job did not complete.")
            return job.output
        backend = self._backend()
        if backend is None:
            return await self._exec_unreserved(command, timeout_sec=timeout_sec, **kwargs)

        async def run():
            if _job:
                return await self._run_gpu_process(_job, **kwargs)
            if backend.policy.mode == "episode" and _cancel_process is None:
                # Direct episode calls also need an owned process group for timeout cleanup.
                transient = GpuJob(
                    uuid.uuid4().hex,
                    command,
                    "",
                    timeout_sec,
                    wait_timeout_sec,
                    kwargs.get("cwd"),
                    dict(kwargs.get("env") or {}),
                )
                return await self._run_gpu_process(transient, **kwargs)
            return await self._exec_unreserved(command, timeout_sec=timeout_sec, **kwargs)

        try:
            return await backend.execute(
                run,
                self._gpu_container_id,
                timeout_sec=timeout_sec,
                wait_timeout_sec=wait_timeout_sec,
                overlap_return=overlap_return,
                job=_job,
                cancel=_cancel_process,
            )
        finally:
            self.last_gpu_command = backend.last_record

    async def exec_gpu_prepared(self, module_path, payload, **kwargs):
        """Prepare a cooperative Python program before reserving GPU execution."""
        from psrl.workers.env_service.prepared_gpu import execute_prepared

        self._require_gpu_execution()
        task = asyncio.current_task()
        self._prepared_commands.add(task)
        try:
            return await execute_prepared(self, module_path, payload, **kwargs)
        finally:
            self._prepared_commands.discard(task)

    def start_gpu_job(self, command, **kwargs):
        """Queue a bounded GPU job without waiting for capacity or command completion."""
        self._require_gpu_execution()
        if self._backend() is None:
            raise RuntimeError("GPU jobs require a managed GPU reservation lease.")
        if kwargs.get("gpu_memory_mb") is not None and not getattr(self, "_mps", None):
            raise ValueError("Per-job memory budgets require the MPS backend.")
        return self._gpu_jobs.start(command, **kwargs)

    def gpu_job_status(self, job_id):
        return self._gpu_jobs.status(job_id)

    async def wait_gpu_job(self, job_id, **kwargs):
        return await self._gpu_jobs.wait(job_id, **kwargs)

    async def cancel_gpu_job(self, job_id):
        return await self._gpu_jobs.cancel(job_id)

    async def gpu_job_telemetry(self, job_id):
        """Expose only this job's host attributed process metrics to its caller."""
        import ray

        status = self._gpu_jobs.status(job_id)
        backend = self._backend()
        if backend.policy.mode == "episode" and not getattr(self, "_mps", None):
            status["telemetry"] = {"available": False, "reason": "Per-job host attribution requires command mode."}
            return status
        router = ray.get_actor(backend.policy.router_name)
        status["telemetry"] = await router.gpu_job_telemetry.remote(self._gpu_execution["lease_id"], job_id)
        return status

    async def finish_gpu_jobs(self):
        """Join bounded jobs and surface failures before dependent verification begins."""
        for job in self._gpu_jobs.jobs.values():
            await asyncio.shield(asyncio.gather(job.task, return_exceptions=True))
            if job.error:
                raise RuntimeError(job.error)

    async def _execute_gpu_job(self, job):
        if getattr(self, "_mps", None):
            return await self._execute_mps_job(job)
        await self.exec_gpu(
            job.command,
            timeout_sec=job.timeout_sec,
            wait_timeout_sec=job.wait_timeout_sec,
            overlap_return=True,
            _job=job,
            cwd=job.cwd,
            env=job.env,
        )
        returning = asyncio.create_task(self.wait_for_gpu_return())
        try:
            await asyncio.shield(returning)
        except asyncio.CancelledError:
            await returning
            raise

    async def _execute_mps_job(self, job):
        import ray
        from harbor.environments.base import ExecResult

        router = ray.get_actor(self._mps["router_name"])
        lease_id = self._mps["lease_id"]
        container = await self._gpu_container_id()
        try:
            status = await router.submit_mps_job.remote(
                lease_id,
                container,
                job.job_id,
                job.command,
                job.gpu_memory_mb or self._mps["gpu_memory_mb"],
                job.timeout_sec,
                job.wait_timeout_sec,
                job.cwd,
                job.env,
                job.tag,
            )
            while status["state"] not in ("completed", "cancelled", "failed", "quarantined"):
                if status["state"] == "running" and job.state != "running":
                    job.transition("running")
                await asyncio.sleep(0.05)
                status = await router.mps_job_status.remote(lease_id, job.job_id)
            command_scope = self._mps["mode"] == "command"
            job.reservation = {
                "reservation_mode": self._mps["mode"],
                "job_capacity_released": status["returned"] is not None,
                "job_released_at": status["returned"],
                "capacity_released": command_scope and status["returned"] is not None,
                "released_at": status["returned"] if command_scope else None,
            }
            if status["state"] != "completed":
                raise RuntimeError(status["error"] or "MPS job was cancelled.")
            job.output = ExecResult(**status["output"])
        except BaseException:
            cancelled = await asyncio.shield(router.cancel_mps_job.remote(lease_id, job.job_id))
            if cancelled["state"] == "quarantined":
                raise RuntimeError(cancelled["error"]) from None
            raise

    async def _run_gpu_process(self, job, **kwargs):
        source = Path(gpu_job_process.__file__).read_text()
        prefix = ["python3", "-c", source]
        options = json.dumps({"cwd": job.cwd, "env": job.env})
        execution = asyncio.create_task(
            self._exec_unreserved(
                shlex.join([*prefix, "run", job.job_id, job.command, options]),
                timeout_sec=job.timeout_sec + 15,
                **kwargs,
            )
        )
        try:
            return await asyncio.shield(execution)
        except BaseException:

            async def terminate():
                await self._exec_unreserved(shlex.join([*prefix, "cancel", job.job_id]), timeout_sec=5)
                await asyncio.wait_for(asyncio.shield(execution), timeout=5)

            cleanup = asyncio.create_task(terminate())
            try:
                await asyncio.shield(cleanup)
            finally:
                if not execution.done():
                    execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            raise

    async def _close_gpu_execution(self):
        """Drain owned jobs before either lifecycle provider removes its containers."""
        self._gpu_stopping = True
        pending = tuple(getattr(self, "_prepared_commands", ()))
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        try:
            if getattr(self, "_gpu_jobs", None):
                await self._gpu_jobs.close()
        finally:
            await self.wait_for_gpu_return()
            if getattr(self, "_mps", None):
                await self._confirm_mps_return()

    async def _confirm_mps_return(self):
        import ray

        router = ray.get_actor(self._mps["router_name"])
        await router.wait_gpu_command_returns.remote(self._mps["lease_id"])
