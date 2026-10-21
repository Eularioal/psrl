"""Attach lease resources and ownership labels to every Harbor environment."""

import asyncio
import json
import shlex
import tempfile
import threading
import time
import uuid
from pathlib import Path

import yaml
from harbor.environments.docker.docker import DockerEnvironment

from psrl.workers.env_service import gpu_job_process
from psrl.workers.env_service.gpu_jobs import GpuJobManager
from psrl.workers.env_service.overlay import command_security_options

_KERNEL_PROBE_LOCK = threading.Lock()


def _warm_kernel_probe():
    """Populate Harbor's existing cache once even when several jobs arrive together."""
    with _KERNEL_PROBE_LOCK:
        DockerEnvironment._egress_control_kernel_support()


class ManagedDockerEnvironment(DockerEnvironment):
    """Keep ownership metadata when Harbor creates a separate verifier environment."""

    @staticmethod
    async def prepare_runtime():
        """Warm the synchronous constructor probe before creating Harbor jobs."""
        probe = asyncio.create_task(asyncio.to_thread(_warm_kernel_probe))
        try:
            await asyncio.shield(probe)
        except asyncio.CancelledError:
            # The probe runs a temporary container, so join it before caller cleanup.
            await probe
            raise

    def __init__(
        self,
        *args,
        lease_overlay: dict | None = None,
        actor_id: str = "",
        gpu_lending=None,
        gpu_commands=None,
        **kwargs,
    ):
        self._gpu_lending = gpu_lending
        self._daemon_os = None
        self._gpu_commands = gpu_commands
        self._gpu_command_lock = asyncio.Lock()
        self._pending_gpu_return = None
        self.last_gpu_command = None
        self._gpu_jobs = GpuJobManager(self._execute_gpu_job)
        self._lease_overlay = lease_overlay or {}
        self._actor_id = actor_id
        self._ownership_dir = tempfile.TemporaryDirectory(prefix="psrl-compose-")
        self._ownership_path = Path(self._ownership_dir.name) / "ownership.yaml"
        super().__init__(*args, **kwargs)

    async def _wait_gpu_return(self):
        """Surface deferred failure before another GPU command or environment teardown."""
        if self._pending_gpu_return is None:
            return
        router, lease_id, record = self._pending_gpu_return
        status = await router.release_gpu_command.remote(lease_id, record["request_id"], record["epoch"])
        record["released_at"] = status["returned_at"]
        self._pending_gpu_return = None

    async def wait_for_gpu_return(self):
        """Confirm all issued GPU work before accepting dependent results."""
        async with self._gpu_command_lock:
            await self._wait_gpu_return()

    async def exec_gpu(
        self, command, *, timeout_sec=300, wait_timeout_sec=300, overlap_return=False, _job=None, **kwargs
    ):
        """
        Execute a foreground GPU command with an optional overlap of planning and return.

        Tasks must route GPU work through this method when command reservations are enabled.
        Persistent CUDA processes prevent return and cause the lease to await cleanup.
        Set overlap_return only when output is final and independent CPU work is safe.
        Deferred errors surface at wait_for_gpu_return, the next GPU command, or stop.
        """
        if not self._gpu_commands:
            return await super().exec(command, timeout_sec=timeout_sec, **kwargs)
        if timeout_sec is None or timeout_sec <= 0 or wait_timeout_sec <= 0:
            raise ValueError("GPU commands require positive execution and queue timeouts.")
        import ray

        async with self._gpu_command_lock:
            await self._wait_gpu_return()
            router = ray.get_actor(self._gpu_commands["router_name"])
            lease_id = self._gpu_commands["lease_id"]
            result = await self._run_docker_compose_command(["ps", "-q", "main"], timeout_sec=10)
            container_ids = (result.stdout or "").split()
            if len(container_ids) != 1:
                raise RuntimeError("GPU command needs exactly one running main container.")
            request_id = _job.job_id if _job else uuid.uuid4().hex
            record = {"request_id": request_id, "requested_at": time.time()}
            self.last_gpu_command = record
            try:
                async with asyncio.timeout(wait_timeout_sec):
                    while True:
                        admission = await router.acquire_gpu_command.remote(
                            lease_id, container_ids[0], request_id, *([_job.tag] if _job else [])
                        )
                        if admission["status"] == "granted":
                            break
                        await asyncio.sleep(0.1)
                record.update(epoch=admission["epoch"], admitted_at=time.time())
                if _job:
                    _job.transition("running")
                async with asyncio.timeout(timeout_sec):
                    if _job:
                        output = await self._run_gpu_process(_job, **kwargs)
                    else:
                        output = await super().exec(command, timeout_sec=timeout_sec, **kwargs)
                    record["command_finished_at"] = time.time()
                record["return_requested_at"] = time.time()
                if _job:
                    _job.output = output
                    _job.transition("returning")
                    _job.output_ready.set()
                if overlap_return:
                    status = await router.begin_gpu_command_return.remote(lease_id, request_id, admission["epoch"])
                    if status["error"]:
                        raise RuntimeError(f"GPU command return failed: {status['error']}")
                    self._pending_gpu_return = (router, lease_id, record)
                    record["return_accepted_at"] = time.time()
                else:
                    status = await router.release_gpu_command.remote(lease_id, request_id, admission["epoch"])
                    record["released_at"] = status["returned_at"]
                return output
            except BaseException:
                # The router serializes cancellation with any activation already in flight.
                await asyncio.shield(router.cancel_gpu_command.remote(lease_id, request_id))
                raise

    def start_gpu_job(self, command, **kwargs):
        """Queue a bounded GPU job without waiting for capacity or command completion."""
        if not self._gpu_commands:
            raise RuntimeError("GPU jobs require a command reservation lease.")
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
        router = ray.get_actor(self._gpu_commands["router_name"])
        status["telemetry"] = await router.gpu_job_telemetry.remote(self._gpu_commands["lease_id"], job_id)
        return status

    async def finish_gpu_jobs(self):
        """Join bounded jobs and surface failures before dependent verification begins."""
        for job in self._gpu_jobs.jobs.values():
            await asyncio.shield(asyncio.gather(job.task, return_exceptions=True))
            if job.error:
                raise RuntimeError(job.error)

    async def _execute_gpu_job(self, job):
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

    async def _run_gpu_process(self, job, **kwargs):
        source = Path(gpu_job_process.__file__).read_text()
        prefix = ["python3", "-c", source]
        options = json.dumps({"cwd": job.cwd, "env": job.env})
        execution = asyncio.create_task(
            super().exec(
                shlex.join([*prefix, "run", job.job_id, job.command, options]),
                timeout_sec=job.timeout_sec + 15,
                **kwargs,
            )
        )
        try:
            return await asyncio.shield(execution)
        except BaseException:

            async def terminate():
                await super(ManagedDockerEnvironment, self).exec(
                    shlex.join([*prefix, "cancel", job.job_id]), timeout_sec=5
                )
                await asyncio.wait_for(asyncio.shield(execution), timeout=5)

            cleanup = asyncio.create_task(terminate())
            try:
                await asyncio.shield(cleanup)
            finally:
                if not execution.done():
                    execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            raise

    async def stop(self, delete: bool):
        """Join pending return and still destroy the environment when verification fails."""
        try:
            try:
                if getattr(self, "_gpu_jobs", None):
                    await self._gpu_jobs.close()
            finally:
                await self.wait_for_gpu_return()
        finally:
            await super().stop(delete=delete)

    async def start(self, force_build: bool):
        """Keep GPU ownership until every asynchronous container start is acknowledged."""
        self._daemon_os = await asyncio.to_thread(super()._detect_daemon_os)
        if not self._gpu_lending:
            return await super().start(force_build)
        import ray

        router = ray.get_actor(self._gpu_lending["router_name"])
        lease_id = self._gpu_lending["lease_id"]
        token = uuid.uuid4().hex
        await router.begin_gpu_container.remote(lease_id, token)
        try:
            result = await super().start(force_build)
        except BaseException:
            await router.fail_gpu_container.remote(lease_id, token)
            raise
        await router.end_gpu_container.remote(lease_id, token)
        return result

    def _detect_daemon_os(self):
        """Use the fresh query from `start` during Harbor's synchronous validation."""
        return self._daemon_os

    @property
    def _docker_compose_paths(self) -> list[Path]:
        """Apply ownership labels after all task and network services are known."""
        paths = super()._docker_compose_paths
        main = dict(self._lease_overlay.get("services", {}).get("main", {}))
        labels = dict(item.split("=", 1) for item in main.get("labels", []))
        if self._actor_id:
            labels["psrl.actor_id"] = self._actor_id
        services = {"main"}
        for path in paths:
            # Only service names are needed, including files with Compose-specific YAML tags.
            document = yaml.load(path.read_text(), Loader=yaml.BaseLoader) or {}
            services.update(document.get("services", {}))
        overlay = {name: {"labels": labels} for name in services}
        overlay["main"].update(main)
        if self._gpu_commands:
            overlay["main"].update(command_security_options())
        overlay["main"]["labels"] = labels
        self._ownership_path.write_text(yaml.safe_dump({"services": overlay}))
        return [*paths, self._ownership_path]
