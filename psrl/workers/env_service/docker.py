"""Attach lease resources and ownership labels to every Harbor environment."""

import asyncio
import json
import shlex
import tempfile
import threading
import uuid
from pathlib import Path

import yaml
from harbor.environments.docker.docker import DockerEnvironment

from psrl.workers.env_service import gpu_job_process
from psrl.workers.env_service.container_budget import ContainerBudget
from psrl.workers.env_service.gpu_execution import GpuExecutionBackend, reservation_policy
from psrl.workers.env_service.gpu_jobs import GpuJob, GpuJobManager
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
        container_budget: dict | None = None,
        actor_id: str = "",
        gpu_lending=None,
        gpu_commands=None,
        gpu_execution=None,
        **kwargs,
    ):
        self._gpu_lending = gpu_lending
        self._daemon_os = None
        if gpu_execution and gpu_commands:
            if gpu_execution != {**gpu_commands, "mode": "command"}:
                raise ValueError("GPU execution and legacy command bindings disagree.")
        binding = gpu_execution or ({**gpu_commands, "mode": "command"} if gpu_commands else None)
        if binding and gpu_lending:
            raise ValueError("Managed GPU jobs require dedicated environment GPUs.")
        self._gpu_execution = binding
        self._execution_backend = GpuExecutionBackend(reservation_policy(binding)) if binding else None
        self._gpu_commands = binding if binding and binding["mode"] == "command" else None
        self._gpu_command_lock = asyncio.Lock()
        self._pending_gpu_return = None
        self.last_gpu_command = None
        self._gpu_jobs = GpuJobManager(self._execute_gpu_job)
        self._prepared_commands = set()
        self._lease_overlay = lease_overlay or {}
        self._container_budget = ContainerBudget(**container_budget) if container_budget is not None else None
        self._actor_id = actor_id
        self._ownership_dir = tempfile.TemporaryDirectory(prefix="psrl-compose-")
        self._ownership_path = Path(self._ownership_dir.name) / "ownership.yaml"
        super().__init__(*args, **kwargs)

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
        backend = self._backend()
        if backend is None:
            return await super().exec(command, timeout_sec=timeout_sec, **kwargs)

        async def container_id():
            result = await self._run_docker_compose_command(["ps", "-q", "main"], timeout_sec=10)
            identifiers = (result.stdout or "").split()
            if len(identifiers) != 1:
                raise RuntimeError("GPU execution needs exactly one running main container.")
            return identifiers[0]

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
            return await super(ManagedDockerEnvironment, self).exec(command, timeout_sec=timeout_sec, **kwargs)

        try:
            return await backend.execute(
                run,
                container_id,
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

        task = asyncio.current_task()
        self._prepared_commands.add(task)
        try:
            return await execute_prepared(self, module_path, payload, **kwargs)
        finally:
            self._prepared_commands.discard(task)

    def start_gpu_job(self, command, **kwargs):
        """Queue a bounded GPU job without waiting for capacity or command completion."""
        if self._backend() is None:
            raise RuntimeError("GPU jobs require a managed GPU reservation lease.")
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
        if backend.policy.mode == "episode":
            status["telemetry"] = {"available": False, "reason": "Per-job host attribution requires command mode."}
            return status
        router = ray.get_actor(backend.policy.router_name)
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
            pending = tuple(getattr(self, "_prepared_commands", ()))
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
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
        budget = getattr(self, "_container_budget", None)
        if budget is not None:
            for name, limits in budget.service_limits(services).items():
                overlay[name].update(limits)
        if self._gpu_commands:
            overlay["main"].update(command_security_options())
        overlay["main"]["labels"] = labels
        self._ownership_path.write_text(yaml.safe_dump({"services": overlay}))
        return [*paths, self._ownership_path]
