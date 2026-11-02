"""Adapt Harbor's environment contract to the shared EnvWorker container backend."""

import asyncio
import shlex
import time
import uuid
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.models.trial.config import ResourceMode

from psrl.workers.env_service.managed_execution import ManagedGpuExecution
from psrl.workers.env_worker.sandbox import ExecResult as ShellExecResult
from psrl.workers.env_worker.sandbox import SandboxSpec
from psrl.workers.env_worker.worker import EnvWorker


class EnvWorkerEnvironment(ManagedGpuExecution, BaseEnvironment):
    """Run prebuilt CPU task images through EnvWorker with explicit capability boundaries."""

    def __init__(
        self,
        *args,
        actor_id="",
        lease_labels=None,
        container_budget=None,
        lease_overlay=None,
        gpu_execution=None,
        gpu_commands=None,
        **kwargs,
    ):
        unsupported = {
            "gpu_lending",
        } & kwargs.keys()
        if unsupported:
            raise ValueError(f"EnvWorker Harbor adapter does not yet implement bindings {sorted(unsupported)!r}.")
        if gpu_commands and gpu_execution != {**gpu_commands, "mode": "command"}:
            raise ValueError("Native GPU command bindings must match the execution policy.")
        self._init_gpu_execution(gpu_execution)
        self._lease_overlay = lease_overlay or {}
        if set(self._lease_overlay.get("services", {})) - {"main"}:
            raise ValueError("Native lifecycle accepts exactly one main service.")
        if gpu_execution and not self._lease_overlay.get("services", {}).get("main", {}).get("devices"):
            raise ValueError("Native GPU execution requires leased device mappings.")
        if self._lease_overlay.get("services", {}).get("main", {}).get("devices") and not gpu_execution:
            raise ValueError("Leased GPU devices require a managed execution policy.")
        if gpu_execution:
            main = self._lease_overlay["services"]["main"]
            labels = dict(item.split("=", 1) for item in main.get("labels", []))
            if labels.get("psrl.env_lease") != gpu_execution["lease_id"]:
                raise ValueError("GPU execution and device mapping must have the same lease owner.")
            if gpu_execution["mode"] == "command" and any(not d.endswith(":m") for d in main["devices"]):
                raise ValueError("Command GPU mappings must start without read or write permission.")
        self._worker = EnvWorker(0, 1, 0, 300)
        self._sandbox_id = f"harbor-{uuid.uuid4().hex}"
        self._actor_id = actor_id
        self._startup_task = None
        self._lease_labels = dict(lease_labels or {})
        self._container_budget = container_budget
        self._cleanup_pending = False
        self.timings = {}
        self._started = False
        self._closed = False
        self._lifecycle_lock = asyncio.Lock()
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "psrl-env-worker"

    @property
    def capabilities(self):
        return EnvironmentCapabilities(disable_internet=True, mounted=True)

    @classmethod
    def resource_capabilities(cls):
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_definition(self):
        if not self.task_env_config.docker_image:
            raise ValueError("EnvWorker Harbor adapter requires a prebuilt docker_image.")
        if any(
            (self.environment_dir / name).exists()
            for name in ("Dockerfile", "docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
        ):
            raise ValueError("EnvWorker Harbor adapter does not build images or interpret Compose definitions.")
        if self.task_env_config.gpus or self._override_gpus:
            raise ValueError("EnvWorker Harbor adapter does not yet support GPU resource bindings.")
        if not self._network_disabled:
            raise ValueError("EnvWorker Harbor adapter currently requires no-network policy.")
        if any(policy != self.network_policy for policy in self._phase_network_policies):
            raise ValueError("EnvWorker Harbor adapter does not switch network policy between phases.")
        if any(mount.get("type") != "bind" for mount in self._mounts):
            raise ValueError("EnvWorker Harbor adapter supports explicit bind mounts only.")
        if self.task_env_config.storage_mb is not None:
            raise ValueError("EnvWorker Harbor adapter does not enforce disk quotas.")

    def _spec(self):
        mounts = []
        for mount in self._mounts:
            source = Path(mount["source"]).resolve()
            if not source.exists():
                if mount.get("bind", {}).get("create_host_path") is False:
                    raise FileNotFoundError(source)
                source.mkdir(parents=True)
            mounts.append((str(source), mount["target"], "ro" if mount.get("read_only") else "rw"))
        memory = self._resource_limit_value("memory", auto_mode=ResourceMode.LIMIT)
        cpu = self._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT)
        if self._container_budget is not None:
            from psrl.workers.env_service.container_budget import ContainerBudget

            budget = ContainerBudget(**self._container_budget)
            if cpu is None or memory is None or cpu > budget.main_cpu or memory > budget.main_memory_mb:
                raise ValueError("Environment limits exceed the admitted main-container budget.")
        labels = {"psrl.harbor_session": self.session_id}
        main = self._lease_overlay.get("services", {}).get("main", {})
        if set(main) - {"labels", "devices", "volumes", "environment", "ipc", "cap_drop", "security_opt"}:
            raise ValueError("Unsupported native lease resource fields.")
        if set(main.get("cap_drop", [])) - {"ALL", "SYS_ADMIN", "MKNOD"}:
            raise ValueError("Unsupported native capability policy.")
        if set(main.get("security_opt", [])) - {"no-new-privileges:true"}:
            raise ValueError("Unsupported native security policy.")
        labels.update(dict(item.split("=", 1) for item in main.get("labels", [])))
        labels.update(self._lease_labels)
        for volume in main.get("volumes", []):
            host, target, mode = volume.rsplit(":", 2)
            mounts.append((host, target, mode))
        startup_env = self._startup_env()
        startup_env.update(dict(item.split("=", 1) for item in main.get("environment", [])))
        if self._actor_id:
            labels["psrl.actor_id"] = self._actor_id
        return SandboxSpec(
            image=self.task_env_config.docker_image,
            cpus=cpu,
            memory=f"{memory}m" if memory is not None else None,
            mounts=tuple(mounts),
            env=startup_env,
            devices=tuple(main.get("devices", ())),
            ipc=main.get("ipc", "private"),
            network="none",
            labels=labels,
            startup_timeout_s=30,
        )

    async def start(self, force_build=False):
        if force_build:
            raise ValueError("EnvWorker Harbor adapter uses cached images and cannot force a build.")
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("A stopped Harbor environment cannot be restarted.")
            if self._started:
                return
            self._startup_task = asyncio.current_task()
            started = time.monotonic()
            try:
                self._cleanup_pending = True
                await self._worker.create_sandbox(self._spec(), self._sandbox_id)
                directories = self._mount_targets(writable_only=True)
                if self.task_env_config.workdir:
                    directories.append(self.task_env_config.workdir)
                if directories:
                    command = "mkdir -p -- " + " ".join(shlex.quote(str(path)) for path in directories)
                    result = await self._worker.exec_process(
                        self._sandbox_id, command, cwd="/", user="0", timeout_s=10
                    )
                    if result.return_code:
                        raise RuntimeError(f"Cannot prepare environment directories: {result.stderr!r}.")
                    result = await self.ensure_dirs(directories)
                    if result is not None and result.return_code:
                        raise RuntimeError(f"Cannot prepare environment permissions: {result.stderr!r}.")
                await self._upload_environment_dir_after_start()
                self._started = True
                self.timings["startup_s"] = time.monotonic() - started
            except BaseException:
                self._invalidate()
                await self._cleanup()
                raise
            finally:
                self._startup_task = None

    async def stop(self, delete=True):
        """Always remove owned containers, including after partial startup or timeout."""
        if not delete:
            raise ValueError("EnvWorker Harbor adapter requires delete=True for confirmed cleanup.")
        async with self._lifecycle_lock:
            self._invalidate()
            try:
                await self._close_gpu_execution()
            finally:
                if getattr(self, "_mps", None):
                    await self._confirm_mps_return()
                self._invalidate()
                await self._cleanup()

    async def _exec_unreserved(self, command, **kwargs):
        return await self._execute_command(command, **kwargs)

    def _require_gpu_execution(self):
        super()._require_gpu_execution()
        self._require_running()

    async def _gpu_container_id(self):
        from psrl.workers.env_service.docker_api import inspect_container

        self._require_running()
        info = await inspect_container(self._worker.container_name(self._sandbox_id))
        return info["Id"]

    async def _cleanup(self):
        """Keep admission ownership until physical deletion has been acknowledged."""

        async def finish():
            started = time.monotonic()
            await self._worker.destroy_sandbox(self._sandbox_id)
            self._cleanup_pending = False
            self.timings["cleanup_s"] = time.monotonic() - started

        cleanup = asyncio.create_task(finish())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    def _require_running(self):
        if self._closed or (not self._started and self._startup_task is not asyncio.current_task()):
            raise RuntimeError("Harbor environment is not running.")

    def _invalidate(self):
        self._closed = True
        self._started = False

    async def exec_shell(
        self,
        command: str,
        timeout_sec: float | None = None,
        no_output_timeout_sec: float | None = None,
    ) -> ShellExecResult:
        """
        Execute in the container's persistent shell with combined output streams.

        Shell state starts from the image user, directory and startup environment.
        Harbor execution scopes do not apply. Change state explicitly with shell commands.
        A timeout invalidates the whole environment and preserves the timeout result.
        """
        self._require_running()
        try:
            result = await self._worker.exec(
                self._sandbox_id,
                command,
                timeout_s=timeout_sec,
                no_output_timeout_s=no_output_timeout_sec,
            )
        except BaseException:
            self._invalidate()
            raise
        if result.timed_out:
            self._invalidate()
        return result

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self._require_running()
        return await self._execute_command(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    async def _execute_command(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        try:
            # Harbor owns the outer deadline when no command deadline is supplied.
            result = await self._worker.exec_process(
                self._sandbox_id,
                command,
                timeout_s=float("inf") if timeout_sec is None else timeout_sec,
                cwd=cwd or self.task_env_config.workdir,
                env=self._merge_env(env),
                user=self._resolve_user(user),
                output_callback=self._output_callback(),
            )
        except BaseException:
            self._invalidate()
            raise
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.return_code)

    async def upload_file(self, source_path, target_path):
        self._require_running()
        await self._worker.copy_path(self._sandbox_id, str(source_path), target_path, True)

    async def upload_dir(self, source_dir, target_dir):
        result = await self.exec(f"mkdir -p -- {shlex.quote(target_dir)}", user="0")
        if result.return_code:
            raise RuntimeError(f"Cannot prepare upload directory: {result.stderr!r}.")
        # Docker copies directory contents when the source ends with a dot component.
        source = str(Path(source_dir).resolve()) + "/."
        await self._worker.copy_path(self._sandbox_id, source, target_dir, True)

    async def download_file(self, source_path, target_path):
        self._require_running()
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._worker.copy_path(self._sandbox_id, str(target), source_path, False)

    async def download_dir(self, source_dir, target_dir):
        self._require_running()
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        await self._worker.copy_path(self._sandbox_id, str(target), source_dir.rstrip("/") + "/.", False)
