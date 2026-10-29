"""Adapt Harbor's environment contract to the shared EnvWorker container backend."""

import asyncio
import shlex
import time
import uuid
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.models.trial.config import ResourceMode

from psrl.workers.env_worker.sandbox import ExecResult as ShellExecResult
from psrl.workers.env_worker.sandbox import SandboxSpec
from psrl.workers.env_worker.worker import EnvWorker


class EnvWorkerEnvironment(BaseEnvironment):
    """Run prebuilt CPU task images through EnvWorker with explicit capability boundaries."""

    def __init__(self, *args, actor_id="", lease_labels=None, container_budget=None, **kwargs):
        unsupported = {
            "lease_overlay",
            "gpu_execution",
            "gpu_commands",
            "gpu_lending",
        } & kwargs.keys()
        if unsupported:
            raise ValueError(f"EnvWorker Harbor adapter does not yet implement bindings {sorted(unsupported)!r}.")
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
        labels.update(self._lease_labels)
        if self._actor_id:
            labels["psrl.actor_id"] = self._actor_id
        return SandboxSpec(
            image=self.task_env_config.docker_image,
            cpus=cpu,
            memory=f"{memory}m" if memory is not None else None,
            mounts=tuple(mounts),
            env=self._startup_env(),
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
            await self._cleanup()

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
        try:
            result = await self._worker.exec_process(
                self._sandbox_id,
                command,
                timeout_s=timeout_sec,
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
