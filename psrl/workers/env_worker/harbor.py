"""Adapt Harbor's environment contract to the shared EnvWorker container backend."""

import asyncio
import shlex
import uuid
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.models.trial.config import ResourceMode

from psrl.workers.env_worker.sandbox import SandboxSpec
from psrl.workers.env_worker.worker import EnvWorker


class EnvWorkerEnvironment(BaseEnvironment):
    """Run prebuilt CPU task images through EnvWorker with explicit capability boundaries."""

    def __init__(self, *args, actor_id="", **kwargs):
        unsupported = {
            "lease_overlay",
            "gpu_execution",
            "gpu_commands",
            "gpu_lending",
            "container_budget",
        } & kwargs.keys()
        if unsupported:
            raise ValueError(f"EnvWorker Harbor adapter does not yet implement bindings {sorted(unsupported)!r}.")
        self._worker = EnvWorker(0, 1, 0, 300)
        self._sandbox_id = f"harbor-{uuid.uuid4().hex}"
        self._actor_id = actor_id
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
        labels = {"psrl.harbor_session": self.session_id}
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
            try:
                await self._worker.create_sandbox(self._spec(), self._sandbox_id)
                self._started = True
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
            except BaseException:
                await self._worker.destroy_sandbox(self._sandbox_id)
                self._closed = True
                self._started = False
                raise

    async def stop(self, delete=True):
        """Always remove owned containers, including after partial startup or timeout."""
        if not delete:
            raise ValueError("EnvWorker Harbor adapter requires delete=True for confirmed cleanup.")
        async with self._lifecycle_lock:
            await self._worker.destroy_sandbox(self._sandbox_id)
            self._closed = True
            self._started = False

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if not self._started or self._closed:
            raise RuntimeError("Harbor environment is not running.")
        result = await self._worker.exec_process(
            self._sandbox_id,
            command,
            timeout_s=timeout_sec,
            cwd=cwd or self.task_env_config.workdir,
            env=self._merge_env(env),
            user=self._resolve_user(user),
            output_callback=self._output_callback(),
        )
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.return_code)

    async def upload_file(self, source_path, target_path):
        await self._worker.copy_path(self._sandbox_id, str(source_path), target_path, True)

    async def upload_dir(self, source_dir, target_dir):
        result = await self.exec(f"mkdir -p -- {shlex.quote(target_dir)}", user="0")
        if result.return_code:
            raise RuntimeError(f"Cannot prepare upload directory: {result.stderr!r}.")
        # Docker copies directory contents when the source ends with a dot component.
        source = str(Path(source_dir).resolve()) + "/."
        await self._worker.copy_path(self._sandbox_id, source, target_dir, True)

    async def download_file(self, source_path, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._worker.copy_path(self._sandbox_id, str(target), source_path, False)

    async def download_dir(self, source_dir, target_dir):
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        await self._worker.copy_path(self._sandbox_id, str(target), source_dir.rstrip("/") + "/.", False)
