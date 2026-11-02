"""Attach lease resources and ownership labels to every Harbor environment."""

import asyncio
import tempfile
import threading
import uuid
from pathlib import Path

import yaml
from harbor.environments.docker.docker import DockerEnvironment

from psrl.workers.env_service.container_budget import ContainerBudget
from psrl.workers.env_service.managed_execution import ManagedGpuExecution
from psrl.workers.env_service.overlay import command_security_options

_KERNEL_PROBE_LOCK = threading.Lock()


def _warm_kernel_probe():
    """Populate Harbor's existing cache once even when several jobs arrive together."""
    with _KERNEL_PROBE_LOCK:
        DockerEnvironment._egress_control_kernel_support()


class ManagedDockerEnvironment(ManagedGpuExecution, DockerEnvironment):
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
        self._init_gpu_execution(binding)
        self._lease_overlay = lease_overlay or {}
        self._container_budget = ContainerBudget(**container_budget) if container_budget is not None else None
        self._actor_id = actor_id
        self._ownership_dir = tempfile.TemporaryDirectory(prefix="psrl-compose-")
        self._ownership_path = Path(self._ownership_dir.name) / "ownership.yaml"
        super().__init__(*args, **kwargs)

    async def _exec_unreserved(self, command, **kwargs):
        return await super().exec(command, **kwargs)

    async def _gpu_container_id(self):
        result = await self._run_docker_compose_command(["ps", "-q", "main"], timeout_sec=10)
        identifiers = (result.stdout or "").split()
        if len(identifiers) != 1:
            raise RuntimeError("GPU execution needs exactly one running main container.")
        return identifiers[0]

    async def stop(self, delete: bool):
        try:
            await self._close_gpu_execution()
        finally:
            if getattr(self, "_mps", None):
                await self._confirm_mps_return()
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
        if self._gpu_commands or getattr(self, "_mps", None):
            overlay["main"].update(command_security_options())
            if getattr(self, "_mps", None):
                overlay["main"]["cap_drop"] = ["ALL"]
        overlay["main"]["labels"] = labels
        self._ownership_path.write_text(yaml.safe_dump({"services": overlay}))
        return [*paths, self._ownership_path]
