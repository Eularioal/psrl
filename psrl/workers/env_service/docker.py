"""Attach lease resources and ownership labels to every Harbor environment."""

import tempfile
import uuid
from pathlib import Path

import yaml
from harbor.environments.docker.docker import DockerEnvironment


class ManagedDockerEnvironment(DockerEnvironment):
    """Keep ownership metadata when Harbor creates a separate verifier environment."""

    def __init__(self, *args, lease_overlay: dict | None = None, actor_id: str = "", gpu_lending=None, **kwargs):
        self._gpu_lending = gpu_lending
        self._lease_overlay = lease_overlay or {}
        self._actor_id = actor_id
        self._ownership_dir = tempfile.TemporaryDirectory(prefix="psrl-compose-")
        self._ownership_path = Path(self._ownership_dir.name) / "ownership.yaml"
        super().__init__(*args, **kwargs)

    async def start(self, force_build: bool):
        """Keep GPU ownership until every asynchronous container start is acknowledged."""
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
        overlay["main"]["labels"] = labels
        self._ownership_path.write_text(yaml.safe_dump({"services": overlay}))
        return [*paths, self._ownership_path]
