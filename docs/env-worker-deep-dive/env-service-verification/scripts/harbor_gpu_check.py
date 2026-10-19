"""Does real Harbor accept a GPU task once override_gpus=0 is set, and where does the lease overlay land?"""

import tempfile
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.types import EnvDemand, EnvLease

root = Path(tempfile.mkdtemp(prefix="harbor_gpu_check_"))
env_dir = root / "environment"
env_dir.mkdir()
(env_dir / "Dockerfile").write_text("FROM scratch\n")
trial_dir = root / "trial"
trial_dir.mkdir()


def build(**kwargs):
    return DockerEnvironment(
        environment_dir=env_dir,
        environment_name="gpu-task",
        session_id="verify__env",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(gpus=1),
        **kwargs,
    )


try:
    build()
    print("WITHOUT override_gpus: accepted (no GPU validation error)")
except Exception as exc:  # noqa: BLE001
    print("WITHOUT override_gpus: REFUSED ->", type(exc).__name__, str(exc)[:120])

lease = EnvLease("g0-1-aaaa", "s", "g0", "127.0.0.1", (1,), EnvDemand(needs_gpu=True), 0.0, 0.0)
overlay = write_overlay(build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version())))
try:
    env = build(override_gpus=0, extra_docker_compose=[overlay])
    print("WITH override_gpus=0 + lease overlay: accepted; task gpus now =", env.task_env_config.gpus)
    print("compose files in order:")
    for p in env._docker_compose_paths:
        tag = "  <-- lease overlay" if p == overlay else ""
        print("   ", p.name, tag)
except Exception as exc:  # noqa: BLE001
    print("WITH override_gpus=0: REFUSED ->", type(exc).__name__, str(exc)[:160])
finally:
    overlay.unlink()
