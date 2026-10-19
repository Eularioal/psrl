"""
Render what Docker Compose would actually start, before and after the env service.

Uses the real Harbor base compose, the real overlay builder, the real driver-library
resolver and the real `docker compose config` (client side, needs no daemon).
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.types import EnvDemand, EnvLease

harbor_spec = importlib.util.find_spec("harbor")
if harbor_spec is None or not harbor_spec.submodule_search_locations:
    raise RuntimeError("Install Harbor before rendering its Docker Compose configuration.")
HARBOR_BASE = (
    Path(next(iter(harbor_spec.submodule_search_locations))) / "environments/docker/docker-compose-prebuilt.yaml"
)
STATIC_OVERRIDE = Path(os.environ["BASELINE_REPO"]) / "examples/sciaccel_rl/config/gpu-compose-override.yaml"
NEW_REPO_STATIC_OVERRIDE = Path("examples/sciaccel_rl/config/gpu-compose-override.yaml")
WORK = Path("/tmp/env_verify/compose")
WORK.mkdir(parents=True, exist_ok=True)


def render(*files: Path) -> dict:
    cmd = ["docker", "compose", "-p", "verify"]
    for f in files:
        cmd += ["-f", str(f)]
    cmd += ["config", "--format", "json"]
    env = dict(os.environ, PREBUILT_IMAGE_NAME="verify:latest")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:])
    return json.loads(proc.stdout)["services"]["main"]


def summarize(tag: str, main: dict) -> None:
    devices = [d if isinstance(d, str) else d for d in main.get("devices", [])]
    gpu_devs = sorted(
        d
        for d in (str(x).split(":")[0] for x in devices)
        if d[len("/dev/nvidia") :].isdigit()
        if d.startswith("/dev/nvidia")
    )
    libs = [v["source"] if isinstance(v, dict) else v for v in main.get("volumes", [])]
    print(f"[{tag}]")
    print("  labels      :", json.dumps(main.get("labels", {}), sort_keys=True))
    print("  gpu devices :", gpu_devs)
    print("  mem_limit   :", main.get("mem_limit"))
    cuda = [x for x in libs if "libcuda.so" in str(x)]
    print("  libcuda src :", sorted(set(cuda)))


def lease(lease_id: str, unit: str, gpus: tuple[int, ...], shared: bool = False) -> EnvLease:
    return EnvLease(lease_id, "sess-" + lease_id, unit, "127.0.0.1", gpus, EnvDemand(needs_gpu=True), 0.0, 0.0, shared)


def main() -> int:
    version = read_kernel_driver_version()
    libs = resolve_driver_libs(version)
    print("host kernel driver version :", version)
    print("resolved libcuda mount     :", [m for m in libs.mounts if "libcuda.so.1" in m[1]])
    print("missing optional libs      :", libs.missing_optional)

    static_text = STATIC_OVERRIDE.read_text()
    print("static override still hardcodes 535.247.01 :", "535.247.01" in static_text)
    print("new tree keeps the static file unchanged   :", NEW_REPO_STATIC_OVERRIDE.read_text() == static_text)
    print()

    actor_overlay = WORK / "actor_id.yaml"
    actor_overlay.write_text(
        yaml.dump({"services": {"main": {"labels": ["psrl.actor_id=actor-7"], "mem_limit": "4096m"}}})
    )

    # BEFORE (c00354c): every GPU episode gets the same static override.
    for i in range(3):
        summarize(
            f"BEFORE episode {i}: static gpu-compose-override.yaml",
            render(HARBOR_BASE, STATIC_OVERRIDE, actor_overlay),
        )
    print()

    # AFTER: one overlay per lease, built from the lease.
    specs = [
        lease("g0-1-aaaa", "g0", (0,)),
        lease("g1-2-bbbb", "g1", (1,)),
        lease("g0-3-cccc", "g0", (0,), shared=True),
    ]
    for spec in specs:
        path = write_overlay(build_lease_overlay(spec, libs))
        try:
            summarize(
                f"AFTER lease {spec.lease_id} (unit {spec.unit_id}, cards {list(spec.gpu_indices)})",
                render(HARBOR_BASE, path, actor_overlay),
            )
        finally:
            path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
