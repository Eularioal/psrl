"""Read local container identity without spawning a Docker command line client."""

import os
from urllib.parse import quote, urlsplit

import aiohttp


async def _local_get(path, *, timeout_s=10, json_response=True):
    """Use a bounded local daemon request without retaining an event loop bound session."""
    host = urlsplit(os.environ.get("DOCKER_HOST") or "unix:///var/run/docker.sock")
    if host.scheme != "unix" or not host.path or host.netloc or os.environ.get("DOCKER_CONTEXT"):
        raise RuntimeError("GPU command permissions require an explicit local Docker Unix socket.")
    connector = aiohttp.UnixConnector(path=host.path)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
        url = f"http://localhost{path}"
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json() if json_response else await response.text()


async def inspect_container(container_id, *, timeout_s=10):
    """Read current identity and state for device cgroup validation."""
    return await _local_get(f"/containers/{quote(container_id, safe='')}/json", timeout_s=timeout_s)


async def ping_docker():
    """Require a live local daemon before declaring a node service ready."""
    if (await _local_get("/_ping", timeout_s=5, json_response=False)).strip() != "OK":
        raise RuntimeError("Local Docker daemon returned an invalid readiness response.")
