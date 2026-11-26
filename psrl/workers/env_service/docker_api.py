"""Read local container identity without spawning a Docker command line client."""

import asyncio
import json
import logging
import math
import os
import time
from urllib.parse import quote, urlencode, urlsplit

import aiohttp

psrl_logger = logging.getLogger(__name__)


def _request_trace(record):
    """Record connection and response boundaries without retaining response content."""
    trace = aiohttp.TraceConfig()

    def callback(phase):
        async def mark(*args):
            record["phase"] = phase
            record.setdefault("events", []).append({"phase": phase, "at": time.time()})

        return mark

    trace.on_connection_create_start.append(callback("connect"))
    trace.on_connection_create_end.append(callback("connected"))
    trace.on_request_headers_sent.append(callback("response_headers"))
    return trace


async def _local_get(path, *, timeout_s=10, json_response=True, trace=None):
    """Use a bounded local daemon request without retaining an event loop bound session."""
    host = urlsplit(os.environ.get("DOCKER_HOST") or "unix:///var/run/docker.sock")
    if host.scheme != "unix" or not host.path or host.netloc or os.environ.get("DOCKER_CONTEXT"):
        raise RuntimeError("GPU command permissions require an explicit local Docker Unix socket.")
    connector = aiohttp.UnixConnector(path=host.path)
    timeout = aiohttp.ClientTimeout(total=timeout_s, ceil_threshold=math.inf)
    traces = [_request_trace(trace)] if trace is not None else []
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, trust_env=False, trace_configs=traces
    ) as session:
        url = f"http://localhost{path}"
        async with session.get(url) as response:
            if trace is not None:
                trace.update(phase="response_body", http_status=response.status)
            response.raise_for_status()
            return await response.json() if json_response else await response.text()


async def find_main_container(project, lease_id, actor_id, *, timeout_s=10, attempts=2, trace=None):
    """Resolve the live owned service with bounded retries of read requests only."""
    if not project or not lease_id:
        raise ValueError("Container lookup requires project and lease ownership.")
    if attempts < 1 or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Container lookup requires positive attempts and timeout.")
    labels = {
        "com.docker.compose.project": project,
        "com.docker.compose.service": "main",
        "psrl.env_lease": lease_id,
    }
    if actor_id:
        labels["psrl.actor_id"] = actor_id
    query = urlencode({"filters": json.dumps({"label": [f"{key}={value}" for key, value in labels.items()]})})
    records = trace if trace is not None else []
    for attempt in range(attempts):
        started = time.monotonic()
        record = {"attempt": attempt + 1, "started_at": time.time(), "phase": "request"}
        records.append(record)
        try:
            rows = await _local_get(f"/containers/json?{query}", timeout_s=timeout_s, trace=record)
            record["phase"] = "validate"
            if not isinstance(rows, list) or len(rows) != 1:
                raise RuntimeError("GPU execution needs exactly one running owned main container.")
            row = rows[0]
            if row.get("State") != "running" or any(row.get("Labels", {}).get(k) != v for k, v in labels.items()):
                raise RuntimeError("GPU container state or ownership does not match the requested service.")
            identifier = row.get("Id", "")
            if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
                raise RuntimeError("Docker returned an invalid full container identity.")
            record.update(status="complete", container_id=identifier)
            return identifier
        except BaseException as error:
            record.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed")
            record.update(error_type=type(error).__name__, error=str(error))
            transient = isinstance(error, (TimeoutError, aiohttp.ClientConnectionError)) or (
                isinstance(error, aiohttp.ClientResponseError) and error.status in (500, 502, 503, 504)
            )
            if not transient or attempt + 1 == attempts:
                raise
        finally:
            record["elapsed_s"] = time.monotonic() - started
            if record["status"] != "complete":
                psrl_logger.warning(f"Container lookup for lease {lease_id!r} and project {project!r}: {record!r}.")
        # Cancellation from the enclosing admission deadline must stop the retry sequence.
        await asyncio.sleep(0.25)


async def inspect_container(container_id, *, timeout_s=10):
    """Read current identity and state for device cgroup validation."""
    return await _local_get(f"/containers/{quote(container_id, safe='')}/json", timeout_s=timeout_s)


async def ping_docker():
    """Require a live local daemon before declaring a node service ready."""
    if (await _local_get("/_ping", timeout_s=5, json_response=False)).strip() != "OK":
        raise RuntimeError("Local Docker daemon returned an invalid readiness response.")


async def list_unit_containers(*, timeout_s=3):
    """Read environment containers without inspecting unrelated daemon workloads."""
    query = urlencode({"all": "1", "filters": json.dumps({"label": ["psrl.env_unit"]})})
    return await _local_get(f"/containers/json?{query}", timeout_s=timeout_s)


async def container_stats(container_id, *, timeout_s=3):
    """Read one bounded Docker resource sample without keeping a stream open."""
    return await _local_get(f"/containers/{quote(container_id, safe='')}/stats?stream=false", timeout_s=timeout_s)
