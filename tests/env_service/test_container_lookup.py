"""Exercise identity changes, transport faults and cancellation over a Unix socket."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from psrl.workers.env_service import docker_api
from psrl.workers.env_service.docker import ManagedDockerEnvironment

LABELS = {
    "com.docker.compose.project": "project",
    "com.docker.compose.service": "main",
    "psrl.env_lease": "lease",
    "psrl.actor_id": "actor",
}


def container(identifier="a", **changes):
    return {"Id": identifier * 64, "Labels": dict(LABELS), "State": "running", **changes}


@asynccontextmanager
async def daemon(tmp_path, monkeypatch, handler):
    socket = tmp_path / "d.sock"
    monkeypatch.setenv("DOCKER_HOST", f"unix://{socket}")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    app = web.Application()
    app.router.add_get("/containers/json", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.UnixSite(runner, str(socket)).start()
        yield
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_provider_requeries_identity_and_filters_all_owners(tmp_path, monkeypatch):
    identity = "a"

    async def handler(request):
        filters = json.loads(request.query["filters"])
        assert set(filters["label"]) == {f"{k}={v}" for k, v in LABELS.items()}, "All owners must be filtered."
        return web.json_response([container(identity)])

    env = ManagedDockerEnvironment.__new__(ManagedDockerEnvironment)
    env._init_gpu_execution({"mode": "episode", "router_name": "r", "lease_id": "lease"})
    env.session_id = "Project"
    env._actor_id = "actor"
    async with daemon(tmp_path, monkeypatch, handler):
        assert await env._gpu_container_id() == "a" * 64, "The first generation must resolve."
        identity = "b"
        assert await env._gpu_container_id() == "b" * 64, "Recreation must not reuse a stale ID."
        assert env.last_container_query[-1]["status"] == "complete", "The query must retain its result."


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["timeout", "server_error"])
async def test_transient_read_retries_and_keeps_transport_evidence(tmp_path, monkeypatch, fault):
    requests = 0

    async def handler(request):
        nonlocal requests
        requests += 1
        if requests == 1:
            if fault == "server_error":
                raise web.HTTPServiceUnavailable()
            await asyncio.sleep(0.06)
        return web.json_response([container()])

    trace = []
    async with daemon(tmp_path, monkeypatch, handler):
        found = await docker_api.find_main_container("project", "lease", "actor", timeout_s=0.02, trace=trace)
    assert found == "a" * 64 and requests == 2, "Only the failed read should be retried."
    assert [r["status"] for r in trace] == ["failed", "complete"], "Recovery must not erase the failure."
    assert trace[0]["phase"] in ("response_headers", "response_body"), "Transport wait must be located."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [[], [container(), container("b")], [container(State="exited")], [container(Labels={})]],
)
async def test_missing_ambiguous_stopped_or_wrong_owner_is_not_retried(tmp_path, monkeypatch, rows):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.json_response(rows)

    async with daemon(tmp_path, monkeypatch, handler):
        with pytest.raises(RuntimeError):
            await docker_api.find_main_container("project", "lease", "actor")
    assert calls == 1, "Identity violations must fail without retrying or choosing another container."


@pytest.mark.asyncio
async def test_enclosing_deadline_cancels_request_without_retry(tmp_path, monkeypatch):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.06)
        return web.json_response([container()])

    trace = []
    async with daemon(tmp_path, monkeypatch, handler):
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                await docker_api.find_main_container("project", "lease", "actor", trace=trace)
    assert calls == 1 and trace[0]["status"] == "cancelled", "The outer budget must stop further attempts."


@pytest.mark.asyncio
async def test_permanent_transport_failure_exhausts_only_two_attempts(tmp_path, monkeypatch):
    async def handler(request):
        raise web.HTTPServiceUnavailable()

    trace = []
    async with daemon(tmp_path, monkeypatch, handler):
        with pytest.raises(docker_api.aiohttp.ClientResponseError):
            await docker_api.find_main_container("project", "lease", "actor", trace=trace)
    assert len(trace) == 2, "Persistent faults must stop within the attempt budget."


@pytest.mark.asyncio
async def test_direct_caller_without_actor_still_checks_lease_and_project(tmp_path, monkeypatch):
    labels = {k: v for k, v in LABELS.items() if k != "psrl.actor_id"}

    async def handler(request):
        filters = json.loads(request.query["filters"])
        assert set(filters["label"]) == {f"{k}={v}" for k, v in labels.items()}, "The lease must remain mandatory."
        return web.json_response([container(Labels=labels)])

    async with daemon(tmp_path, monkeypatch, handler):
        found = await docker_api.find_main_container("project", "lease", "")
    assert found == "a" * 64, "A standalone leased environment need not belong to a Ray actor."
