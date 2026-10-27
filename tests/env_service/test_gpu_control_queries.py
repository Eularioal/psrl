"""Check live GPU telemetry and local daemon queries without a physical GPU."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from psrl.workers.env_service.docker_api import inspect_container
from psrl.workers.env_service.probe import NvmlProbe, read_gpu_devices, read_gpu_snapshot


def fake_nvml(monkeypatch):
    state = {"used": 4, "pids": [17], "initializations": 0, "shutdowns": 0, "queried": []}

    def initialize():
        state["initializations"] += 1

    def handle(index):
        state["queried"].append(index)
        if index not in (0, 1):
            raise RuntimeError("GPU is unavailable.")
        return index

    def shutdown():
        state["shutdowns"] += 1

    def memory(handle, *, version):
        assert version == 2, "Use application memory without driver reserved memory."
        return SimpleNamespace(used=state["used"] * 1024**2, total=80 * 1024**2)

    module = SimpleNamespace(
        nvmlInit=initialize,
        nvmlShutdown=shutdown,
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=handle,
        nvmlDeviceGetMinorNumber=lambda handle: (7, 3)[handle],
        nvmlDeviceGetUUID=lambda handle: f"GPU-{handle}",
        nvmlDeviceGetMemoryInfo=memory,
        nvmlMemory_v2=2,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [
            SimpleNamespace(pid=pid, usedGpuMemory=3 * 1024**2) for pid in state["pids"]
        ],
    )
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return state


def test_inventory_preserves_physical_minor_mapping(monkeypatch):
    state = fake_nvml(monkeypatch)
    devices = read_gpu_devices()
    assert devices[0].path == "/dev/nvidia7", "NVML index must not be used as the device minor."
    assert devices[1].uuid == "GPU-1" and devices[1].minor == 3, "Physical identity must be preserved."
    assert state["shutdowns"] == 1, "The driver session must close."


def test_snapshot_reads_live_selected_devices_and_fails_closed(monkeypatch):
    state = fake_nvml(monkeypatch)
    assert read_gpu_snapshot((1,))[1] == {
        "uuid": "GPU-1",
        "used_mb": 4,
        "total_mb": 80,
        "pids": [17],
        "memory_mb_by_pid": {17: 3},
    }, "Telemetry must retain the admission and return fields."
    state.update(used=0, pids=[])
    assert read_gpu_snapshot((1,))[1]["pids"] == [], "A returned process must disappear without cached telemetry."
    with pytest.raises(RuntimeError, match="unavailable"):
        read_gpu_snapshot((1, 2))
    assert state["queried"] == [1, 1, 1, 2], "Queries must be limited to requested devices."
    assert state["shutdowns"] == 3, "Partial query failures must still close the driver session."
    with pytest.raises(ValueError, match="indices"):
        read_gpu_snapshot(())


@pytest.mark.asyncio
async def test_inspect_uses_local_socket_and_propagates_missing_container(monkeypatch, tmp_path):
    socket = tmp_path / "docker.sock"
    monkeypatch.setenv("DOCKER_HOST", f"unix://{socket}")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    container = {"Id": "owned", "State": {"Running": True}, "Config": {"Labels": {"psrl.env_lease": "lease"}}}

    async def inspect(request):
        if request.match_info["id"] != "owned":
            raise web.HTTPNotFound()
        return web.json_response(container)

    app = web.Application()
    app.router.add_get("/containers/{id}/json", inspect)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.UnixSite(runner, str(socket)).start()
        assert await inspect_container("owned") == container, "Identity and state must reach the existing fence."
        with pytest.raises(aiohttp.ClientResponseError) as error:
            await inspect_container("missing")
        assert error.value.status == 404, "Missing containers must not be accepted as empty state."
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_device_fence_rejects_wrong_owner_before_device_query(monkeypatch):
    from psrl.workers.env_service import command_gpu_node

    async def inspect(container_id):
        return {"Config": {"Labels": {"psrl.env_lease": "someone-else"}}}

    def inventory(self):
        raise AssertionError("An unowned container must be rejected before reading devices.")

    monkeypatch.setattr(command_gpu_node, "inspect_container", inspect)
    monkeypatch.setattr(NvmlProbe, "read_gpu_devices", inventory)
    with pytest.raises(RuntimeError, match="does not belong"):
        await command_gpu_node.GpuCommandNodeMixin()._command_container("ours", "id", (0,))


@pytest.mark.asyncio
async def test_remote_daemon_cannot_control_local_devices(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:2375")
    with pytest.raises(RuntimeError, match="local Docker"):
        await inspect_container("owned")


def test_retained_connection_observes_changes_and_releases_once(monkeypatch):
    state = fake_nvml(monkeypatch)
    with NvmlProbe() as probe:
        assert probe.read_gpu_snapshot((1,))[1]["pids"] == [17], "The first query must observe live processes."
        state.update(used=0, pids=[])
        assert probe.read_gpu_snapshot((1,))[1]["used_mb"] == 0, "Memory readings must not be cached."
        assert probe.read_gpu_snapshot((1,))[1]["pids"] == [], "A departed process must not remain cached."
        assert probe.read_gpu_devices()[1].minor == 3, "Inventory must retain physical device identity."
        assert state["shutdowns"] == 0, "The owner must retain its connection between queries."
        with pytest.raises(RuntimeError, match="unavailable"):
            probe.read_gpu_snapshot((2,))
    probe.close()
    assert state["initializations"] == 1, "Queries must share one initialized connection."
    assert state["shutdowns"] == 1, "Explicit teardown must release the connection exactly once."


def test_close_waits_for_an_inflight_driver_query(monkeypatch):
    state = fake_nvml(monkeypatch)
    import pynvml

    entered, finish = threading.Event(), threading.Event()
    original = pynvml.nvmlDeviceGetMemoryInfo

    def memory(*args, **kwargs):
        entered.set()
        assert finish.wait(3), "The test must release the blocked driver query."
        assert state["shutdowns"] == 0, "The connection must stay alive during a query."
        return original(*args, **kwargs)

    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo", memory)
    probe = NvmlProbe()
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(probe.read_gpu_snapshot, (1,))
        assert entered.wait(3), "The query must enter before teardown starts."
        closing = pool.submit(probe.close)
        try:
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.05)
        finally:
            finish.set()
        assert reading.result(timeout=3)[1]["used_mb"] == 4, "The in-flight reading must complete."
        closing.result(timeout=3)
    assert state["shutdowns"] == 1, "Teardown must run after the query exits."
