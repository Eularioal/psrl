"""Reject stale registrations, foreign cgroups and unhealthy identity collectors."""

import asyncio
import ctypes as c
import threading
from types import SimpleNamespace

import pytest
from psrl.workers.env_service.bpf_identity import BpfOwner, Record
from psrl.workers.env_service.process_identity import build_identity_backend, validate_identity_backend
from psrl.workers.env_service.task_telemetry import TaskGpuTelemetry, read_command_memory


class Collector:
    """Model kernel records while exercising the production attribution consumer."""

    name = "ebpf"

    def __init__(self):
        self.lock = threading.RLock()
        self.records = {}
        self.registrations = {}
        self.bound = {}
        self.error = None
        self.lib = SimpleNamespace(bpf_map_delete_elem=self.delete)
        self.maps = {"registrations": 1}

    def delete(self, _, key):
        del self.registrations[c.cast(key, c.POINTER(c.c_uint64)).contents.value]
        return 0

    def check_support(self):
        if self.error:
            raise RuntimeError(self.error)

    def bind(self, fence):
        owner = BpfOwner(self, fence.path)
        self.registrations[owner.inode] = owner.cookie
        self.bound[owner.cookie] = owner
        return owner

    def lookup(self, name, key, _):
        if name == "registrations":
            value = self.registrations.get(key.value)
            return None if value is None else c.c_uint64(value)
        return self.records.get(key.value)


@pytest.fixture
def task(tmp_path):
    collector = Collector()
    item = TaskGpuTelemetry(None, SimpleNamespace(path=tmp_path), (0,), identity_backend=collector)
    try:
        yield item, collector
    finally:
        item.close_identity()


def test_foreign_cgroup_and_old_command_cookie_never_earn_memory_credit(task):
    item, collector = task
    owner = item.owner
    collector.records = {
        10: Record(1, owner.inode, owner.cookie),
        11: Record(1, owner.inode + 1, owner.cookie),
        12: Record(1, owner.inode, owner.cookie + 1),
    }
    assert set(item._owners([10, 11, 12])) == {10}


def test_kernel_birth_identity_changes_even_when_wall_time_is_close(task):
    item, collector = task
    owner = item.owner
    collector.records[10] = Record(1000000001, owner.inode, owner.cookie)
    first = item._owners([10])[10][0]
    collector.records[10] = Record(1000000002, owner.inode, owner.cookie)
    assert item._owners([10])[10][0] != first
    del collector.records[10]
    assert item._owners([10]) == {}


def test_replaced_registration_is_rejected_and_old_close_preserves_new_owner(task):
    item, collector = task
    owner = item.owner
    collector.registrations[owner.inode] = owner.cookie + 1
    with pytest.raises(RuntimeError, match="replaced"):
        item._owners([10])
    item.close_identity()
    assert collector.registrations[owner.inode] == owner.cookie + 1


def test_closed_owner_does_not_reuse_cached_samples(task):
    item, collector = task
    owner = item.owner
    collector.records[10] = Record(1, owner.inode, owner.cookie)
    item.close_identity()
    with pytest.raises(RuntimeError, match="no longer valid"):
        item._owners([10])


def test_collector_failure_retains_full_reservation(task):
    item, collector = task
    collector.error = "Collector lost records."
    probe = SimpleNamespace(
        read_gpu_snapshot=lambda _: {0: {"uuid": "gpu", "used_mb": 80, "pids": [10], "memory_mb_by_pid": {10: 20}}}
    )
    result = read_command_memory(probe, {"job": item}, (0,))
    assert result["owned_mb"] == {} and result["devices"][0]["used_mb"] == 80
    assert "Collector lost records" in result["attribution_errors"]["job"]


def test_collector_failure_during_second_identity_read_withholds_credit(task, monkeypatch):
    item, collector = task
    owner = item.owner
    collector.records[10] = Record(1, owner.inode, owner.cookie)
    original = item._owners
    calls = 0

    def owners(pids):
        nonlocal calls
        calls += 1
        if calls == 2:
            collector.error = "Collector detached."
        return original(pids)

    monkeypatch.setattr(item, "_owners", owners)
    probe = SimpleNamespace(
        read_gpu_snapshot=lambda _: {0: {"uuid": "gpu", "used_mb": 80, "pids": [10], "memory_mb_by_pid": {10: 20}}}
    )
    result = read_command_memory(probe, {"job": item}, (0,))
    assert result["owned_mb"] == {} and "job" in result["attribution_errors"]


def test_backend_selection_is_explicit_without_automatic_fallback():
    assert build_identity_backend("procfs").name == "procfs"
    assert validate_identity_backend("ebpf") == "ebpf"
    with pytest.raises(ValueError, match="procfs or ebpf"):
        build_identity_backend("auto")


@pytest.mark.parametrize("capability", ["BPF", "CAP_BPF", "PERFMON", "CAP_SYS_ADMIN", "ALL"])
def test_task_cannot_receive_privileges_to_modify_kernel_attribution(capability):
    from psrl.workers.env_service.command_gpu_node import DeviceCgroupFence

    with pytest.raises(RuntimeError, match="kernel tracing"):
        DeviceCgroupFence({"HostConfig": {"CapAdd": [capability]}}, [])


@pytest.mark.asyncio
async def test_cancelled_initialization_closes_collector_created_in_executor(monkeypatch):
    from psrl.workers.env_service import command_gpu_node

    started, finish, closed = threading.Event(), threading.Event(), threading.Event()

    def build(_):
        started.set()
        if not finish.wait(timeout=5):
            raise TimeoutError("The test did not release initialization.")
        return SimpleNamespace(close=closed.set)

    monkeypatch.setattr(command_gpu_node, "build_identity_backend", build)
    node = command_gpu_node.GpuCommandNodeMixin()
    pending = asyncio.create_task(node._gpu_identity())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        pending.cancel()
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert closed.is_set() and not hasattr(node, "_identity_source")


@pytest.mark.asyncio
async def test_closed_node_cannot_restart_identity_collection():
    from psrl.workers.env_service.node import EnvNodeService

    node = EnvNodeService(None, "127.0.0.1")
    await node.close()
    with pytest.raises(RuntimeError, match="has closed"):
        await node._gpu_identity()
