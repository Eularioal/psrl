"""Verify that failed Docker cleanup remains pending and is retried."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from psrl.workers.env_service import node


def test_cleanup_retains_lease_until_docker_confirms_removal(monkeypatch):
    lease = SimpleNamespace(lease_id="expired-lease", gpu_lending=False, gpu_reservation="episode")
    router = SimpleNamespace(
        pending_reaped=SimpleNamespace(remote=AsyncMock(return_value=[lease])),
        acknowledge_reaped=SimpleNamespace(remote=AsyncMock()),
        reclaim_gpu_phases=SimpleNamespace(remote=AsyncMock()),
    )
    remove = Mock()
    monkeypatch.setattr(node, "force_remove_containers_by_label", remove)
    processes = [
        SimpleNamespace(returncode=1, communicate=AsyncMock(return_value=(b"", b"Daemon unavailable"))),
        SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b"container-id\n", b""))),
        SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b"", b""))),
    ]
    monkeypatch.setattr(node.asyncio, "create_subprocess_exec", AsyncMock(side_effect=processes))

    async def run():
        service = node.EnvNodeService(router, "127.0.0.1")
        assert await service.cleanup() == 1, "A Docker failure released capacity."
        assert await service.cleanup() == 1, "A live container released capacity."
        router.acknowledge_reaped.remote.assert_not_awaited()
        assert await service.cleanup() == 0, "Successful cleanup stayed pending."
        router.acknowledge_reaped.remote.assert_awaited_once_with("127.0.0.1", "expired-lease")

    asyncio.run(run())
    assert remove.call_count == 3, "Failed cleanup was not retried."
    remove.assert_called_with("psrl.env_lease", "expired-lease")


def test_lost_acknowledgement_is_redelivered_to_a_new_node(monkeypatch):
    lease = SimpleNamespace(lease_id="expired", gpu_lending=False, gpu_reservation="episode")
    router = SimpleNamespace(
        pending_reaped=SimpleNamespace(remote=AsyncMock(return_value=[lease])),
        acknowledge_reaped=SimpleNamespace(remote=AsyncMock(side_effect=[TimeoutError("Lost reply"), None])),
        reclaim_gpu_phases=SimpleNamespace(remote=AsyncMock()),
    )
    monkeypatch.setattr(node, "force_remove_containers_by_label", Mock())
    process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b"", b"")))
    monkeypatch.setattr(node.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    async def run():
        first = node.EnvNodeService(router, "node")
        assert await first.cleanup() == 1
        replacement = node.EnvNodeService(router, "node")
        assert await replacement.cleanup() == 0
        assert router.acknowledge_reaped.remote.await_count == 2

    asyncio.run(run())
