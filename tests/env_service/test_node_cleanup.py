"""Verify that failed Docker cleanup remains pending and is retried."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from psrl.workers.env_service import node


def test_cleanup_retains_lease_until_docker_confirms_removal(monkeypatch):
    lease = SimpleNamespace(lease_id="expired-lease", gpu_lending=False)
    router = SimpleNamespace(
        pop_reaped=SimpleNamespace(remote=AsyncMock(side_effect=[[lease], [], []])),
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
        assert await service.cleanup() == 0, "Successful cleanup stayed pending."

    asyncio.run(run())
    assert remove.call_count == 3, "Failed cleanup was not retried."
    remove.assert_called_with("psrl.env_lease", "expired-lease")
