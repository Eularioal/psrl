"""Keep deployment failures and cancellation ahead of GPU validation workloads."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_rejected", "preflight_rejected", "preflight_ready", "cancelled"])
async def test_preflight_closes_node_without_starting_harbor(monkeypatch, mode):
    script = (
        Path(__file__).resolve().parents[2]
        / "docs/env-worker-deep-dive/env-service-verification/scripts/gpu_job_probe.py"
    )
    spec = importlib.util.spec_from_file_location("gpu_job_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    service = SimpleNamespace(
        prepare=AsyncMock(),
        check_gpu_command_support=AsyncMock(),
        check_gpu_attribution_support=AsyncMock(),
        close=AsyncMock(),
    )
    monkeypatch.setattr(module, "EnvNodeService", lambda *_args: service)
    start = AsyncMock(side_effect=AssertionError("Harbor must not start before deployment checks pass."))
    monkeypatch.setattr(module.ManagedDockerEnvironment, "prepare_runtime", start)
    args = SimpleNamespace(gpu=0, allow_unavailable_attribution=False, preflight_only=mode != "full_rejected")
    evidence = {}
    if mode == "cancelled":
        service.prepare.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await module.probe(args, evidence)
    elif mode.endswith("rejected"):
        service.check_gpu_attribution_support.side_effect = RuntimeError("Host procfs is unavailable.")
        with pytest.raises(RuntimeError, match="Host procfs"):
            await module.probe(args, evidence)
    else:
        await module.probe(args, evidence)
        assert evidence["attribution_preflight"] == "passed", "Readiness must be explicit."
    service.close.assert_awaited_once()
    start.assert_not_awaited()
    assert evidence["node_service_closed"] and not evidence["workload_started"], "No workload may leak from preflight."
