"""Check the fixed-step RPC adapter against a CPU service protocol stand-in."""

import asyncio
import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/env-worker-deep-dive/env-service-verification/scripts/warm_router_study/prediction_provider.py"
)
SPEC = importlib.util.spec_from_file_location("test_fixed_step_provider", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FixedStepServiceProvider = MODULE.FixedStepServiceProvider


class Harness:
    """Provide protocol responses and local progress files without using GPUs."""

    def __init__(self, root):
        self.root = root
        self.sizes = {"alpha": 100, "beta": 200}
        self.workspace = 20
        self.budget = 500
        self.caches = [set(), set()]
        self.calls = []
        self.shutdown_calls = 0
        self.frames = []
        self.run_error = None
        self.run_result_changes = {}
        self.run_started = asyncio.Event()
        self.run_release = None
        self.stop_release = None
        self.shutdown_error = None
        self.context = {
            "envs": [object(), object()],
            "record": {"progress_storage": "node_local"},
            "progress_clock": "host_monotonic",
            "request": self.request,
            "progress_file": self.progress_file,
        }

    def progress_file(self, index):
        return self.root / f"progress-{index}.json"

    async def request(self, index, payload):
        self.calls.append((index, payload))
        operation = payload.get("op", "run")
        if operation == "ping":
            return dict(
                status="passed",
                cache_keys=list(self.caches[index]),
                cache_bytes=sum(self.sizes[name] for name in self.caches[index]),
                budget_bytes=self.budget,
                workspace_bytes=self.workspace,
            )
        if operation == "admit":
            required = self.sizes[payload["model"]] + self.workspace
            return dict(
                status="passed", required_bytes=required, available_bytes=self.budget, admits=required <= self.budget
            )
        self.run_started.set()
        for changes in self.frames:
            frame = dict(
                id=payload["id"],
                phase="compute",
                total_steps=payload["steps"],
                completed_steps=64,
                observed_at=time.monotonic(),
                chunk_steps=[16] * 4,
                wall_chunk_s=[0.001] * 4,
            )
            frame.update(changes)
            self.progress_file(index).write_text(json.dumps(frame))
            await asyncio.sleep(0.005)
        if self.run_release is not None:
            await self.run_release.wait()
        if self.run_error is not None:
            raise self.run_error
        self.caches[index].add(payload["model"])
        result = dict(
            status="passed",
            id=payload["id"],
            model=payload["model"],
            steps=payload["steps"],
            optimizer_updates=payload["steps"],
            model_bytes=self.sizes[payload["model"]],
            cache_keys=list(self.caches[index]),
            cache_bytes=sum(self.sizes[name] for name in self.caches[index]),
        )
        result.update(self.run_result_changes)
        return result

    async def shutdown(self):
        self.shutdown_calls += 1
        if self.stop_release is not None:
            await self.stop_release.wait()
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def provider(self, **changes):
        arguments = dict(
            device_class="gpu-class",
            model_versions={"alpha": "weights-v1", "beta": "weights-v2"},
            code_version="service-v1",
            runtime_version="runtime-v1",
            input_signature="fixed-two-rows",
            shutdown=self.shutdown,
            poll_interval_s=0.001,
        )
        arguments.update(changes)
        return FixedStepServiceProvider(self.context, **arguments)


async def ignore_progress(value):
    return None


async def test_initialization_uses_empty_ping_and_cold_admission_to_obtain_exact_sizes(tmp_path):
    harness = Harness(tmp_path)
    provider = harness.provider()
    await provider.initialize()
    assert len(harness.calls) == 6, "Two devices must each receive one ping and two model probes."
    assert provider.model_bytes == {"alpha": 100, "beta": 200}, "Sizes must exclude the declared workspace."
    work = provider.request("command", "alpha", 128)
    assert work.spec.model_id == "alpha:weights-v1", "Cache identity must include the immutable weight version."
    assert work.spec.workload == provider.workload("alpha"), "Timing identity must describe the actual service."
    assert work.model_bytes == 100 and work.workspace_bytes == 20, "The declaration must charge model and workspace."
    assert work.spec.relocatable, "The fixed service resets mutable command state before execution."
    states = await provider.device_snapshot()
    assert [state.available_gpu_bytes for state in states] == [500, 500], "Empty cache budgets must be exact."
    await provider.aclose()


async def test_execution_uses_declared_fields_and_maps_server_cache_to_immutable_identity(tmp_path):
    harness = Harness(tmp_path)
    provider = harness.provider()
    await provider.initialize()
    work = provider.request("command", "alpha", 128)
    result = await provider.execute("0", work, None, ignore_progress)
    assert result["optimizer_updates"] == 128, "The adapter must retain the real service result."
    assert harness.calls[-1] == (0, dict(id="command", model="alpha", steps=128, progress_every=16)), (
        "External payload fields must not alter execution identity."
    )
    states = await provider.device_snapshot()
    assert states[0].resident_models == frozenset({"alpha:weights-v1"}), (
        "The cache must use versioned model identities."
    )
    assert states[0].available_gpu_bytes == 400, "The retained model must remain charged after completion."
    calls = len(harness.calls)
    await provider.device_snapshot()
    assert len(harness.calls) == calls, "A snapshot must not send RPCs to a busy serial service."
    await provider.aclose()
    await provider.aclose()
    assert harness.shutdown_calls == 1, "Provider shutdown must be idempotent."


async def test_busy_service_snapshot_is_cached_and_does_not_claim_inflight_model_is_resident(tmp_path):
    harness = Harness(tmp_path)
    harness.run_release = asyncio.Event()
    provider = harness.provider()
    await provider.initialize()
    running = asyncio.create_task(
        provider.execute("0", provider.request("command", "alpha", 128), None, ignore_progress)
    )
    await asyncio.wait_for(harness.run_started.wait(), 1)
    calls = len(harness.calls)
    state = (await provider.device_snapshot())[0]
    assert state.resident_models == frozenset() and state.available_gpu_bytes == 500, (
        "The dispatcher must reserve in-flight footprints until successful service confirmation."
    )
    assert len(harness.calls) == calls, "A busy service must not receive a blocking ping."
    with pytest.raises(ValueError, match="available owned serial"):
        await provider.execute("0", provider.request("overlap", "alpha", 128), None, ignore_progress)
    harness.run_release.set()
    await running
    await provider.aclose()


async def test_progress_filters_foreign_identity_and_duplicate_counts_and_accepts_empty_terminal(tmp_path):
    harness = Harness(tmp_path)
    harness.frames = [
        {"id": "foreign"},
        {"completed_steps": 64},
        {"completed_steps": 64},
        {"completed_steps": 80, "chunk_steps": [16] * 5, "wall_chunk_s": [0.001] * 5},
        {"phase": "complete", "completed_steps": 128, "chunk_steps": [], "wall_chunk_s": []},
    ]
    provider = harness.provider()
    await provider.initialize()
    reports = []

    async def observe(value):
        reports.append(value)

    await provider.execute("0", provider.request("command", "alpha", 128), None, observe)
    assert [report.completed_steps for report in reports] == [64, 80, 128], (
        "Only distinct matching progress may advance."
    )
    assert all(report.command_id == "command" for report in reports), "Foreign reports must not enter calibration."
    assert all(report.report_age_s >= 0 for report in reports), "Report age must use the confirmed host clock."
    assert reports[-1].chunk_steps == (), "The real service's empty terminal summary must remain supported."
    await provider.aclose()


@pytest.mark.parametrize(
    "changes",
    [
        {"total_steps": 1},
        {"completed_steps": -1},
        {"completed_steps": True},
        {"observed_at": float("nan")},
        {"observed_at": time.monotonic() + 1000},
        {"phase": "rebuild"},
        {"chunk_steps": [16] * 33},
        {"wall_chunk_s": "opaque"},
    ],
)
async def test_invalid_progress_is_ignored_without_inventing_an_estimate(tmp_path, changes):
    harness = Harness(tmp_path)
    harness.frames = [changes]
    provider = harness.provider()
    await provider.initialize()
    reports = []

    async def observe(value):
        reports.append(value)

    await provider.execute("0", provider.request("command", "alpha", 128), None, observe)
    assert not reports, "Invalid progress must not be converted to a usable observation."
    await provider.aclose()


@pytest.mark.parametrize("defect", ["clock", "storage", "cache", "budget", "workspace", "devices"])
async def test_initialization_rejects_incompatible_accounting_and_confirms_cleanup(tmp_path, defect):
    harness = Harness(tmp_path)
    if defect == "clock":
        harness.context.pop("progress_clock")
    elif defect == "storage":
        harness.context["record"]["progress_storage"] = "remote"
    elif defect == "cache":
        harness.caches[0].add("alpha")
    elif defect == "budget":
        harness.budget = True
    elif defect == "workspace":
        harness.workspace = 1000
    else:
        harness.context["envs"] = []
    provider = harness.provider()
    with pytest.raises(ValueError):
        await provider.initialize()
    assert harness.shutdown_calls == 1, "Failed initialization must retain physical cleanup ownership."
    with pytest.raises(RuntimeError, match="stopped"):
        await provider.device_snapshot()


@pytest.mark.parametrize("defect", ["version", "size", "workspace", "steps", "payload", "device"])
async def test_unmatched_execution_declarations_never_reach_rpc(tmp_path, defect):
    harness = Harness(tmp_path)
    provider = harness.provider()
    await provider.initialize()
    work, payload, device = provider.request("command", "alpha", 128), None, "0"
    if defect == "version":
        work = replace(work, spec=replace(work.spec, workload=replace(work.spec.workload, model_version="changed")))
    elif defect == "size":
        work = replace(work, model_bytes=101)
    elif defect == "workspace":
        work = replace(work, workspace_bytes=21)
    elif defect == "steps":
        work = replace(work, spec=replace(work.spec, total_steps=None))
    elif defect == "payload":
        payload = {"steps": 8192}
    else:
        device = "unknown"
    before = len(harness.calls)
    with pytest.raises(ValueError):
        await provider.execute(device, work, payload, ignore_progress)
    assert len(harness.calls) == before, "Mismatched declarations must be rejected before any execution."
    await provider.aclose()


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "failed"},
        {"id": "foreign"},
        {"model": "beta"},
        {"steps": 127},
        {"optimizer_updates": 127},
        {"model_bytes": 99},
        {"cache_keys": ["unknown"]},
        {"cache_keys": ["alpha", "alpha"]},
        {"cache_bytes": 99},
        {"cache_keys": []},
    ],
)
async def test_invalid_completion_poisoning_stops_physical_service_before_propagating(tmp_path, changes):
    harness = Harness(tmp_path)
    harness.run_result_changes = changes
    provider = harness.provider()
    await provider.initialize()
    with pytest.raises(RuntimeError):
        await provider.execute("0", provider.request("command", "alpha", 128), None, ignore_progress)
    assert harness.shutdown_calls == 1, (
        "An unconfirmed completion must stop owned physical work before returning failure."
    )
    with pytest.raises(RuntimeError, match="stopped"):
        await provider.device_snapshot()
    await provider.aclose()


async def test_rpc_exception_waits_for_physical_shutdown_before_reaching_the_caller(tmp_path):
    harness = Harness(tmp_path)
    harness.run_error = OSError("RPC transport failed.")
    harness.stop_release = asyncio.Event()
    provider = harness.provider()
    await provider.initialize()
    running = asyncio.create_task(
        provider.execute("0", provider.request("command", "alpha", 128), None, ignore_progress)
    )
    for _ in range(100):
        if harness.shutdown_calls:
            break
        await asyncio.sleep(0)
    assert harness.shutdown_calls == 1 and not running.done(), "Failure must wait for confirmed physical shutdown."
    harness.stop_release.set()
    with pytest.raises(OSError, match="transport failed"):
        await running


async def test_cancelling_execution_still_confirms_physical_shutdown(tmp_path):
    harness = Harness(tmp_path)
    harness.run_release = asyncio.Event()
    provider = harness.provider()
    await provider.initialize()
    running = asyncio.create_task(
        provider.execute("0", provider.request("command", "alpha", 128), None, ignore_progress)
    )
    await asyncio.wait_for(harness.run_started.wait(), 1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert harness.shutdown_calls == 1, "Execution cancellation must retain mandatory shutdown responsibility."


async def test_cleanup_failure_remains_visible_to_the_caller(tmp_path):
    harness = Harness(tmp_path)
    harness.run_error = OSError("RPC failed.")
    harness.shutdown_error = RuntimeError("Physical release was not confirmed.")
    provider = harness.provider()
    await provider.initialize()
    with pytest.raises(RuntimeError, match="Physical release was not confirmed"):
        await provider.execute("0", provider.request("command", "alpha", 128), None, ignore_progress)


@pytest.mark.parametrize("interval", [0, -1, True, float("nan"), float("inf")])
def test_invalid_polling_budget_is_rejected(tmp_path, interval):
    with pytest.raises(ValueError, match="finite and positive"):
        Harness(tmp_path).provider(poll_interval_s=interval)


def test_shutdown_confirmation_is_mandatory(tmp_path):
    with pytest.raises(ValueError, match="physical shutdown callback"):
        Harness(tmp_path).provider(shutdown=None)
