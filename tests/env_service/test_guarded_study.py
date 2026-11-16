"""Exercise the opt-in GPU study's actual wiring through a CPU RPC stand-in."""

import asyncio
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest

FOLDER = (
    Path(__file__).resolve().parents[2]
    / "docs/env-worker-deep-dive/env-service-verification/scripts/warm_router_study"
)


def load(name):
    spec = importlib.util.spec_from_file_location("_guarded_test_" + name, FOLDER / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARDED = load("guarded")
PROVIDER = load("prediction_provider")


def arguments(root):
    return GUARDED.parser().parse_args(
        [
            "--root",
            str(root),
            "--assets",
            str(root / "assets"),
            "--container-python",
            "/usr/bin/python3",
            "--node-ip",
            "127.0.0.1",
            "--gpus",
            "0",
            "1",
            "--image",
            "test-image",
            "--budget-s",
            "60",
            "--target-version",
            "target-sha",
            "--blocker-version",
            "blocker-sha",
            "--target-steps",
            "2",
            "--blocker-steps",
            "128",
        ]
    )


def test_default_entry_is_off_and_validation_does_not_launch_resources(tmp_path):
    args = arguments(tmp_path)
    assert args.mode == "off"
    GUARDED.validate(args)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field,value",
    [
        ("blocker_steps", 80),
        ("blocker_steps", 8193),
        ("target_steps", 0),
        ("budget_s", float("nan")),
        ("budget_s", 2701),
        ("gpus", [0, 0]),
        ("target_version", "  "),
    ],
)
def test_invalid_workload_or_budget_is_rejected(tmp_path, field, value):
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        GUARDED.validate(args)


class CpuHarness:
    """Implement the existing service wire protocol, without Docker or CUDA."""

    def __init__(self, root):
        self.root = root
        self.caches = [set(), set()]
        self.sizes = {GUARDED.A: 100, GUARDED.B: 40}
        self.calls = []
        self.closed = 0
        self.record = {
            "gpu_before": {0: {"name": "CPU-test-device"}, 1: {"name": "CPU-test-device"}},
            "service_ready": [{"torch": "test", "transformers": "test"}] * 2,
            "image_id": "immutable-test-image",
            "progress_storage": "node_local",
        }
        self.context = {
            "root": root,
            "envs": [object(), object()],
            "request": self.request,
            "record": self.record,
            "save": lambda: None,
            "progress_file": self.progress_file,
            "remaining": lambda: 60,
        }
        for relative in ("warm_router_study/server.py", "joint_gpu_study/worker.py"):
            path = root / "source-snapshot" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# CPU fixture source.\n")
        data = root / "assets/data/gsm8k/public/rows.json"
        data.parent.mkdir(parents=True)
        data.write_text("[]")

    def progress_file(self, index):
        return self.root / f"progress-{index}.json"

    async def request(self, index, payload):
        self.calls.append((index, payload))
        operation = payload.get("op", "run")
        if operation == "clear":
            self.caches[index].clear()
            return {"status": "passed"}
        cache_bytes = sum(self.sizes[name] for name in self.caches[index])
        if operation == "ping":
            return {
                "status": "passed",
                "cache_keys": list(self.caches[index]),
                "cache_bytes": cache_bytes,
                "budget_bytes": 500,
                "workspace_bytes": 20,
            }
        if operation == "admit":
            return {
                "status": "passed",
                "required_bytes": self.sizes[payload["model"]] + 20,
                "available_bytes": 500,
                "admits": True,
            }
        model, steps = payload["model"], payload["steps"]
        started = time.monotonic()
        self.caches[index].add(model)
        chunks, durations = [], []
        for completed in range(16, steps + 1, 16):
            await asyncio.sleep(0.025)
            chunks.append(16)
            durations.append(0.025)
            self.progress_file(index).write_text(
                json.dumps(
                    {
                        "id": payload["id"],
                        "phase": "compute",
                        "total_steps": steps,
                        "completed_steps": completed,
                        "observed_at": time.monotonic(),
                        "chunk_steps": chunks,
                        "wall_chunk_s": durations,
                    }
                )
            )
        await asyncio.sleep(0.002)
        return {
            "status": "passed",
            "id": payload["id"],
            "model": model,
            "steps": steps,
            "optimizer_updates": steps,
            "model_bytes": self.sizes[model],
            "cache_keys": list(self.caches[index]),
            "cache_bytes": sum(self.sizes[name] for name in self.caches[index]),
            "load_s": 0.0001,
            "finished_monotonic": time.monotonic(),
            "started_monotonic": started,
        }

    async def shutdown(self, context):
        assert context is self.context
        self.closed += 1
        self.caches = [set(), set()]


@pytest.mark.parametrize("mode", ["off", "observe", "auto"])
def test_whole_runner_uses_real_provider_dispatcher_profiles_and_cleanup(tmp_path, monkeypatch, mode):
    async def scenario():
        harness = CpuHarness(tmp_path)
        args = arguments(tmp_path)
        args.mode = mode
        monkeypatch.setitem(sys.modules, "prediction_provider", PROVIDER)
        monkeypatch.setattr(GUARDED, "shutdown_services", harness.shutdown)
        await GUARDED.study(harness.context, args)
        result = harness.record["guarded_result"]
        assert result["target"]["id"] == "formal-target"
        assert result["blocker"]["id"] == "formal-blocker"
        assert len(harness.record["guarded_profiles"]["0"]) == 3
        assert len(harness.record["guarded_profiles"]["1"]) == 3
        assert len(harness.record["guarded_calibration"]) == 3
        assert harness.record["guarded_registered_optimizer_updates"] == 8 * 2 + 4 * 128
        runs = [payload for _, payload in harness.calls if payload.get("op", "run") == "run"]
        assert len(runs) == 12
        assert sum(payload["steps"] for payload in runs) == 528
        assert harness.closed == 1 and not any(harness.caches)
        assert (tmp_path / "guarded-policy-source/command_prediction.py").is_file()
        if mode != "auto":
            assert not result["decision"]["prediction_used"]

    asyncio.run(scenario())


def test_physical_shutdown_is_bounded_when_container_destruction_defers_cancellation(monkeypatch):
    async def scenario():
        release = asyncio.Event()
        started = asyncio.Event()
        cleanup_tasks = []
        runtime_guard = load("runtime_guard")

        class Environment:
            async def stop(self, *, delete):
                assert delete
                cleanup_tasks.append(asyncio.current_task())
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    # Actual environment destruction can keep ownership after cancellation.
                    await release.wait()

        async def bounded_cleanup(awaitable, deadline, *, timeout_s):
            assert timeout_s == 35
            return await runtime_guard.await_cleanup(awaitable, min(deadline, time.monotonic() + 0.01))

        def unexpected_docker(*args):
            raise AssertionError("Failed destruction cannot proceed to physical release confirmation.")

        monkeypatch.setitem(sys.modules, "experiment", types.SimpleNamespace(docker=unexpected_docker))
        monkeypatch.setitem(sys.modules, "runtime_guard", types.SimpleNamespace(await_cleanup=bounded_cleanup))
        record = {"tag": "cpu-owned", "gpu_before": {}}
        task = asyncio.create_task(GUARDED.shutdown_services({"envs": [Environment()], "record": record}))
        try:
            await started.wait()
            done, _ = await asyncio.wait((task,), timeout=0.2)
            assert task in done, "Cleanup must not await noncooperative destruction past its budget."
            with pytest.raises(TimeoutError):
                task.result()
            assert "guarded_provider_cleanup" not in record
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, *cleanup_tasks, return_exceptions=True)

    asyncio.run(scenario())
