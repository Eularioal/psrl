"""Teach evidence-gated placement with real modules and simulated CPU devices."""

import asyncio
import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path


def load_modules():
    """Load standard library modules without importing the training runtime."""
    folder = Path(__file__).resolve().parents[4] / "psrl/workers/env_service"
    package_name = "_command_predictability_teaching"
    package = types.ModuleType(package_name)
    package.__path__ = [str(folder)]
    sys.modules[package_name] = package
    loaded = {}
    for name in (
        "warm_placement",
        "step_predictor",
        "warm_dispatch",
        "command_prediction",
        "predictive_scheduler",
        "managed_command",
    ):
        spec = importlib.util.spec_from_file_location(package_name + "." + name, folder / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


MODULES = load_modules()
PREDICTION = MODULES["command_prediction"]
PLACEMENT = MODULES["predictive_scheduler"]
MANAGED = MODULES["managed_command"]
KEY = PREDICTION.WorkloadKey("code-v1", "weights-v1", "fixed-shape-train", "teaching-device", "runtime-v1")


class TeachingClock:
    """Advance logical seconds without sleeping or measuring CPU performance."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TeachingScheduler(PLACEMENT.PredictionAwareScheduler):
    """Expose the admission boundary so the example never races snapshot collection."""

    def __init__(self, config):
        super().__init__(config)
        self.target_admitted = asyncio.Event()

    def plan(self, request, devices, profiles, *, now_s):
        decision = super().plan(request, devices, profiles, now_s=now_s)
        if request.spec.command_id == "target":
            self.target_admitted.set()
        return decision


class TeachingProvider:
    """Simulate cache, compute and confirmed cleanup while using the real dispatcher."""

    def __init__(self, clock, scenario):
        self.clock, self.scenario = clock, scenario
        self.models = {"hot": {"model-v1": 40}, "cold": {}}
        self.active_ready = asyncio.Event()
        self.release_active = asyncio.Event()
        self.closed = False

    async def device_snapshot(self):
        # P4 reads retained cache and a reservation budget before workspace.
        return tuple(
            PLACEMENT.DeviceState(device, "teaching-device", frozenset(models), 200 - sum(models.values()))
            for device, models in self.models.items()
        )

    async def execute(self, device_id, request, payload, report_progress):
        spec = request.spec
        cold = spec.model_id not in self.models[device_id]
        if cold:
            self.clock.now += 1
            self.models[device_id][spec.model_id] = request.model_bytes
        if payload == "target":
            self.clock.now += 2
            return {"completed_at_s": self.clock(), "steps": spec.total_steps}
        self.clock.now += 4
        walls = (1.0, 1.0, 1.0, 3.0) if payload == "active" and self.scenario == "unstable" else (1.0,) * 4
        age = 2.0 if payload == "active" and self.scenario == "stale" else 0.0
        # P2 publishes progress. P3 checks it and freezes the first shadow forecast.
        await report_progress(PREDICTION.StepProgress(spec.command_id, spec.workload, 100, 40, (10,) * 4, walls, age))
        remaining_started = self.clock()
        if payload == "active":
            self.active_ready.set()
            await self.release_active.wait()
        self.clock.now = max(self.clock(), remaining_started + 6)
        # P6 confirms full completion and calibrates actual remaining time.
        return {"completed_at_s": self.clock(), "steps": 100}

    async def aclose(self):
        # Final cleanup owns these simulated devices. No shared services are touched.
        for models in self.models.values():
            models.clear()
        self.closed = True


def request(command_id, total_steps):
    spec = PREDICTION.CommandSpec(command_id, KEY, "model-v1", total_steps, relocatable=True)
    return PLACEMENT.PlacementRequest(spec, model_bytes=40, workspace_bytes=10, home_device_id="hot")


def profiles(req, clock):
    return {
        device: PLACEMENT.CommandCostProfile(
            replace(KEY, device_class="teaching-device"),
            "model-v1",
            req.spec.total_steps,
            40,
            10,
            (1.0,) * 3,
            ((2.0 if req.spec.total_steps == 2 else 10.0),) * 3,
            clock(),
        )
        for device in ("hot", "cold")
    }


async def run_case(mode, scenario="stable", *, calibrated=True):
    clock = TeachingClock()
    scheduler = TeachingScheduler(PLACEMENT.PlacementConfig(mode=mode))
    provider = TeachingProvider(clock, scenario)
    async with MANAGED.ManagedCommandDispatcher(scheduler, provider, clock=clock) as dispatcher:
        if calibrated:
            # Three separate real submit/progress/finish paths provide calibration.
            for index in range(3):
                req = request(f"calibration-{index}", 100)
                await dispatcher.submit(req, profiles(req, clock), "calibration")
        active_req = request("active", 100)
        active = asyncio.create_task(dispatcher.submit(active_req, profiles(active_req, clock), "active"))
        await provider.active_ready.wait()
        submitted_at = clock()
        target_req = request("target", 2)
        # P1 submits work. P4 chooses a device and P5 records its execution queue.
        target = asyncio.create_task(dispatcher.submit(target_req, profiles(target_req, clock), "target"))
        await scheduler.target_admitted.wait()
        provider.release_active.set()
        _, result = await asyncio.gather(active, target)
        outcome = {
            "mode": mode,
            "scenario": scenario,
            "calibrated": calibrated,
            "device": result.decision.device_id,
            "prediction_used": result.decision.prediction_used,
            "reason": result.decision.reason,
            "logical_target_latency_s": result.value["completed_at_s"] - submitted_at,
        }
    assert provider.closed and not any(provider.models.values())
    outcome["simulated_resources_released"] = True
    return outcome


async def main():
    results = [
        await run_case("off"),
        await run_case("observe"),
        await run_case("auto"),
        await run_case("auto", calibrated=False),
        await run_case("auto", "unstable"),
        await run_case("auto", "stale"),
    ]
    assert [row["device"] for row in results] == ["hot", "hot", "cold", "hot", "hot", "hot"]
    assert [row["prediction_used"] for row in results] == [False, False, True, False, False, False]
    assert [row["logical_target_latency_s"] for row in results] == [8, 8, 3, 8, 8, 8]
    print(json.dumps({"kind": "CPU teaching simulation, not GPU performance evidence", "cases": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
