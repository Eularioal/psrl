"""Teach warm placement and structured cleanup using CPU-only simulated devices."""

import asyncio
import importlib.util
import json
import sys
import tempfile
import time
import types
from builtins import ExceptionGroup
from dataclasses import asdict, dataclass
from pathlib import Path


def load_modules():
    """Load only the three standard-library modules, without a PSRL runtime."""
    folder = Path(__file__).resolve().parents[5] / "psrl/workers/env_service"
    package_name = "_warm_asyncio_teaching"
    package = types.ModuleType(package_name)
    package.__path__ = [str(folder)]
    sys.modules[package_name] = package
    loaded = {}
    for name in ("warm_placement", "step_predictor", "warm_dispatch"):
        spec = importlib.util.spec_from_file_location(package_name + "." + name, folder / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


MODULES = load_modules()
DISPATCH, PREDICTOR = MODULES["warm_dispatch"], MODULES["step_predictor"]
INTERVAL = MODULES["warm_placement"].DurationInterval


def estimate(point):
    return DISPATCH.TimeEstimate(point, INTERVAL(point * 0.8, point * 1.3))


def atomic_progress(path, value):
    """W4 publishes a complete JSON file using a sibling temporary file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


@dataclass
class Command:
    name: str
    model_id: str
    model_bytes: int
    steps: int
    result: asyncio.Future
    fail_after: int | None = None
    workspace_bytes: int = 10


class TeachingDevice:
    """W2 serializes one logical device and tracks simulated cache and workspace."""

    def __init__(self, name, folder, events, activity, resident=None):
        self.name, self.events, self.activity = name, events, activity
        self.progress_path = folder / (name + ".json")
        self.budget_bytes = 160
        self.resident = dict(resident or {})
        self.workspace_bytes = 0
        self.running = False
        self.queue = asyncio.Queue()
        self.closed = False

    def emit(self, event, **values):
        self.events.append(dict(device=self.name, event=event, **values))

    def snapshot(self, active=None):
        return DISPATCH.QueueSnapshot(
            self.name,
            frozenset(self.resident),
            self.budget_bytes - sum(self.resident.values()),
            busy=self.running,
            active_remaining=active,
        )

    async def request(self, name, model_id, model_bytes, steps, fail_after=None):
        # W1 enqueues work immediately, then cooperatively waits for its result.
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(Command(name, model_id, model_bytes, steps, future, fail_after))
        return await future

    async def execute(self, command):
        cost = DISPATCH.ModelCost(command.model_id, command.model_bytes, estimate(0.01), estimate(0.05), 10)
        if not DISPATCH.fits_capacity(cost, self.snapshot()):
            raise RuntimeError("The simulated model and workspace do not fit.")
        self.workspace_bytes = command.workspace_bytes
        self.running = True
        self.activity["active"] += 1
        self.activity["maximum"] = max(self.activity["maximum"], self.activity["active"])
        self.emit("start", command=command.name)
        try:
            if command.model_id not in self.resident:
                self.emit("rebuild", command=command.name)
                await asyncio.sleep(0.01)
                self.resident[command.model_id] = command.model_bytes
            chunk_started = time.monotonic()
            chunk_steps, wall_chunks = [], []
            for index in range(command.steps):
                # W3 sleep represents asynchronous external work, not GPU computation.
                await asyncio.sleep(0.002)
                completed = index + 1
                if command.fail_after == completed:
                    raise RuntimeError("Deliberate teaching failure.")
                if completed % 8 == 0 or completed == command.steps:
                    observed_at = time.monotonic()
                    chunk_steps.append(8 if completed % 8 == 0 else completed % 8)
                    wall_chunks.append(observed_at - chunk_started)
                    chunk_started = observed_at
                    atomic_progress(
                        self.progress_path,
                        dict(
                            command=command.name,
                            completed_steps=completed,
                            total_steps=command.steps,
                            chunk_steps=chunk_steps,
                            wall_chunk_s=wall_chunks,
                            observed_at=observed_at,
                        ),
                    )
            self.emit("complete", command=command.name)
            return dict(command=command.name, completed_steps=command.steps)
        finally:
            # W7 command cleanup releases workspace while the model remains cached.
            self.workspace_bytes = 0
            self.running = False
            self.activity["active"] -= 1
            self.emit("workspace_released", command=command.name, cached_models=sorted(self.resident))

    async def serve(self):
        try:
            while True:
                command = await self.queue.get()
                try:
                    if command is None:
                        return
                    try:
                        value = await self.execute(command)
                        if not command.result.done():
                            command.result.set_result(value)
                    except asyncio.CancelledError:
                        command.result.cancel()
                        self.emit("cancelled", command=command.name)
                        raise
                    except Exception as exc:
                        if not command.result.done():
                            command.result.set_exception(exc)
                        raise
                finally:
                    self.queue.task_done()
        finally:
            # W8 environment shutdown also releases retained models and queued futures.
            self.resident.clear()
            while not self.queue.empty():
                pending = self.queue.get_nowait()
                if pending is not None:
                    pending.result.cancel()
                self.queue.task_done()
            self.closed = True
            self.emit("device_closed")


async def placement_case(folder):
    events, activity = [], dict(active=0, maximum=0)
    devices = [
        TeachingDevice("hot", folder, events, activity, {"target@v1": 100, "blocker@v1": 40}),
        TeachingDevice("cold", folder, events, activity),
    ]
    async with asyncio.TaskGroup() as group:
        for device in devices:
            group.create_task(device.serve())
        blocker = group.create_task(devices[0].request("blocker", "blocker@v1", 40, 80))
        while True:
            try:
                progress = json.loads(devices[0].progress_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                progress = {}
            if progress.get("completed_steps", 0) >= 24:
                break
            await asyncio.sleep(0.001)
        # W5 uses only already-published progress, including its report age.
        forecast = PREDICTOR.predict_remaining(
            progress["chunk_steps"][1:],
            progress["wall_chunk_s"][1:],
            progress["completed_steps"],
            80,
            time.monotonic() - progress["observed_at"],
            report_every_steps=8,
        )
        active = DISPATCH.estimate_from_forecast(forecast)
        snapshots = [devices[0].snapshot(active), devices[1].snapshot()]
        cost = DISPATCH.ModelCost("target@v1", 100, estimate(0.01), estimate(0.05), workspace_bytes=10)
        decision = DISPATCH.WarmPlacementController(switch_margin_s=0.002).decide(cost, snapshots)
        selected = next(device for device in devices if device.name == decision.device_id)
        # W6 both requests become tasks before gather waits for both results.
        target = group.create_task(selected.request("target", "target@v1", 100, 20))
        outputs = await asyncio.gather(target, blocker)
        for device in devices:
            await device.queue.put(None)
    assert all(device.closed and not device.resident and device.workspace_bytes == 0 for device in devices)
    return dict(
        decision=asdict(decision),
        forecast=asdict(forecast),
        outputs=outputs,
        maximum_overlapping_commands=activity["maximum"],
        cleanup_confirmed=True,
        events=events,
    )


async def cancellation_case(folder):
    events, activity = [], dict(active=0, maximum=0)
    devices = [TeachingDevice(name, folder, events, activity) for name in ("left", "right")]
    saw_failure = False
    try:
        async with asyncio.TaskGroup() as group:
            for device in devices:
                group.create_task(device.serve())
            group.create_task(devices[0].request("long", "target@v1", 100, 80))
            group.create_task(devices[1].request("failing", "target@v1", 100, 80, fail_after=2))
    except ExceptionGroup as failures:
        if any(not isinstance(error, RuntimeError) for error in failures.exceptions):
            raise
        saw_failure = True
    assert saw_failure and activity["active"] == 0
    assert all(device.closed and not device.resident and device.workspace_bytes == 0 for device in devices)
    return dict(expected_failure_observed=saw_failure, cleanup_confirmed=True, events=events)


async def main():
    with tempfile.TemporaryDirectory(prefix="warm-asyncio-demo-") as temporary:
        folder = Path(temporary)
        async with asyncio.timeout(5):
            placement = await placement_case(folder)
            cancellation = await cancellation_case(folder)
        print(
            json.dumps(
                dict(
                    scope="CPU teaching simulation, no GPU or PSRL services",
                    placement=placement,
                    cancellation=cancellation,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
