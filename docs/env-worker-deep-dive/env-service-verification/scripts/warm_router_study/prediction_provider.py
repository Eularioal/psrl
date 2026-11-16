"""Adapt exclusively owned fixed-step GPU services to guarded command dispatch."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from math import isfinite

from psrl.workers.env_service.command_prediction import CommandSpec, StepProgress, WorkloadKey
from psrl.workers.env_service.predictive_scheduler import DeviceState, PlacementRequest


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _consume(task):
    if not task.cancelled():
        task.exception()


class FixedStepServiceProvider:
    """
    Bridge the study's real RPC and progress protocol without querying a busy RPC.

    One owner exclusively controls both services. Budgets remain fixed and models
    remain resident throughout this provider's lifetime. The harness must confirm
    node local progress files use the same host monotonic clock. Model sizes are
    obtained with cold admission probes before the first execution. A mandatory
    shutdown callback stops owned containers and confirms physical GPU cleanup.
    Outer Router leases remain owned by the experiment's context manager.
    """

    def __init__(
        self,
        context,
        *,
        device_class: str,
        model_versions: Mapping[str, str],
        code_version: str,
        runtime_version: str,
        input_signature: str,
        shutdown: Callable[[], Awaitable[None]],
        poll_interval_s: float = 0.02,
    ):
        if not callable(shutdown):
            raise ValueError("A physical shutdown callback is required.")
        if type(poll_interval_s) not in (int, float) or not isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("Progress polling interval must be finite and positive.")
        if not isinstance(model_versions, Mapping) or not model_versions:
            raise ValueError("At least one immutable model version is required.")
        if any(not isinstance(name, str) or not name.strip() for name in model_versions):
            raise ValueError("Server model names must be nonempty strings.")
        self.context = context
        self.device_class = device_class
        self.model_versions = dict(model_versions)
        self._workloads = {
            name: WorkloadKey(code_version, version, input_signature, device_class, runtime_version)
            for name, version in self.model_versions.items()
        }
        self._model_names = {self.model_id(name): name for name in self.model_versions}
        if len(self._model_names) != len(self.model_versions):
            raise ValueError("Immutable model identities must be unique.")
        self._shutdown = shutdown
        self._poll_interval_s = poll_interval_s
        self.model_bytes = {}
        self._budgets = {}
        self._workspace = {}
        self._states = {}
        self._busy = set()
        self._rpcs = set()
        self._initialized = False
        self._closing = False
        self._poisoned = False
        self._cleanup_task = None

    def model_id(self, server_model):
        """Bind one immutable cache identity to its server model and weight version."""
        return f"{server_model}:{self.model_versions[server_model]}"

    def workload(self, server_model):
        """Return the declared code, weights, input shape, device and runtime identity."""
        return self._workloads[server_model]

    def request(self, command_id, server_model, steps, home="0"):
        """Build a relocatable request for the service's reset between commands."""
        self._ready()
        if home not in self._states:
            raise ValueError("Home service identity is unknown.")
        if not _integer(steps, minimum=1) or steps > 8192:
            raise ValueError("Fixed commands require 1..8192 integer updates.")
        return PlacementRequest(
            CommandSpec(command_id, self.workload(server_model), self.model_id(server_model), steps, True),
            self.model_bytes[server_model],
            self._workspace[home],
            home,
        )

    def _ready(self):
        if self._closing or self._poisoned:
            raise RuntimeError("Fixed step provider has stopped or lost execution integrity.")
        if not self._initialized:
            raise RuntimeError("Fixed step provider must be initialized before use.")

    async def _rpc(self, index, payload):
        if self._closing:
            raise RuntimeError("The fixed step provider is closing.")
        task = asyncio.create_task(self.context["request"](index, payload))
        self._rpcs.add(task)
        task.add_done_callback(_consume)
        try:
            response = await task
            if not isinstance(response, dict) or response.get("status") != "passed":
                raise RuntimeError("The fixed step service did not confirm successful execution.")
            return response
        finally:
            self._rpcs.discard(task)

    async def initialize(self):
        """Probe empty services and obtain device reservations and model sizes."""
        if self._initialized or self._closing:
            raise RuntimeError("Fixed step provider initialization requires a fresh instance.")
        try:
            if len(self.context["envs"]) != 2:
                raise ValueError("This adapter requires exactly two owned fixed step services.")
            if (
                self.context.get("progress_clock") != "host_monotonic"
                or self.context.get("record", {}).get("progress_storage") != "node_local"
            ):
                raise ValueError("Progress requires an explicit node local host monotonic clock contract.")
            for index in range(2):
                state = await self._rpc(index, {"op": "ping"})
                if (
                    state.get("cache_keys") != []
                    or type(state.get("cache_bytes")) is not int
                    or state["cache_bytes"] != 0
                ):
                    raise ValueError("Initialize the provider with empty retained model caches.")
                budget, workspace = state.get("budget_bytes"), state.get("workspace_bytes")
                if not _integer(budget, minimum=1) or not _integer(workspace) or workspace > budget:
                    raise ValueError("The service returned an invalid fixed reservation.")
                self._budgets[str(index)], self._workspace[str(index)] = budget, workspace
                self._states[str(index)] = DeviceState(str(index), self.device_class, frozenset(), budget)
                for name in self.model_versions:
                    admission = await self._rpc(index, {"op": "admit", "model": name})
                    required = admission.get("required_bytes")
                    if (
                        not _integer(required, minimum=workspace + 1)
                        or admission.get("available_bytes") != budget
                        or type(admission.get("admits")) is not bool
                        or admission["admits"] != (required <= budget)
                    ):
                        raise ValueError("The cold admission probe returned inconsistent model accounting.")
                    size = required - workspace
                    if name in self.model_bytes and self.model_bytes[name] != size:
                        raise ValueError("The same immutable model has different sizes across services.")
                    self.model_bytes[name] = size
            if len(set(self._workspace.values())) != 1:
                raise ValueError("This fixed input adapter requires identical workspace declarations.")
            self._initialized = True
        except BaseException:
            self._poisoned = True
            await self.aclose()
            raise

    async def device_snapshot(self):
        """Return cached physical accounting while the serial service is executing."""
        self._ready()
        return tuple(self._states[str(index)] for index in range(2))

    def _decode_progress(self, index, request, previous):
        try:
            path = self.context["progress_file"](index)
            if path.stat().st_size > 262144:
                return None
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeError):
            return None
        if not isinstance(value, dict) or value.get("id") != request.spec.command_id:
            return None
        completed, observed = value.get("completed_steps"), value.get("observed_at")
        if (
            value.get("phase") not in {"compute", "complete"}
            or value.get("total_steps") != request.spec.total_steps
            or not _integer(completed)
            or completed > request.spec.total_steps
            or (previous is not None and completed <= previous)
            or type(observed) not in (int, float)
            or not isfinite(observed)
        ):
            return None
        age = time.monotonic() - observed
        if not isfinite(age) or age < 0:
            return None
        counts, durations = value.get("chunk_steps", []), value.get("wall_chunk_s", [])
        if not isinstance(counts, list) or not isinstance(durations, list) or len(counts) > 32 or len(durations) > 32:
            return None
        return StepProgress(
            request.spec.command_id,
            request.spec.workload,
            request.spec.total_steps,
            completed,
            tuple(counts),
            tuple(durations),
            age,
        )

    def _check_result(self, device_id, request, name, response):
        keys = response.get("cache_keys")
        if (
            response.get("id") != request.spec.command_id
            or response.get("model") != name
            or not _integer(response.get("steps"), minimum=1)
            or response.get("steps") != request.spec.total_steps
            or not _integer(response.get("optimizer_updates"), minimum=1)
            or response.get("optimizer_updates") != request.spec.total_steps
            or not _integer(response.get("model_bytes"), minimum=1)
            or response.get("model_bytes") != request.model_bytes
            or not isinstance(keys, list)
            or any(not isinstance(key, str) or key not in self.model_versions for key in keys)
            or len(keys) != len(set(keys))
            or name not in keys
        ):
            raise RuntimeError("The fixed step service returned inconsistent completion identity.")
        resident = frozenset(self.model_id(key) for key in keys)
        cache_bytes = sum(self.model_bytes[key] for key in keys)
        if (
            not _integer(response.get("cache_bytes"))
            or response.get("cache_bytes") != cache_bytes
            or not self._states[device_id].resident_models <= resident
            or cache_bytes + self._workspace[device_id] > self._budgets[device_id]
        ):
            raise RuntimeError("The fixed step service returned inconsistent retained model accounting.")
        self._states[device_id] = DeviceState(
            device_id, self.device_class, resident, self._budgets[device_id] - cache_bytes
        )

    async def execute(self, device_id, request, payload, report_progress):
        """Run the declared GPU command and stream fresh fixed-step observations."""
        self._ready()
        if device_id not in self._states or device_id in self._busy:
            raise ValueError("Execution requires one available owned serial service.")
        name = self._model_names.get(request.spec.model_id)
        if (
            name is None
            or request.spec.workload != self.workload(name)
            or request.model_bytes != self.model_bytes[name]
            or request.workspace_bytes != self._workspace[device_id]
            or not _integer(request.spec.total_steps, minimum=1)
            or request.spec.total_steps > 8192
        ):
            raise ValueError("Execution declarations do not match this fixed step service.")
        if payload is not None:
            raise ValueError("The fixed step provider accepts only its declared command fields.")
        index = int(device_id)
        self._busy.add(device_id)
        rpc = asyncio.create_task(
            self._rpc(
                index,
                dict(
                    id=request.spec.command_id,
                    model=name,
                    steps=request.spec.total_steps,
                    progress_every=16,
                ),
            )
        )
        rpc.add_done_callback(_consume)
        previous = None
        try:
            while True:
                observation = self._decode_progress(index, request, previous)
                if observation is not None:
                    await report_progress(observation)
                    previous = observation.completed_steps
                if rpc.done():
                    response = rpc.result()
                    self._check_result(device_id, request, name, response)
                    return response
                await asyncio.wait((rpc,), timeout=self._poll_interval_s)
        except BaseException:
            self._poisoned = True
            rpc.cancel()
            await self.aclose()
            raise
        finally:
            self._busy.discard(device_id)

    async def _cleanup(self):
        pending = tuple(self._rpcs)
        for task in pending:
            task.cancel()
        await self._shutdown()
        if pending:
            _, unfinished = await asyncio.wait(pending, timeout=1)
            if unfinished:
                raise RuntimeError("Owned RPC clients did not stop after physical shutdown.")

    async def aclose(self):
        """Confirm owned service shutdown once without releasing outer Router leases."""
        if self._cleanup_task is None:
            self._closing = True
            self._cleanup_task = asyncio.create_task(self._cleanup())
            self._cleanup_task.add_done_callback(_consume)
        await asyncio.shield(self._cleanup_task)
