"""Execute admitted structured commands through owned serial device queues."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from math import isfinite
from time import monotonic
from typing import Protocol

from .command_prediction import StepProgress
from .predictive_scheduler import (
    CommandCostProfile,
    DeviceState,
    GuardedDispatchDecision,
    PlacementRequest,
    PredictionAwareScheduler,
    QueuedWork,
)


class ManagedCommandProvider(Protocol):
    """
    Supply physical observations and execute commands with confirmed boundaries.

    Snapshots contain physical cache residency and admission budgets, without
    software queue entries. Retained models must survive until `aclose` or an
    explicitly reported cache change. Successful execution confirms all declared
    steps and includes preparation, compute and completion tail work. Execution
    failure must stop that command before returning an exception. Progress ages
    are normalized by the adapter without subtracting clocks on different hosts.

    Cancellation alone is not a physical release confirmation. `aclose` must stop
    owned execution, release owned resources and confirm cleanup, or raise.
    """

    async def device_snapshot(self) -> tuple[DeviceState, ...]:
        """Return bounded physical observations without active or queued work."""
        ...

    async def execute(
        self,
        device_id: str,
        request: PlacementRequest,
        payload: object,
        report_progress: Callable[[StepProgress], Awaitable[None]],
    ) -> object:
        """Execute one complete command and await each progress callback."""
        ...

    async def aclose(self) -> None:
        """Confirm that this provider's owned physical resources are released."""
        ...


class CommandDispatchError(RuntimeError):
    """Report failed admission, execution ownership or dispatcher closure."""


@dataclass(frozen=True)
class ManagedCommandResult:
    """Return a provider result alongside its actual admission decision."""

    value: object
    decision: GuardedDispatchDecision


@dataclass
class _Entry:
    work: QueuedWork
    payload: object
    decision: GuardedDispatchDecision
    result: asyncio.Future[ManagedCommandResult]
    progress: StepProgress | None = None
    received_at_s: float | None = None
    reference_time_s: float | None = None


def _consume_exception(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


class ManagedCommandDispatcher:
    """
    Atomically plan and queue work while observing full command completion.

    Each device has one execution coroutine. The lock covers planning and queue
    mutation. Physical snapshots and execution run outside it. A bounded snapshot
    is accepted only if queue ownership has not changed during the observation. Caller cancellation stops
    waiting for a result without relinquishing ownership of running GPU work.
    The provider alone owns physical leases, cache state and device cleanup.
    """

    def __init__(
        self,
        scheduler: PredictionAwareScheduler,
        provider: ManagedCommandProvider,
        *,
        clock: Callable[[], float] = monotonic,
        close_timeout_s: float = 5,
        snapshot_timeout_s: float = 5,
    ) -> None:
        if type(close_timeout_s) not in (int, float) or not isfinite(close_timeout_s) or close_timeout_s <= 0:
            raise ValueError("Close timeout must be finite and positive.")
        if type(snapshot_timeout_s) not in (int, float) or not isfinite(snapshot_timeout_s) or snapshot_timeout_s <= 0:
            raise ValueError("Snapshot timeout must be finite and positive.")
        self.scheduler = scheduler
        self.provider = provider
        self._clock = clock
        self._close_timeout_s = close_timeout_s
        self._snapshot_timeout_s = snapshot_timeout_s
        self._observations: set[asyncio.Task] = set()
        self._generation = 0
        self._lock = asyncio.Lock()
        self._queues: dict[str, deque[_Entry]] = {}
        self._active: dict[str, _Entry] = {}
        self._entries: dict[str, _Entry] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._failed_devices: set[str] = set()
        self._model_bindings: dict[str, tuple[str, int]] = {}
        self._closing = False
        self._close_task: asyncio.Task | None = None

    @staticmethod
    def _aged_progress(entry: _Entry, now_s: float) -> StepProgress | None:
        if entry.progress is None or entry.received_at_s is None:
            return None
        return replace(entry.progress, report_age_s=entry.progress.report_age_s + now_s - entry.received_at_s)

    def _check_request(self, request: PlacementRequest) -> None:
        if self._closing:
            raise CommandDispatchError("Dispatcher is closing.")
        command_id = request.spec.command_id
        if command_id in self._entries:
            raise CommandDispatchError("Command identity is already admitted.")
        if len(self._entries) >= self.scheduler.predictor.config.max_pending:
            raise CommandDispatchError("Managed command capacity is exhausted.")
        binding = (request.spec.workload.model_version, request.model_bytes)
        previous = self._model_bindings.get(request.spec.model_id)
        if previous is not None and previous != binding:
            raise CommandDispatchError("An immutable model identity changed its version or retained size.")
        if previous is None and len(self._model_bindings) >= self.scheduler.predictor.config.max_profiles:
            raise CommandDispatchError("Managed model identity capacity is exhausted.")

    async def _snapshot(self) -> tuple[DeviceState, ...]:
        if self._closing:
            raise CommandDispatchError("Dispatcher is closing.")
        if len(self._observations) >= self.scheduler.predictor.config.max_pending:
            raise CommandDispatchError("Physical observation capacity is exhausted.")
        observation = asyncio.create_task(self.provider.device_snapshot())
        self._observations.add(observation)
        observation.add_done_callback(self._observations.discard)
        observation.add_done_callback(_consume_exception)
        try:
            done, _ = await asyncio.wait((observation,), timeout=self._snapshot_timeout_s)
            if not done:
                observation.cancel()
                raise CommandDispatchError("Physical device snapshot timed out.")
            if observation.cancelled() and self._closing:
                raise CommandDispatchError("Dispatcher closed during physical observation.")
            return observation.result()
        except asyncio.CancelledError:
            observation.cancel()
            raise

    async def submit(
        self,
        request: PlacementRequest,
        profiles: Mapping[str, CommandCostProfile],
        payload: object = None,
    ) -> ManagedCommandResult:
        """Admit one command using a fresh observation and bounded queue retries."""
        for _ in range(3):
            async with self._lock:
                self._check_request(request)
                generation = self._generation
            physical = await self._snapshot()
            async with self._lock:
                self._check_request(request)
                if generation != self._generation:
                    continue
                future = self._admit(request, profiles, payload, physical)
                break
        else:
            raise CommandDispatchError("Queue changed during three physical observations. Retry submission.")
        return await asyncio.shield(future)

    def _admit(
        self,
        request: PlacementRequest,
        profiles: Mapping[str, CommandCostProfile],
        payload: object,
        physical: tuple[DeviceState, ...],
    ) -> asyncio.Future[ManagedCommandResult]:
        if len({state.device_id for state in physical}) != len(physical):
            raise CommandDispatchError("Provider returned duplicate device identities.")
        if any(state.active is not None or state.queued for state in physical):
            raise CommandDispatchError("Provider snapshots must contain physical observations only.")
        known_devices = {state.device_id for state in physical}
        if set(profiles) - known_devices:
            raise ValueError("Cost profiles must map known provider device identities.")
        if set(self._active) - known_devices or any(
            queue and device_id not in known_devices for device_id, queue in self._queues.items()
        ):
            raise CommandDispatchError("Provider lost a device with owned work.")
        now_s = self._clock()
        devices = []
        for state in physical:
            if state.device_id in self._failed_devices:
                continue
            active = self._active.get(state.device_id)
            if active and active.work.request.spec.workload.device_class != state.device_class:
                raise CommandDispatchError("Provider changed the class of an executing device.")
            if any(
                entry.work.request.spec.workload.device_class != state.device_class
                for entry in self._queues.get(state.device_id, ())
            ):
                raise CommandDispatchError("Provider changed the class of a queued device.")
            devices.append(
                replace(
                    state,
                    active=active.work if active else None,
                    progress=self._aged_progress(active, now_s) if active else None,
                    queued=tuple(entry.work for entry in self._queues.get(state.device_id, ())),
                )
            )
        candidate_ids = {state.device_id for state in devices}
        candidate_profiles = {
            device_id: profile for device_id, profile in profiles.items() if device_id in candidate_ids
        }
        decision = self.scheduler.plan(request, tuple(devices), candidate_profiles, now_s=now_s)
        if decision.device_id is None:
            raise CommandDispatchError(f"Command admission failed with reason {decision.reason!r}.")
        selected = next((state for state in devices if state.device_id == decision.device_id), None)
        if selected is None:
            raise CommandDispatchError("Scheduler selected an unavailable device.")
        workload = replace(request.spec.workload, device_class=selected.device_class)
        selected_request = replace(request, spec=replace(request.spec, workload=workload))
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_exception)
        entry = _Entry(QueuedWork(selected_request, profiles.get(selected.device_id)), payload, decision, future)
        self._model_bindings[request.spec.model_id] = (request.spec.workload.model_version, request.model_bytes)
        self._entries[request.spec.command_id] = entry
        self._queues.setdefault(selected.device_id, deque()).append(entry)
        worker = self._workers.get(selected.device_id)
        if worker is None or worker.done():
            worker = asyncio.create_task(self._run_device(selected.device_id))
            worker.add_done_callback(_consume_exception)
            self._workers[selected.device_id] = worker
        self._generation += 1
        return future

    async def _progress(self, device_id: str, entry: _Entry, progress: StepProgress) -> None:
        async with self._lock:
            if self._closing or self._active.get(device_id) is not entry:
                return
            if self.scheduler.config.mode == "off":
                return
            now_s = self._clock()
            assessment = self.scheduler.predictor.assess(entry.work.request.spec, progress, now_s=now_s)
            entry.reference_time_s = assessment.reference_time_s
            if (
                isinstance(progress, StepProgress)
                and type(progress.report_age_s) in (int, float)
                and isfinite(progress.report_age_s)
                and progress.report_age_s >= 0
            ):
                entry.progress = progress
                entry.received_at_s = now_s
            else:
                entry.progress = None
                entry.received_at_s = None

    async def _complete(
        self,
        device_id: str,
        entry: _Entry,
        *,
        value: object = None,
        error: BaseException | None = None,
    ) -> None:
        async with self._lock:
            spec = entry.work.request.spec
            now_s = self._clock()
            if self._closing and error is None:
                error = CommandDispatchError("Dispatcher closed before command completion.")
            if self.scheduler.config.mode != "off":
                self.scheduler.predictor.finish(
                    spec,
                    succeeded=error is None,
                    completed_steps=spec.total_steps if error is None and spec.total_steps is not None else 0,
                    actual_remaining_s=(
                        now_s - entry.reference_time_s
                        if error is None and entry.reference_time_s is not None
                        else None
                    ),
                    now_s=now_s,
                )
            self._active.pop(device_id, None)
            self._entries.pop(spec.command_id, None)
            self._generation += 1
            if not entry.result.done():
                if error is not None:
                    entry.result.set_exception(error)
                else:
                    entry.result.set_result(ManagedCommandResult(value, entry.decision))

    async def _run_device(self, device_id: str) -> None:
        try:
            while True:
                async with self._lock:
                    queue = self._queues[device_id]
                    if self._closing or not queue:
                        return
                    entry = queue.popleft()
                    self._active[device_id] = entry
                    self._generation += 1
                try:
                    value = await self.provider.execute(
                        device_id,
                        entry.work.request,
                        entry.payload,
                        lambda progress, owned_entry=entry: self._progress(device_id, owned_entry, progress),
                    )
                except asyncio.CancelledError:
                    await self._complete(
                        device_id, entry, error=CommandDispatchError("Owned command execution was cancelled.")
                    )
                    raise
                except Exception as error:
                    await self._complete(device_id, entry, error=error)
                else:
                    await self._complete(device_id, entry, value=value)
        except BaseException:
            async with self._lock:
                self._failed_devices.add(device_id)
                self._generation += 1
                abandoned = list(self._queues[device_id])
                self._queues[device_id].clear()
                active = self._active.pop(device_id, None)
                if active is not None:
                    abandoned.append(active)
                for entry in abandoned:
                    command_id = entry.work.request.spec.command_id
                    self.scheduler.predictor.forget(command_id)
                    self._entries.pop(command_id, None)
                    if not entry.result.done():
                        entry.result.set_exception(CommandDispatchError("Device worker stopped before completion."))
            raise

    async def _close(self) -> None:
        async with self._lock:
            owned_tasks = tuple(self._workers.values()) + tuple(self._observations)
            for entry in self._entries.values():
                if not entry.result.done():
                    entry.result.set_exception(CommandDispatchError("Dispatcher closed before command completion."))
            for queue in self._queues.values():
                while queue:
                    entry = queue.popleft()
                    command_id = entry.work.request.spec.command_id
                    self.scheduler.predictor.forget(command_id)
                    self._entries.pop(command_id, None)
        for owned in owned_tasks:
            owned.cancel()
        pending_workers = set()
        if owned_tasks:
            _, pending_workers = await asyncio.wait(owned_tasks, timeout=self._close_timeout_s)
        cleanup = asyncio.create_task(self.provider.aclose())
        cleanup.add_done_callback(_consume_exception)
        done, _ = await asyncio.wait((cleanup,), timeout=self._close_timeout_s)
        if not done:
            cleanup.cancel()
            raise CommandDispatchError("Provider cleanup timed out without physical release confirmation.")
        cleanup.result()
        if any(not worker.done() for worker in pending_workers):
            raise CommandDispatchError("Owned workers or observations did not stop within the cleanup budget.")

    async def aclose(self) -> None:
        """Stop submissions and confirm owned provider cleanup within bounds."""
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(_consume_exception)
        await asyncio.shield(self._close_task)

    async def __aenter__(self) -> "ManagedCommandDispatcher":
        if self._closing:
            raise CommandDispatchError("Dispatcher is closing.")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()
