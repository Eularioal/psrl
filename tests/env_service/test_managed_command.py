"""Exercise owned queues and online calibration without physical GPU resources."""

import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest
from psrl.workers.env_service.command_prediction import (
    CommandRuntimePredictor,
    CommandSpec,
    PredictabilityConfig,
    StepProgress,
    WorkloadKey,
)
from psrl.workers.env_service.managed_command import CommandDispatchError, ManagedCommandDispatcher
from psrl.workers.env_service.predictive_scheduler import (
    CommandCostProfile,
    DeviceState,
    PlacementConfig,
    PlacementRequest,
    PredictionAwareScheduler,
    QueuedWork,
)

WORKLOAD = WorkloadKey("code-v1", "weights-v1", "batch=1,len=256,fp16", "class-a", "runtime-v1")


class Clock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value


def request(command_id, *, steps=128, relocatable=True, home="gpu-0", workload=WORKLOAD):
    return PlacementRequest(CommandSpec(command_id, workload, "model", steps, relocatable), 10, 2, home)


def progress(spec, *, age=0):
    return StepProgress(spec.command_id, spec.workload, spec.total_steps, 64, (16,) * 4, (2.0,) * 4, age)


def profiles(clock, work):
    return {
        device_id: CommandCostProfile(
            work.spec.workload, "model", work.spec.total_steps, 10, 2, (1.0,) * 3, (1.0,) * 3, clock()
        )
        for device_id in ("gpu-0", "gpu-1")
    }


class CpuProvider:
    """Model confirmed provider boundaries with cooperative CPU coroutines."""

    def __init__(self, clock):
        self.clock = clock
        self.states = (
            DeviceState("gpu-0", "class-a", frozenset({"model"}), 100),
            DeviceState("gpu-1", "class-a", frozenset(), 100),
        )
        self.requests = []
        self.running = {}
        self.maximum_running = {}
        self.closed = 0
        self.close_error = None

    async def device_snapshot(self):
        return self.states

    async def execute(self, device_id, work, payload, report_progress):
        payload = payload or {}
        self.requests.append((device_id, work.spec.command_id, work.spec.workload.device_class))
        self.running[device_id] = self.running.get(device_id, 0) + 1
        self.maximum_running[device_id] = max(self.maximum_running.get(device_id, 0), self.running[device_id])
        self.states = tuple(
            replace(
                state,
                resident_models=state.resident_models | {work.spec.model_id},
                available_gpu_bytes=state.available_gpu_bytes - work.model_bytes,
            )
            if state.device_id == device_id and work.spec.model_id not in state.resident_models
            else state
            for state in self.states
        )
        try:
            if payload.get("progress"):
                await report_progress(progress(work.spec))
            if payload.get("started"):
                payload["started"].set()
            if payload.get("release"):
                await payload["release"].wait()
            if payload.get("error"):
                raise payload["error"]
            if payload.get("cancel"):
                raise asyncio.CancelledError
            self.clock.value += payload.get("tail_s", 8)
            return payload.get("value", work.spec.command_id)
        finally:
            self.running[device_id] -= 1

    async def aclose(self):
        self.closed += 1
        if self.close_error:
            raise self.close_error
        if any(self.running.values()):
            raise RuntimeError("Physical release is unconfirmed.")


class RecordingScheduler(PredictionAwareScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots = []

    def plan(self, work, devices, costs, *, now_s):
        self.snapshots.append(devices)
        return super().plan(work, devices, costs, now_s=now_s)


def seed_history(predictor, clock):
    for index in range(3):
        spec = request(f"history-{index}").spec
        predictor.assess(spec, progress(spec), now_s=clock())
        clock.value += 8
        predictor.finish(spec, succeeded=True, completed_steps=128, actual_remaining_s=8, now_s=clock())


async def wait_for_admission(driver, command_id):
    for _ in range(100):
        async with driver._lock:
            if command_id in driver._entries:
                return
        await asyncio.sleep(0)
    raise AssertionError("The bounded CPU command admission did not complete.")


@pytest.mark.parametrize("mode", ["off", "observe", "auto"])
def test_submit_returns_provider_result_and_the_actual_decision(mode):
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = PredictionAwareScheduler(PlacementConfig(mode=mode))
        async with ManagedCommandDispatcher(scheduler, provider, clock=clock) as driver:
            work = request("target", steps=8)
            result = await driver.submit(work, profiles(clock, work), {"value": 42})
            assert result.value == 42, "The provider result must reach the caller."
            assert result.decision.device_id == "gpu-0", "The cached model should win without a busy queue."
            assert result.decision.prediction_used == (mode == "auto"), "Only auto may use the validated comparison."
        assert provider.closed == 1, "Context exit must confirm provider cleanup once."

    asyncio.run(scenario())


def test_observation_automatically_calibrates_independent_full_completions():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = PredictionAwareScheduler(PlacementConfig(mode="observe"))
        async with ManagedCommandDispatcher(scheduler, provider, clock=clock) as driver:
            for index in range(3):
                result = await driver.submit(request(f"shadow-{index}"), {}, {"progress": True})
                assert not result.decision.prediction_used, "Observation must retain ordinary placement."
            spec = request("probe").spec
            assessment = scheduler.predictor.assess(spec, progress(spec), now_s=clock())
            assert assessment.eligible, "Three independently completed commands should calibrate the profile."
            assert assessment.calibration_commands == 3, "Progress polls must not replace independent commands."
            scheduler.predictor.forget(spec.command_id)

    asyncio.run(scenario())


def test_off_does_not_feed_prediction_or_completion_learning():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = PredictionAwareScheduler()
        scheduler.predictor.assess = Mock(side_effect=AssertionError("Off must not assess."))
        scheduler.predictor.finish = Mock(side_effect=AssertionError("Off must not calibrate."))
        async with ManagedCommandDispatcher(scheduler, provider, clock=clock) as driver:
            await driver.submit(request("disabled"), {}, {"progress": True})

    asyncio.run(scenario())


def test_auto_moves_new_command_when_calibrated_hot_queue_is_longer_than_cold_cost():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
        seed_history(scheduler.predictor, clock)
        started, release = asyncio.Event(), asyncio.Event()
        async with ManagedCommandDispatcher(scheduler, provider, clock=clock) as driver:
            blocker = asyncio.create_task(
                driver.submit(
                    request("blocker"),
                    {},
                    {
                        "progress": True,
                        "started": started,
                        "release": release,
                    },
                )
            )
            await asyncio.wait_for(started.wait(), 1)
            target = request("target", steps=8)
            result = await driver.submit(target, profiles(clock, target), {"tail_s": 1})
            assert result.decision.prediction_used, "Calibrated progress and measured costs must enable auto."
            assert result.decision.device_id == "gpu-1", "A two second cold path should beat the eight second queue."
            assert result.decision.action == "rebuild", "The chosen device must rebuild its missing model."
            release.set()
            await blocker
        assert provider.maximum_running == {"gpu-0": 1, "gpu-1": 1}, "Each device must remain serial."

    asyncio.run(scenario())


def test_auto_falls_back_for_an_unobserved_active_command():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = PredictionAwareScheduler(PlacementConfig(mode="auto"))
        started, release = asyncio.Event(), asyncio.Event()
        async with ManagedCommandDispatcher(scheduler, provider, clock=clock) as driver:
            blocker = asyncio.create_task(
                driver.submit(
                    request("opaque-blocker"),
                    {},
                    {
                        "started": started,
                        "release": release,
                    },
                )
            )
            await asyncio.wait_for(started.wait(), 1)
            target = request("target", steps=8)
            pending = asyncio.create_task(driver.submit(target, profiles(clock, target)))
            await wait_for_admission(driver, "target")
            assert len(provider.requests) == 1, "Fallback should queue on the cached device."
            release.set()
            await blocker
            result = await pending
            assert result.decision.device_id == "gpu-0", "Unknown queue duration must retain model affinity."
            assert not result.decision.prediction_used, "No trustworthy remaining time means no prediction placement."
            assert result.decision.reason == "queue_time_unpredictable", "The fallback cause must remain visible."

    asyncio.run(scenario())


def test_owned_queue_is_visible_to_the_next_admission_and_never_executes_in_parallel():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = RecordingScheduler(PlacementConfig(mode="observe"))
        started, release = asyncio.Event(), asyncio.Event()
        async with ManagedCommandDispatcher(scheduler, provider, clock=clock) as driver:
            blocker = asyncio.create_task(
                driver.submit(
                    request("first"),
                    {},
                    {
                        "progress": True,
                        "started": started,
                        "release": release,
                    },
                )
            )
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(driver.submit(request("second"), {}))
            await wait_for_admission(driver, "second")
            clock.value += 0.5
            third = asyncio.create_task(driver.submit(request("third"), {}))
            await wait_for_admission(driver, "third")
            hot = next(state for state in scheduler.snapshots[-1] if state.device_id == "gpu-0")
            assert hot.active.request.spec.command_id == "first", "The owner must report its actual active command."
            assert [work.request.spec.command_id for work in hot.queued] == ["second"], (
                "Queued work must stay ordered."
            )
            assert hot.progress.report_age_s == pytest.approx(0.5), "Progress must age on the driver clock."
            release.set()
            await asyncio.gather(blocker, second, third)
        assert provider.maximum_running["gpu-0"] == 1, "The execution queue must serialize commands."
        assert [item[1] for item in provider.requests] == ["first", "second", "third"], "FIFO order must be preserved."

    asyncio.run(scenario())


@pytest.mark.parametrize("tail_s", [30, 0])
def test_calibration_uses_full_service_tail_and_never_substitutes_zero_for_missing_progress(tail_s):
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = PredictionAwareScheduler(PlacementConfig(mode="observe"))
        async with ManagedCommandDispatcher(scheduler, provider, clock=clock) as driver:
            await driver.submit(request("slow-tail"), {}, {"progress": tail_s != 0, "tail_s": tail_s})
            spec = request("probe").spec
            assessment = scheduler.predictor.assess(spec, progress(spec), now_s=clock())
            if tail_s:
                assert assessment.reason == "history_error", "Completion tail error must disable auto."
                assert assessment.calibration_commands == 1, "The independent failed forecast should be retained."
            else:
                assert assessment.calibration_commands == 0, "No progress means no remaining time observation."
            scheduler.predictor.forget(spec.command_id)

    asyncio.run(scenario())


def test_failed_execution_does_not_become_a_successful_zero_length_calibration():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = PredictionAwareScheduler(PlacementConfig(mode="observe"))
        async with ManagedCommandDispatcher(scheduler, provider, clock=clock) as driver:
            with pytest.raises(ValueError, match="Execution failed"):
                await driver.submit(
                    request("failed"), {}, {"progress": True, "error": ValueError("Execution failed.")}
                )
            spec = request("probe").spec
            assessment = scheduler.predictor.assess(spec, progress(spec), now_s=clock())
            assert assessment.calibration_commands == 0, "Provider failures must not count as timing evidence."
            scheduler.predictor.forget(spec.command_id)
            result = await driver.submit(request("next"), {})
            assert result.value == "next", "A confirmed command failure should not strand the remaining queue."

    asyncio.run(scenario())


def test_caller_cancellation_leaves_execution_owned_until_dispatcher_cleanup():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = PredictionAwareScheduler(PlacementConfig(mode="observe"))
        started, release = asyncio.Event(), asyncio.Event()
        driver = ManagedCommandDispatcher(scheduler, provider, clock=clock)
        caller = asyncio.create_task(
            driver.submit(
                request("cancel-caller"),
                {},
                {
                    "progress": True,
                    "started": started,
                    "release": release,
                },
            )
        )
        await asyncio.wait_for(started.wait(), 1)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert provider.running["gpu-0"] == 1, "Caller cancellation must not imply GPU work stopped."
        await driver.aclose()
        assert provider.closed == 1 and not any(provider.running.values()), "Cleanup must stop owned execution."
        await driver.aclose()
        assert provider.closed == 1, "Cleanup must be idempotent."

    asyncio.run(scenario())


def test_close_fails_queued_commands_and_stops_accepting_submissions():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        scheduler = PredictionAwareScheduler()
        started, release = asyncio.Event(), asyncio.Event()
        driver = ManagedCommandDispatcher(scheduler, provider, clock=clock)
        active = asyncio.create_task(driver.submit(request("active"), {}, {"started": started, "release": release}))
        await asyncio.wait_for(started.wait(), 1)
        queued = asyncio.create_task(driver.submit(request("queued"), {}))
        await wait_for_admission(driver, "queued")
        await driver.aclose()
        results = await asyncio.gather(active, queued, return_exceptions=True)
        assert all(isinstance(result, CommandDispatchError) for result in results), "Closure must settle all callers."
        assert len(provider.requests) == 1, "Queued commands must not execute during shutdown."
        with pytest.raises(CommandDispatchError, match="closing"):
            await driver.submit(request("late"), {})

    asyncio.run(scenario())


def test_duplicate_active_identity_is_rejected_without_starting_another_execution():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        started, release = asyncio.Event(), asyncio.Event()
        async with ManagedCommandDispatcher(PredictionAwareScheduler(), provider, clock=clock) as driver:
            active = asyncio.create_task(driver.submit(request("same"), {}, {"started": started, "release": release}))
            await asyncio.wait_for(started.wait(), 1)
            with pytest.raises(CommandDispatchError, match="already admitted"):
                await driver.submit(request("same"), {})
            release.set()
            await active

    asyncio.run(scenario())


def test_provider_cleanup_failure_is_visible_even_when_no_command_is_running():
    async def scenario():
        provider = CpuProvider(Clock())
        provider.close_error = RuntimeError("Release could not be confirmed.")
        driver = ManagedCommandDispatcher(PredictionAwareScheduler(), provider)
        with pytest.raises(RuntimeError, match="could not be confirmed"):
            await driver.aclose()
        assert provider.closed == 1, "Physical cleanup must be attempted."

    asyncio.run(scenario())


def test_provider_cleanup_timeout_is_bounded_and_reported():
    class SlowClose(CpuProvider):
        async def aclose(self):
            self.closed += 1
            await asyncio.Event().wait()

    async def scenario():
        provider = SlowClose(Clock())
        driver = ManagedCommandDispatcher(PredictionAwareScheduler(), provider, close_timeout_s=0.01)
        with pytest.raises(CommandDispatchError, match="without physical release confirmation"):
            await asyncio.wait_for(driver.aclose(), 1)
        assert provider.closed == 1, "Timeout must retain evidence that cleanup was attempted."

    asyncio.run(scenario())


def test_provider_cancellation_fails_queued_work_instead_of_stranding_it():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        started, release = asyncio.Event(), asyncio.Event()
        async with ManagedCommandDispatcher(PredictionAwareScheduler(), provider, clock=clock) as driver:
            first = asyncio.create_task(
                driver.submit(
                    request("cancel-provider"),
                    {},
                    {
                        "started": started,
                        "release": release,
                        "cancel": True,
                    },
                )
            )
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(driver.submit(request("queued"), {}))
            await wait_for_admission(driver, "queued")
            release.set()
            outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 1)
            assert all(isinstance(item, CommandDispatchError) for item in outcomes), (
                "Worker cancellation must settle work."
            )
            result = await driver.submit(request("healthy-device", home="gpu-1"), {})
            assert result.decision.device_id == "gpu-1", "A failed device must be excluded from later admission."

    asyncio.run(scenario())


def test_device_class_is_rebound_before_provider_execution():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        provider.states = (DeviceState("gpu-0", "class-b", frozenset({"model"}), 100),)
        async with ManagedCommandDispatcher(PredictionAwareScheduler(), provider, clock=clock) as driver:
            await driver.submit(request("heterogeneous"), {})
        assert provider.requests == [("gpu-0", "heterogeneous", "class-b")], (
            "Execution identity must match the device."
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("defect", ["duplicate", "foreign-queue", "lost-active", "class-change"])
def test_provider_snapshot_contract_violations_are_rejected(defect):
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        started, release = asyncio.Event(), asyncio.Event()
        async with ManagedCommandDispatcher(PredictionAwareScheduler(), provider, clock=clock) as driver:
            if defect in {"lost-active", "class-change"}:
                first = asyncio.create_task(
                    driver.submit(
                        request("first"),
                        {},
                        {
                            "started": started,
                            "release": release,
                        },
                    )
                )
                await asyncio.wait_for(started.wait(), 1)
            if defect == "duplicate":
                provider.states = (provider.states[0], provider.states[0])
            elif defect == "foreign-queue":
                provider.states = (replace(provider.states[0], queued=(QueuedWork(request("foreign")),)),)
            elif defect == "lost-active":
                provider.states = (provider.states[1],)
            else:
                provider.states = (replace(provider.states[0], device_class="class-b"), provider.states[1])
            with pytest.raises(CommandDispatchError):
                await driver.submit(request("invalid"), {})
            if defect in {"lost-active", "class-change"}:
                release.set()
                await first

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_close_budget_rejects_invalid_values(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        ManagedCommandDispatcher(PredictionAwareScheduler(), CpuProvider(Clock()), close_timeout_s=timeout)


@pytest.mark.parametrize("change", ["version", "size"])
def test_completed_model_identity_cannot_be_reused_for_different_cached_state(change):
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        async with ManagedCommandDispatcher(PredictionAwareScheduler(), provider, clock=clock) as driver:
            await driver.submit(request("original"), {})
            changed = request("changed")
            if change == "version":
                changed = replace(changed, spec=replace(changed.spec, workload=replace(WORKLOAD, model_version="v2")))
            else:
                changed = replace(changed, model_bytes=11)
            with pytest.raises(CommandDispatchError, match="changed its version or retained size"):
                await driver.submit(changed, {})
            assert len(provider.requests) == 1, (
                "A different cached state requires a different immutable model identity."
            )

    asyncio.run(scenario())


def test_model_identity_memory_is_bounded_without_evicting_cached_bindings():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        predictor = CommandRuntimePredictor(PredictabilityConfig(max_profiles=1))
        async with ManagedCommandDispatcher(
            PredictionAwareScheduler(predictor=predictor), provider, clock=clock
        ) as driver:
            await driver.submit(request("first"), {})
            second = request("second")
            second = replace(second, spec=replace(second.spec, model_id="different-model"))
            with pytest.raises(CommandDispatchError, match="identity capacity"):
                await driver.submit(second, {})
            await driver.submit(request("reuse-known-model"), {})
            assert len(provider.requests) == 2, "A known cache identity should remain usable at the binding limit."

    asyncio.run(scenario())


def test_pending_command_budget_includes_executing_work():
    async def scenario():
        clock = Clock()
        provider = CpuProvider(clock)
        predictor = CommandRuntimePredictor(PredictabilityConfig(max_pending=1))
        started, release = asyncio.Event(), asyncio.Event()
        async with ManagedCommandDispatcher(
            PredictionAwareScheduler(predictor=predictor), provider, clock=clock
        ) as driver:
            first = asyncio.create_task(driver.submit(request("active"), {}, {"started": started, "release": release}))
            await asyncio.wait_for(started.wait(), 1)
            with pytest.raises(CommandDispatchError, match="command capacity"):
                await driver.submit(request("overflow"), {})
            release.set()
            await first
            await driver.submit(request("after-completion"), {})

    asyncio.run(scenario())


def test_cancelling_close_caller_does_not_cancel_physical_cleanup():
    class DelayedClose(CpuProvider):
        def __init__(self, clock):
            super().__init__(clock)
            self.closing_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def aclose(self):
            self.closed += 1
            self.closing_started.set()
            await self.release_close.wait()

    async def scenario():
        provider = DelayedClose(Clock())
        driver = ManagedCommandDispatcher(PredictionAwareScheduler(), provider)
        caller = asyncio.create_task(driver.aclose())
        await asyncio.wait_for(provider.closing_started.wait(), 1)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        provider.release_close.set()
        await driver.aclose()
        assert provider.closed == 1, "Caller cancellation must leave one owned cleanup operation running."

    asyncio.run(scenario())


def test_blocked_physical_snapshot_does_not_block_close_or_admit_late_work():
    class BlockingSnapshot(CpuProvider):
        def __init__(self, clock):
            super().__init__(clock)
            self.snapshot_started = asyncio.Event()

        async def device_snapshot(self):
            self.snapshot_started.set()
            await asyncio.Event().wait()

    async def scenario():
        provider = BlockingSnapshot(Clock())
        driver = ManagedCommandDispatcher(PredictionAwareScheduler(), provider, close_timeout_s=0.01)
        submitting = asyncio.create_task(driver.submit(request("pending-snapshot"), {}))
        await asyncio.wait_for(provider.snapshot_started.wait(), 1)
        await asyncio.wait_for(driver.aclose(), 1)
        with pytest.raises(CommandDispatchError, match="closed during physical observation"):
            await submitting
        assert provider.closed == 1, "Physical cleanup must run even while a snapshot is blocked."
        assert not provider.requests, "A command observed during closure must never enter execution."

    asyncio.run(scenario())


def test_physical_snapshot_timeout_has_a_separate_bounded_budget():
    class BlockingSnapshot(CpuProvider):
        async def device_snapshot(self):
            await asyncio.Event().wait()

    async def scenario():
        provider = BlockingSnapshot(Clock())
        async with ManagedCommandDispatcher(PredictionAwareScheduler(), provider, snapshot_timeout_s=0.01) as driver:
            with pytest.raises(CommandDispatchError, match="snapshot timed out"):
                await asyncio.wait_for(driver.submit(request("timed-out"), {}), 1)
        assert provider.closed == 1, "A snapshot timeout must retain provider cleanup ownership."

    asyncio.run(scenario())


def test_snapshot_can_await_an_active_progress_callback_without_lock_inversion():
    class ReportingSnapshot(CpuProvider):
        def __init__(self, clock):
            super().__init__(clock)
            self.report = None
            self.active_spec = None

        async def execute(self, device_id, work, payload, report_progress):
            self.report, self.active_spec = report_progress, work.spec
            return await super().execute(device_id, work, payload, report_progress)

        async def device_snapshot(self):
            if self.report is not None and any(self.running.values()):
                await self.report(progress(self.active_spec))
            return self.states

    async def scenario():
        clock = Clock()
        provider = ReportingSnapshot(clock)
        started, release = asyncio.Event(), asyncio.Event()
        scheduler = PredictionAwareScheduler(PlacementConfig(mode="observe"))
        async with ManagedCommandDispatcher(scheduler, provider, clock=clock, snapshot_timeout_s=0.1) as driver:
            first = asyncio.create_task(driver.submit(request("active"), {}, {"started": started, "release": release}))
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(driver.submit(request("next"), {}))
            await wait_for_admission(driver, "next")
            release.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), 1)
            assert len(results) == 2, "A provider snapshot must not hold the callback's dispatcher lock."

    asyncio.run(scenario())


def test_queue_completion_during_snapshot_forces_a_fresh_observation():
    class PausingSnapshot(CpuProvider):
        def __init__(self, clock):
            super().__init__(clock)
            self.calls = 0
            self.second_snapshot = asyncio.Event()
            self.release_snapshot = asyncio.Event()

        async def device_snapshot(self):
            self.calls += 1
            captured = self.states
            if self.calls == 2:
                self.second_snapshot.set()
                await self.release_snapshot.wait()
            return captured

    async def scenario():
        clock = Clock()
        provider = PausingSnapshot(clock)
        started, release = asyncio.Event(), asyncio.Event()
        async with ManagedCommandDispatcher(PredictionAwareScheduler(), provider, clock=clock) as driver:
            first = asyncio.create_task(driver.submit(request("active"), {}, {"started": started, "release": release}))
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(driver.submit(request("next"), {}))
            await asyncio.wait_for(provider.second_snapshot.wait(), 1)
            release.set()
            await first
            provider.release_snapshot.set()
            await second
        assert provider.calls == 3, "Ownership changes during a snapshot must trigger a fresh physical observation."

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_snapshot_budget_rejects_invalid_values(timeout):
    with pytest.raises(ValueError, match="Snapshot timeout must be finite and positive"):
        ManagedCommandDispatcher(PredictionAwareScheduler(), CpuProvider(Clock()), snapshot_timeout_s=timeout)
