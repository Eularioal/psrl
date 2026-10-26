"""Same bounded OOM refill policy for the historical and current science loops."""

import asyncio
import json
import time
from pathlib import Path

from examples.sciaccel_rl.agent_loop import SciAccelAgentLoop
from science_pressure_agent import PressureTerminus2
from science_pressure_audit import load_pressure_evidence, save_pressure_evidence

_slots = None


class FixedWorkPressureAgent(PressureTerminus2):
    """Use the same configured CUDA pressure phase in both experiment arms."""

    async def _run_pressure_command(self, command, **kwargs):
        return await super()._exec_pressure(command, **kwargs)

    async def _exec_pressure(self, command, **kwargs):
        result = await self._run_pressure_command(command, **kwargs)
        save_pressure_evidence(self.logs_dir, result.model_dump())
        return result


class CommandPressureAgent(FixedWorkPressureAgent):
    """Return GPU command capacity while the science repair agent continues."""

    async def _run_pressure_command(self, command, **kwargs):
        environment = self._pressure_environment
        try:
            return await environment.exec_gpu(command, wait_timeout_sec=300, overlap_return=True, **kwargs)
        finally:
            (self.logs_dir / "gpu-command-phase.json").write_text(json.dumps(environment.last_gpu_command, indent=2))


class FixedWorkAgentLoop(SciAccelAgentLoop):
    """Discard OOM attempts before reward computation and preserve each GRPO slot."""

    async def create_session(self, request):
        self._attempt_session = await super().create_session(request)
        return self._attempt_session

    async def _run_harbor_in_thread(self, *args, **kwargs):
        result = await super()._run_harbor_in_thread(*args, **kwargs)
        self._attempt_result = result
        self._pressure = load_pressure_evidence(Path(self.runtime_config.harbor.jobs_dir) / self._attempt_session)
        return result

    async def get_training_data(self, session_id):
        # Do not feed failed-attempt tokens to scoring or the optimizer.
        if self._pressure.get("status") == "cuda_oom":
            return []
        return await super().get_training_data(session_id)

    async def run(self, request):
        global _slots
        if _slots is None:
            _slots = asyncio.Semaphore(4)
        uid = str(request["uid"])
        root = Path(self.runtime_config.harbor.jobs_dir).parent / "attempts"
        root.mkdir(parents=True, exist_ok=True)
        slot_start = time.time()
        starts = root.parent / "slot-starts"
        starts.mkdir(exist_ok=True)
        marker = starts / f"{uid}.json"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"uid": uid, "started_at": slot_start}))
        temporary.replace(marker)
        for attempt in range(1, 13):
            self._pressure = {"status": "not_started"}
            self._attempt_result = None
            self._attempt_session = None
            started = time.time()
            async with _slots:
                admitted = time.time()
                output, reason = await super().run(request)
            result = self._attempt_result
            record = {
                "uid": uid,
                "parent_id": str(request.get("parent_id", uid)),
                "version_tag": request.get("version_tag"),
                "attempt": attempt,
                "session_id": self._attempt_session,
                "slot_started_at": slot_start,
                "started_at": started,
                "admitted_at": admitted,
                "finished_at": time.time(),
                "pressure": self._pressure,
                "termination": str(reason),
                "has_output": output is not None,
                "rewards": result.rewards if result else {},
                "exception_type": result.exception_type if result else "missing",
                "accepted": self._pressure.get("status") == "ok"
                and output is not None
                and bool(result and result.rewards)
                and result.exception_type is None,
            }
            import inspect

            import psrl
            import transfer_queue
            import verl

            record["source_paths"] = {
                "psrl": psrl.__file__,
                "verl": verl.__file__,
                "transfer_queue": transfer_queue.__file__,
                "science_loop": inspect.getfile(SciAccelAgentLoop),
            }
            (root / f"{uid}-{attempt:02d}.json").write_text(json.dumps(record, indent=2, default=str))
            print("FIXED_WORK_ATTEMPT " + json.dumps(record, default=str), flush=True)
            if record["accepted"]:
                return output, reason
            if self._pressure.get("status") != "cuda_oom":
                raise RuntimeError(f"Non-OOM infrastructure failure for slot {uid}; fixed work not complete.")
            if attempt < 12:
                # Both arms use equal deterministic backoff without holding the semaphore.
                await asyncio.sleep(12 + sum(uid.encode()) % 9)
        raise RuntimeError(f"OOM retry budget exhausted for slot {uid}; fixed work not complete.")
