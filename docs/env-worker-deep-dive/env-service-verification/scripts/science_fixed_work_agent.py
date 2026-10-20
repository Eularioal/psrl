"""Same bounded OOM refill policy for the historical and current science loops."""

import asyncio
import json
import time
from pathlib import Path

from examples.sciaccel_rl.agent_loop import SciAccelAgentLoop
from science_pressure_agent import PressureTerminus2

_slots = None


class FixedWorkPressureAgent(PressureTerminus2):
    """Use the same 50 GiB / 45 second phase as the preceding experiment."""


class CommandPressureAgent(FixedWorkPressureAgent):
    """Return GPU command capacity while the science repair agent continues."""

    async def _exec_pressure(self, command, **kwargs):
        environment = self._pressure_environment
        try:
            return await environment.exec_gpu(command, wait_timeout_sec=300, **kwargs)
        finally:
            (self.logs_dir / "gpu-command-phase.json").write_text(json.dumps(environment.last_gpu_command, indent=2))


class FixedWorkAgentLoop(SciAccelAgentLoop):
    """Discard OOM attempts before reward computation; preserve each GRPO slot."""

    async def create_session(self, request):
        self._attempt_session = await super().create_session(request)
        return self._attempt_session

    async def _run_harbor_in_thread(self, *args, **kwargs):
        result = await super()._run_harbor_in_thread(*args, **kwargs)
        self._attempt_result = result
        paths = list(
            (Path(self.runtime_config.harbor.jobs_dir) / self._attempt_session).glob(
                "*/*/artifacts/logs/artifacts/gpu-pressure.json"
            )
        )
        self._pressure = json.loads(paths[0].read_text()) if len(paths) == 1 else {"status": "missing"}
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
                and bool(result and result.rewards),
            }
            import inspect

            import psrl
            import verl

            record["source_paths"] = {
                "psrl": psrl.__file__,
                "verl": verl.__file__,
                "science_loop": inspect.getfile(SciAccelAgentLoop),
            }
            (root / f"{uid}-{attempt:02d}.json").write_text(json.dumps(record, indent=2, default=str))
            print("FIXED_WORK_ATTEMPT " + json.dumps(record, default=str), flush=True)
            if record["accepted"]:
                return output, reason
            if self._pressure.get("status") != "cuda_oom":
                raise RuntimeError(f"Non-OOM infrastructure failure for slot {uid}; fixed work not complete.")
            if attempt < 12:
                # Equal deterministic backoff in both arms; the semaphore is not held.
                await asyncio.sleep(12 + sum(uid.encode()) % 9)
        raise RuntimeError(f"OOM retry budget exhausted for slot {uid}; fixed work not complete.")
