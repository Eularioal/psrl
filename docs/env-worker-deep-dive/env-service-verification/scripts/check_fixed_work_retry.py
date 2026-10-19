"""CPU contract test of the experiment's real retry methods, with fake episodes."""

import ast
import asyncio
import json
import sys
import tempfile
import time
import types
from pathlib import Path


class EpisodeDouble:
    async def run(self, request):
        self.calls += 1
        self._pressure = {"status": self.statuses[min(self.calls - 1, len(self.statuses) - 1)]}
        self._attempt_result = types.SimpleNamespace(rewards={"reward_repair": 0}, exception_type=None)
        output = await self.get_training_data("fake-session")
        return output or None, "finished"

    async def get_training_data(self, session_id):
        self.scored += 1
        return ["valid tokens"]


async def no_wait(seconds):
    return None


async def main():
    source = Path(__file__).with_name("science_fixed_work_agent.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "FixedWorkAgentLoop")
    for name in ("psrl", "verl"):
        sys.modules[name] = types.SimpleNamespace(__file__="CPU-test-double")
    namespace = {
        "SciAccelAgentLoop": EpisodeDouble,
        "_slots": None,
        "json": json,
        "Path": Path,
        "time": time,
        "asyncio": types.SimpleNamespace(Semaphore=asyncio.Semaphore, sleep=no_wait),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "actual-retry-methods", "exec"), namespace)
    with tempfile.TemporaryDirectory() as directory:

        def instance(name, statuses):
            obj = namespace["FixedWorkAgentLoop"]()
            obj.calls = obj.scored = 0
            obj.statuses = statuses
            obj.runtime_config = types.SimpleNamespace(
                harbor=types.SimpleNamespace(jobs_dir=str(Path(directory) / name / "jobs"))
            )
            return obj

        obj = instance("recover", ["cuda_oom", "ok"])
        output, _ = await obj.run({"uid": 0, "parent_id": 0})
        assert output and obj.calls == 2 and obj.scored == 1
        records = [json.loads(p.read_text()) for p in (Path(directory) / "recover/attempts").glob("*.json")]
        assert sum(r["accepted"] for r in records) == 1
        for name, statuses, expected in [("exhaust", ["cuda_oom"], 12), ("infra", ["missing"], 1)]:
            obj = instance(name, statuses)
            try:
                await obj.run({"uid": 1, "parent_id": 0})
            except RuntimeError:
                pass
            else:
                raise AssertionError("An incomplete slot must fail, not count as completion.")
            assert obj.calls == expected
    print(
        json.dumps(
            {
                "scope": "CPU retry contract with fake episodes",
                "oom_tokens_not_scored": True,
                "replacement_preserves_slot": True,
                "attempt_limit": 12,
                "non_oom_failure_stops": True,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
