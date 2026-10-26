"""Run original science tasks without injected commands or environment GPU demand."""

import asyncio
import hashlib
import inspect
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import psrl
import transfer_queue
import verl
from examples.sciaccel_rl.agent_loop import SciAccelAgentLoop

_slots = None


def prepare_native(args, out):
    """Keep task bytes unchanged while isolating paths and recording source hashes."""
    names = [
        "sciaccel/laps-repair-coef-2d-rktmod-l17k1",
        "sciaccel/laps-repair-subtle-2d-rktmod-l17k1",
        "sciaccel/laps-repair-coef-2d-rktmod-l18k4",
        "sciaccel/laps-repair-subtle-2d-rktmod-l18k3",
    ]
    frame = pd.read_parquet(args.dataset).set_index("task_name").loc[names].reset_index()
    manifest = {}
    for i, row in frame.iterrows():
        info = dict(row.extra_info)
        if int(info.get("gpus", 0)):
            raise ValueError("The native comparison requires the original CPU task declarations.")
        source = Path(info["task_path"])
        target = out / "tasks" / source.name
        if not target.exists():
            shutil.copytree(source, target)
        hashes = {}
        for path in sorted(source.rglob("*")):
            if path.is_file():
                relative = path.relative_to(source)
                content = path.read_bytes()
                if content != (target / relative).read_bytes():
                    raise ValueError(f"Task copy differs from its original: {relative!s}.")
                hashes[str(relative)] = hashlib.sha256(content).hexdigest()
        manifest[row.task_name] = {"source": str(source), "copy": str(target), "sha256": hashes}
        info["task_path"] = str(target)
        frame.at[i, "extra_info"] = info
        frame.at[i, "agent_name"] = f"science-pool-{i}"
    repeated = pd.concat([frame] * args.steps, ignore_index=True)
    for split in ("train", "eval"):
        directory = out / "data" / split
        directory.mkdir(parents=True, exist_ok=True)
        repeated.to_parquet(directory / "L1.parquet", index=False)
    design = {
        "scope": "Original CPU LAPS repair with real model inference and training. No injected CUDA or sleep.",
        "workload": "native",
        "task_names": names,
        "steps_per_arm": args.steps,
        "batch_size": 4,
        "rollout_n": args.rollout_n,
        "accepted_rollouts": 4 * args.rollout_n * args.steps,
        "common_active_attempt_limit": 4,
        "max_attempts_per_slot": 1,
        "model_gpus": [0, 1, 2],
        "environment_gpus": [],
        "cpu_unit": {"cpu": 16, "memory_mb": 32768, "max_containers": 4},
        "max_turns": args.turns,
        "response_length": args.response_length,
        "task_timeout_s": args.task_timeout,
        "initialization_timeout_s": args.initialization_timeout,
        "work_timeout_s": args.work_timeout,
        "timeout_per_arm_s": args.timeout,
        "modes": args.modes,
        "baseline_repo": str(args.baseline_repo),
        "current_repo": str(args.repo),
        "data_shuffle": False,
        "seed": 17,
        "comparison": "Fixed historical OFF and frozen Router ON. No GPU packing claim for CPU tasks.",
    }
    (out / "design.json").write_text(json.dumps(design, indent=2))
    (out / "task-integrity.json").write_text(json.dumps(manifest, indent=2))
    return ""


class NativeWorkAgentLoop(SciAccelAgentLoop):
    """Observe original task results with equal concurrency and no injected commands."""

    async def create_session(self, request):
        self._attempt_session = await super().create_session(request)
        return self._attempt_session

    async def _run_harbor_in_thread(self, *args, **kwargs):
        self._attempt_result = await super()._run_harbor_in_thread(*args, **kwargs)
        return self._attempt_result

    async def run(self, request):
        global _slots
        if _slots is None:
            _slots = asyncio.Semaphore(4)
        uid = str(request["uid"])
        root = Path(self.runtime_config.harbor.jobs_dir).parent
        starts, attempts = root / "slot-starts", root / "attempts"
        starts.mkdir(parents=True, exist_ok=True)
        attempts.mkdir(parents=True, exist_ok=True)
        started = time.time()
        marker = starts / f"{uid}.json"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"uid": uid, "started_at": started}))
        temporary.replace(marker)
        self._attempt_result = None
        self._attempt_session = None
        async with _slots:
            admitted = time.time()
            output, reason = await super().run(request)
        result = self._attempt_result
        record = {
            "uid": uid,
            "parent_id": str(request.get("parent_id", uid)),
            "version_tag": request.get("version_tag"),
            "attempt": 1,
            "session_id": self._attempt_session,
            "slot_started_at": started,
            "admitted_at": admitted,
            "finished_at": time.time(),
            "pressure_injected": False,
            "termination": str(reason),
            "has_output": output is not None,
            "rewards": result.rewards if result else {},
            "exception_type": result.exception_type if result else "missing",
            "accepted": output is not None and bool(result and result.rewards) and result.exception_type is None,
            "source_paths": {
                "psrl": psrl.__file__,
                "verl": verl.__file__,
                "transfer_queue": transfer_queue.__file__,
                "science_loop": inspect.getfile(SciAccelAgentLoop),
            },
        }
        (attempts / f"{uid}-01.json").write_text(json.dumps(record, indent=2, default=str))
        if not record["accepted"]:
            raise RuntimeError(f"Native science slot {uid} failed. No automatic refill is allowed.")
        return output, reason
