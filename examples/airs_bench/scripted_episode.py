"""Exercise sandbox commands and isolated CSV grading with fixed actions and no LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any

from examples.airs_bench.config import AirsBenchRuntimeConfig, build_runtime_config
from examples.airs_bench.grading import build_verifier_spec, grade_submission
from psrl.environments.mlgym_env import AGENT_WORKSPACE, build_sandbox_spec
from psrl.workers.env_worker.coordinator import acquire_sandbox

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

# The container login shell does not activate Conda, so use the environment's
# absolute Python path.
_PYTHON = "/home/agent/miniconda3/envs/mlgym_generic/bin/python"

# This sandbox script infers the required submission shape from the task data.
_MAKE_SUBMISSION_SRC = """\
import csv, json, os
from datasets import load_from_disk

ws = "/home/agent/workspace"
test_path = os.path.join(ws, "data", "test")
queries_path = os.path.join(test_path, "queries")
is_retrieval = os.path.isdir(queries_path) and os.path.isfile(
    os.path.join(queries_path, "dataset_info.json")
)
ds = load_from_disk(queries_path if is_retrieval else test_path)
n = len(ds)
cols = ds.column_names
# CodeGenerationAPPSPassAt5 is uniquely identified by a `starter_code` column.
is_code_gen = not is_retrieval and "starter_code" in cols
# DuoRC needs answer + has_answer columns; identified by `no_answer` in input.
is_duorc = not is_retrieval and "no_answer" in cols
# Eli5 needs free-form string answers; identified by `answers_urls` in input.
is_eli5 = not is_retrieval and "answers_urls" in cols
# Time series tasks store a `target` column that is a list of floats.
is_ts = not is_retrieval and "target" in cols and isinstance(ds[0]["target"], list)
prediction_length = __PREDICTION_LENGTH__
if is_ts and prediction_length <= 0:
    raise ValueError("Provide --prediction-length from the public task description.")

out = open(os.path.join(ws, "submission.csv"), "w", newline="")
writer = csv.writer(out)
if is_code_gen:
    writer.writerow(["code1", "code2", "code3", "code4", "code5"])
    for _ in range(n):
        writer.writerow(["print(42)"] * 5)
elif is_retrieval:
    writer.writerow(["query", "rankings"])
    for i in range(n):
        writer.writerow([ds[i]["query"], json.dumps([])])
elif is_duorc:
    writer.writerow(["answer", "has_answer"])
    for _ in range(n):
        writer.writerow(["unknown", 0])
elif is_eli5:
    writer.writerow(["prediction"])
    for _ in range(n):
        writer.writerow(["unknown"])
elif is_ts:
    writer.writerow(["prediction"])
    dummy = str([0.0] * prediction_length)
    for _ in range(n):
        writer.writerow([dummy])
else:
    writer.writerow(["prediction"])
    for _ in range(n):
        writer.writerow([0])
out.close()
print("wrote", n, "rows")
"""

# A deliberately dumb but valid solution: write a format-aware dummy submission,
# then grade it. It should score poorly and still grade cleanly.
SCRIPTED_ACTIONS: list[str] = [
    f"cd {AGENT_WORKSPACE}",
    "ls -la",
    "ls -la data",
    (
        f'{_PYTHON} -c "'
        "from datasets import load_from_disk; import os; "
        f"ws = '{AGENT_WORKSPACE}'; "
        "qp = os.path.join(ws, 'data', 'test', 'queries'); "
        "ok = os.path.isdir(qp) and os.path.isfile(os.path.join(qp, 'dataset_info.json')); "
        "ds = load_from_disk(qp if ok else os.path.join(ws, 'data', 'test')); "
        "print('rows', len(ds)); "
        "print('cols', ds.column_names)\""
    ),
    f"{_PYTHON} {AGENT_WORKSPACE}/_make_submission.py",
    "ls -la submission.csv",
]


async def run_scripted_episode(
    task_id: str,
    runtime_config: AirsBenchRuntimeConfig,
    coordinator: Any,
    prediction_length: int = 0,
) -> dict[str, Any]:
    """
    Drive one sandbox through the scripted action sequence.

    Args:
        task_id (str): AIRS-Bench task identifier.
        runtime_config (AirsBenchRuntimeConfig): Recipe runtime settings.
        coordinator (Any): Env worker coordinator actor handle.

    Returns:
        dict[str, Any]: Result with `actions_run`, `submission_found`, `score`, and
            per-action observations.
    """
    dataset_data_path = os.path.join(runtime_config.data_root, "airs_prepared", task_id)
    verifier_spec, verifier_command = build_verifier_spec(
        runtime_config,
        dataset_data_path,
        f"tasks/{task_id}.yaml",
        {"psrl.airs_task_id": task_id},
    )
    spec = build_sandbox_spec(
        runtime_config,
        dataset_data_path=dataset_data_path,
        labels={"psrl.airs_task_id": task_id, "psrl.airs_scripted": "true"},
        verifier_spec=verifier_spec,
    )
    handle = await acquire_sandbox(coordinator, spec, runtime_config.episode_timeout_s)

    observations: list[dict[str, Any]] = []
    submission_found = False
    score: dict[str, Any] | None = None
    try:
        source = _MAKE_SUBMISSION_SRC.replace("__PREDICTION_LENGTH__", str(int(prediction_length)))
        await handle.write_file(f"{AGENT_WORKSPACE}/_make_submission.py", source.encode())

        for action in SCRIPTED_ACTIONS:
            result = await handle.exec(action, runtime_config.per_action_timeout_s)
            observations.append(
                {
                    "action": action,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "stdout_tail": result.stdout[-400:],
                }
            )
            psrl_logger.info(
                f"[{task_id}] action exit={result.exit_code} timed_out={result.timed_out}: {action[:70]!r}"
            )
            if result.timed_out or result.exit_code != 0:
                raise RuntimeError(f"Scripted action failed: {action!r}.")

        check = await handle.exec(f"test -f {AGENT_WORKSPACE}/submission.csv", 60.0)
        submission_found = check.exit_code == 0

        score = await grade_submission(
            handle,
            verifier_spec,
            verifier_command,
            None,
            runtime_config.per_action_timeout_s,
        )
    finally:
        await handle.destroy()

    return {
        "task_id": task_id,
        "actions_run": len(observations),
        "submission_found": submission_found,
        "score": score,
        "observations": observations,
    }


async def _main_async(task_id: str, prediction_length: int) -> None:
    import ray
    from omegaconf import OmegaConf
    from psrl.workers.env_worker.manager import EnvWorkerManager

    ray.init(address="auto", ignore_reinit_error=True)
    runtime_config = build_runtime_config(None)
    config = OmegaConf.create(
        {
            "psrl": {
                "env_worker": {
                    "enable": True,
                    "placement": "colocated",
                    "dedicated_node_ips": [],
                    "workers_per_node": 1,
                    "cpu_slots_per_worker": 2,
                    "gpu_slots_per_worker": 0,
                    "worker_num_cpus": 1,
                    "routing": {"method": "least_loaded"},
                    "exec_default_timeout_s": 3600,
                    "max_observation_chars": 8000,
                    "coordinator_max_concurrency": 16,
                }
            }
        }
    )
    manager = EnvWorkerManager(config)
    try:
        result = await asyncio.wait_for(
            run_scripted_episode(task_id, runtime_config, manager.coordinator_handle(), prediction_length),
            runtime_config.episode_timeout_s,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2))
        assert result["submission_found"], "The scripted episode produced no submission.csv."
        assert result["score"] is not None, "The task's evaluate.py did not return a score."
        print("SCRIPTED EPISODE OK")
    finally:
        manager.shutdown()


def main() -> None:
    """Run one scripted episode against a live Ray cluster."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run a scripted AIRS-Bench episode.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--prediction-length", type=int, default=0, help="CSV list length from the public task spec.")
    args = parser.parse_args()
    asyncio.run(_main_async(args.task_id, args.prediction_length))


if __name__ == "__main__":
    main()
