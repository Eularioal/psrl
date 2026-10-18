"""
Synchronous MLGym invocation inside a cancellable episode process.

MLGym's `BaseAgent` is used exactly as published. Only two things are supplied from
outside: its LLM `host_url`, which points at a PSRL session so TITO captures the
token stream, and its environment, which is a `PSRLMLGymEnv` whose container lives
in an env worker sandbox.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from examples.airs_bench.grading import grade_submission, task_config

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _ensure_mlgym_importable(mlgym_repo: str) -> None:
    """Put the read-only MLGym checkout on `sys.path` exactly once."""
    if mlgym_repo not in sys.path:
        sys.path.insert(0, mlgym_repo)


def build_mlgym_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Build MLGym model arguments that route generation through a PSRL session.

    MLGym's LiteLLM backend forwards `host_url` to litellm as `api_base`, so pointing
    it at the session URL is all that is required for TITO to observe every turn.

    Args:
        payload (dict[str, Any]): Runner payload with `base_url`, `model`,
            `sampling_params`, and `trajectory_id_strategy`.

    Returns:
        dict[str, Any]: Keyword arguments for MLGym's model construction.
    """
    sampling = dict(payload["sampling_params"])
    top_k = sampling.pop("top_k", None)

    completion_kwargs: dict[str, Any] = {
        "max_tokens": int(sampling.get("max_tokens", 4096)),
        "logprobs": True,
        "top_logprobs": 1,
    }
    if top_k is not None and int(top_k) >= 0:
        completion_kwargs["extra_body"] = {"top_k": int(top_k)}
    if payload.get("trajectory_id_strategy", "manual") == "manual":
        completion_kwargs["extra_headers"] = {"x-smg-tito-trajectory-id": "0"}

    return {
        "model_name": f"openai/{payload['model']}",
        "host_url": payload["base_url"],
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "completion_kwargs": completion_kwargs,
    }


def collect_scores_from_info(
    info: Any,
    submit_count: int,
    validate_count: int,
) -> dict[str, Any]:
    """
    Extract graded scores from MLGym's `AgentInfo`.

    `AgentInfo.score` is a list because MLGym appends one entry per submit or
    validate call. The reward function consumes the last entry.

    Args:
        info (Any): MLGym `AgentInfo`.
        submit_count (int): Observed submit actions.
        validate_count (int): Observed validate actions.

    Returns:
        dict[str, Any]: Fields merged into `extra_info` for the reward function.
    """
    scores = list(getattr(info, "score", []) or [])
    if getattr(info, "exit_status", "") in {
        "invalid_submission",
        "submission_not_found",
        "exit_forfeit",
        "skipped",
        "skip",
    }:
        scores = []
    return {
        "airs_scores": scores,
        "airs_submit_count": submit_count,
        "airs_validate_count": validate_count,
    }


def prepare_mlgym_task(payload: dict, directory: Path) -> Path:
    """Translate AIRS asset paths into the upstream MLGym task schema."""
    from dataclasses import fields

    from mlgym.environment.tasks import TaskConfig

    raw, root = task_config(payload["runtime_config"], payload["task_config_path"])
    allowed = {field.name for field in fields(TaskConfig) if field.init}
    task = {key: value for key, value in raw.items() if key in allowed}
    for key in ("dataset_configs", "starter_code"):
        task[key] = [str((root / path).resolve()) for path in task.get(key, [])]
    task["use_generic_conda"] = True
    task["requirements_path"] = None
    task["memory_path"] = None
    path = directory / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    return path


def run_mlgym_agent(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Run one full MLGym episode synchronously.

    Args:
        payload (dict[str, Any]): Everything the episode needs, including `base_url`,
            `model`, `sampling_params`, `task_config_path`, `runtime_config`,
            `sandbox_handle`, `event_loop`, and `max_turns`.

    Returns:
        dict[str, Any]: Episode outcome with `exit_status`, graded scores, submit and
            validate counts, any evaluation error, turn count, and timing.
    """
    runtime_config = payload["runtime_config"]
    _ensure_mlgym_importable(runtime_config.mlgym_repo)

    started_at = time.monotonic()
    outcome: dict[str, Any] = {
        "exit_status": "exit_error",
        "airs_scores": [],
        "airs_submit_count": 0,
        "airs_validate_count": 0,
        "airs_eval_error": "",
        "turns": 0,
        "timing": {},
    }

    env = None
    try:
        from mlgym.agent.base import AgentArguments, BaseAgent
        from mlgym.backend.base import ModelArguments
        from mlgym.environment.env import EnvironmentArguments
        from psrl.environments.mlgym_env import build_psrl_mlgym_env_class

        directory = Path(payload["episode_dir"])
        prepared = prepare_mlgym_task(payload, directory)
        env_args = EnvironmentArguments(
            image_name=runtime_config.image,
            max_steps=int(payload["max_turns"]),
            task_config_path=str(prepared),
            container_type="docker",
            verbose=False,
            cache_baseline_scores=False,
        )

        def grader():
            future = asyncio.run_coroutine_threadsafe(
                grade_submission(
                    payload["sandbox_handle"],
                    payload["verifier_spec"],
                    payload["verifier_command"],
                    payload["metric"],
                    runtime_config.per_action_timeout_s,
                ),
                payload["event_loop"],
            )
            try:
                return future.result(timeout=runtime_config.per_action_timeout_s + 60)
            except BaseException:
                future.cancel()
                raise

        env_class = build_psrl_mlgym_env_class()
        env = env_class(
            env_args,
            sandbox_handle=payload["sandbox_handle"],
            loop=payload["event_loop"],
            per_action_timeout_s=runtime_config.per_action_timeout_s,
            grader=grader,
        )

        model_args = ModelArguments(**build_mlgym_model_config(payload))
        agent_args = AgentArguments(
            model=model_args,
            agent_config_path=runtime_config.agent_config_path,
        )
        agent = BaseAgent("primary", agent_args)

        observation, _ = env.reset()
        info, trajectory = agent.run(
            env=env,
            observation=observation,
            traj_dir=directory,
            return_type="info_trajectory",
        )

        submit_count = sum(1 for step in trajectory if "submit" in str(step.get("action", "")))
        validate_count = sum(1 for step in trajectory if "validate" in str(step.get("action", "")))
        outcome.update(collect_scores_from_info(info, submit_count, validate_count))
        outcome["exit_status"] = getattr(info, "exit_status", "") or "exit_error"
        outcome["turns"] = len(trajectory)
    except Exception as error:  # noqa: BLE001
        psrl_logger.exception("MLGym episode failed.")
        outcome["exit_status"] = "exit_error"
        outcome["airs_eval_error"] = f"runner_error: {type(error).__name__}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                psrl_logger.warning("MLGym env close failed.", exc_info=True)
        outcome["timing"] = {"episode_s": time.monotonic() - started_at}

    return outcome
