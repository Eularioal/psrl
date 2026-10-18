"""
Agent loop that runs MLGym as a black box against AIRS-Bench tasks.

MLGym's agent drives the episode over a session-scoped OpenAI-compatible URL, so
TITO records every turn's tokens and logprobs server side. This loop only sets the
episode up, waits for it, and turns the graded outcome into reward inputs.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import ray
from examples.airs_bench.config import AirsBenchRuntimeConfig, build_runtime_config
from examples.airs_bench.grading import build_verifier_spec
from examples.airs_bench.process_runner import run_in_process

from psrl.environments import Environment
from psrl.environments.mlgym_env import build_sandbox_spec
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason, register
from psrl.workers.env_worker.coordinator import acquire_sandbox
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_EXIT_STATUS_TO_REASON: dict[str, TerminateReason] = {
    "submitted": TerminateReason.FINISHED,
    "submission_not_found": TerminateReason.FINISHED,
    "invalid_submission": TerminateReason.FINISHED,
    "submit": TerminateReason.FINISHED,
    "exit_context": TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
    "exit_cost": TerminateReason.MAX_TURNS_EXCEEDED,
    "max_steps": TerminateReason.MAX_TURNS_EXCEEDED,
    "exit_forfeit": TerminateReason.FINISHED,
    "skip": TerminateReason.ABORTED,
    "exit_error": TerminateReason.ROLLOUT_ERROR,
    "exit_api": TerminateReason.ROLLOUT_ERROR,
    "exit_format": TerminateReason.ROLLOUT_ERROR,
}


def classify_exit_status(exit_status: str) -> TerminateReason:
    """
    Map an MLGym exit status onto a PSRL terminal reason.

    Statuses that indicate infrastructure trouble map to `ROLLOUT_ERROR` so the group
    is re-dispatched rather than teaching the policy that its actions were bad.

    Args:
        exit_status (str): Value from MLGym's `AgentInfo.exit_status`.

    Returns:
        TerminateReason: Matching terminal reason, `UNKNOWN` when unrecognized.
    """
    return _EXIT_STATUS_TO_REASON.get(exit_status, TerminateReason.UNKNOWN)


@register("mlgym_agent")
class MLGymAgentLoop(SessionAgentLoop):
    """Run one MLGym episode per request and collect its TITO trajectory."""

    def __init__(
        self,
        context: AgentLoopContext,
        runtime: Any = None,
        **kwargs: Any,
    ):
        super().__init__(context=context)
        self.runtime_config: AirsBenchRuntimeConfig = build_runtime_config(runtime)
        multi_turn = context.config.gen_actor_rollout_ref.rollout.multi_turn
        if not getattr(multi_turn, "enable", False):
            raise ValueError("The MLGym agent loop requires rollout.multi_turn.enable=True.")
        if not context.config.psrl.env_worker.enable:
            raise ValueError("The MLGym agent loop requires psrl.env_worker.enable=True.")
        self._coordinator = ray.get_actor("env_worker_coordinator")

    def get_generate_fields(self) -> list[str]:
        """Include the AIRS-Bench task identity carried in the dataset row."""
        return [*super().get_generate_fields(), "extra_info"]

    async def _run_episode_logic(self, request: dict, observation: dict, session_id: str, handle: Any) -> dict:
        """Run MLGym after sandbox admission and session creation."""
        verifier_spec, verifier_command = build_verifier_spec(
            self.runtime_config,
            observation["dataset_data_path"],
            observation["task_config_path"],
            {"psrl.airs_episode_id": observation["episode_id"]},
        )
        payload = {
            "base_url": self.session_api_url(session_id),
            "model": self.model_config.path,
            "sampling_params": self.get_session_sampling_params(request),
            "task_config_path": observation["task_config_path"],
            "runtime_config": self.runtime_config,
            "max_turns": self.max_turns,
            "trajectory_id_strategy": self.trajectory_id_strategy,
            "verifier_spec": verifier_spec,
            "verifier_command": verifier_command,
            "metric": observation["metric"],
        }
        return await run_in_process(payload, handle)

    async def run(
        self,
        request: dict,
    ) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """
        Run one AIRS-Bench episode and report its graded outcome.

        Creates the environment, resets it to obtain the sandbox spec, creates a TITO
        session, drives the MLGym runner, fetches TITO training data, and scores the
        trajectory. On timeout the method returns immediately with
        `TRAJECTORY_TIMEOUT` without fetching TITO data.

        Args:
            request (dict): Dataset row plus routing metadata.

        Returns:
            tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
                The scored trajectory output and the terminal reason.
                Returns `(None, TerminateReason.TRAJECTORY_TIMEOUT)` on timeout,
                `(None, TerminateReason.ROLLOUT_ERROR)` when training data is absent,
                and `(None, TerminateReason.ABORTED)` when reward scoring rejects the
                trajectory.
        """
        env_class = request.get("env_class") or self.config.gen_actor_rollout_ref.rollout.agent.env.name
        env = Environment.get_environment(
            env_class,
            self.config,
            self.reward_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            runtime_config=self.runtime_config,
        )
        handle = None

        async def episode():
            nonlocal handle
            observation, _ = await env.reset(task=request)
            verifier_spec, _ = build_verifier_spec(
                self.runtime_config,
                observation["dataset_data_path"],
                observation["task_config_path"],
                {},
            )
            spec = build_sandbox_spec(
                self.runtime_config,
                observation["dataset_data_path"],
                {"psrl.airs_task_id": observation["airs_task_id"], "psrl.airs_episode_id": observation["episode_id"]},
                verifier_spec=verifier_spec,
            )
            handle = await acquire_sandbox(self._coordinator, spec, self.runtime_config.episode_timeout_s)
            async with self.session_scope(request) as session_id:
                result = await self._run_episode_logic(request, observation, session_id, handle)
                if classify_exit_status(result["exit_status"]) in {
                    TerminateReason.ROLLOUT_ERROR,
                    TerminateReason.UNKNOWN,
                }:
                    return None, TerminateReason.ROLLOUT_ERROR

                psrl_logger.info(
                    f"AIRS-Bench episode: task={observation['airs_task_id']!r}, "
                    f"status={result['exit_status']!r}, turns={result['turns']}, "
                    f"elapsed={result['timing'].get('episode_s', 0.0):.0f}s."
                )

                training_data_list = await self.get_training_data(session_id)
                if not training_data_list or any(
                    item["num_turns"] == 0 or not item["response_ids"] for item in training_data_list
                ):
                    return None, TerminateReason.ROLLOUT_ERROR

                extra_fields: dict[str, Any] = {
                    "airs_task_id": observation["airs_task_id"],
                    "airs_scores": result["airs_scores"],
                    "airs_submit_count": result["airs_submit_count"],
                    "airs_validate_count": result["airs_validate_count"],
                    "airs_eval_error": result["airs_eval_error"],
                    "airs_episode_s": result["timing"].get("episode_s", 0.0),
                    "num_turns": result["turns"],
                }
                outputs = [self.build_token_output(item, extra_fields=extra_fields) for item in training_data_list]
                output: TokenOutput | list[TokenOutput] = outputs[0] if len(outputs) == 1 else outputs
                scored_output = await self.compute_reward_score(output, **request)
                if scored_output is None:
                    return None, TerminateReason.ABORTED
                return scored_output, classify_exit_status(result["exit_status"])

        try:
            return await asyncio.wait_for(episode(), self.runtime_config.episode_timeout_s)
        except TimeoutError:
            return None, TerminateReason.TRAJECTORY_TIMEOUT
        finally:
            try:
                if handle is not None:
                    await handle.destroy()
            finally:
                await env.close()
