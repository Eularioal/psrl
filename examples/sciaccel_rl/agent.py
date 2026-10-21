"""
A terminus-2 variant whose terminal observations are truncated more aggressively.

Harbor caps observations at 10000 bytes inside `Terminus2._limit_output_length`, as a
hardcoded default that `AgentConfig.kwargs` cannot reach. Graded sources run 15 KB to
40 KB, so the cap decides how much of the context window one careless read consumes.
Subclassing keeps the override in this repository.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from harbor.agents.terminus_2 import Terminus2
from psrl.utils.agent.overflow import is_prompt_overflow

psrl_logger = logging.getLogger("psrl.sciaccel_rl.agent")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

HARBOR_DEFAULT_MAX_OBS_BYTES = 10000


class ContextWindowExhausted(Exception):
    """
    Raised when the served prompt no longer fits the model window.

    Terminus-2 does not recognize vLLM's 400 wording, so it treats the rejection as
    transient and keeps issuing turns that re-send an already-too-long transcript.
    """


class TruncatingTerminus2(Terminus2):
    """
    Terminus-2 with a configurable observation byte cap.

    Harbor calls `_limit_output_length` in five places and always relies on its
    default argument, so overriding the default is enough to move every call
    site. The signature keeps `max_bytes` so an explicit caller still wins.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_observation_bytes: int = HARBOR_DEFAULT_MAX_OBS_BYTES,
        gpu_lending: dict | None = None,
        gpu_jobs: bool = False,
        **kwargs,
    ):
        """
        Args:
            logs_dir (Path): Directory harbor stores agent logs in.
            model_name (str | None): Model name passed through to terminus-2.
            max_observation_bytes (int): Byte cap for a single terminal
                observation. Harbor's default is 10000.
        """
        if max_observation_bytes <= 0:
            raise ValueError(f"max_observation_bytes must be positive, got {max_observation_bytes}.")
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._gpu_phase_client = None
        self._gpu_jobs_enabled = gpu_jobs
        self._gpu_job_environment = None
        if gpu_lending:
            import ray
            from psrl.workers.env_service.lending import GpuPhaseClient

            self._gpu_phase_client = GpuPhaseClient(ray.get_actor(gpu_lending["router_name"]), gpu_lending["lease_id"])
        self._max_observation_bytes = int(max_observation_bytes)
        psrl_logger.info(
            "TruncatingTerminus2 capping observations at %d bytes (harbor default %d).",
            self._max_observation_bytes,
            HARBOR_DEFAULT_MAX_OBS_BYTES,
        )

    def _limit_output_length(self, output: str, max_bytes: int | None = None) -> str:
        """
        Truncate one observation, defaulting to this agent's configured cap.

        Args:
            output (str): Raw terminal output.
            max_bytes (int | None): Explicit cap. None uses the configured one.

        Returns:
            str: The output, truncated in the middle when it exceeds the cap.
        """
        return super()._limit_output_length(
            output,
            self._max_observation_bytes if max_bytes is None else max_bytes,
        )

    async def run(self, instruction, environment, context):
        if self._gpu_jobs_enabled:
            from examples.sciaccel_rl.gpu_tools import GPU_JOB_INSTRUCTIONS

            self._gpu_job_environment = environment
            instruction += "\n" + GPU_JOB_INSTRUCTIONS
        await super().run(instruction, environment, context)
        if self._gpu_jobs_enabled:
            await environment.finish_gpu_jobs()

    async def _execute_commands(self, commands, session):
        if not self._gpu_jobs_enabled:
            return await super()._execute_commands(commands, session)
        from examples.sciaccel_rl.gpu_tools import dispatch_gpu_tool

        observations = []
        for command in commands:
            text = command.keystrokes.strip()
            if text == "psrl-gpu" or text.startswith("psrl-gpu "):
                try:
                    result = await dispatch_gpu_tool(self._gpu_job_environment, text)
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    result = {"error": str(exc)}
                observations.append(json.dumps(result))
            else:
                timed_out, observation = await super()._execute_commands([command], session)
                observations.append(observation)
                if timed_out:
                    return True, self._limit_output_length("\n".join(observations))
        return False, self._limit_output_length("\n".join(observations))

    async def _query_llm(self, *args, **kwargs):
        """
        Query the model, ending the episode when the window is exhausted.

        A prompt overflow is terminal by nature: the transcript only grows, so retrying
        cannot succeed. Raising a distinct exception stops the turn loop immediately
        instead of letting it spin until the agent timeout. The partial trajectory is
        still graded, because `SciAccelAgentLoop` recognizes the overflow and trains the
        turns captured before it.
        """
        phase_client = getattr(self, "_gpu_phase_client", None)
        if phase_client is not None:
            await phase_client.set_phase("model")
        try:
            return await super()._query_llm(*args, **kwargs)
        except Exception as exc:
            if is_prompt_overflow(exc):
                psrl_logger.warning("Context window exhausted, ending the episode: %s", exc)
                raise ContextWindowExhausted(str(exc)) from exc
            raise
        finally:
            if phase_client is not None:
                await phase_client.set_phase("env")
