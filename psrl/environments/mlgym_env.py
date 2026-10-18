"""Adapt MLGym lifecycle, shell commands and grading to remotely owned sandboxes."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import shlex
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from examples.airs_bench.config import AirsBenchRuntimeConfig, build_runtime_config
from omegaconf import DictConfig

from psrl.environments.base import Environment, EnvStepOutput
from psrl.workers.env_worker.sandbox import ExecResult, SandboxSpec

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

REQUIRED_HISTORY_PROCESSOR = "DefaultHistoryProcessor"
AGENT_WORKSPACE = "/home/agent/workspace"


def assert_default_history_processor(agent_config_path: str) -> None:
    """
    Fail loudly unless an MLGym agent config pins the safe history processor.

    Any processor that rewrites earlier messages, such as `Last5Observations`,
    invalidates TITO's message prefix hashes. The corruption is silent, producing
    misaligned training tokens rather than an error, so it is checked up front.

    Args:
        agent_config_path (str): Path to the MLGym agent config yaml.

    Raises:
        ValueError: If the key is absent or set to anything else.
    """
    payload = yaml.safe_load(Path(agent_config_path).read_text()) or {}
    processor = payload.get("history_processor")
    if processor is None:
        raise ValueError(
            f"Agent config {agent_config_path!r} does not set history_processor, "
            f"which must be {REQUIRED_HISTORY_PROCESSOR} for TITO correctness."
        )
    if processor != REQUIRED_HISTORY_PROCESSOR:
        raise ValueError(
            f"Agent config {agent_config_path!r} sets history_processor to {processor!r}. "
            f"RL training requires {REQUIRED_HISTORY_PROCESSOR} because any processor that "
            "rewrites earlier messages silently corrupts TITO training data."
        )


def build_sandbox_spec(
    runtime_config: AirsBenchRuntimeConfig,
    dataset_data_path: str,
    labels: dict[str, str],
    verifier_spec: SandboxSpec | None = None,
) -> SandboxSpec:
    """Mount only named public splits into the agent container."""
    root = Path(dataset_data_path).resolve()
    mounts = []
    for split in ("train", "validation", "test"):
        path = root / split
        if split == "validation" and not path.exists():
            continue
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"Public split {split!r} must be an existing directory, not a symlink.")
        mounts.append((str(path), f"{AGENT_WORKSPACE}/data/{split}", "ro"))
    from psrl.workers.env_worker.coordinator import memory_bytes

    auxiliary_cpus = runtime_config.verifier_cpus
    auxiliary_memory = runtime_config.verifier_memory
    if verifier_spec is not None:
        auxiliary_cpus += verifier_spec.auxiliary_cpus
        if verifier_spec.auxiliary_memory != "0":
            auxiliary_memory = str(memory_bytes(auxiliary_memory) + memory_bytes(verifier_spec.auxiliary_memory))
    return SandboxSpec(
        image=runtime_config.image,
        cpus=runtime_config.sandbox_cpus,
        memory=runtime_config.sandbox_memory,
        mounts=tuple(mounts),
        network=runtime_config.sandbox_network,
        labels=labels,
        startup_timeout_s=runtime_config.startup_timeout_s,
        idle_timeout_s=runtime_config.lease_timeout_s,
        auxiliary_cpus=auxiliary_cpus,
        auxiliary_memory=auxiliary_memory,
    )


def sync_exec(
    handle: Any,
    loop: asyncio.AbstractEventLoop,
    command: str,
    timeout_s: float,
    no_output_timeout_s: float | None = None,
) -> ExecResult:
    """
    Run one sandbox command from synchronous code.

    MLGym's agent is synchronous and runs on a worker thread, while the sandbox
    handle is async and owned by the agent loop's event loop. This bridges the two
    by scheduling onto that loop and blocking the calling thread.

    Args:
        handle (Any): Sandbox handle.
        loop (asyncio.AbstractEventLoop): The loop the handle belongs to.
        command (str): Shell command.
        timeout_s (float): Seconds allowed for the command.
        no_output_timeout_s (float | None): Silent-seconds allowance.

    Returns:
        ExecResult: Command outcome.
    """
    future: Future = asyncio.run_coroutine_threadsafe(handle.exec(command, timeout_s, no_output_timeout_s), loop)
    # Add slack so the local wait never fires before the remote timeout reports.
    try:
        return future.result(timeout=timeout_s + 30.0)
    except BaseException:
        future.cancel()
        raise


@dataclass
class RemoteShell:
    """Expose the shell identity required by MLGym's agent logging."""

    pid: str

    def terminate(self) -> None:
        """End the episode rather than attempting local process cleanup."""
        raise RuntimeError("Remote shell termination ends the episode.")


def build_psrl_mlgym_env_class() -> type:
    """Implement MLGym's environment contract without local Docker access."""
    from mlgym.environment.env import MLGymEnv

    class PSRLMLGymEnv(MLGymEnv):
        def __init__(self, args, sandbox_handle, loop, per_action_timeout_s, grader):
            self._sandbox_handle = sandbox_handle
            self._loop = loop
            self._per_action_timeout_s = per_action_timeout_s
            self._grader = grader
            super().__init__(args, devices=["cpu_0"])

        def _reset_container(self) -> None:
            if self.container is not None:
                raise RuntimeError("A failed remote sandbox cannot be reset in place.")
            self.container = RemoteShell(self._sandbox_handle.sandbox_id)
            self.container_obj = None
            self.container_name = self._sandbox_handle.sandbox_id
            self._communicate(
                "mkdir -p /home/agent/commands && touch /home/agent/commands/__init__.py && "
                "export PATH=$PATH:/home/agent/commands"
            )
            if self.args.aliases_file is not None:
                self.add_commands(
                    [
                        {
                            "name": "bash_aliases.sh",
                            "type": "source_file",
                            "contents": Path(self.args.aliases_file).read_text(),
                        }
                    ]
                )

        def reset_container(self) -> None:
            """Require a new episode after any remote shell failure."""
            raise RuntimeError("A failed remote sandbox cannot be reset in place.")

        def reset(self, *, seed=None, options=None):
            self.current_step = 0
            self.task = self.task_entrypoint(
                seed=self.seed if seed is None else seed,
                args=copy.deepcopy(self.task_args),
                task_workspace=str(self.task_workspace),
                _communicate=self.communicate,
                _communicate_with_handling=self.communicate_with_handling,
            )
            self._reset_environment_variables()
            self.task.setup()
            for path in self.task.args.starter_code:
                source = Path(path)
                if source.is_dir():
                    raise ValueError("AIRS starter code must contain individual files.")
                self._write_file(str(self.task_workspace / source.name), source.read_bytes())
            self.communicate_with_handling(
                "source /home/agent/miniconda3/etc/profile.d/conda.sh && "
                "conda activate mlgym_generic && cd /home/agent/workspace",
                error_msg="Failed to activate the preinstalled AIRS environment.",
            )
            return {"observation": None}, {}

        def _write_file(self, path: str, data: bytes) -> None:
            future = asyncio.run_coroutine_threadsafe(self._sandbox_handle.write_file(path, data), self._loop)
            try:
                future.result(timeout=60)
            except BaseException:
                future.cancel()
                raise

        def add_commands(self, commands: list[dict]) -> None:
            for command in commands:
                name = command["name"]
                if Path(name).name != name:
                    raise ValueError("Command names must be basenames.")
                path = f"/home/agent/commands/{name}"
                self._write_file(path, command["contents"].encode())
                if command["type"] == "source_file":
                    self.communicate_with_handling(
                        f"source {shlex.quote(path)}",
                        error_msg="Failed to install command functions.",
                    )
                elif command["type"] == "script":
                    self.communicate_with_handling(
                        f"chmod +x {shlex.quote(path)}",
                        error_msg="Failed to install command script.",
                    )
                elif command["type"] != "utility":
                    raise ValueError(f"Invalid command type {command['type']!r}.")

        def _communicate(self, input: str, timeout_duration=25, no_output_timeout_duration=25) -> str:
            result = sync_exec(
                self._sandbox_handle,
                self._loop,
                input,
                min(float(timeout_duration), self._per_action_timeout_s),
                float(no_output_timeout_duration),
            )
            self.returncode = result.exit_code
            if result.timed_out:
                raise RuntimeError("Sandbox command timed out. The sandbox has been destroyed.")
            return result.stdout

        def interrupt(self) -> str:
            raise RuntimeError("Timed out remote sandboxes cannot be reused.")

        def _evaluate_with_error_handling(self, info, action):
            terminal = action != "validate"
            self._communicate("test -f /home/agent/workspace/submission.csv")
            if self.returncode != 0:
                if terminal:
                    info["exit_status"] = "submission_not_found"
                return "Create submission.csv before submitting.", 0, terminal, info
            try:
                metrics = self._grader()
            except ValueError:
                if terminal:
                    info["exit_status"] = "invalid_submission"
                return "Invalid submission. Check the documented CSV format.", 0, terminal, info
            info["score"].append(metrics)
            info["submission"] = "/home/agent/workspace/submission.csv"
            if terminal:
                info["exit_status"] = "submitted" if action == "submit" else action
            return f"Evaluation score: {metrics}", 0, terminal, info

        def close(self) -> None:
            """The episode owner releases the remote reservation."""

    return PSRLMLGymEnv


@Environment.register("mlgym_env")
class MLGymEnvironment(Environment[dict, None]):
    """
    PSRL environment adapter for AIRS-Bench tasks.

    Like `MiniSWEEnvironment`, this prepares per-episode state and hands it to the
    agent loop. `ActType` is None because MLGym's own agent owns the step loop, so
    `step` is never called.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: Any,
        tokenizer: Any = None,
        processor: Any = None,
        dataset_cls: Any = None,
        runtime_config: AirsBenchRuntimeConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, reward_manager, tokenizer, processor, dataset_cls)
        self._runtime_config = runtime_config or build_runtime_config(None)
        self._task_id: str | None = None
        self._episode_id: str | None = None

    async def reset(self, task: dict, **kwargs: Any) -> tuple[dict, dict]:
        """
        Prepare one AIRS-Bench episode.

        Args:
            task (dict): Dataset row, whose `extra_info` carries the task identity and
                its normalization constants.
            **kwargs (Any): Ignored framework extras.

        Returns:
            tuple[dict, dict]: Observation for the agent loop, plus an empty info dict.
        """
        extra_info = task.get("extra_info") or {}
        self._task_id = extra_info.get("airs_task_id")
        self._episode_id = f"{uuid.uuid4().hex[:12]}-{int(time.time())}"

        if not self._task_id:
            raise ValueError("Dataset row is missing extra_info.airs_task_id.")
        assert_default_history_processor(self._runtime_config.agent_config_path)
        from examples.airs_bench.grading import task_config

        task_config(self._runtime_config, extra_info["task_config_path"])

        observation = {
            "airs_task_id": self._task_id,
            "episode_id": self._episode_id,
            "task_config_path": extra_info["task_config_path"],
            "dataset_data_path": extra_info["dataset_data_path"],
            "metric": extra_info["metric"],
            "runtime_config": self._runtime_config,
        }
        return observation, {}

    async def step(self, action: None, **kwargs: Any) -> EnvStepOutput:
        """Never called. MLGym's own agent owns the step loop."""
        raise NotImplementedError("MLGymEnvironment does not implement step because MLGym's agent owns the loop.")

    async def close(self) -> None:
        """Remote resources belong to the agent loop reservation."""

    @property
    def state(self) -> Any:
        """Return the current episode identity."""
        return {"airs_task_id": self._task_id, "episode_id": self._episode_id}
