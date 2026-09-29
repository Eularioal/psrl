"""
Harbor Job execution wrapper for SciAccel-RL.

Runs one Harbor episode (agent container + verifier) and returns the shaped
reward from the verifier. The model endpoint is pointed at PSRL's
SessionRouter so TITO captures all tokens.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from examples.sciaccel_rl.config import SciAccelRuntimeConfig
from harbor.job import Job
from harbor.models.job.config import AgentConfig, JobConfig, SourceJobConfig
from harbor.models.trial.config import TaskConfig
from psrl.utils.agent.thinking import MULTI_TRAJ, harness_extra_body
from psrl.utils.common.docker_utils import (
    CLEANUP_EXECUTOR,
    force_remove_compose_images,
    force_remove_compose_project,
    prune_dangling_images,
)
from psrl.utils.common.http_utils import post
from psrl.utils.concurrency.gpu_device_gate import AllocatedGpu, GpuDeviceGate, build_gpu_override

psrl_logger = logging.getLogger("psrl.sciaccel_rl.runner")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Tokens held back from the window advertised to terminus-2, which counts with litellm's
# estimator rather than the served tokenizer and so drifts low.
_CONTEXT_SAFETY_MARGIN = 8192


async def _bind_gate_b(session_router_url: str, session_id: str, device: str | None) -> None:
    """
    Best-effort: tell SessionRouter which physical GPU device (if any) this
    session is using, so Gate B (psrl/workers/gen/session_router.py's
    `control_gpu_bind` + `_wait_for_gpu_idle`) can hold the model's reply
    until that device looks idle before letting the agent issue its next
    tool command. Called once when Gate A reserves a device, and again with
    `device=None` to unbind once the episode's containers are gone.

    Never raises: Gate B is a soft throttle layered on top of Gate A, and a
    broken bind/unbind call (SessionRouter unreachable, session already torn
    down, ...) must not fail the episode itself.
    """
    if not session_router_url or not session_id:
        return
    try:
        await post(
            f"{session_router_url.rstrip('/')}/control/gpu_bind",
            payload=[{"session_id": session_id, "gpu_device": device}],
            max_retries=1,
        )
    except Exception as exc:  # noqa: BLE001
        psrl_logger.debug(
            "Gate B %s call failed for session=%r: %s.",
            "bind" if device else "unbind",
            session_id,
            exc,
        )


async def _in_cleanup_pool(fn, *args):
    """
    Run a blocking Docker call on the dedicated cleanup pool.

    Not `asyncio.to_thread`: that shares the loop's default executor with everything else
    on the Harbor loop, and a `docker rm` against a loaded daemon blocks for tens of seconds.
    """
    return await asyncio.get_running_loop().run_in_executor(CLEANUP_EXECUTOR, fn, *args)


@dataclass
class HarborEpisodeResult:
    """
    Result of one Harbor episode.
    """

    task_name: str
    reward: float
    rewards: dict = field(default_factory=dict)
    exception: str | None = None
    # Exception class name, kept separate because Harbor's own classes distinguish
    # cases the message cannot. `HarborExceptionClassifier` prefers this over text.
    exception_type: str | None = None


async def _regrade_from_artifacts(
    task_path: str,
    trial_uri: str | None,
    jobs_dir: str | Path,
    timeout_sec: float,
) -> dict:
    """
    Recover a verifier score for a trial whose verification was skipped.

    Harbor skips verification whenever the agent raises, so the trial reports no rewards
    at all. The artifacts survive, and a `regrade` source job grades exactly those with no
    agent and no live container, which tasks declaring
    `[verifier] environment_mode = "separate"` support.

    It runs only when the reward dict is empty, and any failure degrades to `{}`. An empty
    dict and a graded 0.0 are different facts, and only the graded result carries `floor`.

    Args:
        task_path: Harbor task directory, providing the verifier to grade with.
        trial_uri: The failed trial's directory, as a `file://` URI or a path.
        jobs_dir: Root for the regrade job's own scratch directory.
        timeout_sec: Backstop for the regrade job.

    Returns:
        dict: The recovered reward dict, or empty when grading could not run at all.
    """
    if not trial_uri:
        return {}
    parsed = urlparse(str(trial_uri))
    trial_dir = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(str(trial_uri))

    # The regrade job runs its own verifier under its own Compose project, so naming it
    # here gives the cleanup in `finally` something to key on.
    job_name = f"regrade_{trial_dir.name}_{uuid.uuid4().hex[:8]}"
    try:
        config = JobConfig(
            jobs_dir=Path(jobs_dir) / "regrade",
            job_name=job_name,
            tasks=[TaskConfig(path=task_path)],
            source_jobs=[SourceJobConfig(action="regrade", type="local", path=trial_dir.parent.resolve())],
            n_concurrent_trials=1,
            quiet=True,
        )
        job = await Job.create(config)
        result = await asyncio.wait_for(job.run(), timeout=timeout_sec)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        psrl_logger.warning("Regrade from artifacts failed for %s: %s.", trial_dir.name, exc)
        return {}
    finally:
        # The regrade builds its own verifier under its own Compose project, so it leaks
        # both containers and images independently of the episode that triggered it.
        await _in_cleanup_pool(force_remove_compose_project, job_name)
        await _in_cleanup_pool(force_remove_compose_images, job_name)

    for trial in result.trial_results or []:
        if trial.verifier_result and trial.verifier_result.rewards:
            rewards = dict(trial.verifier_result.rewards)
            psrl_logger.info(
                "Recovered a verifier score by regrading %s: reward=%s.",
                trial_dir.name,
                rewards.get("reward"),
            )
            return rewards
    return {}


async def run_harbor_episode(
    task_path: str,
    model_base_url: str,
    model_name: str,
    config: SciAccelRuntimeConfig,
    needs_gpu: bool = False,
    session_id: str = "",
    session_router_url: str = "",
    max_model_len: int = 40960,
    max_turns: int | None = None,
    regrade_unverified: bool = True,
    actor_id: str = "",
    thinking_template: str = MULTI_TRAJ,
    hint: str = "",
    guidance: str = "",
) -> HarborEpisodeResult:
    """
    Run one Harbor episode and return the verifier reward.

    Args:
        task_path: Path to the Harbor task directory (containing task.toml).
        model_base_url: SessionRouter session URL for the model endpoint.
        model_name: Model name for the agent (e.g. `Qwen/Qwen3-8B`).
        config: Runtime config with Harbor and timeout settings.
        needs_gpu: Whether the task container needs GPU device access. When
            true, Gate A (`psrl.utils.concurrency.gpu_device_gate`) reserves
            one physical GPU device exclusively for this episode instead of
            always mounting the static override's hardcoded device.
        session_id: TITO session ID, used as the job directory name for traceability.
        session_router_url: SessionRouter base URL (no `/sessions/...` suffix),
            used only to call Gate B's `/control/gpu_bind` so the router can
            throttle this session's replies while its GPU device is busy.
            Empty disables Gate B for this call without affecting Gate A.
        max_model_len: vLLM context window size, forwarded to terminus-2 as its
            `model_info` limits. Note this is metadata only on the litellm path:
            nothing sends it as a per-request `max_tokens`, so generation is bounded
            by the server's own `--max-model-len`. `finish_reason == "length"` from
            that boundary is what terminus-2 reports as `OutputLengthExceededError`.
        max_turns: Hard cap on agent turns. Must be set: terminus-2 otherwise defaults
            to `max_episodes=1000000`, so an agent that never calls the completion
            tool grinds on until the context window overflows.
        regrade_unverified: When the verifier never ran, try to recover a score from
            the artifacts the episode delivered before failing. Training an empty
            reward dict as 0.0 is an incorrect label, not just a missing one. See
            `_regrade_from_artifacts`.
        thinking_template: How the model's chain-of-thought is carried across turns.
            Decides the `extra_body` sent to the gateway: `multi_thinking` turns the
            reasoning parser off so <think> stays inline in `content`,
            `disable_thinking` turns thinking off, and the two trajectory-oriented
            modes send nothing. See `psrl/utils/agent/thinking.py`.
        hint: Localization text appended to the task instruction by Harbor. The
            dataset's `prompt` column never reaches the model, because Harbor
            re-reads `instruction.md` from `task_path` itself, so this is the
            only path that puts extra text in front of the agent. An empty hint
            leaves the job config untouched, which keeps the unhinted level a
            true control.
        guidance: Terminal and build discipline, delivered at every hint level so
            it stays orthogonal to localization. It describes how to spend turns,
            never where the defect is, so it does not weaken the unhinted control.

    Returns:
        HarborEpisodeResult with the verifier's shaped reward.
    """
    extra_body = harness_extra_body(thinking_template)
    env_kwargs: dict = {}
    extra_compose_paths: list[Path] = []
    allocated_gpu: AllocatedGpu | None = None
    gpu_override_path: Path | None = None
    if needs_gpu and config.harbor.gpu_compose_override:
        # Gate A: reserve one physical GPU device before this episode's
        # container exists, instead of always mounting the static override's
        # hardcoded /dev/nvidia0. Without this, N concurrent GPU episodes on
        # an N>1-GPU host would all get handed the identical device string,
        # because nothing here previously tracked per-device occupancy (only
        # `_acquire_episode_slot`'s semaphore counted *how many*, never
        # *which*). See docs/psrl/psrl-env-service-envworker-design.md §16.
        gpu_gate = GpuDeviceGate(jobs_dir=str(config.harbor.jobs_dir))
        allocated_gpu = await gpu_gate.acquire()
        if allocated_gpu is not None:
            gpu_override_path = Path(build_gpu_override(config.harbor.gpu_compose_override, allocated_gpu.device))
            extra_compose_paths.append(gpu_override_path)
            await _bind_gate_b(session_router_url, session_id, allocated_gpu.device)
        else:
            # No discoverable device on this host (e.g. a dev box with no
            # GPU): degrade to the old static mapping rather than silently
            # dropping GPU access, but this episode gets no per-device
            # exclusivity guarantee.
            psrl_logger.warning(
                "Gate A found no GPU devices; falling back to the static override %r "
                "(no per-device exclusivity for this episode).",
                config.harbor.gpu_compose_override,
            )
            extra_compose_paths.append(Path(config.harbor.gpu_compose_override))

    main_override: dict = {}
    if actor_id:
        main_override["labels"] = [f"psrl.actor_id={actor_id}"]
    if config.harbor.memory_mb_override > 0:
        main_override["mem_limit"] = f"{config.harbor.memory_mb_override}m"

    if main_override:
        import tempfile

        import yaml

        compose_override = {"services": {"main": main_override}}
        override_file = Path(tempfile.mktemp(suffix=".yaml", prefix="harbor-override-"))
        override_file.write_text(yaml.dump(compose_override))
        extra_compose_paths.append(override_file)

    if extra_compose_paths:
        env_kwargs["extra_docker_compose"] = extra_compose_paths

    job_name = session_id or uuid.uuid4().hex[:12]
    job_dir = Path(config.harbor.jobs_dir) / job_name
    job_config = JobConfig(
        jobs_dir=job_dir,
        tasks=[TaskConfig(path=task_path)],
        agents=[
            AgentConfig(
                name=config.harbor.agent_name,
                model_name=f"openai/{model_name}",
                env={"OPENAI_API_KEY": "EMPTY"},
                # Let Harbor enforce the external budget so timed-out jobs remain gradable.
                override_timeout_sec=config.task_timeout_sec,
                kwargs={
                    "api_base": model_base_url,
                    "enable_summarize": False,
                    "collect_rollout_details": True,
                    # Disable terminal recording because offline task images omit asciinema.
                    "record_terminal_session": False,
                    "temperature": 1.0,
                    # Only the local `TruncatingTerminus2` accepts this. Stock terminus-2
                    # swallows it in `**kwargs`, so this is gated on the subclass.
                    **(
                        {"max_observation_bytes": config.harbor.max_observation_bytes}
                        if ":" in config.harbor.agent_name
                        else {}
                    ),
                    # Bound turns before context overflow.
                    **({"max_turns": max_turns} if max_turns else {}),
                    "suppress_max_turns_warning": True,
                    "model_info": {
                        # NOTE(lhy): Declare a window smaller than the server's, because
                        # terminus-2 estimates tokens and would build a prompt vLLM rejects.
                        # Do NOT raise it: this is a budget the agent spends.
                        "max_input_tokens": max(1024, max_model_len - _CONTEXT_SAFETY_MARGIN),
                        "max_output_tokens": max_model_len,
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "llm_kwargs": {
                        "timeout": 900,
                        "max_retries": 0,
                        # Apply gateway overrides required by `thinking_template`.
                        **({"extra_body": extra_body} if extra_body else {}),
                    },
                },
            )
        ],
        n_attempts=1,
        n_concurrent_trials=1,
        # Harbor appends these to the on-disk instruction before the agent sees it.
        # Guidance leads, so the localization hint stays closest to the task.
        **({"extra_instructions": [t for t in (guidance, hint) if t]} if (guidance or hint) else {}),
        **({"environment": env_kwargs} if env_kwargs else {}),
    )

    # This Harbor backstop exceeds the agent and verifier budgets plus container slack.
    timeout = config.task_timeout_sec + config.verifier_timeout_sec + 600.0

    # Every return path runs through `finally`. Cancelling the coroutine leaves the
    # verifier running `sleep infinity` and the next one queues behind it.
    try:
        try:
            job = await Job.create(job_config)
            result = await asyncio.wait_for(job.run(), timeout=timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            psrl_logger.warning("Harbor job failed before returning a trial: %s.", exc, exc_info=True)
            return HarborEpisodeResult(
                task_name="unknown",
                reward=0.0,
                exception=str(exc) or type(exc).__name__,
                exception_type=type(exc).__name__,
            )

        if not result.trial_results:
            return HarborEpisodeResult(
                task_name="unknown",
                reward=0.0,
                exception="no_trials",
                exception_type="NoTrialsError",
            )

        tr = result.trial_results[0]
        rewards = dict(tr.verifier_result.rewards) if tr.verifier_result and tr.verifier_result.rewards else {}
        # Preserve the Harbor exception type because LiteLLM strips transport metadata.
        exception = tr.exception_info.exception_message if tr.exception_info else None
        exception_type = tr.exception_info.exception_type if tr.exception_info else None

        # Regrade missing verifier output because an agent exception may skip verification.
        if not rewards and regrade_unverified:
            rewards = await _regrade_from_artifacts(
                task_path=task_path,
                trial_uri=tr.trial_uri,
                jobs_dir=job_config.jobs_dir,
                timeout_sec=config.verifier_timeout_sec + 600.0,
            )

        return HarborEpisodeResult(
            task_name=tr.task_name,
            reward=float(rewards.get("reward", 0.0)),
            rewards=rewards,
            exception=exception,
            exception_type=exception_type,
        )
    finally:
        # Runs in a thread because `docker rm -f` on a wedged container can block for
        # tens of seconds, and this sits on the shared Harbor event loop.
        removed = await _in_cleanup_pool(force_remove_compose_project, job_name)
        if removed:
            psrl_logger.info(
                "Reclaimed %d container(s) for episode %s after the job returned.",
                len(removed),
                job_name,
            )
        # Each image is tagged after the per-episode Compose project, and on the cancelled
        # path Harbor's teardown never runs, leaving a tag no dangling sweep can reach.
        await _in_cleanup_pool(force_remove_compose_images, job_name)
        # Backstop for images this episode did not own, mainly ones orphaned by a rebuild
        # elsewhere. Self-throttled, so it is a cheap no-op on most episodes.
        await _in_cleanup_pool(prune_dangling_images)
        # Gate A: release the reserved physical GPU device only after the container
        # that used it is confirmed gone (the `force_remove_compose_project` call
        # above), so a waiting episode never gets the device while the old one's
        # container teardown is still in flight.
        if allocated_gpu is not None:
            GpuDeviceGate.release(allocated_gpu)
            await _bind_gate_b(session_router_url, session_id, None)
        if gpu_override_path is not None:
            gpu_override_path.unlink(missing_ok=True)
