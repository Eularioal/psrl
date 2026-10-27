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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from examples.sciaccel_rl.config import SciAccelRuntimeConfig
from examples.sciaccel_rl.gpu_grading import LAPS_CUDA_CHECKS, apply_gpu_activity_gate, is_laps_cuda, save_gpu_reports
from harbor.job import Job
from harbor.models.job.config import AgentConfig, JobConfig, SourceJobConfig
from harbor.models.trial.config import TaskConfig
from psrl.utils.agent.thinking import MULTI_TRAJ, harness_extra_body
from psrl.utils.common.docker_utils import (
    CLEANUP_EXECUTOR,
    force_remove_compose_images,
    force_remove_compose_project,
)
from psrl.utils.common.http_utils import post
from psrl.utils.concurrency.gpu_device_gate import AllocatedGpu, GpuDeviceGate, build_gpu_override
from psrl.workers.env_service.client import EnvLeaseHandle, EnvServiceClient
from psrl.workers.env_service.harbor_resources import demand_from_task, with_container_budget
from psrl.workers.env_service.overlay import (
    DriverLibraryError,
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.types import EnvDemand, EnvLease

psrl_logger = logging.getLogger("psrl.sciaccel_rl.runner")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Tokens held back from the window advertised to terminus-2, which counts with litellm's
# estimator rather than the served tokenizer and so drifts low.
_CONTEXT_SAFETY_MARGIN = 8192


async def _bind_gate_b(
    session_router_url: str,
    session_id: str,
    device: str | None,
    shared: bool = False,
    node_service_url: str = "",
    gpu_indices: tuple[int, ...] = (),
) -> None:
    """
    Best-effort: tell SessionRouter which physical GPU device (if any) this
    session is using, so Gate B (psrl/workers/gen/session_router.py's
    `control_gpu_bind` + `_wait_for_gpu_idle`) can hold the model's reply
    until that device looks idle before letting the agent issue its next
    tool command. Called once when Gate A reserves a device, and again with
    `device=None` to unbind once the episode's containers are gone.

    `device` may list several devices joined by commas (an env unit that owns more than
    one card). `shared` marks a card that other containers of the same unit also use, so
    Gate B judges it on utilization alone.

    Never raises: Gate B is a soft throttle layered on top of Gate A, and a
    broken bind/unbind call (SessionRouter unreachable, session already torn
    down, ...) must not fail the episode itself.
    """
    if not session_router_url or not session_id:
        return
    try:
        await post(
            f"{session_router_url.rstrip('/')}/control/gpu_bind",
            payload=[
                {
                    "session_id": session_id,
                    "gpu_device": device,
                    "shared": shared,
                    "node_service_url": node_service_url,
                    "gpu_indices": list(gpu_indices),
                }
            ],
            max_retries=1,
        )
    except Exception as exc:  # noqa: BLE001
        psrl_logger.debug(
            "Gate B %s call failed for session=%r: %s.",
            "bind" if device else "unbind",
            session_id,
            exc,
        )


async def _apply_env_lease(
    lease: EnvLease,
    extra_compose_paths: list[Path],
    env_kwargs: dict,
    session_router_url: str,
    session_id: str,
) -> Path:
    """
    Turn a granted lease into the Compose overlay and Harbor settings that realise it.

    A GPU lease gets its unit's devices and the driver libraries matching the loaded kernel
    module, resolved now rather than hardcoded (design document B1 to B3). Harbor's local
    Docker backend declares no GPU capability and would refuse a task whose `task.toml`
    asks for one, so `override_gpus: 0` tells it none is needed while the overlay supplies
    the devices (design document section 7.3). Gate B is bound here too, so a shared card
    is judged on utilization alone.

    Args:
        lease: The granted lease.
        extra_compose_paths: Compose files, appended to.
        env_kwargs: Harbor environment settings, updated in place.
        session_router_url: SessionRouter base URL for Gate B. Empty disables Gate B.
        session_id: The TITO session id Gate B binds to.

    Returns:
        Path of the overlay file. The caller deletes it once the containers are gone.

    Raises:
        DriverLibraryError: When a GPU lease cannot find driver libraries matching the kernel.
    """

    def prepare_overlay():
        driver_libs = None
        if lease.gpu_indices:
            version = read_kernel_driver_version()
            if version is None:
                raise DriverLibraryError(
                    "A GPU lease was granted but /proc/driver/nvidia/version is unreadable. "
                    "This node has no NVIDIA kernel driver loaded."
                )
            driver_libs = resolve_driver_libs(version)
        return build_lease_overlay(lease, driver_libs)

    overlay = await asyncio.to_thread(prepare_overlay)
    # Keep file creation and cleanup registration together if the caller is cancelled.
    overlay_path = write_overlay(overlay)
    extra_compose_paths.append(overlay_path)
    env_kwargs["import_path"] = "psrl.workers.env_service.docker:ManagedDockerEnvironment"
    env_kwargs.setdefault("kwargs", {})["lease_overlay"] = overlay
    env_kwargs["kwargs"]["container_budget"] = (
        asdict(lease.demand.container_budget) if lease.demand.container_budget else None
    )
    if lease.gpu_indices:
        env_kwargs["override_gpus"] = 0
        devices = ",".join(
            item.split(":", 1)[0]
            for item in overlay["services"]["main"]["devices"]
            if item.split(":", 1)[0].removeprefix("/dev/nvidia").isdigit()
        )
        await _bind_gate_b(
            "" if lease.gpu_lending or lease.gpu_reservation == "command" else session_router_url,
            session_id,
            devices,
            shared=lease.shared,
            node_service_url=lease.node_service_url,
            gpu_indices=lease.gpu_indices,
        )
    return overlay_path


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
    env_client: EnvServiceClient | None = None,
    env_demand: EnvDemand | None = None,
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
        env_client: Env service client. When given, Gate A is the router's resource-keyed
            admission (`psrl.workers.env_service`): the episode waits, at no cost, until a
            unit on this node can hold it, then runs on that unit's devices with a
            per-lease Compose overlay. This replaces the per-device `GpuDeviceGate` path
            below, and lets several episodes share one card. `None` keeps the legacy path.
        env_demand: What the episode charges against its unit. Defaults to one CPU, plus a
            GPU when `needs_gpu` is set.

    Returns:
        HarborEpisodeResult with the verifier's shaped reward.
    """
    extra_body = harness_extra_body(thinking_template)
    env_kwargs: dict = {}
    if config.harbor.override_gpus is not None:
        env_kwargs["override_gpus"] = config.harbor.override_gpus
    extra_compose_paths: list[Path] = []
    allocated_gpu: AllocatedGpu | None = None
    gpu_override_path: Path | None = None
    lease_handle: EnvLeaseHandle | None = None
    lease_overlay_path: Path | None = None
    job_name = session_id or uuid.uuid4().hex[:12]
    lending_agent_kwargs = {}
    agent_name = config.harbor.agent_name
    if env_client is not None:
        # Wait for admission before creating containers, and classify admission failures as episode errors.
        try:
            demand = with_container_budget(
                task_path, env_demand or demand_from_task(task_path, needs_gpu_hint=needs_gpu)
            )
            lease_handle = await env_client.acquire(job_name, demand)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            psrl_logger.warning("Env service admission failed for %s: %s.", job_name, exc)
            return HarborEpisodeResult(
                task_name="unknown",
                reward=0.0,
                exception=f"env_admission_failed: {exc}",
                exception_type=type(exc).__name__,
            )
    elif needs_gpu and config.harbor.gpu_compose_override:
        # Device ownership must be acquired before the container maps the selected GPU.
        gpu_gate = GpuDeviceGate(jobs_dir=str(config.harbor.jobs_dir))
        allocated_gpu = await gpu_gate.acquire()
        if allocated_gpu is not None:
            gpu_override_path = Path(build_gpu_override(config.harbor.gpu_compose_override, allocated_gpu.device))
            extra_compose_paths.append(gpu_override_path)
            await _bind_gate_b(session_router_url, session_id, allocated_gpu.device)
        else:
            # Static mapping without a discoverable device cannot guarantee exclusive access.
            psrl_logger.warning(
                "Gate A found no GPU devices. Using the static override %r "
                "(no per-device exclusivity for this episode).",
                config.harbor.gpu_compose_override,
            )
            extra_compose_paths.append(Path(config.harbor.gpu_compose_override))

    # Include setup in the cleanup scope so a failed overlay cannot leave a heartbeating lease.
    try:
        if lease_handle is not None and lease_handle.lease.gpu_lending:
            if agent_name not in ("terminus-2", "examples.sciaccel_rl.agent:TruncatingTerminus2"):
                raise ValueError("GPU lending requires the science Terminus2 phase hook.")
            agent_name = "examples.sciaccel_rl.agent:TruncatingTerminus2"
            lending_agent_kwargs["gpu_lending"] = {
                "router_name": env_client.router_name,
                "lease_id": lease_handle.lease.lease_id,
            }
            await env_client.router.set_gpu_phase.remote(lease_handle.lease.lease_id, "env", 0)
            env_kwargs.setdefault("kwargs", {})["gpu_lending"] = lending_agent_kwargs["gpu_lending"]
        if lease_handle is not None:
            if lease_handle.lease.gpu_indices and not lease_handle.lease.gpu_lending:
                if agent_name == "terminus-2":
                    agent_name = "examples.sciaccel_rl.agent:TruncatingTerminus2"
                lending_agent_kwargs["gpu_jobs"] = True
                binding = {
                    "router_name": env_client.router_name,
                    "lease_id": lease_handle.lease.lease_id,
                }
                env_kwargs.setdefault("kwargs", {})["gpu_execution"] = {
                    **binding,
                    "mode": lease_handle.lease.gpu_reservation,
                }
                if lease_handle.lease.gpu_reservation == "command":
                    env_kwargs["kwargs"]["gpu_commands"] = binding
            try:
                if lease_handle.lease.gpu_reservation == "command" and is_laps_cuda(task_path):
                    if config.harbor.gpu_attribution_mode == "required":
                        await env_client.router.require_gpu_attribution.remote(lease_handle.lease.lease_id)
                    else:
                        psrl_logger.warning(
                            "GPU attribution preflight skipped for lease %s in observation mode. "
                            "Task GPU scores remain unverified and can include neighboring activity.",
                            lease_handle.lease.lease_id,
                        )
                lease_overlay_path = await _apply_env_lease(
                    lease_handle.lease, extra_compose_paths, env_kwargs, session_router_url, session_id
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                psrl_logger.error("Cannot prepare containers for lease %s: %s.", lease_handle.lease.lease_id, exc)
                return HarborEpisodeResult(
                    task_name="unknown",
                    reward=0.0,
                    exception=f"env_lease_setup_failed: {exc}",
                    exception_type=type(exc).__name__,
                )

        main_override: dict = {}
        if actor_id:
            main_override["labels"] = [f"psrl.actor_id={actor_id}"]
            env_kwargs["import_path"] = "psrl.workers.env_service.docker:ManagedDockerEnvironment"
            env_kwargs.setdefault("kwargs", {})["actor_id"] = actor_id
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

        job_dir = Path(config.harbor.jobs_dir) / job_name
        job_config = JobConfig(
            jobs_dir=job_dir,
            tasks=[TaskConfig(path=task_path)],
            agents=[
                AgentConfig(
                    name=agent_name,
                    model_name=f"openai/{model_name}",
                    env={"OPENAI_API_KEY": "EMPTY"},
                    # Let Harbor enforce the external budget so timed-out jobs remain gradable.
                    override_timeout_sec=config.task_timeout_sec,
                    kwargs={
                        **lending_agent_kwargs,
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

        try:
            if env_kwargs.get("import_path") == "psrl.workers.env_service.docker:ManagedDockerEnvironment":
                from psrl.workers.env_service.docker import ManagedDockerEnvironment

                await ManagedDockerEnvironment.prepare_runtime()
            job = await Job.create(job_config)
            result = await asyncio.wait_for(job.run(), timeout=timeout)
            if lease_handle is not None and lease_handle.lease.gpu_reservation == "command":
                await env_client.router.wait_gpu_command_returns.remote(lease_handle.lease.lease_id)
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

        if lease_handle is not None and lease_handle.lease.gpu_reservation == "command":
            reports = await env_client.router.gpu_job_reports.remote(lease_handle.lease.lease_id)
            if is_laps_cuda(task_path):
                rewards = apply_gpu_activity_gate(
                    rewards, reports, LAPS_CUDA_CHECKS, mode=config.harbor.gpu_attribution_mode
                )
            if reports or is_laps_cuda(task_path):
                await asyncio.to_thread(
                    save_gpu_reports,
                    Path(job_config.jobs_dir) / job_name,
                    reports,
                    rewards,
                    mode=config.harbor.gpu_attribution_mode if is_laps_cuda(task_path) else None,
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
        # Physical container removal must precede release of the device gate.
        if allocated_gpu is not None:
            GpuDeviceGate.release(allocated_gpu)
            await _bind_gate_b(session_router_url, session_id, None)
        if gpu_override_path is not None:
            gpu_override_path.unlink(missing_ok=True)
        # Session resources remain charged until the owned containers are confirmed gone.
        if lease_handle is not None:
            if lease_handle.lease.gpu_indices:
                await _bind_gate_b(session_router_url, session_id, None)
            await lease_handle.release()
        if lease_overlay_path is not None:
            lease_overlay_path.unlink(missing_ok=True)
