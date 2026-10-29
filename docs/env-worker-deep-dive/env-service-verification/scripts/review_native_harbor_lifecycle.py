"""Probe adapter readiness failures with CPU fakes, without Docker or shared services."""

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_worker.harbor import EnvWorkerEnvironment
from psrl.workers.env_worker.process import ProcessResult


def environment(root):
    """Construct the real adapter with a fake container worker."""
    root.mkdir()
    env = EnvWorkerEnvironment(
        environment_dir=root,
        environment_name="review",
        session_id="review",
        trial_paths=TrialPaths(root / "trial"),
        task_env_config=EnvironmentConfig(docker_image="review:unused", cpus=1, memory_mb=64),
        network_policy=NetworkPolicy(network_mode="no-network"),
    )
    env._worker = SimpleNamespace(
        create_sandbox=AsyncMock(),
        destroy_sandbox=AsyncMock(),
        exec_process=AsyncMock(return_value=ProcessResult("ran", "", 0)),
    )
    return env


async def probe(root):
    # Diagram A: observe readiness while environment upload is still blocked.
    env = environment(root / "starting")
    entered, finish = asyncio.Event(), asyncio.Event()

    async def upload():
        entered.set()
        await finish.wait()

    env._upload_environment_dir_after_start = upload
    start = asyncio.create_task(env.start())
    try:
        await entered.wait()
        try:
            result = await env.exec("cat input.txt")
            early_execution = result.stdout == "ran" and not start.done()
        except RuntimeError:
            early_execution = False
    finally:
        finish.set()
        await start
        await env.stop()

    # Diagram B: a failed cleanup must not make a partial environment ready.
    broken = environment(root / "failed")
    broken._upload_environment_dir_after_start = AsyncMock(side_effect=RuntimeError("upload failed"))
    broken._worker.destroy_sandbox.side_effect = RuntimeError("docker unavailable")
    try:
        await broken.start()
    except RuntimeError as exc:
        failure = str(exc)
    second_success = False
    try:
        await broken.start()
        second_success = True
    except RuntimeError:
        pass
    return {
        "scope": "Real adapter with fake Docker operations, no physical resource experiment",
        "exec_accepted_before_upload_finished": early_execution,
        "first_start_error": failure,
        "second_start_returned_success": second_success,
        "create_calls_after_two_starts": broken._worker.create_sandbox.await_count,
        "started_after_failed_cleanup": broken._started,
        "closed_after_failed_cleanup": broken._closed,
    }


async def main():
    with tempfile.TemporaryDirectory(prefix="harbor-lifecycle-review-") as root:
        result = await asyncio.wait_for(probe(Path(root)), 5)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
