"""Verify partial-start cleanup and retirement against a real CPU Docker container."""

import argparse
import asyncio
import json
import subprocess
import uuid
from pathlib import Path

from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.router import EnvRouterCore
from psrl.workers.env_service.types import EnvDemand
from psrl.workers.env_worker.harbor import EnvWorkerEnvironment


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    tag = "lifecycle-" + uuid.uuid4().hex[:10]
    core = EnvRouterCore(
        {
            "node_policy": "cpu_only",
            "units": [{"unit_id": tag, "node_ip": "local", "cpu_quota": 1, "memory_mb": 256, "max_containers": 1}],
        }
    )
    demand = EnvDemand(1, 256, False, 0)
    lease = (await core.try_admit("first", demand, "local")).lease
    env = EnvWorkerEnvironment(
        environment_dir=args.output,
        environment_name=tag,
        session_id=tag,
        trial_paths=TrialPaths(args.output / "trial"),
        actor_id=tag,
        task_env_config=EnvironmentConfig(docker_image=args.image, cpus=1, memory_mb=256),
        network_policy=NetworkPolicy(network_mode="no-network"),
        lease_labels={"psrl.env_lease": lease.lease_id, "psrl.env_unit": tag},
    )
    entered, proceed = asyncio.Event(), asyncio.Event()
    real_delete = env._worker.destroy_sandbox
    attempts = 0
    evidence = {"tag": tag, "scope": "Real Docker, in-process EnvRouterCore, injected upload/delete exceptions"}

    async def upload():
        entered.set()
        await proceed.wait()
        raise RuntimeError("injected upload failure")

    async def delete(sandbox):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected deletion failure")
        await real_delete(sandbox)

    env._upload_environment_dir_after_start = upload
    env._worker.destroy_sandbox = delete
    start = asyncio.create_task(env.start())
    try:
        await asyncio.wait_for(entered.wait(), 30)
        try:
            await env.exec("true")
        except RuntimeError:
            evidence["partial_start_rejected"] = True
        else:
            raise AssertionError("Partial environment must not execute commands.")
        proceed.set()
        try:
            await start
        except RuntimeError as exc:
            evidence["injected_error"] = str(exc)
        assert env._cleanup_pending and not env._started, "Cleanup ownership must remain visible."
        evidence["wait_while_owned"] = (await core.try_admit("second", demand, "local")).status
        await env.stop()
        await core.release(lease.lease_id)
        await core.acknowledge_reaped("local", lease.lease_id)
        second = await core.try_admit("second", demand, "local")
        evidence["after_physical_cleanup"] = second.status
        assert evidence["wait_while_owned"] == "wait", "Unreturned capacity must remain blocked."
        assert second.status == "granted", "Confirmed cleanup must make capacity reusable."
        await core.release(second.lease.lease_id)
        await core.acknowledge_reaped("local", second.lease.lease_id)
        evidence["passed"] = True
    finally:
        proceed.set()
        await asyncio.gather(start, return_exceptions=True)
        env._worker.destroy_sandbox = real_delete
        await env.stop()
        if core._reaper:
            core._reaper.cancel()
            await asyncio.gather(core._reaper, return_exceptions=True)
        evidence["remaining_containers"] = subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.actor_id={tag}"], text=True, timeout=10
        ).strip()
        (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2))
        assert not evidence["remaining_containers"], "Owned containers must be physically removed."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(run(args), 60))
