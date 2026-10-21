"""Compare persistent shell state through EnvWorker and Harbor without GPU or models."""

import argparse
import asyncio
import json
import subprocess
import time
import uuid
from pathlib import Path

from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_worker.shell import parse_sentinel, wrap_command
from psrl.workers.env_worker.worker import ContainerShell


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    directory = args.output / "environment"
    directory.mkdir()
    await ManagedDockerEnvironment.prepare_runtime()
    env = ManagedDockerEnvironment(
        environment_dir=directory,
        environment_name="shell-contract",
        session_id="shell-contract-" + uuid.uuid4().hex[:12],
        trial_paths=TrialPaths(args.output / "trial"),
        task_env_config=EnvironmentConfig(docker_image=args.image, cpus=1, memory_mb=512),
        network_policy=NetworkPolicy(network_mode="no-network"),
        actor_id="shell-contract",
    )
    result = {"scope": "CPU shell state only. Not an MLGym episode or training equivalence test."}
    commands = [
        "mkdir -p /tmp/state; cd /tmp/state; export STATE_VALUE=retained; state_fn() { echo function-retained; }",
        'printf "state=%s cwd=%s\\n" "$STATE_VALUE" "$PWD"; state_fn',
    ]
    try:
        async with asyncio.timeout(120):
            await env.start(force_build=False)
            preflight = await env.exec("command -v tmux", timeout_sec=5)
            if preflight.return_code:
                raise RuntimeError("Shell contract probe requires a cached image with tmux installed.")
            session = TmuxSession("contract", env, "/tmp/contract.log", None, None)
            session._disable_recording = True
            await session.start()
            try:
                for command in commands:
                    await session.send_keys(command + "\n", block=True, max_timeout_sec=5)
                result["harbor"] = await session.get_incremental_output()
            finally:
                await session.stop()
            container = (await env._run_docker_compose_command(["ps", "-q", "main"], timeout_sec=5)).stdout.strip()
            process = subprocess.Popen(
                ["docker", "exec", "-i", container, "bash", "-l"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            shell = ContainerShell(process, container)
            try:
                for command in commands:
                    command_id = uuid.uuid4().hex
                    deadline = time.monotonic() + 5
                    assert await shell.write(wrap_command(command, command_id), deadline)
                    output = await shell.read_until(deadline, 5, command_id)
                    parsed = parse_sentinel(output, command_id)
                    assert parsed and parsed[1] == 0, output
                result["envworker"] = parsed[0]
            finally:
                shell.close()
                await asyncio.to_thread(process.wait, timeout=5)
            for backend in ("harbor", "envworker"):
                assert "state=retained cwd=/tmp/state" in result[backend], result
                assert "function-retained" in result[backend], result
            result["passed"] = True
    finally:
        await env.stop(delete=True)
        (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
