"""Verify a shared container backend and one Harbor oracle trial without GPUs or models."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import subprocess
import time
import traceback
import uuid
from pathlib import Path

from harbor.job import Job
from harbor.models.job.config import AgentConfig, JobConfig
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.config import TaskConfig
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_worker.harbor import EnvWorkerEnvironment


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=10).strip()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


async def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tag = f"native-harbor-{uuid.uuid4().hex[:8]}"
    evidence = {
        "passed": False,
        "tag": tag,
        "scope": "Local EnvWorker backend and Harbor oracle, no Ray or GPU training.",
        "docker": docker("info", "--format", "{{.ServerVersion}}"),
        "image": docker("image", "inspect", args.image, "--format", "{{.Id}}"),
        "sources": {
            name: importlib.util.find_spec(name).origin for name in ("psrl", "harbor", "verl", "transfer_queue")
        },
    }
    repo = Path(__file__).resolve().parents[4]
    evidence["source_sha256"] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (repo / "psrl/workers/env_worker").glob("*.py")
    }
    save(out / "ownership.json", {"actor_id": tag, "label": f"psrl.actor_id={tag}"})
    environments = []
    started = time.monotonic()

    def environment(name):
        root = out / name
        (root / "environment").mkdir(parents=True)
        (root / "environment/seed.txt").write_text("seed")
        env = EnvWorkerEnvironment(
            environment_dir=root / "environment",
            environment_name=name,
            session_id=f"{tag}-{name}",
            trial_paths=TrialPaths(root / "trial"),
            task_env_config=EnvironmentConfig(docker_image=args.image, cpus=1, memory_mb=256, workdir="/work"),
            network_policy=NetworkPolicy(network_mode="no-network"),
            actor_id=tag,
        )
        environments.append(env)
        return env

    async def workflow():
        # The two frontends below reach the very same container and filesystem.
        env = environment("contract")
        await env.start()
        container = env._worker.container_name(env._sandbox_id)
        details = json.loads(docker("inspect", container))[0]
        evidence["limits"] = {key: details["HostConfig"][key] for key in ("NanoCpus", "Memory", "NetworkMode")}
        assert evidence["limits"] == {"NanoCpus": 1000000000, "Memory": 256 * 1024**2, "NetworkMode": "none"}, (
            "Resource limits and network isolation must reach Docker."
        )
        legacy = await env._worker.exec(env._sandbox_id, "export PERSIST=kept; cd /tmp; echo 42 > /work/answer")
        assert legacy.exit_code == 0, "Persistent frontend must execute successfully."
        legacy_again = await env._worker.exec(env._sandbox_id, 'printf "%s:%s" "$PERSIST" "$PWD"')
        assert legacy_again.stdout.strip() == "kept:/tmp", "EnvWorker shell state must persist."
        result = await env.exec(
            'printf "%s:%s:%s" "${PERSIST-unset}" "$PWD" "$INPUT"; cat answer; echo err >&2', env={"INPUT": "a b"}
        )
        assert result.stdout == "unset:/work:a b42\n" and result.stderr == "err\n", (
            "Harbor shell state must be isolated."
        )
        evidence["shared_container"] = {
            "name": container,
            "persistent": legacy_again.stdout,
            "harbor": result.model_dump(),
        }
        download = out / "download"
        await env.download_dir("/work", download)
        assert (download / "seed.txt").read_text() == "seed", "Upload must copy contents, not nest a directory."
        assert (download / "answer").read_text().strip() == "42", "Download must preserve shared files."
        await env.stop()
        await env.stop()

        # Timeout must remove the container, not only its Docker client process.
        timed = environment("timeout")
        await timed.start()
        timeout_container = timed._worker.container_name(timed._sandbox_id)
        try:
            await timed.exec("sleep 20", timeout_sec=0.2)
        except TimeoutError:
            evidence["timeout_confirmed"] = True
        else:
            raise AssertionError("The bounded command must time out.")
        assert not docker("ps", "-aq", "--filter", f"name=^{timeout_container}$"), (
            "Timed out container must be removed."
        )

        # Harbor retains task orchestration, oracle execution, grading and artifact collection.
        task = out / "oracle-task"
        for name in ("environment", "solution", "tests"):
            (task / name).mkdir(parents=True)
        (task / "instruction.md").write_text("Write 42 to /work/answer.txt.\n")
        (task / "solution/solve.sh").write_text("#!/bin/bash\nset -eu\necho 42 > /work/answer.txt\n")
        (task / "tests/test.sh").write_text(
            '#!/bin/bash\nset -eu\ntest "$(cat /work/answer.txt)" = 42\necho 1 > /logs/verifier/reward.txt\n'
        )
        (task / "task.toml").write_text(
            f'''schema_version = "1.4"
[task]
name = "verification/native-harbor"
[environment]
docker_image = "{args.image}"
cpus = 1
memory_mb = 256
workdir = "/work"
network_mode = "no-network"
[agent]
timeout_sec = 20
[verifier]
timeout_sec = 20
'''
        )
        job = await Job.create(
            JobConfig(
                job_name=tag,
                jobs_dir=out / "jobs",
                tasks=[TaskConfig(path=task)],
                agents=[AgentConfig(name="oracle")],
                n_attempts=1,
                n_concurrent_trials=1,
                quiet=True,
                environment={
                    "import_path": "psrl.workers.env_worker.harbor:EnvWorkerEnvironment",
                    "kwargs": {"actor_id": tag},
                },
            )
        )
        result = await job.run()
        evidence["harbor_job"] = result.model_dump(mode="json")
        trial_files = [p for p in (out / "jobs").rglob("result.json") if p.parent.parent.name == tag]
        trials = [json.loads(p.read_text()) for p in trial_files]
        evidence["trials"] = trials
        assert len(trials) == 1 and trials[0].get("exception_info") is None, "Exactly one Harbor trial must finish."
        assert trials[0]["verifier_result"]["rewards"]["reward"] == 1, "Harbor must grade the oracle successfully."
        evidence["passed"] = True

    try:
        await asyncio.wait_for(workflow(), 150)
    except BaseException:
        evidence["error"] = traceback.format_exc()
        raise
    finally:
        evidence["cleanup_errors"] = []
        for env in environments:
            try:
                await asyncio.wait_for(env.stop(), 25)
            except Exception as exc:
                evidence["cleanup_errors"].append(repr(exc))
        owned = docker("ps", "-aq", "--filter", f"label=psrl.actor_id={tag}").split()
        if owned:
            docker("rm", "-f", *owned)
        evidence["remaining_containers"] = docker("ps", "-aq", "--filter", f"label=psrl.actor_id={tag}")
        evidence["passed"] &= not evidence["remaining_containers"] and not evidence["cleanup_errors"]
        evidence["elapsed_s"] = time.monotonic() - started
        save(out / "evidence.json", evidence)
    assert evidence["passed"], "Backend or cleanup verification failed."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
