#!/usr/bin/env python3
"""Run a bounded Docker smoke test using current AIRS production code and tiny CSV fixtures."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def command(argv, timeout=10, check=True):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"Command failed: {argv!r}: {result.stderr[-2000:]}")
    return result


async def scenario(args, root, record):
    from examples.airs_bench.config import AirsBenchRuntimeConfig
    from examples.airs_bench.grading import build_verifier_spec, grade_submission
    from psrl.environments.mlgym_env import build_sandbox_spec
    from psrl.workers.env_worker.worker import EnvWorker

    # Diagram [3]: public and private splits are separate host directories.
    data = root / "airs_prepared/tiny"
    assets = root / "airs_configs/data/tiny"
    assets.mkdir(parents=True)
    (root / "airs_configs/tasks").mkdir(parents=True)
    (root / "airs_configs/tasks/tiny.yaml").write_text("id: tiny\nname: Tiny\nevaluation_paths: [evaluate.py]\n")
    for split in ("train", "test", "test_with_labels"):
        (data / split).mkdir(parents=True)
    (data / "test/inputs.csv").write_text("id\n0\n1\n2\n")
    (data / "test_with_labels/labels.csv").write_text("label\n0\n1\n1\n")
    (assets / "evaluate.py").write_text(
        "import argparse, csv, json\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--submission-file', required=True)\n"
        "a = p.parse_args()\n"
        "pred = list(csv.DictReader(open(a.submission_file)))\n"
        "gold = list(csv.DictReader(open('data/test_with_labels/labels.csv')))\n"
        "assert len(pred) == len(gold) == 3\n"
        "score = sum(x['label'] == y['label'] for x, y in zip(pred, gold)) / len(gold)\n"
        "print(json.dumps({'Accuracy': score}))\n"
    )
    runtime = AirsBenchRuntimeConfig(
        image=args.image,
        data_root=str(root),
        sandbox_cpus=1,
        sandbox_memory="256m",
        verifier_cpus=1,
        verifier_memory="256m",
        startup_timeout_s=15,
    )
    labels = {"psrl.airs_smoke": args.run_id}
    agent_spec = build_sandbox_spec(runtime, str(data), labels)
    verifier_spec, grading_command = build_verifier_spec(runtime, str(data), "tasks/tiny.yaml", labels)
    worker = EnvWorker(0, 1, 0, 10)
    sandbox_id = await worker.create_sandbox(agent_spec)
    record["sandbox_id"] = sandbox_id
    try:
        name = f"psrl-env-{sandbox_id}"
        inspection = json.loads(command(["docker", "inspect", name]).stdout)[0]
        record["agent_container"] = {
            "user": inspection["Config"]["User"],
            "network": inspection["HostConfig"]["NetworkMode"],
            "nano_cpus": inspection["HostConfig"]["NanoCpus"],
            "memory": inspection["HostConfig"]["Memory"],
            "mounts": inspection["Mounts"],
        }

        async def execute(text):
            result = await worker.exec(sandbox_id, text, 5)
            assert result.exit_code == 0 and not result.timed_out, dataclasses.asdict(result)
            return result.stdout

        # Diagram [5]: real container shell state and filesystem isolation.
        record["public_data"] = await execute("cat data/test/inputs.csv")
        record["private_data_denied"] = await execute(
            "test ! -e data/test_with_labels/labels.csv && "
            "test ! -e /opt/airs-verifier/evaluate.py && "
            "test ! -S /var/run/docker.sock && echo PRIVATE_DATA_UNAVAILABLE"
        )
        record["read_only"] = await execute(
            "if echo changed >> data/test/inputs.csv 2>/dev/null; then false; else echo READ_ONLY; fi"
        )
        await execute("export AIRS_MARKER=persisted; cd /tmp")
        record["shell_state"] = await execute('test "$AIRS_MARKER" = persisted && pwd')
        await execute("cd /home/agent/workspace")
        failed = await worker.exec(sandbox_id, "echo EXPECTED_NONZERO; false", 5)
        assert failed.exit_code == 1 and not failed.timed_out
        record["nonzero_command"] = dataclasses.asdict(failed)

        # Diagram [6]: only CSV crosses into a fresh, real verifier container.
        await worker.write_file(sandbox_id, "/home/agent/workspace/submission.csv", b"label\n0\n1\n0\n")
        await worker.write_file(sandbox_id, "/home/agent/workspace/evaluate.py", b'print("FAKE_SCORE=1")\n')

        async def evaluate(spec, text, timeout_s):
            return await worker.evaluate(sandbox_id, spec, text, timeout_s)

        score = await grade_submission(
            SimpleNamespace(evaluate=evaluate),
            verifier_spec,
            grading_command,
            "Accuracy",
            10,
        )
        assert score == {"Accuracy": 2 / 3}, score
        record["metrics_despite_agent_fake_evaluator"] = score
        live = command(["docker", "ps", "-aq", "--filter", f"label=psrl.airs_smoke={args.run_id}"]).stdout.split()
        assert len(live) == 1 and await worker.live_sandbox_count() == 1, live
        record["verifier_removed_before_agent_exit"] = True

        # Diagram [5]/[8]: timeout physically removes the container before return.
        result = await worker.exec(sandbox_id, "echo A-start; sleep 3; echo A-end", 0.3)
        assert result.timed_out, dataclasses.asdict(result)
        record["timeout"] = dataclasses.asdict(result)
        try:
            await worker.exec(sandbox_id, "echo B-output", 2)
        except KeyError:
            record["next_command_rejected"] = True
        else:
            raise AssertionError("Timed out sandbox accepted another command.")
        assert await worker.live_sandbox_count() == 0
        record["containers_after_timeout"] = command(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.airs_smoke={args.run_id}"]
        ).stdout.split()
        assert record["containers_after_timeout"] == []
    finally:
        await worker.destroy_sandbox(sandbox_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-image", default="python:3.13-slim")
    args = parser.parse_args()
    args.image = f"airs-smoke-{args.run_id}:local"
    sys.path.insert(0, str(args.repo.resolve()))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    record = {
        "run_id": args.run_id,
        "status": "failed",
        "scope": "real Docker, synthetic CSV, no Ray/LLM/GPU",
    }
    with tempfile.TemporaryDirectory(prefix="airs-smoke-") as tmp:
        root = Path(tmp)
        root.chmod(0o755)
        try:
            record["docker_version"] = command(["docker", "info", "--format", "{{.ServerVersion}}"], 5).stdout.strip()
            record["imports"] = {
                name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue")
            }
            paths = [
                "psrl/workers/env_worker/worker.py",
                "psrl/workers/env_worker/shell.py",
                "psrl/workers/env_worker/sandbox.py",
                "psrl/environments/mlgym_env.py",
                "examples/airs_bench/grading.py",
            ]
            record["source_sha256"] = {p: hashlib.sha256((args.repo / p).read_bytes()).hexdigest() for p in paths}
            command(["docker", "image", "inspect", args.base_image], 5)
            (root / "Dockerfile").write_text(
                f"FROM {args.base_image}\n"
                "RUN useradd -m -u 1000 agent && mkdir -p /home/agent/workspace "
                "/home/agent/miniconda3/envs/mlgym_generic/bin && "
                "ln -s /usr/local/bin/python /home/agent/miniconda3/envs/mlgym_generic/bin/python && "
                "chown -R agent:agent /home/agent\nUSER agent\nWORKDIR /home/agent/workspace\n"
            )
            build = command(
                [
                    "docker",
                    "build",
                    "--network=none",
                    "--pull=false",
                    "-t",
                    args.image,
                    str(root),
                ],
                60,
            )
            record["build_log"] = build.stdout + build.stderr
            record["image_id"] = command(
                ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], 5
            ).stdout.strip()
            asyncio.run(asyncio.wait_for(scenario(args, root, record), timeout=90))
            record["status"] = "passed"
        except BaseException as error:
            record["error"] = repr(error)
            raise
        finally:
            owned = command(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "--filter",
                    f"label=psrl.airs_smoke={args.run_id}",
                ]
            ).stdout.split()
            if owned:
                command(["docker", "rm", "-f", *owned], 20)
            record["cleanup_remaining_containers"] = command(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "--filter",
                    f"label=psrl.airs_smoke={args.run_id}",
                ]
            ).stdout.split()
            record["image_remove_exit_code"] = command(
                ["docker", "image", "rm", args.image], 15, check=False
            ).returncode
            record["duration_s"] = round(time.monotonic() - started, 3)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record, indent=2), flush=True)
            assert record["cleanup_remaining_containers"] == []


if __name__ == "__main__":
    main()
