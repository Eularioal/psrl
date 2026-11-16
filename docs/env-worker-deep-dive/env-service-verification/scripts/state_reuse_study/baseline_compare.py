"""Exercise the exact fixed EnvWorker backend with a real CPU model and edited code."""

import argparse
import asyncio
import hashlib
import importlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from workspace import Workspace

BASELINE = "0eeead861ffafed6bf31c962d259fd847775dcf9"
MODEL_REVISION = "6a5829f5079c66e78d911e06fe21931cc00232f7"
MODEL = f"/hf-cache/hub/models--tomaarsen--Qwen3-Reranker-0.6B-seq-cls/snapshots/{MODEL_REVISION}"
INITIAL = dict(
    **{
        "helper.py": "OFFSET = 0\n",
        "scorer.py": "from helper import OFFSET\ndef score(logit): return logit + OFFSET\n",
    },
    **{"loader.json": '{"length":8}', "README.md": "Unrelated documentation.\n"},
)
CASES = [
    ("initial", {}, 0, True),
    ("input", {}, 1, True),
    ("unrelated", {"README.md": "Edited documentation.\n"}, 0, True),
    ("helper", {"helper.py": "OFFSET = 1\n"}, 0, True),
    ("scorer", {"scorer.py": "from helper import OFFSET\ndef score(logit): return 2 * logit + OFFSET\n"}, 0, True),
    ("loader", {"loader.json": '{"length":12}'}, 0, True),
    ("uncertain", {}, 0, False),
    ("revert-code", {"helper.py": INITIAL["helper.py"], "scorer.py": INITIAL["scorer.py"]}, 0, True),
]


def load_baseline(repo, target):
    """Isolate original standard-library backend files without modifying their contents."""
    package = target / "baseline-package"
    for path in [package / "psrl", package / "psrl/workers", package / "psrl/workers/env_worker"]:
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").write_text("")
    hashes = {}
    for name in ("worker.py", "shell.py", "sandbox.py"):
        relative = f"psrl/workers/env_worker/{name}"
        source = subprocess.check_output(["git", "show", f"{BASELINE}:{relative}"], cwd=repo)
        (package / relative).write_bytes(source)
        hashes[relative] = hashlib.sha256(source).hexdigest()
    sys.path.insert(0, str(package))
    worker = importlib.import_module("psrl.workers.env_worker.worker")
    sandbox = importlib.import_module("psrl.workers.env_worker.sandbox")
    if Path(worker.__file__).resolve() != package / "psrl/workers/env_worker/worker.py":
        raise RuntimeError("The actual backend did not load from the fixed source package.")
    return worker.EnvWorker, sandbox.SandboxSpec, hashes


async def scenario(args):
    """Bound all comparisons and preserve an independent fresh-process reference."""
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = dict(status="running", baseline=BASELINE, blocks=[], faults={})

    def save():
        temporary = args.output / "evidence.tmp"
        temporary.write_text(json.dumps(evidence, indent=2) + "\n")
        temporary.replace(args.output / "evidence.json")

    scratch = tempfile.TemporaryDirectory(prefix="state-study-", dir="/tmp")
    local = Path(scratch.name)
    worker_type, spec_type, hashes = load_baseline(args.repo, args.output)
    evidence["baseline_source_sha256"] = hashes
    evidence["image"] = args.image
    evidence["model_revision"] = MODEL_REVISION
    evidence["scope"] = (
        "Actual fixed EnvWorker CPU Docker execution, scripted edits, real model. Not science training."
    )
    evidence["resources"] = dict(cpus=8, memory="16g", gpus=0, network="none")
    token = f"state-study-{uuid.uuid4().hex[:10]}"
    evidence["token"] = token
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    worker = worker_type(0, 8, 0, 90)
    sandbox = None
    started = time.monotonic()
    retained = {}

    def remote(path):
        return "/run/" + str(path.relative_to(local))

    async def execute(argv, label, timeout=90):
        command = argv if isinstance(argv, str) else shlex.join(argv)
        result = await worker.exec(sandbox, command, timeout_s=timeout)
        (args.output / f"{label}.log").write_text(result.stdout)
        if result.timed_out or result.exit_code:
            raise RuntimeError(f"Baseline command {label!r} failed or timed out.")
        return result

    async def stop_service(state, label):
        if state.get("socket"):
            request = state["directory"] / "stop.json"
            request.write_text('{"stop":true}')
            await execute(
                [
                    "python3",
                    "/assets/model_process.py",
                    "request",
                    "--socket",
                    remote(state["socket"]),
                    "--request",
                    remote(request),
                ],
                label,
            )
            pid = state["pid"]
            await execute(f"wait {pid}", label + "-reaped", 10)
            state.pop("socket")

    async def request(policy, workspace, state, seed, complete, label, gate=None):
        version = workspace.capture()
        code = version["snapshot"] if policy == "versioned" else workspace.live
        request_path = workspace.path / f"request-{label}.json"
        request_data = dict(seed=seed, code=remote(code))
        if gate:
            request_data["gate"] = remote(gate)
        request_path.write_text(json.dumps(request_data))
        begin = time.monotonic()
        if policy == "fresh":
            result = await execute(
                [
                    "python3",
                    "/assets/model_process.py",
                    "once",
                    "--model",
                    MODEL,
                    "--config",
                    remote(workspace.live / "loader.json"),
                    "--code",
                    remote(code),
                    "--request",
                    remote(request_path),
                ],
                label,
            )
            state["builds"] += 1
        else:
            key = (version["code"], version["model"]) if policy == "restart" else version["model"]
            rebuild = "socket" not in state or (policy != "blind" and (state.get("key") != key or not complete))
            if rebuild:
                await stop_service(state, label + "-stop")
                state["builds"] += 1
                service = workspace.path / f"service-{state['builds']}"
                service.mkdir()
                state.update(directory=service, socket=service / "model.sock", key=key)
                ready = service / "ready.json"
                argv = [
                    "python3",
                    "-u",
                    "/assets/model_process.py",
                    "serve",
                    "--model",
                    MODEL,
                    "--config",
                    remote(version["snapshot"] / "loader.json"),
                    "--code",
                    remote(code),
                    "--socket",
                    remote(state["socket"]),
                    "--ready",
                    remote(ready),
                ]
                if policy in {"restart", "blind"}:
                    argv.append("--frozen")
                shell = f"nohup {shlex.join(argv)} > {shlex.quote(remote(service / 'model.log'))} 2>&1 < /dev/null &"
                await execute(shell, label + "-start")
                deadline = time.monotonic() + 80
                while not ready.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("The bounded CPU model service did not become ready.")
                    await asyncio.sleep(0.05)
                state["pid"] = json.loads(ready.read_text())["pid"]
            result = await execute(
                [
                    "python3",
                    "/assets/model_process.py",
                    "request",
                    "--socket",
                    remote(state["socket"]),
                    "--request",
                    remote(request_path),
                ],
                label,
            )
        elapsed = time.monotonic() - begin
        payload = json.loads(
            next(
                line[len("STATE_RESULT ") :] for line in result.stdout.splitlines() if line.startswith("STATE_RESULT ")
            )
        )
        if payload.get("cuda_initialized"):
            raise RuntimeError("The CPU experiment initialized CUDA.")
        if policy == "versioned":
            accepted = workspace.publish(version, payload, workspace.path / f"accepted-{label}.json")
        else:
            accepted = dict(status="unmanaged", **payload)
        return dict(
            elapsed_s=elapsed,
            result=payload,
            publication=accepted["status"],
            requested_code=version["code"],
            requested_model=version["model"],
        )

    save()
    try:
        sandbox = await worker.create_sandbox(
            spec_type(
                image=args.image,
                cpus=8,
                memory="16g",
                gpus=0,
                network="none",
                mounts=(
                    (str(Path(__file__).resolve().parent), "/assets", "ro"),
                    (str(local), "/run", "rw"),
                    (str(args.hf_cache), "/hf-cache", "ro"),
                ),
                env={
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HOME": "/hf-cache",
                    "OMP_NUM_THREADS": "4",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                },
                labels={"psrl.state_study": token},
            )
        )
        inspected = json.loads(
            subprocess.check_output(["docker", "inspect", worker._container_names[sandbox]], text=True, timeout=10)
        )[0]["HostConfig"]
        evidence["actual_container_resources"] = {
            name: inspected[name] for name in ("NanoCpus", "Memory", "NetworkMode", "Devices", "DeviceRequests")
        }
        save()
        warm = Workspace(local / "warm")
        warm.commit(INITIAL)
        await request("fresh", warm, dict(builds=0), 0, True, "warm")
        evidence["warmup_complete"] = True
        save()
        reference = None
        for block, policy in enumerate(("fresh", "restart", "split", "versioned", "versioned", "split", "blind")):
            workspace = Workspace(local / f"block-{block}")
            workspace.commit(INITIAL)
            state = dict(builds=0)
            retained[block] = state
            rows = []
            block_started = time.monotonic()
            for index, (name, changes, seed, complete) in enumerate(CASES):
                if changes:
                    workspace.commit(changes)
                row = await request(policy, workspace, state, seed, complete, f"{block}-{index}")
                row["case"] = name
                rows.append(row)
            await stop_service(state, f"{block}-stop-final")
            wall = time.monotonic() - block_started
            if reference is None:
                reference = [row["result"]["value"] for row in rows]
            wrong = [
                row["case"]
                for row, expected in zip(rows, reference, strict=True)
                if not math.isclose(row["result"]["value"], expected, rel_tol=1e-6, abs_tol=1e-5)
            ]
            if policy != "blind" and (wrong or any(row["publication"] == "rejected" for row in rows)):
                raise RuntimeError(f"Correctness failed for the main control {policy!r}.")
            evidence["blocks"].append(
                dict(block=block, policy=policy, wall_s=wall, builds=state["builds"], wrong=wrong, commands=rows)
            )
            save()
        workspace = Workspace(local / "faults")
        workspace.commit(INITIAL)
        state = dict(builds=0)
        retained["faults"] = state
        await request("versioned", workspace, state, 0, True, "fault-warm")
        gate = workspace.path / "publication-gate"
        pending = asyncio.create_task(request("versioned", workspace, state, 0, True, "fault-race", gate))
        try:
            deadline = time.monotonic() + 20
            while not Path(str(gate) + ".ready").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("The publication race did not reach its bounded gate.")
                await asyncio.sleep(0.02)
            workspace.commit({"helper.py": "OFFSET = 10\n"})
            Path(str(gate) + ".continue").write_text("continue")
            race = await pending
        finally:
            Path(str(gate) + ".continue").write_text("continue")
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        if race["publication"] != "rejected":
            raise RuntimeError("The obsolete result was incorrectly accepted.")
        current = await request("versioned", workspace, state, 0, True, "fault-retry")
        if not math.isclose(current["result"]["value"], reference[0] + 10, rel_tol=1e-6, abs_tol=1e-5):
            raise RuntimeError("The retried result did not use the committed helper code.")
        evidence["faults"] = dict(race_rejected=True, retry_correct=True)
        await stop_service(state, "fault-stop")
        evidence["status"] = "passed"
    except BaseException as error:
        evidence.update(status="failed", error=repr(error))
        raise
    finally:
        evidence["elapsed_s"] = time.monotonic() - started
        if sandbox:
            shell_process = worker._shells[sandbox].process
            container = worker._container_names[sandbox]
            await asyncio.wait_for(worker.destroy_sandbox(sandbox), timeout=30)
            await asyncio.to_thread(shell_process.wait, timeout=5)
            remaining = subprocess.check_output(
                ["docker", "ps", "-aq", "--filter", f"label=psrl.state_study={token}"], text=True, timeout=10
            ).strip()
            evidence["cleanup"] = dict(
                remaining_containers=remaining.splitlines(), shell_exit=shell_process.returncode
            )
            if remaining:
                evidence["status"] = "failed"
            evidence["container"] = container
        for path in local.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        scratch.cleanup()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--budget-seconds", type=int, default=480)
    args = parser.parse_args()
    if not 60 <= args.budget_seconds <= 480:
        raise ValueError("The active experiment budget must be between 60 and 480 seconds.")
    asyncio.run(asyncio.wait_for(scenario(args), timeout=args.budget_seconds))
