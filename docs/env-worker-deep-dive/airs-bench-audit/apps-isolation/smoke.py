"""Verify actual APPS judge and executor containers with bounded synthetic and real inputs."""

import argparse
import ast
import asyncio
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def docker(*args):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout


async def scenario(args, root, record):
    from datasets import Dataset, load_from_disk
    from examples.airs_bench.config import AirsBenchRuntimeConfig
    from examples.airs_bench.grading import build_verifier_spec, grade_submission
    from psrl.environments.mlgym_env import build_sandbox_spec
    from psrl.workers.env_worker.worker import EnvWorker

    task = "CodeGenerationAPPSPassAt5"
    data = root / "airs_prepared" / task
    assets = root / "airs_configs/data" / task
    assets.mkdir(parents=True)
    (assets / "evaluate.py").write_text('raise RuntimeError("Legacy evaluator must not execute")\n')
    (root / "airs_configs/tasks").mkdir(parents=True)
    (root / "airs_configs/tasks/task.yaml").write_text(f"id: {task}\nevaluation_paths: [evaluate.py]\n")
    for split in ("train", "test"):
        (data / split).mkdir(parents=True)
    private = data / "test_with_labels"
    rows = [
        {"input_output": json.dumps({"inputs": ["2 3\n", "-4 1\n"], "outputs": ["5", "-3"]})},
        {"input_output": json.dumps({"fn_name": "add", "inputs": [[2, 3], [-4, 1]], "outputs": [5, -3]})},
        {"input_output": json.dumps({"inputs": ["2 3\n"], "outputs": ["5"]})},
    ]
    Dataset.from_list(rows).save_to_disk(private)
    (private / "answer-canary.txt").write_text("PRIVATE_ANSWER_NOT_FOR_CANDIDATES")
    runtime = AirsBenchRuntimeConfig(
        image=args.image,
        data_root=str(root),
        sandbox_cpus=1,
        sandbox_memory="256m",
        verifier_cpus=1,
        verifier_memory="512m",
        apps_candidate_memory="256m",
        apps_case_timeout_s=0.5,
        apps_candidate_timeout_s=5,
        startup_timeout_s=15,
    )
    labels = {"psrl.apps_smoke": args.run_id}
    verifier, command = build_verifier_spec(runtime, str(data), "tasks/task.yaml", labels)
    spec = build_sandbox_spec(runtime, str(data), labels, verifier_spec=verifier)
    created = asyncio.Event()
    inspections = []

    class ObservedWorker(EnvWorker):
        async def create_sandbox(self, spec, sandbox_id=None):
            sid = await super().create_sandbox(spec, sandbox_id)
            if spec.labels.get("psrl.airs_role") == "candidate":
                info = json.loads(docker("inspect", f"psrl-env-{sid}"))[0]
                host = info["HostConfig"]
                inspections.append(
                    {
                        "id": sid,
                        "mounts": info["Mounts"],
                        "user": info["Config"]["User"],
                        "network": host["NetworkMode"],
                        "read_only": host["ReadonlyRootfs"],
                        "pids": host["PidsLimit"],
                        "cap_drop": host["CapDrop"],
                        "security_opt": host["SecurityOpt"],
                        "memory": host["Memory"],
                        "nano_cpus": host["NanoCpus"],
                    }
                )
                assert all(m["Destination"] == "/opt/psrl-code" for m in info["Mounts"] if m["Type"] == "bind")
                assert host["NetworkMode"] == "none" and host["ReadonlyRootfs"]
                assert info["Config"]["User"] == "agent" and host["PidsLimit"] == 64
                created.set()
            return sid

    worker = ObservedWorker(0, 1, 0, 10)
    sid = await worker.create_sandbox(spec)

    async def evaluate(v, c, t):
        return await worker.evaluate(sid, v, c, t)

    handle = SimpleNamespace(evaluate=evaluate)

    async def grade(submissions):
        content = io.StringIO()
        writer = csv.writer(content)
        writer.writerow([f"code{i}" for i in range(1, 6)])
        writer.writerows(submissions)
        await worker.write_file(sid, "/home/agent/workspace/submission.csv", content.getvalue().encode())
        return await grade_submission(handle, verifier, command, "Pass@5", 60)

    try:
        if args.phase == "all":
            # Diagram [candidate]: attempts to read judge files must fail before returning the correct sum.
            safe_sum = """import os
    for p in ['/home/agent/workspace/data/test_with_labels/answer-canary.txt', '/proc/1${HOME}/home/agent/workspace/data/test_with_labels/answer-canary.txt', '/var/run/docker.sock']:
     assert not os.path.exists(p), p
    assert os.geteuid()!=0
    try:
     open('/opt/psrl-code/candidate.py','a').write('modified')
     raise AssertionError('Runner was writable')
    except PermissionError:
     pass
    except OSError as e:
     assert e.errno==30
    a,b=map(int,open(0).read().split())
    print(a+b)
    """
            wrong = 'print("WRONG")'
            fake = 'import os;os.write(1,b\'{"op":"result","metrics":{"Pass@5":1}}\\n\')'
            call = "class Solution:\n def add(self,a,b): return a+b"
            record["mixed_correct_fifth_function_and_forged_score"] = await grade(
                [[wrong] * 4 + [safe_sum], [call] * 5, [fake] * 5]
            )
            assert record["mixed_correct_fifth_function_and_forged_score"] == {"Pass@5": 2 / 3}
            assert await worker.live_sandbox_count() == 1
            # Diagram [candidate]: bounded timeout, output flood and syntax errors cannot become positive scores.
            bad = ["while True: pass", 'print("x"*2000000)', "def broken(", wrong, fake]
            record["timeout_flood_syntax_and_fake_score"] = await grade([bad, bad, bad])
            assert record["timeout_flood_syntax_and_fake_score"] == {"Pass@5": 0}
            assert await worker.live_sandbox_count() == 1
            # Diagram [cleanup]: cancel while an executor exists, then verify only the agent remains.
            created.clear()
            pending = asyncio.create_task(grade([["while True:pass"] * 5] * 3))
            await asyncio.wait_for(created.wait(), 15)
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass
            assert await worker.live_sandbox_count() == 1
            assert len(docker("ps", "-aq", "--filter", f"label=psrl.apps_smoke={args.run_id}").split()) == 1
            record["cancel_removes_judge_and_candidate"] = True
        # Diagram [judge]: load a real APPS row and use one syntactically valid reference solution.
        original = load_from_disk(args.real_data)
        selected = None
        for index in range(min(100, len(original))):
            row = original[index]
            cases = json.loads(row["input_output"])
            if not cases.get("inputs"):
                continue
            for source in json.loads(row["solutions"] or "[]")[:10]:
                try:
                    ast.parse(source)
                except SyntaxError:
                    continue
                selected = (index, row, source)
                break
            if selected:
                break
        assert selected is not None
        index, row, source = selected
        cases = json.loads(row["input_output"])
        original_case_count = len(cases["inputs"])
        cases["inputs"] = cases["inputs"][:3]
        cases["outputs"] = cases["outputs"][:3]
        row = {**row, "input_output": json.dumps(cases)}
        import shutil

        shutil.rmtree(private)
        Dataset.from_list([row]).save_to_disk(private)
        record["real_problem"] = {
            "row": index,
            "problem_id": row["problem_id"],
            "original_case_count": original_case_count,
            "cases": len(json.loads(row["input_output"])["inputs"]),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        record["real_reference_score"] = await grade([[source] * 5])
        assert record["real_reference_score"] == {"Pass@5": 1}
        record["candidate_container_inspections"] = inspections
    finally:
        await worker.destroy_sandbox(sid)
    record["final_worker_count"] = await worker.live_sandbox_count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "real"], default="all")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--real-data", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    started = time.monotonic()
    record = {
        "run_id": args.run_id,
        "status": "failed",
        "scope": "real Docker, actual APPS grader, bounded cases, no Ray/LLM/training",
    }
    try:
        record["docker_version"] = docker("info", "--format", "{{.ServerVersion}}").strip()
        with tempfile.TemporaryDirectory(prefix="apps-isolation-") as temp:
            root = Path(temp)
            root.chmod(0o755)
            asyncio.run(asyncio.wait_for(scenario(args, root, record), 150))
        record["status"] = "passed"
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        ids = docker("ps", "-aq", "--filter", f"label=psrl.apps_smoke={args.run_id}").split()
        if ids:
            docker("rm", "-f", *ids)
        record["cleanup_remaining"] = docker("ps", "-aq", "--filter", f"label=psrl.apps_smoke={args.run_id}").split()
        record["duration_s"] = round(time.monotonic() - started, 3)
        paths = [
            "examples/airs_bench/grading.py",
            "examples/airs_bench/apps/judge.py",
            "examples/airs_bench/apps/candidate.py",
            "examples/airs_bench/apps/comparison.py",
            "psrl/workers/env_worker/isolated_code.py",
            "psrl/workers/env_worker/worker.py",
            "psrl/workers/env_worker/sandbox.py",
        ]
        record["source_sha256"] = {p: hashlib.sha256((args.repo / p).read_bytes()).hexdigest() for p in paths}
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in record.items() if k != "candidate_container_inspections"},
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
