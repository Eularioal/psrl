"""Compare bounded APPS fixtures and replay one MLGym episode in the AIRS image."""

import argparse
import ast
import asyncio
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

PYTHON = "/home/agent/miniconda3/envs/mlgym_generic/bin/python"
TASK = "CodeGenerationAPPSPassAt5"


def docker(*args, timeout=20):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return result.stdout


def legacy_case(path, index):
    # Run only inside the disposable reference container, never on the host.
    sys.path.insert(0, "/reference")
    from utils import solves_testcases

    fixture = json.loads(Path(path).read_text())[index]
    print(json.dumps({"passed": bool(solves_testcases(fixture["source"], fixture["sample"]))}))


def replay_runner(payload):
    # Substitute fixed model actions only. All runner and Docker operations are real.
    from examples.airs_bench import runner

    runner.build_mlgym_model_config = lambda _: {"model_name": "replay", "replay_path": payload["replay_path"]}
    return runner.run_mlgym_agent(payload)


def csv_bytes(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow([f"code{i}" for i in range(1, 6)])
    writer.writerows(rows)
    return stream.getvalue().encode()


def select_fixtures(path):
    from datasets import load_from_disk

    dataset = load_from_disk(path)
    selected = []
    counts = {"stdin": 0, "function": 0}
    for index, row in enumerate(dataset):
        cases = json.loads(row["input_output"])
        kind = "function" if cases.get("fn_name") is not None else "stdin"
        if counts[kind] >= 2 or not cases.get("inputs"):
            continue
        for source in json.loads(row["solutions"] or "[]")[:10]:
            try:
                ast.parse(source)
            except SyntaxError:
                continue
            original_count = len(cases["inputs"])
            cases["inputs"] = cases["inputs"][:3]
            cases["outputs"] = cases["outputs"][:3]
            sample = {"input_output": json.dumps(cases)}
            selected.append(
                {"row": index, "kind": kind, "original_cases": original_count, "source": source, "sample": sample}
            )
            counts[kind] += 1
            break
        if all(value == 2 for value in counts.values()):
            break
    assert len(selected) == 4, "Need two real problems for each execution style."
    return selected


async def scenario(args, root, record):
    import yaml
    from datasets import Dataset
    from examples.airs_bench.config import AirsBenchRuntimeConfig
    from examples.airs_bench.grading import build_verifier_spec, grade_submission
    from examples.airs_bench.process_runner import run_in_process
    from psrl.environments.mlgym_env import build_sandbox_spec
    from psrl.workers.env_worker.worker import EnvWorker

    fixtures = select_fixtures(args.data)
    record["fixtures"] = [
        {k: v for k, v in f.items() if k not in ("source", "sample")}
        | {
            "source_sha256": hashlib.sha256(f["source"].encode()).hexdigest(),
            "cases": len(json.loads(f["sample"]["input_output"])["inputs"]),
        }
        for f in fixtures
    ]
    # Diagram: legacy comparison runs separately with only these bounded fixtures.
    legacy = fixtures + [
        {
            "source": "def solve(x): return {'not_an_integer': 1}",
            "sample": {"input_output": json.dumps({"fn_name": "solve", "inputs": [[1]], "outputs": [1]})},
        },
        {"source": "def broken(", "sample": fixtures[0]["sample"]},
    ]
    (root / "legacy.json").write_text(json.dumps(legacy))
    record["legacy"] = []
    for index in range(len(legacy)):
        started = time.monotonic()
        out = await asyncio.to_thread(
            docker,
            "run",
            "--rm",
            "--name",
            f"{args.run_id}-legacy-{index}",
            "--label",
            f"psrl.airs_followup={args.run_id}",
            "--network",
            "none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--cpus",
            "1",
            "--memory",
            "1g",
            "--pids-limit",
            "64",
            "--mount",
            f"type=bind,src={args.assets},dst=/reference,readonly",
            "--mount",
            f"type=bind,src={root},dst=/fixture,readonly",
            "--mount",
            f"type=bind,src={Path(__file__).resolve().parent},dst=/experiment,readonly",
            args.image,
            PYTHON,
            "/experiment/verify.py",
            "--legacy-case",
            str(index),
            timeout=25,
        )
        result = json.loads(out.strip().splitlines()[-1])
        record["legacy"].append({**result, "elapsed_s": time.monotonic() - started})
        print("legacy", index, record["legacy"][-1], flush=True)

    data = root / "airs_prepared" / TASK
    private = data / "test_with_labels"
    assets = root / "airs_configs/data" / TASK
    assets.mkdir(parents=True)
    (assets / "evaluate.py").write_text("raise RuntimeError('Unused legacy evaluator')\n")
    (assets / "requirements.txt").write_text("")
    tasks = root / "airs_configs/tasks"
    tasks.mkdir(parents=True)
    task = {
        "id": TASK,
        "name": "Bounded APPS replay",
        "description": "Submit five Python programs. {dataset_docs}",
        "use_generic_conda": True,
        "requirements_path": f"data/{TASK}/requirements.txt",
        "evaluation_paths": ["evaluate.py"],
        "use_separate_eval_container": True,
    }
    (tasks / f"{TASK}.yaml").write_text(yaml.safe_dump(task))
    for split in ("train", "test"):
        Dataset.from_list([{"question": "Bounded replay", "starter_code": ""}]).save_to_disk(data / split)
    runtime = AirsBenchRuntimeConfig(
        image=args.image,
        data_root=str(root),
        mlgym_repo=args.mlgym,
        sandbox_cpus=1,
        sandbox_memory="1g",
        verifier_cpus=1,
        verifier_memory="1g",
        per_action_timeout_s=60,
        startup_timeout_s=15,
    )
    labels = {"psrl.airs_followup": args.run_id}
    Dataset.from_list([fixtures[0]["sample"]]).save_to_disk(private)
    verifier, command = build_verifier_spec(runtime, str(data), f"tasks/{TASK}.yaml", labels)
    spec = build_sandbox_spec(runtime, str(data), labels, verifier_spec=verifier)
    timings = []

    class ObservedWorker(EnvWorker):
        async def create_sandbox(self, spec, sandbox_id=None):
            started = time.monotonic()
            sid = await super().create_sandbox(spec, sandbox_id)
            timings.append(
                {
                    "op": "create",
                    "role": spec.labels.get("psrl.airs_role", "agent"),
                    "elapsed_s": time.monotonic() - started,
                }
            )
            return sid

        async def destroy_sandbox(self, sandbox_id):
            started = time.monotonic()
            await super().destroy_sandbox(sandbox_id)
            timings.append({"op": "destroy", "id": sandbox_id, "elapsed_s": time.monotonic() - started})

    worker = ObservedWorker(0, 1, 0, 60)
    sid = await worker.create_sandbox(spec)

    async def evaluate(v, c, timeout):
        return await worker.evaluate(sid, v, c, timeout)

    async def execute(c, timeout, no_output_timeout_s=None):
        return await worker.exec(sid, c, timeout, no_output_timeout_s)

    async def write_file(path, value):
        return await worker.write_file(sid, path, value)

    handle = SimpleNamespace(sandbox_id=sid, evaluate=evaluate, exec=execute, write_file=write_file)

    async def grade(sample, sources):
        shutil.rmtree(private)
        Dataset.from_list([sample]).save_to_disk(private)
        await write_file("/home/agent/workspace/submission.csv", csv_bytes([sources]))
        started = time.monotonic()
        result = await grade_submission(handle, verifier, command, "Pass@5", 60)
        return {"score": result["Pass@5"], "elapsed_s": time.monotonic() - started}

    try:
        record["isolated"] = []
        for index, fixture in enumerate(fixtures):
            result = await grade(fixture["sample"], [fixture["source"]] * 5)
            result["matches_legacy"] = result["score"] == int(record["legacy"][index]["passed"])
            record["isolated"].append(result)
            print("isolated", index, result, flush=True)
        sample = legacy[4]["sample"]
        record["bad_then_good"] = await grade(sample, [legacy[4]["source"], "def solve(x): return x"] + ["pass"] * 3)
        record["all_bad"] = await grade(sample, [legacy[4]["source"]] * 5)
        assert record["bad_then_good"]["score"] == 1 and record["all_bad"]["score"] == 0
        # Diagram: actual MLGym process -> real remote shell -> private judge -> cleanup.
        shutil.rmtree(private)
        Dataset.from_list([fixtures[0]["sample"]]).save_to_disk(private)
        script = "import csv\nwith open('submission.csv','w',newline='') as f:\n w=csv.writer(f)\n"
        script += " w.writerow(['code1','code2','code3','code4','code5'])\n"
        script += f" w.writerow({[fixtures[0]['source']] * 5!r})\n"
        await write_file("/home/agent/workspace/make_submission.py", script.encode())
        actions = ["python make_submission.py", "validate", "submit"]
        replay = root / "replay.jsonl"
        replay.write_text(json.dumps({TASK: [f"DISCUSSION\nTake this step.\n```\n{a}\n```" for a in actions]}) + "\n")
        payload = {
            "runtime_config": runtime,
            "task_config_path": f"tasks/{TASK}.yaml",
            "max_turns": 5,
            "verifier_spec": verifier,
            "verifier_command": command,
            "metric": "Pass@5",
            "replay_path": str(replay),
        }
        record["episode"] = await asyncio.wait_for(run_in_process(payload, handle, replay_runner), 90)
        assert record["episode"]["exit_status"] == "submitted", record["episode"]
        assert record["episode"]["airs_scores"][-1]["Pass@5"] == record["isolated"][0]["score"]
        assert record["episode"]["airs_validate_count"] == record["episode"]["airs_submit_count"] == 1
    finally:
        await worker.destroy_sandbox(sid)
        record["timings"] = timings
        record["worker_remaining"] = await worker.live_sandbox_count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-case", type=int)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--data")
    parser.add_argument("--assets")
    parser.add_argument("--mlgym")
    parser.add_argument("--image", default="psrl/airs-bench-agent:latest")
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.legacy_case is not None:
        legacy_case("/fixture/legacy.json", args.legacy_case)
        return
    sys.path.insert(0, str(args.repo))
    started = time.monotonic()
    record = {
        "status": "failed",
        "run_id": args.run_id,
        "scope": "bounded real Docker and MLGym replay, no Ray/TITO/LLM/GPU",
    }
    try:
        record["image"] = docker("image", "inspect", args.image, "--format", "{{.Id}} {{.Config.User}} ").strip()
        with tempfile.TemporaryDirectory(prefix="airs-followup-") as temp:
            root = Path(temp)
            root.chmod(0o755)
            asyncio.run(asyncio.wait_for(scenario(args, root, record), 300))
        record["status"] = "passed" if all(r["matches_legacy"] for r in record["isolated"]) else "score_mismatch"
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        ids = docker("ps", "-aq", "--filter", f"label=psrl.airs_followup={args.run_id}").split()
        if ids:
            docker("rm", "-f", *ids)
        record["cleanup_remaining"] = docker(
            "ps", "-aq", "--filter", f"label=psrl.airs_followup={args.run_id}"
        ).split()
        record["duration_s"] = time.monotonic() - started
        record["source_sha256"] = {
            str(p.relative_to(args.repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.repo / "examples/airs_bench/apps").glob("*.py")
        }
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
