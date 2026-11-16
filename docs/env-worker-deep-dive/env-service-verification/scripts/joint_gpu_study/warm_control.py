"""Compare dispatch after retaining every model on each GPU without block resets."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

from experiment import MODELS, Worker, query
from planner import plan


def original_check(args, workers, executor, deadline, expected):
    """Compare the adapter against the existing local program after freeing models."""

    def remaining():
        return max(0.001, deadline - time.monotonic())

    cleared = [executor.submit(worker.call, {"op": "clear"}) for worker in workers]
    for future in cleared:
        future.result(timeout=remaining())
    source = Path(__file__).resolve().parents[5] / "examples/posttrain_bench/local_model.py"
    subprocess.run(
        ["docker", "cp", str(source), f"{workers[0].name}:/tmp/reference.py"],
        check=True,
        capture_output=True,
        timeout=remaining(),
    )
    prefix = ["docker", "exec", workers[0].name, f"{args.python_prefix}/bin/python"]
    result = subprocess.run(
        prefix
        + [
            "/tmp/reference.py",
            "train",
            "--model",
            "/assets/models/Qwen3-1.7B-Base",
            "--data",
            "/assets/data/aime2025/public/rows.json",
            "--checkpoint",
            "/tmp/reference-checkpoint",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=remaining(),
    )
    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    digest_source = (
        "import hashlib,torch;from safetensors.torch import load_file;"
        "value=load_file('/tmp/reference-checkpoint/adapter.safetensors')['model.norm.weight'];"
        "print(hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())"
    )
    digest = subprocess.check_output(prefix + ["-c", digest_source], text=True, timeout=remaining()).strip()
    matched = digest == expected["norm_sha256"] and metrics["answer_nll"] == expected["answer_nll"]
    if metrics.get("optimizer_updates") != 2 or not matched:
        raise RuntimeError("The cached adapter disagrees with the existing independent training program.")
    return dict(
        status="passed",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        stdout=result.stdout,
        metrics=metrics,
        norm_sha256=digest,
        exact_adapter_match=matched,
        scope="Original local two-update training program. No independent hidden verifier was run.",
    )


def run_block(workers, jobs, profiles, caches, policy, executor, deadline):
    """Time planning and real execution while preserving all resident models."""
    started = time.monotonic()
    queues = None if policy == "fifo" else plan(jobs, profiles, caches, 23 * 1024**3, policy)
    planning_s = time.monotonic() - started
    pending, active, rows, free = list(jobs), {}, [], {0, 1}
    while active or (pending if queues is None else any(queues)):
        if time.monotonic() >= deadline:
            raise TimeoutError("The warm control exceeded its declared budget.")
        for index in sorted(free.copy()):
            queue = pending if queues is None else queues[index]
            if queue:
                job = queue.pop(0)
                future = executor.submit(workers[index].call, dict(op="run", **job))
                active[future] = index, job, time.monotonic() - started
                free.remove(index)
        done, _ = concurrent.futures.wait(active, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            index, job, submitted = active.pop(future)
            result = future.result()
            if not result["cache_hit"] or result["load_s"] or result["evicted"]:
                raise RuntimeError("The all-fit control unexpectedly reloaded or evicted a model.")
            caches[index] = result["cache_keys"]
            rows.append(
                dict(job=job, gpu=index, submitted_s=submitted, finished_s=time.monotonic() - started, result=result)
            )
            free.add(index)
    return dict(policy=policy, wall_s=time.monotonic() - started, planning_s=planning_s, rows=rows)


def launch(args):
    """Prewarm both devices, execute fixed jobs, and always reclaim owned containers."""
    if not 1 <= args.budget_seconds <= 120:
        raise ValueError("The warm control budget must be between one and 120 seconds.")
    args.output.mkdir(parents=True, exist_ok=False)
    args.script_dir = args.output.resolve() / "source"
    args.script_dir.mkdir()
    for script in Path(__file__).parent.glob("*.py"):
        shutil.copy2(script, args.script_dir / script.name)
    profile_raw = args.profiles.read_bytes()
    parent = json.loads(profile_raw)
    if parent.get("status") != "passed":
        raise ValueError("Use frozen profiles from a completed main experiment.")
    profiles = parent["profiles"]["24576"]
    record = dict(
        status="running",
        scope="Independent workers with all models retained. No Agent, EnvRouter, or official benchmark score.",
        parent_evidence_sha256=hashlib.sha256(profile_raw).hexdigest(),
        profiles=profiles,
        image=args.image,
        budget_mb=24576,
        budget_s=args.budget_seconds,
        blocks=[],
        warmup=[],
        source_hashes={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("worker.py", "experiment.py", "planner.py", "warm_control.py")
        },
        timing_boundary="Each block includes planning and six commands. Prewarm and final cleanup are separate.",
    )
    workers = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    started = time.monotonic()
    deadline = started + args.budget_seconds

    def save():
        temporary = args.output / "evidence.tmp"
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(args.output / "evidence.json")

    def remaining():
        return max(0.001, deadline - time.monotonic())

    def warm(worker):
        return [
            worker.call(dict(op="run", id=f"warm-{index}", model=model, data="/assets/data/aime2025/public/rows.json"))
            for index, model in enumerate(MODELS)
        ]

    try:
        record["docker"] = query(["docker", "info", "--format", "{{.ServerVersion}}"])
        record["image_id"] = query(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"])
        record["before"] = query(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"])
        devices = []
        for index in args.gpus:
            row = next(row.split(", ") for row in record["before"].splitlines() if int(row.split(",")[0]) == index)
            if int(row[2]) > 32:
                raise RuntimeError("A selected GPU is occupied before the warm control.")
            info = next(
                path.read_text()
                for path in Path("/proc/driver/nvidia/gpus").glob("*/information")
                if row[1] in path.read_text()
            )
            devices.append(dict(index=index, uuid=row[1], minor=int(re.search(r"Device Minor:\s*(\d+)", info)[1])))
        record["devices"] = devices
        applications = query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
        if any(device["uuid"] in applications for device in devices):
            raise RuntimeError("A selected GPU already has a compute process.")
        record["compute_processes_before"] = applications
        save()
        workers = [Worker(args, device, 24576, index) for index, device in enumerate(devices)]
        ready = [executor.submit(worker.read) for worker in workers]
        record["worker_ready"] = [future.result(timeout=remaining()) for future in ready]
        warmed = [executor.submit(warm, worker) for worker in workers]
        record["warmup"] = [future.result(timeout=remaining()) for future in warmed]
        record["startup_and_warmup_s"] = time.monotonic() - started
        caches = [group[-1]["cache_keys"] for group in record["warmup"]]
        if any(set(cache) != set(MODELS) for cache in caches):
            raise RuntimeError("Prewarm failed to retain every model on each GPU.")
        save()
        jobs = [
            dict(id=f"{model}-{data}", model=model, data=f"/assets/data/{data}/public/rows.json")
            for data in ("aime2025", "gsm8k")
            for model in MODELS
        ]
        for repeat in range(2):
            policies = ["fifo", "affinity", "joint"]
            random.Random(913 + repeat).shuffle(policies)
            for policy in policies:
                block = run_block(workers, jobs, profiles, caches, policy, executor, deadline)
                block.update(repeat=repeat, budget_mb=24576, workload="balanced")
                record["blocks"].append(block)
                save()
        identities = {}
        for block in record["blocks"]:
            for row in block["rows"]:
                result = row["result"]
                identity = tuple(
                    json.dumps(result[key], sort_keys=True)
                    for key in ("norm_sha256", "data_sha256", "loss", "losses", "optimizer_updates")
                )
                key = row["job"]["id"]
                if key in identities and identities[key] != identity:
                    raise RuntimeError("Independent warm commands produced inconsistent numerical results.")
                identities.setdefault(key, identity)
        if remaining() > 20:
            record["original_program_check"] = original_check(
                args, workers, executor, deadline, record["warmup"][0][0]
            )
        else:
            record["original_program_check"] = dict(status="not_run", reason="Insufficient remaining time budget.")
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for worker in workers:
            worker.close()
        executor.shutdown(wait=False, cancel_futures=True)
        record["elapsed_s"] = time.monotonic() - started
        record["after"] = query(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"])
        record["owned_containers"] = query(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.joint_gpu={args.output.name}"]
        )
        record["compute_processes_after"] = query(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
        )
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "assets", "python-prefix", "profiles"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--budget-seconds", type=int, default=120)
    launch(parser.parse_args())
