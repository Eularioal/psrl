"""Compare bounded real model commands on two private GPUs with equal cache budgets."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import shutil
import statistics
import subprocess
import time
import uuid
from pathlib import Path

from planner import plan

MODELS = ["Qwen3-1.7B-Base", "Qwen3-4B-Base", "SmolLM3-3B-Base"]


def query(argv):
    return subprocess.check_output(argv, text=True, timeout=15).strip()


class Worker:
    """Own a single GPU container and serialize its JSON requests."""

    def __init__(self, args, gpu, budget, index):
        self.name = f"joint-gpu-{uuid.uuid4().hex[:10]}"
        self.stderr = (args.output / f"{self.name}.stderr").open("w")
        self.process = None
        self.calls = []
        self.index = index
        gpu_uuid = gpu["uuid"]
        prefix = str(args.python_prefix)
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            self.name,
            "--label",
            f"psrl.joint_gpu={args.output.name}",
            "--network=none",
            "--cpus=8",
            "--memory=48g",
            "--memory-swap=48g",
            "--pids-limit=512",
            "-e",
            f"CUDA_VISIBLE_DEVICES={gpu_uuid}",
            "-e",
            f"EXPECTED_GPU_UUID={gpu_uuid}",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "TRANSFORMERS_OFFLINE=1",
            "-e",
            "OMP_NUM_THREADS=2",
            "-e",
            "TOKENIZERS_PARALLELISM=false",
            "-e",
            "LD_LIBRARY_PATH=/usr/local/cuda/compat:/host-driver:/usr/local/cuda/lib64",
            "-v",
            f"{args.script_dir}:/study:ro",
            "-v",
            f"{args.assets.resolve()}:/assets:ro",
            "-v",
            f"{prefix}:{prefix}:ro",
        ]
        for device in [f"/dev/nvidia{gpu['minor']}", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]:
            if Path(device).exists():
                command.extend(["--device", device])
        version = re.search(r"Kernel Module\s+([0-9.]+)", Path("/proc/driver/nvidia/version").read_text())[1]
        for library in ("libcuda", "libnvidia-ml", "libnvidia-ptxjitcompiler"):
            candidates = [Path(d) / f"{library}.so.{version}" for d in ("/usr/lib64", "/usr/lib/x86_64-linux-gnu")]
            library_path = next((p for p in candidates if p.is_file()), None)
            if library_path:
                command.extend(["-v", f"{library_path}:/host-driver/{library}.so.1:ro"])
        command.extend(
            [
                "--entrypoint",
                f"{prefix}/bin/python",
                args.image,
                "-u",
                "/study/worker.py",
                "--budget-mb",
                str(budget),
            ]
        )
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1
        )

    def read(self):
        for line in self.process.stdout:
            if line.startswith("JRESULT "):
                value = json.loads(line.removeprefix("JRESULT "))
                if value.get("status") == "error" or value.get("error"):
                    raise RuntimeError(value)
                return value
        raise RuntimeError(f"Worker {self.name!r} exited unexpectedly.")

    def call(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self.read()
        self.calls.append(dict(request=request, result=result))
        return result

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=20)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.stderr.close()


def run_block(workers, jobs, profiles, budget, policy, executor, deadline):
    for worker in workers:
        worker.call({"op": "clear"})
    planning_started = time.monotonic()
    queues = None if policy in {"fifo", "lpt"} else plan(jobs, profiles, [[], []], budget - 1024**3, policy)
    pending = (
        sorted(jobs, key=lambda j: -profiles[j["model"]]["load_s"] - profiles[j["model"]]["compute_s"])
        if policy == "lpt"
        else list(jobs)
    )
    planning_s = time.monotonic() - planning_started
    started = planning_started
    active, rows = {}, []
    free = {0, 1}
    while active or (pending if queues is None else any(queues)):
        if time.monotonic() > deadline:
            raise TimeoutError("The declared experiment budget expired.")
        for index in sorted(free.copy()):
            queue = pending if queues is None else queues[index]
            if queue:
                job = queue.pop(0)
                submitted = time.monotonic() - started
                future = executor.submit(workers[index].call, dict(op="run", **job))
                active[future] = (index, job, submitted)
                free.remove(index)
        if not active:
            break
        done, _ = concurrent.futures.wait(active, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            index, job, submitted = active.pop(future)
            rows.append(
                dict(
                    job=job,
                    gpu=index,
                    submitted_s=submitted,
                    finished_s=time.monotonic() - started,
                    result=future.result(),
                )
            )
            free.add(index)
    clears = list(executor.map(lambda w: w.call({"op": "clear"}), workers))
    return dict(
        policy=policy,
        budget_mb=budget,
        wall_s=time.monotonic() - started,
        planning_s=planning_s,
        rows=rows,
        final_clears=clears,
    )


def launch(args):
    args.output.mkdir(parents=True, exist_ok=False)
    args.script_dir = args.output.resolve() / "source"
    args.script_dir.mkdir()
    for script in Path(__file__).parent.glob("*.py"):
        shutil.copy2(script, args.script_dir / script.name)
    record = dict(
        status="running",
        image=args.image,
        blocks=[],
        profiles={},
        scope="Independent GPU command replay of existing local PostTrain programs, no Agent or EnvRouter.",
        budget_s=args.budget_seconds,
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
    )
    workers = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def save():
        temporary = args.output / "evidence.tmp"
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(args.output / "evidence.json")

    started = time.monotonic()
    deadline = started + args.budget_seconds
    try:
        record["docker"] = query(["docker", "info", "--format", "{{.ServerVersion}}"])
        record["image_id"] = query(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"])
        before = query(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"])
        record["before"] = before
        devices = []
        for index in args.gpus:
            row = next(row.split(", ") for row in before.splitlines() if int(row.split(",")[0]) == index)
            if int(row[2]) > 32:
                raise RuntimeError("A selected GPU is occupied.")
            info = next(
                p.read_text()
                for p in Path("/proc/driver/nvidia/gpus").glob("*/information")
                if row[1] in p.read_text()
            )
            devices.append(dict(index=index, uuid=row[1], minor=int(re.search(r"Device Minor:\s*(\d+)", info)[1])))
        record["devices"] = devices
        save()
        for budget in (10240, 24576):
            workers = [Worker(args, gpu, budget, i) for i, gpu in enumerate(devices)]
            record.setdefault("worker_ready", []).append([w.read() for w in workers])
            profiles = {}
            for model in MODELS:
                samples = []
                for index, worker in enumerate(workers):
                    for repeat in range(2):
                        result = worker.call(
                            dict(
                                op="profile",
                                id=f"profile-{index}-{repeat}-{model}",
                                model=model,
                                data="/assets/data/aime2025/public/rows.json",
                            )
                        )
                        samples.append(result)
                profiles[model] = dict(
                    load_s=statistics.median(v["load_s"] for v in samples if not v["cache_hit"]),
                    compute_s=statistics.median(v["compute_s"] for v in samples if v["cache_hit"]),
                    bytes=max(v["model_bytes"] for v in samples),
                    samples=samples,
                )
            record["profiles"][str(budget)] = profiles
            save()
            sets = {
                "balanced": [(m, d) for d in ("aime2025", "gsm8k") for m in MODELS],
                "skewed": [(MODELS[0], d) for d in ("aime2025", "gsm8k", "humaneval")]
                + [(MODELS[1], d) for d in ("aime2025", "gsm8k")]
                + [(MODELS[2], "aime2025")],
            }
            for workload, pairs in sets.items():
                jobs = [dict(id=f"{m}-{d}", model=m, data=f"/assets/data/{d}/public/rows.json") for m, d in pairs]
                for repeat in range(2 if budget == 10240 else 1):
                    policies = ["fifo", "lpt", "affinity", "joint"]
                    random.Random(130 + repeat).shuffle(policies)
                    for policy in policies:
                        block = run_block(workers, jobs, profiles, budget * 1024**2, policy, executor, deadline)
                        block.update(workload=workload, repeat=repeat, budget_mb=budget)
                        record["blocks"].append(block)
                        save()
            for worker in workers:
                worker.close()
            workers = []
        record["status"] = "passed"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
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
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--python-prefix", type=Path, required=True)
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--budget-seconds", type=int, default=900)
    launch(parser.parse_args())
