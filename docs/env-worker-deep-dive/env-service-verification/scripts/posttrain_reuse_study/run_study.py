"""Run bounded container comparisons of the unchanged local PostTrain verifier."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def write_json(path, value):
    """Publish each result without exposing a partially written JSON file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_state(uuid):
    """Retain real process observations without treating telemetry errors as zero usage."""
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=8,
    )
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=8,
    )
    rows = [line.split(",") for line in memory.stdout.splitlines() if line.startswith(uuid)]
    return {
        "exit_code": query.returncode,
        "processes": [line for line in query.stdout.splitlines() if uuid in line],
        "stderr": query.stderr,
        "memory_exit_code": memory.returncode,
        "memory_used_mib": int(rows[0][1].strip()) if memory.returncode == 0 and len(rows) == 1 else None,
    }


def idle(state):
    """Require both complete telemetry and physical device release."""
    return (
        state["exit_code"] == 0
        and not state["processes"]
        and state["memory_exit_code"] == 0
        and state["memory_used_mib"] is not None
        and state["memory_used_mib"] <= 16
    )


def main():
    """Stop at a fixed deadline and clean only the containers created by this study."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    frozen = output / "study"
    shutil.copytree(Path(__file__).parent, frozen, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(config["source"], frozen / "local_model.py")
    candidates = output / "candidates"
    for key in ("A", "B"):
        (candidates / key).mkdir(parents=True)
        shutil.copy2(Path(config["candidates"][key]) / "adapter.safetensors", candidates / key)
    provenance = {
        "config": config,
        "sources": {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in frozen.glob("*.py")},
        "scope": "Real local likelihood verifier. No PSRL training or EnvRouter scheduling comparison.",
    }
    write_json(output / "provenance.json", provenance)
    start = time.monotonic()
    deadline = start + config.get("budget_s", 600)
    records, owned = [], []
    status, error = "running", None
    image = subprocess.check_output(["docker", "image", "inspect", config["image"], "--format", "{{.Id}}"], text=True)
    provenance["image_id"] = image.strip()
    write_json(output / "provenance.json", provenance)
    sequence = [
        ("native-A1", "fresh", ["A"]),
        ("native-B", "fresh", ["B"]),
        ("native-A2", "fresh", ["A"]),
        ("process-reload", "fresh", ["A", "B", "A"]),
        ("resident-base", "resident", ["A", "B", "A"]),
    ]
    try:
        for label, mode, keys in sequence:
            before = gpu_state(config["gpu_uuid"])
            if not idle(before):
                raise RuntimeError(f"GPU preflight rejected {label!r}: {before!r}.")
            name = f"psrl-posttrain-reuse-{config['run_id']}-{label.lower()}"
            owned.append(name)
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--label",
                f"psrl.study={config['run_id']}",
                *config["gpu_argv"],
                "-e",
                f"CUDA_VISIBLE_DEVICES={config['gpu_uuid']}",
                "--network",
                "none",
                "--cpus",
                "4",
                "--memory",
                "24g",
                "--shm-size",
                "1g",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "HF_DATASETS_OFFLINE=1",
                "-e",
                "TOKENIZERS_PARALLELISM=false",
                "-v",
                f"{config['python_prefix']}:{config['python_prefix']}:ro",
                "-v",
                f"{config['model']}:/model:ro",
                "-v",
                f"{config['data_root']}:/data:ro",
                "-v",
                f"{frozen}:/study:ro",
                "-v",
                f"{candidates}:/candidates:ro",
                "-v",
                f"{output}:/results",
                config["image"],
                f"{config['python_prefix']}/bin/python",
                "/study/bounded_worker.py",
                "--mode",
                mode,
                "--model",
                "/model",
                "--source",
                "/study/local_model.py",
                "--checkpoints",
                *[f"/candidates/{key}" for key in keys],
                "--data-paths",
                *[f"/data/{config['benchmarks'][key]}/private/rows.json" for key in keys],
                "--expected-gpu-uuid",
                config["gpu_uuid"],
                "--output",
                f"/results/{label}.json",
            ]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The complete study budget expired.")
            began = time.monotonic()
            with (output / f"{label}.log").open("w") as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=min(180, remaining))
            record = {"label": label, "container_wall_s": time.monotonic() - began, "exit_code": run.returncode}
            records.append(record)
            if run.returncode:
                raise RuntimeError(f"Worker {label!r} failed with exit code {run.returncode}.")
            for _ in range(6):
                after = gpu_state(config["gpu_uuid"])
                if idle(after):
                    break
                time.sleep(1)
            record["physical_release"] = after
            write_json(output / "status.json", {"status": status, "records": records})
            if not idle(after):
                raise RuntimeError("Physical GPU release was not verified before the next group.")
        status = "complete"
    except Exception as exc:
        status, error = "failed", repr(exc)
    finally:
        cleanups = []
        for name in owned:
            run = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=12)
            cleanups.append({"name": name, "exit_code": run.returncode, "stderr": run.stderr})
        write_json(
            output / "status.json",
            {
                "status": status,
                "error": error,
                "records": records,
                "cleanup": cleanups,
                "elapsed_s": time.monotonic() - start,
                "gpu_after": gpu_state(config["gpu_uuid"]),
            },
        )
    if status != "complete":
        raise SystemExit(error)


if __name__ == "__main__":
    main()
