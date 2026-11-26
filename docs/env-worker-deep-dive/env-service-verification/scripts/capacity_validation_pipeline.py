"""Finish bounded calibration, freeze capacity, then run the integrated PSRL pair."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def cleanup_processes(owned):
    remaining = []
    for pid, created in owned.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
                process.terminate()
                remaining.append(process)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(remaining, timeout=10)
    for process in alive:
        process.kill()


def main(args):
    root = args.output.resolve()
    evidence = root / "calibration/evidence.json"
    scripts = Path(__file__).parent
    initial = json.loads(evidence.read_text())
    deadline = initial["started_at"] + 4300
    state = {"status": "waiting_for_calibration", "started_at": time.time(), "pid": os.getpid()}
    save(root / "pipeline.json", state)
    while time.time() < deadline:
        current = json.loads(evidence.read_text())
        if current["status"] != "running":
            break
        time.sleep(5)
    else:
        raise TimeoutError("Calibration did not finish within its budget.")
    if current["status"] != "complete" or not current.get("physical_release_verified"):
        raise RuntimeError("Calibration failed or physical cleanup is incomplete.")
    selection = root / "capacity-selection.json"
    with (root / "selection.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(scripts / "analyze_environment_capacity.py"),
                str(evidence),
                "--output",
                str(selection),
                "--groups",
                "6",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=30,
        )
    state.update(status="integrated_training", selected=json.loads(selection.read_text()))
    save(root / "pipeline.json", state)
    integrated = root / args.integrated_name
    command = [
        sys.executable,
        "-u",
        str(scripts / "integrated_partition_training.py"),
        "--repo",
        str(args.repo),
        "--output",
        str(integrated),
        "--node-ip",
        args.node_ip,
        "--model",
        str(args.model),
        "--env-model",
        str(args.env_model),
        "--data",
        str(args.data),
        "--steps",
        "4",
        "--groups",
        "6",
        "--arm-timeout",
        "3000",
        "--total-timeout",
        str(args.pair_timeout),
        "--capacity-selection",
        str(selection),
    ]
    owned = {}
    child = None
    try:
        with (root / "training-background.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(training_pid=child.pid, command=command)
            save(root / "pipeline.json", state)
            deadline = time.time() + args.pair_timeout + 300
            manifest = integrated / "source-manifest.json"
            while child.poll() is None:
                processes = [psutil.Process(child.pid), *psutil.Process(child.pid).children(recursive=True)]
                for process in processes:
                    try:
                        owned[process.pid] = process.create_time()
                    except psutil.NoSuchProcess:
                        pass
                save(root / "owned-processes.json", owned)
                if (integrated / "design.json").exists() and not manifest.exists():
                    files = {
                        str(path.relative_to(integrated / "source")): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (integrated / "source").rglob("*")
                        if path.is_file() and "__pycache__" not in path.parts
                    }
                    save(manifest, files)
                if time.time() > deadline:
                    raise TimeoutError("Integrated pair exceeded its total budget.")
                time.sleep(5)
            state["training_exit_code"] = child.returncode
    finally:
        cleanup_processes(owned)
        if child:
            child.wait(timeout=10)
        ids = subprocess.check_output(["docker", "ps", "-aq"], text=True, timeout=10).split()
        removed = []
        if ids:
            containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True, timeout=20))
            projects = {
                (container["Config"].get("Labels") or {}).get("com.docker.compose.project")
                for container in containers
                if any(
                    Path(mount.get("Source", "/")).is_relative_to(integrated) for mount in container.get("Mounts", [])
                )
            } - {None}
            for container in containers:
                if (container["Config"].get("Labels") or {}).get("com.docker.compose.project") in projects:
                    subprocess.run(
                        ["docker", "rm", "-f", container["Id"]], capture_output=True, timeout=30, check=True
                    )
                    removed.append(container["Id"])
        snapshot = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        save(
            root / "final-resource-check.json",
            {
                "at": time.time(),
                "owned_containers_removed": removed,
                "gpu_snapshot": snapshot.stdout,
                "gpu_snapshot_exit": snapshot.returncode,
            },
        )
    if state["training_exit_code"] != 0:
        state.update(status="failed", finished_at=time.time())
        save(root / "pipeline.json", state)
        raise RuntimeError("Integrated pair failed. See the preserved arm status for its cause.")
    for name in ("fixed", "resident"):
        status_path = integrated / name / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if "ray_tmp" in status:
            logs = Path(status["ray_tmp"]) / "session_latest/logs"
            texts = []
            for path in logs.glob("worker-*.out"):
                text = path.read_text(errors="replace")
                if "timing_s/step" in text:
                    texts.append(text)
            (integrated / name / "trainer-console.log").write_text("\n".join(texts))
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_integrated_partition.py"),
            str(integrated),
            "--output",
            str(root / "analysis.json"),
        ],
        check=True,
        timeout=60,
    )
    state.update(status="complete" if state["training_exit_code"] == 0 else "failed", finished_at=time.time())
    save(root / "pipeline.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "output", "model", "env-model", "data"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--integrated-name", default="integrated")
    parser.add_argument("--pair-timeout", type=int, default=6600)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        save(args.output / "pipeline-failure.json", {"at": time.time(), "error": repr(error)})
        raise
