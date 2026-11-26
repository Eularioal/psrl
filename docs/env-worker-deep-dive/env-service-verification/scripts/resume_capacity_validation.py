"""Run one authorized resident attempt while preserving the completed static arm."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
from capacity_validation_pipeline import cleanup_processes, save


def main(root, attempt_name):
    root = root.resolve()
    integrated = root / "integrated-retry"
    fixed = json.loads((integrated / "fixed/status.json").read_text())
    if fixed.get("exit_code") != 0 or not fixed.get("physical_release_verified"):
        raise RuntimeError("The completed static arm must be preserved and physically released.")
    failed = integrated / "resident"
    if list(failed.glob("episodes/*/episode.json")):
        raise RuntimeError("Resident trajectory evidence already exists.")
    source = integrated / "source"
    manifest = json.loads((integrated / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name!r}.")
    attempt = root / attempt_name
    attempt.mkdir()
    if failed.exists():
        failed.rename(attempt / "previous-resident")
    preflight = integrated / "container-preflight.json"
    if preflight.exists():
        shutil.copy2(preflight, attempt / "previous-container-preflight.json")
    fixed_hash = hashlib.sha256((integrated / "fixed/status.json").read_bytes()).hexdigest()
    started_at = time.time()
    init_deadline = started_at + 1200
    deadline = started_at + 3000
    child_program = """
import argparse
import json
import sys
from pathlib import Path
from integrated_partition_training import preflight, run_arm
out = Path(sys.argv[1])
design = json.loads((out / 'design.json').read_text())
args = argparse.Namespace(**design['args'])
for name in ('repo', 'output', 'model', 'env_model', 'data', 'capacity_selection'):
    value = getattr(args, name, None)
    if value is not None:
        setattr(args, name, Path(value))
selected = json.loads(args.capacity_selection.read_text())
args.dedicated_cap = selected['selected_static_cap']
args.shared_cap = selected['shared_cap']
args.arm_timeout = 3000
args.deadline_epoch = float(sys.argv[2])
preflight(args, out)
result = run_arm(args, out, 'resident')
if result.get('exit_code') != 0 or not result.get('physical_release_verified'):
    raise RuntimeError('The resident arm failed or did not release its resources.')
"""
    command = [sys.executable, "-u", "-c", child_program, str(integrated), str(deadline)]
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join(map(str, (source / "bootstrap", source / "dependencies", source))),
        PSRL_WORKSPACE=str(source),
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        RAY_enable_worker_prestart="0",
        RAY_prestart_worker_first_driver="0",
    )
    state = {
        "status": "launching",
        "started_at": started_at,
        "pid": os.getpid(),
        "command": command,
        "initialization_deadline": init_deadline,
        "deadline": deadline,
        "frozen_training_source_changed": False,
        "fixed_status_sha256": fixed_hash,
        "frozen_files_checked": len(manifest),
        "attempt_limit": 1,
        "authorization": "User approved continuation after the previous bounded attempt stopped.",
        "startup_change": "Fresh driver with worker prestart disabled. Frozen training source is unchanged.",
    }
    save(attempt / "status.json", state)
    owned = {}
    child = None
    try:
        with (attempt / "driver.log").open("w") as log:
            child = subprocess.Popen(
                command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            state.update(status="initializing", child_pid=child.pid)
            save(attempt / "status.json", state)
            while child.poll() is None:
                process = psutil.Process(child.pid)
                for member in [process, *process.children(recursive=True)]:
                    try:
                        owned[member.pid] = member.create_time()
                    except psutil.NoSuchProcess:
                        pass
                save(attempt / "owned-processes.json", owned)
                started = bool(list((integrated / "resident").glob("episodes/*/episode.json")))
                if started and state["status"] != "training":
                    state.update(status="training", first_observed_training_at=time.time())
                    save(attempt / "status.json", state)
                if time.time() >= deadline or (not started and time.time() >= init_deadline):
                    raise TimeoutError("The authorized initialization or arm budget expired.")
                time.sleep(3)
            state["exit_code"] = child.returncode
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
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
                    Path(mount.get("Source", "/")).is_relative_to(integrated / "resident")
                    for mount in container.get("Mounts", [])
                )
            } - {None}
            for container in containers:
                if (container["Config"].get("Labels") or {}).get("com.docker.compose.project") in projects:
                    subprocess.run(
                        ["docker", "rm", "-f", container["Id"]], capture_output=True, timeout=30, check=True
                    )
                    removed.append(container["Id"])
        snapshot = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        save(
            attempt / "resource-check.json",
            {"at": time.time(), "removed": removed, "gpu_snapshot": snapshot.stdout, "exit_code": snapshot.returncode},
        )
        state["finished_at"] = time.time()
        state["fixed_status_unchanged"] = (
            hashlib.sha256((integrated / "fixed/status.json").read_bytes()).hexdigest() == fixed_hash
        )
        save(attempt / "status.json", state)
    if state["exit_code"] != 0:
        state["status"] = "failed"
        save(attempt / "status.json", state)
        raise RuntimeError("The remaining resident arm did not complete.")
    subprocess.run(
        [
            sys.executable,
            str(source.parent.parent / "runner-retry/analyze_integrated_partition.py"),
            str(integrated),
            "--output",
            str(root / "analysis.json"),
        ],
        check=True,
        timeout=60,
    )
    complete = json.loads((root / "analysis.json").read_text())["paired_complete"]
    state["status"] = "complete" if complete else "failed"
    save(attempt / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--attempt-name", required=True)
    args = parser.parse_args()
    if Path(args.attempt_name).name != args.attempt_name:
        parser.error("The attempt name must be a single directory name.")
    main(args.root, args.attempt_name)
