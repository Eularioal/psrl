"""Run one bounded partition using a verified frozen PSRL workload."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
from capacity_validation_pipeline import cleanup_processes, save

SIX_GPU_LAYOUT = range(6)


def freeze(
    reference,
    output,
    rollout_replicas,
    mode="fixed",
    sync_policy="baseline",
    headroom_mib=8192,
    budget_s=3000,
    dedicated_cap=None,
):
    """Verify the reference before copying it and replacing only the experiment runner."""
    manifest = json.loads((reference / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((reference / "source" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference source changed: {name!r}.")
    design = json.loads((reference / "design.json").read_text())
    if design["steps_per_arm"] != 4 or design["args"]["groups"] != 6:
        raise ValueError("The reference must contain four steps with six groups per step.")
    for arm in ("fixed", "resident"):
        status = json.loads((reference / arm / "status.json").read_text())
        if status.get("exit_code") != 0 or not status.get("physical_release_verified"):
            raise ValueError("Both preceding arms must have completed and released their resources.")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    shutil.copytree(reference / "source", source, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(
        Path(__file__).with_name("integrated_partition_training.py"), source / "integrated_partition_training.py"
    )
    runner = source / "integrated_partition_training.py"
    runner.write_text(runner.read_text().replace("model_headroom_mb=8192", f"model_headroom_mb={headroom_mib}"))
    expected_changes = ["integrated_partition_training.py"]
    if sync_policy != "baseline":
        from sync_overlap_study import patch_gate

        gate = source / "psrl/workers/gen/gpu_handoff.py"
        gate.write_text(patch_gate(gate.read_text()))
        enabled = sync_policy == "overlap"
        runner.write_text(
            runner.read_text().replace(
                '"psrl.env_service.enable=True",',
                f'"psrl.env_service.enable=True", "++psrl.env_service.gpu_sharing.sync_overlap={enabled}",',
            )
        )
        expected_changes.append("psrl/workers/gen/gpu_handoff.py")
    shutil.copy2(reference / "tasks.parquet", output / "tasks.parquet")
    args = design["args"]
    args.update(
        output=str(output), rollout_replicas=rollout_replicas, mode=mode, total_timeout=budget_s, arm_timeout=budget_s
    )
    selected = json.loads(Path(args["capacity_selection"]).read_text())
    args.update(dedicated_cap=dedicated_cap or selected["selected_static_cap"], shared_cap=selected["shared_cap"])
    design.update(
        args=args,
        comparison="Fixed workload comparison across static and resident GPU partitions.",
        sync_policy=sync_policy,
        model_headroom_mib=headroom_mib,
        reference=str(reference),
        gpu_budget={
            "training": [0, 1],
            "rollout": list(range(2, 2 + rollout_replicas)),
            "dedicated_environment": [i for i in (4, 5, 3) if i >= 2 + rollout_replicas],
        },
        reference_source_changes=expected_changes,
        command_limit_per_gpu=args["dedicated_cap"],
        limitations=["Sequential comparison without repeat statistics.", "Generated actions may differ."],
    )
    save(output / "design.json", design)
    frozen = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    changed = sorted(name for name in manifest if frozen.get(name) != manifest[name])
    if changed != sorted(expected_changes):
        raise ValueError(f"Unexpected source changes: {changed!r}.")
    save(output / "source-manifest.json", frozen)
    save(output / "reference-audit.json", {"files": len(manifest), "changed": changed, "data_unchanged": True})
    return source


def remove_owned_containers(output):
    """Remove only Compose projects with mounts under this experiment output."""
    ids = subprocess.check_output(["docker", "ps", "-aq"], text=True, timeout=10).split()
    if not ids:
        return []
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True, timeout=20))
    projects = {
        (c["Config"].get("Labels") or {}).get("com.docker.compose.project")
        for c in containers
        if any(Path(m.get("Source", "/")).is_relative_to(output) for m in c.get("Mounts", []))
    } - {None}
    removed = []
    for container in containers:
        if (container["Config"].get("Labels") or {}).get("com.docker.compose.project") in projects:
            subprocess.run(["docker", "rm", "-f", container["Id"]], capture_output=True, timeout=20, check=True)
            removed.append(container["Id"])
    return removed


def main(args):
    output = args.output.resolve()
    source = freeze(
        args.reference.resolve(),
        output,
        args.rollout_replicas,
        args.mode,
        args.sync_policy,
        args.headroom_mib,
        args.budget_s,
        args.dedicated_cap,
    )
    run_prepared(args, source)


def verify_release(source, output, gpu_indices=SIX_GPU_LAYOUT):
    """Inspect physical GPUs using the frozen probe, independent of installed PSRL."""
    spec = importlib.util.spec_from_file_location(
        "_partition_frozen_probe", source / "psrl/workers/env_service/probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    probe = module.NvmlProbe()
    released = False
    try:
        for _ in range(30):
            snapshot = probe.read_gpu_snapshot(gpu_indices)
            released = set(snapshot) == set(gpu_indices) and all(
                not value["pids"] and value["used_mb"] < 64 for value in snapshot.values()
            )
            if released:
                break
            time.sleep(2)
        save(output / "final-resource-check.json", {"at": time.time(), "gpus": snapshot})
        return released
    finally:
        probe.close()


def run_prepared(args, source, *, gpu_indices=SIX_GPU_LAYOUT):
    """Execute an already frozen arm under the bounded supervisor."""
    output = args.output.resolve()
    started = time.time()
    state = {"status": "initializing", "pid": os.getpid(), "started_at": started, "deadline": started + args.budget_s}
    save(output / "supervisor.json", state)
    environment = dict(os.environ)
    runtime_paths = [source / "bootstrap", source / "dependencies", source]
    local_site = getattr(args, "local_python_site", None)
    if local_site:
        runtime_paths.insert(1, Path(local_site))
    environment.update(
        PYTHONPATH=os.pathsep.join(map(str, runtime_paths)),
        PSRL_WORKSPACE=str(source),
        CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(2 + args.rollout_replicas)),
        RAY_enable_worker_prestart="0",
        RAY_prestart_worker_first_driver="0",
    )
    program = """
import argparse, json, sys
from pathlib import Path
from integrated_partition_training import preflight, run_arm
out = Path(sys.argv[1])
args = argparse.Namespace(**json.loads((out / 'design.json').read_text())['args'])
args.deadline_epoch = float(sys.argv[2])
preflight(args, out)
result = run_arm(args, out, args.mode)
if result.get('exit_code') != 0 or not result.get('physical_release_verified'):
    raise RuntimeError('The partition arm failed or did not release its devices.')
"""
    child = None
    owned = {}

    def stop(signum, frame):
        raise InterruptedError(f"Experiment interrupted by signal {signum!r}.")

    signal.signal(signal.SIGTERM, stop)
    try:
        with (output / "driver.log").open("w") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", "-c", program, str(output), str(started + args.budget_s)],
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state["child_pid"] = child.pid
            save(output / "supervisor.json", state)
            while child.poll() is None:
                try:
                    root = psutil.Process(child.pid)
                    for member in [root, *root.children(recursive=True)]:
                        try:
                            owned[member.pid] = member.create_time()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                save(output / "owned-processes.json", owned)
                training = bool(list((output / args.mode / "episodes").glob("*/episode.json")))
                if training and state["status"] == "initializing":
                    state.update(status="training", first_episode_observed_at=time.time())
                    save(output / "supervisor.json", state)
                if time.time() >= started + (args.budget_s if training else 1200):
                    raise TimeoutError("The initialization or training budget expired.")
                time.sleep(3)
            state["exit_code"] = child.returncode
            if child.returncode:
                raise RuntimeError("The partition arm failed.")
            state["status"] = "completed_pending_audit"
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        if isinstance(exc, (InterruptedError, KeyboardInterrupt)):
            state["interrupted"] = True
    finally:
        cleanup_processes(owned)
        if child:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        try:
            state["removed_containers"] = remove_owned_containers(output)
            state["physical_release_verified"] = verify_release(source, output, gpu_indices)
        except Exception as exc:
            state["cleanup_error"] = repr(exc)
        state["finished_at"] = time.time()
        save(output / "supervisor.json", state)
    from analyze_integrated_partition import analyze

    if state["status"] == "completed_pending_audit":
        result = analyze(output / args.mode)
        expected_steps = int(json.loads((output / "design.json").read_text())["steps_per_arm"])
        complete = sorted(result["steps"]) == list(range(1, expected_steps + 1)) and state.get(
            "physical_release_verified", False
        )
        state["status"] = "complete" if complete else "failed"
        save(output / "analysis.json", result)
    manifest = json.loads((output / "source-manifest.json").read_text())
    state["source_changes"] = [
        name
        for name, expected in manifest.items()
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected
    ]
    if state["source_changes"]:
        state["status"] = "failed"
    save(output / "supervisor.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-replicas", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--mode", choices=("fixed", "resident"), default="fixed")
    parser.add_argument("--sync-policy", choices=("baseline", "wait", "overlap"), default="baseline")
    parser.add_argument("--headroom-mib", type=int, default=8192)
    parser.add_argument("--budget-s", type=int, default=3000)
    parser.add_argument("--dedicated-cap", type=int, choices=(1, 2, 3, 4, 5))
    main(parser.parse_args())
