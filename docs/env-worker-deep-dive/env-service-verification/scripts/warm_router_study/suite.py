"""Freeze and run the bounded three-round, nine-block placement comparison."""

import argparse
import difflib
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

WORK_BUDGETS = {"router": 480.0, "baseline": 360.0}
CLEANUP_RESERVE_S = 190.0
MINIMUM_WORK_S = 90.0


def planned_runs(gpus):
    """Preregister all six processes before running any GPU workload."""
    plan = []
    for round_id in (1, 2, 3):
        backends = ("router", "baseline") if round_id == 2 else ("baseline", "router")
        devices = list(reversed(gpus)) if round_id == 2 else list(gpus)
        for backend in backends:
            run_id = f"r{round_id}-{backend}"
            labels = ["baseline"] if backend == "baseline" else (["on", "off"] if round_id == 2 else ["off", "on"])
            plan.append(
                dict(
                    id=run_id,
                    round=round_id,
                    backend=backend,
                    path=f"runs/{run_id}/status.json",
                    planned_labels=labels,
                    gpus=devices,
                    work_budget_cap_s=WORK_BUDGETS[backend],
                    status="planned",
                    stdout=f"logs/{run_id}.stdout.log",
                )
            )
    return plan


def save_manifest(root, manifest):
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(root / "manifest.json")


def freeze_sources(root):
    """Copy the complete scripts tree and the two policy modules once."""
    source = Path(__file__).resolve().parent.parent
    repository = Path(__file__).resolve().parents[5]
    if root.is_relative_to(source):
        raise ValueError("The result root must be outside the helper scripts directory.")
    target = root / "source-snapshot"
    target.mkdir()
    hashes = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    modules = {}
    for source_name, target_name in (("warm_placement", "placement-policy"), ("step_predictor", "step-predictor")):
        original = repository / "psrl/workers/env_service" / (source_name + ".py")
        destination = root / (target_name + ".py")
        shutil.copy2(original, destination)
        modules[target_name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    entry = target / "warm_router_study/long_study.py"
    if not entry.is_file():
        raise FileNotFoundError("The frozen long_study.py entry point is missing.")
    return entry, hashes, modules


def cleanup_passed(record, backend, gpus):
    """Require physical release and, for Router, an empty lease ledger."""
    if record.get("cleanup_errors") != [] or record.get("containers_remaining") not in ("", []):
        return False
    snapshots = record.get("gpu_after")
    if not isinstance(snapshots, dict) or set(map(str, snapshots)) != set(map(str, gpus)):
        return False
    for snapshot in snapshots.values():
        if not isinstance(snapshot, dict) or snapshot.get("pids") != []:
            return False
        memory = snapshot.get("used_mb")
        if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not math.isfinite(memory):
            return False
        if not 0 <= memory < 128:
            return False
    if backend == "router":
        ledger = record.get("ledger_after_cleanup")
        if not isinstance(ledger, list) or not ledger:
            return False
        if any(not isinstance(unit, dict) or unit.get("leases_active") != 0 for unit in ledger):
            return False
    return True


def child_command(args, root, entry, run, work_budget_s, image_id):
    command = [
        sys.executable,
        "-u",
        str(entry),
        "--root",
        str(root / "runs" / run["id"]),
        "--assets",
        str(args.assets.resolve()),
        "--node-ip",
        args.node_ip,
        "--gpus",
        *map(str, run["gpus"]),
        "--image",
        image_id,
        "--budget-s",
        str(work_budget_s),
        "--backend",
        run["backend"],
        "--baseline-repo",
        str(args.baseline_repo.resolve()),
        "--round",
        str(run["round"]),
        "--policies",
        *run["planned_labels"],
        "--policy-source",
        str(root / "placement-policy.py"),
        "--predictor-source",
        str(root / "step-predictor.py"),
    ]
    for option in ("container_python", "python_prefix"):
        value = getattr(args, option)
        if value is not None:
            command.extend(("--" + option.replace("_", "-"), str(value)))
    return command


def run_child(command, stdout, timeout_s):
    """Let child cleanup finish, with a final process-group watchdog."""
    hard_deadline = time.monotonic() + timeout_s
    with stdout.open("wb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout_s), False
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            try:
                os.killpg(process.pid, signal.SIGINT)
                grace = min(180, max(0, hard_deadline - time.monotonic() - 5)) if interrupted else 3
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
            except ProcessLookupError:
                process.wait(timeout=2)
            if interrupted:
                raise
            return process.returncode, True


def resume_startup(args, root):
    """Resume once after a verified pre-work import failure, keeping the original deadline."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("resume_count", 0) or manifest.get("budget_s") != args.budget_s:
        raise ValueError("Only one startup repair is allowed, with the original suite budget.")
    original_plan = planned_runs(args.gpus)
    runs = manifest.get("runs", [])
    if len(runs) != len(original_plan):
        raise ValueError("The original six-run registration is required.")
    for actual, planned in zip(runs, original_plan, strict=True):
        for field in ("id", "round", "backend", "path", "planned_labels", "gpus", "work_budget_cap_s"):
            if actual.get(field) != planned[field]:
                raise ValueError(f"The preregistered {field!r} cannot change during startup repair.")
    failed = [index for index, value in enumerate(runs) if value.get("status") == "failed"]
    if len(failed) != 1:
        raise ValueError("Exactly one failed startup run is required.")
    index = failed[0]
    if any(item.get("status") != "passed" for item in runs[:index]) or any(
        item.get("status") != "planned" for item in runs[index + 1 :]
    ):
        raise ValueError("Completed runs must precede the failed startup and untouched planned runs.")
    item = runs[index]
    record = json.loads((root / item["path"]).read_text())
    if item["backend"] != "router" or "long_study_backend" not in record.get("error", ""):
        raise ValueError("This repair is limited to the known Router startup import failure.")
    if any(record.get(field, []) != [] for field in ("profiles", "blocks", "requests", "service_ready")):
        raise ValueError("A run that submitted workload cannot be retried.")
    if not cleanup_passed(record, "router", item["gpus"]):
        preflight = json.loads((root / "startup-repair-preflight.json").read_text())
        if any(record.get(field) for field in ("units", "leases", "containers", "ledger_after_cleanup")):
            raise ValueError("Fresh preflight can only replace cleanup evidence before resources were created.")
        checked = preflight.get("checked_at", 0)
        if (
            preflight.get("tag") != record.get("tag")
            or not isinstance(checked, (int, float))
            or not record.get("finished", math.inf) <= checked <= time.time() + 1
            or time.time() - checked > 300
            or not cleanup_passed(preflight, "baseline", item["gpus"])
        ):
            raise ValueError("Fresh physical cleanup evidence is required for the exact failed startup.")
    command = runs[0].get("command", [])
    expected = {
        "--assets": str(args.assets.resolve()),
        "--node-ip": args.node_ip,
        "--baseline-repo": str(args.baseline_repo.resolve()),
        "--container-python": str(args.container_python) if args.container_python else None,
        "--python-prefix": str(args.python_prefix) if args.python_prefix else None,
    }
    for flag, value in expected.items():
        recorded = command[command.index(flag) + 1] if flag in command else None
        if recorded != value:
            raise ValueError(f"The original public parameter {flag!r} cannot change.")
    if args.image not in (manifest.get("image_name"), manifest.get("image_id")):
        raise ValueError("The original immutable Docker image must be preserved.")
    snapshot = root / "source-snapshot"
    for relative, digest in manifest["source_hashes"].items():
        if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"The frozen source {relative!r} changed before repair.")
    for name, digest in manifest["module_hashes"].items():
        if hashlib.sha256((root / (name + ".py")).read_bytes()).hexdigest() != digest:
            raise ValueError(f"The frozen policy module {name!r} cannot change.")
    relative = "warm_router_study/experiment.py"
    destination = snapshot / relative
    original = destination.read_bytes()
    updated = Path(__file__).with_name("experiment.py").read_bytes()
    if original == updated:
        raise ValueError("No startup repair is present in experiment.py.")
    attempt = root / "attempts" / (item["id"] + "-startup-1")
    attempt.mkdir(parents=True, exist_ok=False)
    shutil.move(str((root / item["path"]).parent), str(attempt / "run"))
    shutil.move(str(root / item["stdout"]), str(attempt / "stdout.log"))
    patch = "".join(
        difflib.unified_diff(
            original.decode().splitlines(keepends=True),
            updated.decode().splitlines(keepends=True),
            fromfile="original/" + relative,
            tofile="repaired/" + relative,
        )
    )
    (attempt / "startup-repair.patch").write_text(patch)
    evidence = dict(
        original_path=item["path"],
        archived_path=str((attempt / "run/status.json").relative_to(root)),
        archived_stdout=str((attempt / "stdout.log").relative_to(root)),
        failure_summary="Router startup could not import dynamically loaded long_study_backend in a Ray worker.",
        no_training_work=True,
        profiles=0,
        blocks=0,
        requests=0,
        physical_cleanup_passed=True,
        changed_source=relative,
        original_sha256=hashlib.sha256(original).hexdigest(),
        repaired_sha256=hashlib.sha256(updated).hexdigest(),
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
        patch_path=str((attempt / "startup-repair.patch").relative_to(root)),
    )
    (attempt / "repair.json").write_text(json.dumps(evidence, indent=2) + "\n")
    destination.write_bytes(updated)
    manifest["source_hashes"][relative] = evidence["repaired_sha256"]
    item.clear()
    item.update(original_plan[index], startup_attempts=[evidence])
    manifest.update(resume_count=1, resumed_at=time.time(), startup_repair=evidence, status="running")
    manifest.pop("stop_reason", None)
    manifest.pop("finished", None)
    save_manifest(root, manifest)
    return manifest, snapshot / "warm_router_study/long_study.py"


def run(args):
    root = args.root.resolve()
    if args.resume:
        manifest, entry = resume_startup(args, root)
        elapsed_before = max(0, time.time() - manifest["started"])
        began = time.monotonic() - elapsed_before
        deadline = began + args.budget_s
        image_id = manifest["image_id"]
        return execute_runs(args, root, manifest, entry, image_id, began, deadline)
    root.mkdir(parents=True, exist_ok=False)
    (root / "runs").mkdir()
    (root / "logs").mkdir()
    began = time.monotonic()
    deadline = began + args.budget_s
    manifest = dict(
        status="preparing",
        started=time.time(),
        budget_s=args.budget_s,
        rounds=3,
        plan=dict(rounds=3, blocks=9, runs=6, budget_s=args.budget_s, retry_count=0),
        protocol=dict(
            target_model="Qwen3-4B-Base",
            blocker_model="Qwen3-1.7B-Base",
            target_steps=1536,
            blocker_steps=2048,
            profile_steps=128,
            progress_every=16,
            arrival_after_completed_steps=64,
            switch_margin_s=2.0,
            decision_mode="median_point_with_2s_margin",
            interval_role="audit only, empirical padded quantiles with no coverage guarantee",
        ),
        image_name=args.image,
        runs=planned_runs(args.gpus),
    )
    save_manifest(root, manifest)
    try:
        entry, hashes, modules = freeze_sources(root)
        manifest.update(source_hashes=hashes, module_hashes=modules)
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True, timeout=15
        ).strip()
        if not image_id.startswith("sha256:"):
            raise RuntimeError("Docker did not resolve the image to an immutable image ID.")
        manifest.update(image_id=image_id, status="running")
        save_manifest(root, manifest)
    except BaseException as exc:
        manifest.update(status="incomplete", stop_reason="suite_exception", error=repr(exc))
        manifest.update(finished=time.time(), elapsed_s=time.monotonic() - began)
        save_manifest(root, manifest)
        raise
    return execute_runs(args, root, manifest, entry, image_id, began, deadline)


def execute_runs(args, root, manifest, entry, image_id, began, deadline):
    try:
        for item in manifest["runs"]:
            if item["status"] == "passed":
                continue
            work_budget = min(item["work_budget_cap_s"], deadline - time.monotonic() - CLEANUP_RESERVE_S)
            if work_budget < MINIMUM_WORK_S:
                manifest.update(status="incomplete", stop_reason="insufficient_remaining_budget")
                break
            command = child_command(args, root, entry, item, work_budget, image_id)
            item.update(status="running", started=time.time(), budget_s=work_budget, command=command)
            save_manifest(root, manifest)
            print(f"Starting {item['id']}, policies={item['planned_labels']}, GPUs={item['gpus']}", flush=True)
            return_code, timed_out = run_child(command, root / item["stdout"], work_budget + 185)
            item.update(return_code=return_code, timed_out=timed_out, finished=time.time())
            try:
                record = json.loads((root / item["path"]).read_text())
            except (OSError, json.JSONDecodeError):
                record = {}
            physical = isinstance(record, dict) and cleanup_passed(record, item["backend"], item["gpus"])
            item["physical_cleanup_passed"] = physical
            item["raw_status"] = record.get("status") if isinstance(record, dict) else None
            raw_finished = record.get("finished") if isinstance(record, dict) else None
            finished = (
                not isinstance(raw_finished, bool)
                and isinstance(raw_finished, (int, float))
                and math.isfinite(raw_finished)
            )
            complete = return_code == 0 and not timed_out and item["raw_status"] == "passed" and physical and finished
            item["status"] = "passed" if complete else "failed"
            save_manifest(root, manifest)
            if not complete:
                manifest.update(status="incomplete", stop_reason="child_failed_or_cleanup_unconfirmed")
                break
        else:
            manifest["status"] = "passed"
    except BaseException as exc:
        manifest.update(status="incomplete", stop_reason="suite_exception", error=repr(exc))
        raise
    finally:
        manifest.update(finished=time.time(), elapsed_s=time.monotonic() - began)
        save_manifest(root, manifest)
    return 0 if manifest["status"] == "passed" else 1


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--container-python", type=Path)
    parser.add_argument("--python-prefix", type=Path)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, default=1500)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if not math.isfinite(args.budget_s) or not 0 < args.budget_s <= 1500:
        parser.error("--budget-s must be positive and no greater than 1500 seconds.")
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error("--gpus requires two distinct nonnegative GPU indices.")
    if args.container_python is not None and not args.container_python.is_absolute():
        parser.error("--container-python must be an absolute container path.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
