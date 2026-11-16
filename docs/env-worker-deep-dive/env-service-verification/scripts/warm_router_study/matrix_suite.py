"""Preregister, freeze and supervise the bounded placement experiment matrix."""

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from matrix import plan_cells

psrl_logger = logging.getLogger(__name__)
CAPS = {"baseline": 850, "native": 2160}
MINIMUM_WORK = {"baseline": 700, "native": 1900}
CLEANUP_RESERVE = 180


def save(root, manifest):
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(root / "manifest.json")


def critical_hashes(repository):
    paths = [repository / "examples/sciaccel_rl/runner.py", repository / "psrl/__init__.py"]
    for directory in ("psrl/workers/env_service", "psrl/workers/env_worker"):
        paths.extend(sorted((repository / directory).glob("*.py")))
    return {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def freeze(root):
    repository = Path(__file__).resolve().parents[5]
    scripts = Path(__file__).resolve().parent.parent
    snapshot = root / "source-snapshot"
    hashes = {}
    for path in sorted(scripts.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(scripts)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    policies = root / "policy"
    policies.mkdir()
    policy_hashes = {}
    for name in ("warm_dispatch.py", "step_predictor.py", "warm_placement.py"):
        source = repository / "psrl/workers/env_service" / name
        shutil.copy2(source, policies / name)
        policy_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=repository, timeout=20)
    (root / "current-worktree.patch").write_bytes(patch)
    return snapshot / "warm_router_study/matrix.py", hashes, policy_hashes, critical_hashes(repository)


def cleanup_passed(record, backend):
    if record.get("cleanup_errors") != [] or record.get("containers_remaining") not in ("", []):
        return False
    phases = record.get("cleanup_phases", {})
    if not isinstance(phases, dict) or phases.get("physical_release_confirmed") is not True:
        return False
    snapshots = record.get("gpu_after", {})
    if not isinstance(snapshots, dict) or len(snapshots) != 2:
        return False
    for value in snapshots.values():
        if not isinstance(value, dict) or value.get("pids") != []:
            return False
        used_mb = value.get("used_mb")
        if type(used_mb) not in (int, float) or not math.isfinite(used_mb) or not 0 <= used_mb < 128:
            return False
    if backend == "native":
        ledger = record.get("ledger_after_cleanup")
        if phases.get("leases_released") is not True or not isinstance(ledger, list) or not ledger:
            return False
        return all(
            isinstance(unit, dict) and type(unit.get("leases_active")) is int and unit["leases_active"] == 0
            for unit in ledger
        )
    return backend == "baseline"


def stop_owned(root, run):
    path = root / run["path"]
    try:
        raw = json.loads(path.read_text())
        tag = raw["tag"]
        ids = subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", "label=psrl.actor_id=" + tag], text=True, timeout=15
        ).split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], check=False, timeout=30, capture_output=True)
        return dict(attempted=True, container_count=len(ids), physical_release_confirmed=False)
    except Exception as exc:
        return dict(attempted=True, error_type=type(exc).__name__, physical_release_confirmed=False)


def stop_backend(process, root, item, *, grace_s=10, kill_s=5):
    """Reap the owned process group and attempt its actor cleanup without raising."""
    recovery = dict(errors=[], physical_release_confirmed=False)

    def attempt(phase, operation):
        try:
            return operation()
        except BaseException as exc:
            recovery["errors"].append(dict(phase=phase, error=repr(exc)))
            return None

    def signal_group(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def group_exists():
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    if process is not None:
        attempt("interrupt", lambda: signal_group(signal.SIGINT))
        attempt("interrupt_join", lambda: process.wait(timeout=grace_s))
        if attempt("group_probe", group_exists) is not False:
            attempt("kill", lambda: signal_group(signal.SIGKILL))
            attempt("kill_join", lambda: process.wait(timeout=kill_s))
        recovery["return_code"] = attempt("poll", process.poll)
    recovery["owned_cleanup"] = attempt("owned_cleanup", lambda: stop_owned(root, item))
    return recovery


def supervise_backend(command, output, timeout_s, root, item):
    """Own a backend through any wait failure, preserving its original exception."""
    process = None

    def recover(trigger):
        try:
            item["watchdog"] = stop_backend(process, root, item)
        except BaseException as cleanup_exc:
            item["watchdog"] = dict(errors=[dict(phase="recovery", error=repr(cleanup_exc))])
        item["watchdog"]["trigger"] = trigger

    try:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        process.wait(timeout=timeout_s)
    except BaseException as exc:
        recover(type(exc).__name__)
        if not isinstance(exc, subprocess.TimeoutExpired):
            raise
    if process is not None and process.returncode != 0 and "watchdog" not in item:
        recover("BackendNonzeroExit")
    return process.returncode if process is not None else None


def run(args):
    from gpu_sampler import GpuSampler

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    started = time.monotonic()
    deadline = started + args.budget_s
    manifest = dict(
        schema_version=1,
        status="preparing",
        started=time.time(),
        budget_s=args.budget_s,
        retries=0,
        policy_dir="policy",
        cells=plan_cells(),
        runs=[dict(backend=name, path=name + "/status.json", status="planned") for name in CAPS],
        protocol=dict(
            target_model="Qwen3-4B-Base",
            blocker_model="Qwen3-1.7B-Base",
            rounds=3,
            progress_every=16,
            arrival_after_completed_steps=64,
            profile_steps=128,
            switch_margin_s=2.0,
            mode="point",
            hot_device_by_round=[0, 1, 0],
            core_mix="Equal count of the four registered cases per strategy.",
            primary_metric="Before blocker submission through both RPC responses.",
            scope="Fixed public inputs, norm-only updates, service budget control, no physical OOM stress.",
        ),
    )
    save(root, manifest)
    sampler = None
    primary_error = None
    try:
        entry, hashes, policies, core_hashes = freeze(root)
        manifest.update(source_hashes=hashes, policy_hashes=policies, critical_psrl_hashes=core_hashes)
        manifest["current_worktree_patch_sha256"] = hashlib.sha256(
            (root / "current-worktree.patch").read_bytes()
        ).hexdigest()
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True, timeout=15
        ).strip()
        manifest.update(image_id=image_id, status="running")
        sampler = GpuSampler(args.gpus, root / "gpu-samples.jsonl", interval_s=2)
        sampler.start()
        save(root, manifest)
        for item in manifest["runs"]:
            backend = item["backend"]
            remaining = deadline - time.monotonic()
            if remaining < MINIMUM_WORK[backend] + CLEANUP_RESERVE:
                manifest.update(status="incomplete", stop_reason="backend_admission_budget")
                break
            repository = Path(__file__).resolve().parents[5]
            if critical_hashes(repository) != core_hashes:
                raise RuntimeError("Live PSRL source changed after the protocol was frozen.")
            work_budget = min(CAPS[backend], remaining - CLEANUP_RESERVE - 10)
            command = [
                sys.executable,
                "-u",
                str(entry),
                "--backend",
                backend,
                "--root",
                str(root / backend),
                "--assets",
                str(args.assets.resolve()),
                "--node-ip",
                args.node_ip,
                "--gpus",
                *map(str, args.gpus),
                "--image",
                image_id,
                "--container-python",
                str(args.container_python),
                "--baseline-repo",
                str(args.baseline_repo.resolve()),
                "--policy-dir",
                str(root / "policy"),
                "--budget-s",
                str(work_budget),
            ]
            item.update(status="running", started=time.time(), budget_s=work_budget)
            save(root, manifest)
            psrl_logger.info(f"[matrix] Starting {backend!r} with a {work_budget:.0f} second work budget.")
            with (root / "logs" / (backend + ".log")).open("w") as output:
                return_code = supervise_backend(
                    command,
                    output,
                    min(work_budget + CLEANUP_RESERVE, deadline - time.monotonic()),
                    root,
                    item,
                )
            item.update(return_code=return_code, finished=time.time())
            try:
                raw = json.loads((root / item["path"]).read_text())
            except (OSError, json.JSONDecodeError):
                raw = {}
            item.update(raw_status=raw.get("status"), physical_cleanup_passed=cleanup_passed(raw, backend))
            passed = (
                return_code == 0
                and "watchdog" not in item
                and raw.get("status") == "passed"
                and item["physical_cleanup_passed"]
            )
            item["status"] = "passed" if passed else "failed"
            save(root, manifest)
            if not passed:
                manifest.update(status="incomplete", stop_reason="backend_failed_or_cleanup_unconfirmed")
                break
        else:
            manifest["status"] = "passed"
    except BaseException as exc:
        primary_error = exc
        manifest.update(status="incomplete", stop_reason="suite_exception", error=repr(exc))
        raise
    finally:
        finalization_error = None
        if sampler is not None:
            try:
                manifest["gpu_sampler_stop"] = sampler.stop()
            except BaseException as exc:
                finalization_error = exc
                manifest["gpu_sampler_stop"] = dict(error=repr(exc))
        manifest.update(finished=time.time(), elapsed_s=time.monotonic() - started)
        try:
            save(root, manifest)
        except BaseException as exc:
            finalization_error = finalization_error or exc
        if primary_error is None and finalization_error is not None:
            raise finalization_error
    return 0 if manifest["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--image", required=True)
    parser.add_argument("--container-python", type=Path, required=True)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, default=3000)
    args = parser.parse_args()
    if not 0 < args.budget_s <= 3000 or len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error("A budget up to 3000 seconds and two distinct nonnegative GPU indices are required.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def terminate(
        signum,
        _frame,
    ):
        raise SystemExit(128 + signum)

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        return run(args)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
