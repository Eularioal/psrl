"""Freeze and run bounded PSRL comparisons of resident and whole replica sharing."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from capacity_validation_pipeline import save
from partition_followup import freeze, run_prepared, verify_release

SOURCE_FILES = (
    "psrl/workers/gen/gpu_handoff.py",
    "psrl/workers/gen/rollout_coordination/env_partition.py",
    "psrl/workers/gen/rollout_coordination/coordinator.py",
    "psrl/workers/gen/rollout_coordination/status_loop.py",
    "psrl/workers/gen/rollout_coordination/command_loop.py",
    "psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py",
    "psrl/workers/env_service/router.py",
    "psrl/workers/env_service/command_gpu.py",
    "psrl/trainer/ppo/ray_trainer.py",
    "psrl/trainer/config/psrl/env_service.yaml",
)


def prepare(args):
    """Freeze all code before waiting for the previous resource owner."""
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).parent
    runner = root / "runner"
    runner.mkdir()
    for name in (
        "live_partition_study.py",
        "partition_followup.py",
        "integrated_partition_training.py",
        "analyze_integrated_partition.py",
        "capacity_validation_pipeline.py",
    ):
        shutil.copy2(scripts / name, runner / name)
    arms = []
    for name, mode, elastic in (
        ("static", "fixed", False),
        ("resident", "resident", False),
        ("elastic", "resident", True),
    ):
        output = root / name
        source = freeze(args.reference.resolve(), output, 2, mode, "baseline", 16384, 2700, 5)
        for path in SOURCE_FILES:
            shutil.copy2(args.repo / path, source / path)
        tasks = pq.read_table(output / "tasks.parquet").slice(0, 12)
        if tasks.num_rows != 12:
            raise ValueError("Two training steps require twelve task groups.")
        pq.write_table(tasks, output / "tasks.parquet")
        design = json.loads((output / "design.json").read_text())
        design["args"].update(steps=2, live_partition_protocol=True, elastic_partition=elastic)
        design.update(
            steps_per_arm=2,
            comparison="Two PSRL steps with static, resident or adaptive partitions.",
            source_overlays=list(SOURCE_FILES),
            sync_policy="production_opt_in_overlap",
        )
        save(output / "design.json", design)
        manifest = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        save(output / "source-manifest.json", manifest)
        reference = json.loads((args.reference / "source-manifest.json").read_text())
        save(
            output / "reference-audit.json",
            {
                "changed": sorted(name for name in manifest if manifest[name] != reference.get(name)),
                "data_sha256": hashlib.sha256((output / "tasks.parquet").read_bytes()).hexdigest(),
            },
        )
        arms.append({"name": name, "mode": mode, "elastic": elastic, "budget_s": 2700})
    manifests = [json.loads((root / arm["name"] / "source-manifest.json").read_text()) for arm in arms]
    if any(value != manifests[0] for value in manifests[1:]):
        raise RuntimeError("Compared arms do not use identical source snapshots.")
    protocol = {
        "arms": arms,
        "steps_per_arm": 2,
        "groups_per_step": 6,
        "rollouts_per_group": 2,
        "expected_per_arm": {"episodes": 24, "gpu_commands": 48, "model_requests": 72},
        "initial_partition": "2 environment + 2 rollout + 2 fixed training GPUs",
        "dedicated_cap": 5,
        "borrowed_cap": 5,
        "resident_cap": 2,
        "model_headroom_mib": 16384,
        "prior_matrix": str(args.wait_for.resolve()),
        "wait_budget_s": 4500,
        "initialization_budget_s": 1200,
        "attempts_per_arm": 1,
        "acceptance": "Complete equal work, observe borrowing and version updates, verify cleanup.",
        "limitations": [
            "Single sequential comparison.",
            "No automatic retry or strong causal speed claim.",
            "Restoration may not trigger under stable low inference load.",
        ],
    }
    save(root / "protocol.json", protocol)
    save(root / "status.json", {"status": "prepared", "arms": {}})


def summarize_partition(output, mode):
    """Audit actual controller transitions without treating missing telemetry as success."""
    telemetry = output / mode / "telemetry.jsonl"
    candidates = list((output / mode).glob("*telemetry*.jsonl"))
    if not telemetry.exists() and candidates:
        telemetry = candidates[0]
    events = {}
    failed = []
    sleeping_commands = False
    resident_commands = False
    for line in telemetry.read_text().splitlines() if telemetry.exists() else []:
        row = json.loads(line)
        modes = [
            mode for model in row.get("models", {}).values() for mode in model.get("shared_commands", {}).values()
        ]
        sleeping_commands = sleeping_commands or "sleeping" in modes
        resident_commands = resident_commands or "resident" in modes
        for event in row.get("partition", {}).get("events", []):
            events[json.dumps(event, sort_keys=True)] = event
        for event in row.get("partition", {}).get("events", []):
            if "error" in event or "controller_error" in event:
                failed.append(event)
    ordered = sorted(events.values(), key=lambda e: e["at"])
    result = {
        "sleeping_commands_observed": sleeping_commands,
        "resident_commands_observed": resident_commands,
        "events": ordered,
        "borrow_observed": any(e.get("state") == "borrowed" for e in ordered),
        "restore_observed": any("restored_version" in e for e in ordered),
        "errors": failed,
    }
    save(output / "partition-audit.json", result)
    return result


def recover_static(root):
    """Audit a completed static arm after the known outer probe import failure."""
    original = json.loads((root / "status.json").read_text())
    if original["status"] != "stopped" or set(original["arms"]) != {"static"}:
        raise RuntimeError("Recovery requires only the completed static arm to have run.")
    output = root / "static"
    supervisor = json.loads((output / "supervisor.json").read_text())
    expected_error = repr(ModuleNotFoundError("No module named 'psrl.workers.env_service'"))
    if supervisor.get("exit_code") != 0 or supervisor.get("cleanup_error") != expected_error:
        raise RuntimeError("This recovery only handles the known supervisor probe import failure.")
    inner = json.loads((output / "fixed/status.json").read_text())
    analysis = json.loads((output / "analysis.json").read_text())
    if inner.get("exit_code") != 0 or not inner.get("physical_release_verified"):
        raise RuntimeError("The inner training process did not complete and release its devices.")
    if set(analysis["steps"]) != {"1", "2"} or (
        analysis["episodes_completed"], analysis["commands_completed"]
    ) != (24, 48):
        raise RuntimeError("The static arm did not complete the frozen workload.")
    for name in ("resident", "elastic"):
        if (root / name / "supervisor.json").exists():
            raise RuntimeError("Recovery refuses to rerun an attempted arm.")
    for name in ("static", "resident", "elastic"):
        manifest = json.loads((root / name / "source-manifest.json").read_text())
        if any(hashlib.sha256((root / name / "source" / p).read_bytes()).hexdigest() != h
               for p, h in manifest.items()):
            raise RuntimeError("Frozen source changed.")
    audit = root / "recovery"
    audit.mkdir(exist_ok=False)
    shutil.copy2(root / "status.json", audit / "original-status.json")
    shutil.copy2(output / "supervisor.json", audit / "original-static-supervisor.json")
    if not verify_release(output / "source", audit):
        raise RuntimeError("Current physical release could not be verified.")
    # The regular arm preflight also verifies Docker, node identity and container CUDA.
    save(audit / "audit.json", {
        "at": time.time(), "static_training_rerun": False, "source_changes": [],
        "remaining_arms": ["resident", "elastic"], "physical_release_verified": True,
        "reason": "Outer supervisor imported an unrelated installed PSRL package.",
    })
    recovered = dict(supervisor, status="complete", physical_release_verified=True,
                     recovered_after_probe_import_failure=True, recovery_audit=str(audit))
    return recovered


def run(root, resume=False):
    """Wait for verified resource release, then execute each frozen arm once."""
    protocol = json.loads((root / "protocol.json").read_text())
    state = {"status": "waiting_for_previous_matrix", "pid": os.getpid(), "started_at": time.time(), "arms": {}}
    if resume:
        state["arms"]["static"] = recover_static(root)
        state["recovery"] = str(root / "recovery")
    save(root / "status.json", state)
    deadline = time.monotonic() + protocol["wait_budget_s"]
    try:
        while True:
            try:
                previous = json.loads(Path(protocol["prior_matrix"]).read_text())
            except json.JSONDecodeError:
                previous = {"status": "running"}
            if previous["status"] not in ("running", "initializing"):
                if not previous.get("arms") or any(
                    not a.get("physical_release_verified") for a in previous["arms"].values()
                ):
                    raise RuntimeError("Previous matrix has not verified physical release.")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Previous matrix did not release within the waiting budget.")
            time.sleep(10)
        state["status"] = "running"
        for arm in protocol["arms"]:
            name = arm["name"]
            if name in state["arms"]:
                continue
            if (root / name / "supervisor.json").exists():
                raise RuntimeError("Refusing to overwrite an attempted arm.")
            with (root / f"{name}.log").open("w") as log:
                child = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "--arm", str(root / name)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                state["arms"][name] = {"pid": child.pid, "status": "running"}
                save(root / "status.json", state)
                try:
                    child.wait(timeout=arm["budget_s"] + 300)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=180)
            result = json.loads((root / name / "supervisor.json").read_text())
            state["arms"][name].update(result)
            save(root / "status.json", state)
            if result["status"] != "complete" or not result.get("physical_release_verified"):
                raise RuntimeError(f"Arm {name!r} failed or did not physically release.")
            audit = summarize_partition(root / name, arm["mode"])
            if audit["errors"]:
                raise RuntimeError("Partition controller reported an error.")
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="stopped", error=repr(exc))
    state["finished_at"] = time.time()
    save(root / "status.json", state)
    results = {
        arm["name"]: json.loads((root / arm["name"] / "analysis.json").read_text())
        for arm in protocol["arms"]
        if (root / arm["name"] / "analysis.json").exists()
    }
    save(root / "analysis.json", results)
    elastic_audit = root / "elastic" / "partition-audit.json"
    if elastic_audit.exists():
        audit = json.loads(elastic_audit.read_text())
        save(
            root / "mechanism-verdict.json",
            {
                "training_complete": state["status"] == "complete",
                "borrow_and_environment_use_observed": audit["borrow_observed"]
                and audit["sleeping_commands_observed"],
                "restore_observed": audit["restore_observed"],
                "resident_sharing_observed": audit["resident_commands_observed"],
                "limitations": protocol["limitations"],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--arm", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.run.resolve(), resume=args.resume)
    elif args.arm:
        design = json.loads((args.arm / "design.json").read_text())
        run_prepared(
            argparse.Namespace(output=args.arm, mode=design["args"]["mode"], budget_s=2700, rollout_replicas=2),
            args.arm / "source",
        )
    else:
        if not all((args.repo, args.reference, args.output, args.wait_for)):
            parser.error("Preparation requires repo, reference, output and wait-for paths.")
        prepare(args)
