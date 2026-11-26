"""Run a bounded partition validation matrix with predictions frozen before launch."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from capacity_validation_pipeline import save
from partition_prelaunch import plan


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    runner = root / "runner"
    runner.mkdir()
    scripts = Path(__file__).parent
    for name in (
        "partition_followup.py",
        "integrated_partition_training.py",
        "capacity_validation_pipeline.py",
        "analyze_integrated_partition.py",
        "sync_overlap_study.py",
        "partition_prelaunch.py",
        "partition_matrix_study.py",
    ):
        shutil.copy2(scripts / name, runner / name)
    selection = args.reference.parent / "capacity-selection.json"
    prediction = plan(json.loads(selection.read_text()))
    prediction["calibration_sha256"] = hashlib.sha256(selection.read_bytes()).hexdigest()
    save(root / "prelaunch-prediction.json", prediction)
    arms = [
        {"name": "three-env-static", "replicas": 1, "mode": "fixed", "cap": 4, "budget_s": 3000},
        {"name": "three-env-dynamic", "replicas": 1, "mode": "resident", "cap": 4, "budget_s": 3000},
        {"name": "three-env-static-cap5", "replicas": 1, "mode": "fixed", "cap": 5, "budget_s": 3000},
        {"name": "one-env-static", "replicas": 3, "mode": "fixed", "cap": 4, "budget_s": 4200},
    ]
    save(
        root / "protocol.json",
        {
            "created_at": time.time(),
            "reference": str(args.reference.resolve()),
            "arms": arms,
            "steps_per_arm": 4,
            "groups_per_step": 6,
            "trajectories_per_group": 2,
            "command_cap": "Per arm. Four for matched partition comparison, five for predicted best static candidate.",
            "shared_command_cap": 2,
            "headroom_mib": 16384,
            "sync_policy": "overlap",
            "attempts_per_arm": 1,
            "initialization_budget_s": 1200,
            "stop": "Stop on failure or missing physical release. No retries or automatic scale increases.",
            "comparison": "Same frozen code and work. Only GPU partition and pool membership differ.",
            "limitations": ["Sequential runs without repeat statistics.", "Model generated actions may differ."],
            "launcher_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in runner.iterdir()},
        },
    )
    state = {"status": "running", "pid": os.getpid(), "started_at": time.time(), "arms": {}}
    save(root / "status.json", state)
    for arm in arms:
        name = arm["name"]
        command = [
            sys.executable,
            "-u",
            str(runner / "partition_followup.py"),
            "--reference",
            str(args.reference.resolve()),
            "--output",
            str(root / name),
            "--rollout-replicas",
            str(arm["replicas"]),
            "--mode",
            arm["mode"],
            "--sync-policy",
            "overlap",
            "--headroom-mib",
            "16384",
            "--budget-s",
            str(arm["budget_s"]),
            "--dedicated-cap",
            str(arm["cap"]),
        ]
        with (root / f"{name}.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["arms"][name] = {"pid": child.pid, "status": "running"}
            save(root / "status.json", state)
            try:
                child.wait(timeout=arm["budget_s"] + 300)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    state["status"] = "cleanup_unconfirmed"
                    save(root / "status.json", state)
                    return
        path = root / name / "supervisor.json"
        result = json.loads(path.read_text()) if path.exists() else {"status": "failed"}
        state["arms"][name].update(result)
        save(root / "status.json", state)
        if result["status"] != "complete" or not result.get("physical_release_verified"):
            state["status"] = "stopped_on_failure"
            break
    else:
        state["status"] = "complete"
    summary = {}
    for arm in arms:
        path = root / arm["name"] / "analysis.json"
        if path.exists():
            summary[arm["name"]] = json.loads(path.read_text())
    save(root / "analysis.json", summary)
    evaluations = []
    for arm in arms:
        if arm["mode"] != "fixed" or arm["name"] not in summary:
            continue
        row = next(r for r in prediction["rows"] if r["cap"] == arm["cap"] and r["model_cards"] == arm["replicas"])
        observed = summary[arm["name"]]["timing_totals"]["timing_s/step"]
        evaluations.append(
            {
                "arm": arm["name"],
                "predicted_step_s": row["predicted_step_s"],
                "observed_step_s": observed,
                "relative_error": row["predicted_step_s"] / observed - 1,
            }
        )
    save(root / "prediction-validation.json", evaluations)
    state["finished_at"] = time.time()
    save(root / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
