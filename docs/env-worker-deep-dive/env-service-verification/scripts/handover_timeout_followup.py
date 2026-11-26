"""Repeat the incomplete handover arm with bounded command and trajectory deadlines."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from analyze_handover_overlap import handovers
from analyze_integrated_partition import analyze
from capacity_validation_pipeline import save
from partition_followup import run_prepared, verify_release

COMMAND_TIMEOUT_S = 285
SHARED_TIMEOUT_S = 300
EPISODE_TIMEOUT_S = 1200
BUDGET_S = 2700

TIMEOUT_EDITS = {
    "integrated_partition_agent.py": (
        ("asyncio.timeout(720)", f"asyncio.timeout({EPISODE_TIMEOUT_S})"),
        ("timeout_sec=240,", f"timeout_sec={COMMAND_TIMEOUT_S},"),
    ),
    "integrated_partition_training.py": (
        (
            "psrl.env_service.gpu_sharing.max_command_s=270",
            f"psrl.env_service.gpu_sharing.max_command_s={SHARED_TIMEOUT_S}",
        ),
    ),
}


def digest(path):
    """Hash the bytes used in the actual run."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(reference, output):
    """Change only time limits on the verified previous overlap snapshot."""
    state = json.loads((reference / "supervisor.json").read_text())
    design = json.loads((reference / "design.json").read_text())
    if state["status"] != "failed" or not state["physical_release_verified"] or state["source_changes"]:
        raise ValueError("The reference must have stopped and released devices with its source intact.")
    if not design["handover_overlap"] or design["args"]["steps"] != 2 or design["args"]["groups"] != 6:
        raise ValueError("Expected the two step overlap arm with six trajectory groups per step.")
    if design["limits"]["command_s"] != 240 or design["limits"]["episode_s"] != 720:
        raise ValueError("The reference command or trajectory deadline changed.")
    manifest = json.loads((reference / "source-manifest.json").read_text())
    ref_source = reference / "source"
    for name, expected in manifest.items():
        if digest(ref_source / name) != expected:
            raise ValueError(f"Reference source changed at {name!r}.")
    local_site = Path(design["args"]["local_python_site"])
    deps = json.loads((reference.parent / "transformers-local-manifest.json").read_text())
    if Path(deps["local"]) != local_site:
        raise ValueError("Unexpected node local dependency path.")
    for name, expected in deps["sha256"].items():
        if digest(local_site / name) != expected:
            raise ValueError(f"Local runtime dependency changed at {name!r}.")

    output.mkdir(exist_ok=False, parents=True)
    source = output / "source"
    shutil.copytree(ref_source, source, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
    for name, edits in TIMEOUT_EDITS.items():
        path = source / name
        code = path.read_text()
        for old, new in edits:
            if code.count(old) != 1:
                raise ValueError(f"Unexpected number of deadline settings {old!r} in {name!r}.")
            code = code.replace(old, new)
        compile(code, str(path), "exec")
        path.write_text(code)
    actual = {name: digest(source / name) for name in manifest}
    changed = {name for name, expected in manifest.items() if actual[name] != expected}
    if changed != set(TIMEOUT_EDITS):
        raise ValueError(f"Unexpected source delta {sorted(changed)!r}.")
    shutil.copy2(reference / "tasks.parquet", output / "tasks.parquet")
    design["args"]["output"] = str(output)
    design.update(
        comparison="Completion and timing of the same overlap workload with longer execution deadlines.",
        reference=str(reference),
    )
    design["limits"].update(
        command_s=COMMAND_TIMEOUT_S,
        shared_command_s=SHARED_TIMEOUT_S,
        episode_s=EPISODE_TIMEOUT_S,
    )
    save(output / "design.json", design)
    save(output / "source-manifest.json", actual)
    save(
        output / "reference-audit.json",
        {
            "changed": sorted(changed),
            "data_sha256": digest(output / "tasks.parquet"),
            "reference_data_sha256": digest(reference / "tasks.parquet"),
            "limits": design["limits"],
        },
    )
    return source, local_site


def main(args):
    """Run the one authorized arm and summarize the observed work."""
    output = args.output.resolve()
    arm = output / "overlap"
    source, local_site = prepare(args.reference, arm)
    if not verify_release(source, arm):
        raise RuntimeError("Experiment devices are occupied.")
    run_prepared(
        argparse.Namespace(
            output=arm, mode="resident", budget_s=BUDGET_S, rollout_replicas=2, local_python_site=local_site
        ),
        source,
    )
    timing = handovers(arm)
    save(output / "handover-timing.json", timing)
    state = json.loads((arm / "supervisor.json").read_text())
    status = {
        "supervisor": state,
        "attempts": 1,
        "expected": {"steps": 2, "trajectories": 24, "commands": 48, "model_requests": 72},
    }
    summary = arm / "analysis.json"
    if not summary.exists() and (arm / "resident/telemetry.jsonl").exists():
        save(summary, analyze(arm / "resident"))
    if summary.exists():
        observed = json.loads(summary.read_text())
        status["observed"] = {
            "steps": len(observed["steps"]),
            "trajectories": observed["episodes_completed"],
            "commands": observed["commands_completed"],
            "model_requests": observed["queries_completed"],
        }
    status["workload_completed"] = state["status"] == "complete" and status.get("observed") == status["expected"]
    save(output / "status.json", status)
    if not status["workload_completed"]:
        raise RuntimeError("The overlap arm did not complete the planned workload.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
