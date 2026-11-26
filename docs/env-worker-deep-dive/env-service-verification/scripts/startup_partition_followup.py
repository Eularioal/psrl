"""Repeat one completed elastic arm with only the audited startup changes."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from capacity_validation_pipeline import save
from live_partition_study import summarize_partition
from partition_followup import run_prepared, verify_release


def prepare(reference, output, repo):
    """Freeze the successful workload and preserve its scheduling and timeouts."""
    state = json.loads((reference / "supervisor.json").read_text())
    if state["status"] != "complete" or not state.get("physical_release_verified"):
        raise ValueError("The reference must have completed and physically released its resources.")
    manifest = json.loads((reference / "source-manifest.json").read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((reference / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reference source changed at {name!r}.")
    design = json.loads((reference / "design.json").read_text())
    if not design["args"].get("elastic_partition") or design["args"]["steps"] != 2:
        raise ValueError("This followup requires the completed two-step elastic arm.")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    shutil.copytree(reference / "source", source, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
    shutil.copy2(reference / "tasks.parquet", output / "tasks.parquet")
    scripts = Path(__file__).parent
    runner = (scripts / "integrated_partition_training.py").read_text()
    (source / "integrated_partition_training.py").write_text(
        runner.replace("model_headroom_mb=8192", "model_headroom_mb=16384")
    )
    shutil.copy2(repo / "psrl/trainer/main_ppo.py", source / "psrl/trainer/main_ppo.py")
    design["args"].update(output=str(output), ray_worker_register_timeout_s=90)
    design.update(
        reference=str(reference),
        comparison="Repeat the elastic arm after lazy startup imports and a 90-second registration window.",
        training_episodes_per_arm=24,
        scheduling_policy_unchanged=True,
        startup_instrumentation=True,
    )
    design["limits"].update(arm_s=2700, initialization_s=1200, episode_s=720, attempts_per_arm=1)
    save(output / "design.json", design)
    frozen = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    changed = sorted(name for name in frozen if frozen[name] != manifest.get(name))
    expected = ["integrated_partition_training.py", "psrl/trainer/main_ppo.py"]
    if changed != expected:
        raise ValueError(f"Unexpected source changes: {changed!r}.")
    for name in changed:
        compile((source / name).read_text(), name, "exec")
    save(output / "source-manifest.json", frozen)
    save(
        output / "reference-audit.json",
        {
            "source_changes": changed,
            "data_unchanged": (output / "tasks.parquet").read_bytes() == (reference / "tasks.parquet").read_bytes(),
            "expected": {"steps": 2, "episodes": 24, "commands": 48, "queries": 72},
            "budget_s": 2700,
            "initialization_budget_s": 1200,
            "attempts": 1,
            "trajectory_timeout_s": 720,
            "transition_overlap_implemented": False,
        },
    )
    if not verify_release(source, output):
        raise RuntimeError("The selected GPUs are occupied.")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    source = prepare(args.reference.resolve(), args.output, args.repo.resolve())
    run_prepared(argparse.Namespace(output=args.output, mode="resident", budget_s=2700, rollout_replicas=2), source)
    summarize_partition(args.output, "resident")
    state = json.loads((args.output / "supervisor.json").read_text())
    if state["status"] != "complete" or not state.get("physical_release_verified"):
        raise SystemExit(1)
