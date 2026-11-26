"""Add an episode static pool arm to a completed frozen command comparison."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from capacity_validation_pipeline import save
from partition_followup import run_prepared, verify_release


def replace_once(path, old, new):
    """Require the audited reference text before applying an experiment patch."""
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"Unexpected source shape in {str(path)!r}.")
    path.write_text(text.replace(old, new))


def prepare(reference, output, trajectory_timeout_s=720):
    """Preserve prior evidence and disclose the isolated baseline policy changes."""
    state = json.loads((reference / "status.json").read_text())
    if state["status"] != "complete" or any(
        not arm.get("physical_release_verified") for arm in state["arms"].values()
    ):
        raise ValueError("The reference must complete and physically release every arm.")
    arm = reference / "static"
    manifest = json.loads((arm / "source-manifest.json").read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((arm / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reference source changed at {name!r}.")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    shutil.copytree(arm / "source", source, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
    shutil.copy2(arm / "tasks.parquet", output / "tasks.parquet")
    runner = source / "integrated_partition_training.py"
    maintained_runner = Path(__file__).with_name("integrated_partition_training.py")
    runner.write_text(maintained_runner.read_text().replace("model_headroom_mb=8192", "model_headroom_mb=16384"))
    replace_once(runner, 'gpu_reservation="command",', 'gpu_reservation="episode",')
    replace_once(runner, 'command_gpu_pool="environment",', 'command_gpu_pool="",')
    replace_once(runner, "max_gpu_commands=args.dedicated_cap,", "max_gpu_commands=0,")
    replace_once(
        runner,
        'for gpu in layout["dedicated_environment"] + layout["rollout"]:',
        'for gpu in layout["dedicated_environment"]:',
    )
    agent = source / "integrated_partition_agent.py"
    replace_once(agent, "async with asyncio.timeout(720):", f"async with asyncio.timeout({trajectory_timeout_s}):")
    replace_once(agent, '"mode": "command",', '"mode": handle.lease.gpu_reservation,')
    replace_once(
        agent,
        '            record["host_unit"] = handle.lease.unit_id',
        '            record["host_unit"] = handle.lease.unit_id\n'
        '            record["gpu_reservation"] = handle.lease.gpu_reservation\n'
        '            record["lease_granted_at"] = time.time()\n'
        "            save()",
    )
    replace_once(
        agent,
        '"admission": environment.last_gpu_command,',
        '"admission": {**environment.last_gpu_command, "unit_id": handle.lease.unit_id},',
    )
    # This frozen baseline fills the first fitting GPU before considering the next GPU.
    ledger = source / "psrl/workers/env_service/ledger.py"
    replace_once(ledger, "                    self._load(item[1], demand, now),\n", "")
    design = json.loads((arm / "design.json").read_text())
    design["args"].update(output=str(output), mode="fixed", elastic_partition=False, ray_worker_register_timeout_s=90)
    design.update(
        comparison="Episode static first-fit pool versus the recorded command arms.",
        gpu_reservation="episode",
        reference=str(reference),
        trajectory_timeout_s=trajectory_timeout_s,
        training_episodes_per_arm=24,
        startup_instrumentation=True,
        timing_boundary="Environment acquisition after the first model reply through final reply and cleanup.",
        limitations=["Sequential single run.", "Generated tokens may differ."],
    )
    save(output / "design.json", design)
    frozen = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    changed = sorted(name for name in frozen if frozen[name] != manifest.get(name))
    expected = [
        "integrated_partition_agent.py",
        "integrated_partition_training.py",
        "psrl/workers/env_service/ledger.py",
    ]
    if changed != expected:
        raise ValueError(f"Unexpected baseline changes {changed!r}.")
    save(output / "source-manifest.json", frozen)
    save(
        output / "reference-audit.json",
        {
            "source_changes": changed,
            "data_sha256": hashlib.sha256((output / "tasks.parquet").read_bytes()).hexdigest(),
            "expected": {"steps": 2, "episodes": 24, "commands": 48, "queries": 72},
            "budget_s": 2700,
            "initialization_budget_s": 1200,
            "attempts": 1,
            "trajectory_timeout_s": trajectory_timeout_s,
            "startup_instrumentation": True,
        },
    )
    for name in changed:
        compile((source / name).read_text(), name, "exec")
    if not verify_release(source, output):
        raise RuntimeError("The selected GPUs are occupied.")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-timeout-s", type=int, choices=(720, 1200), default=720)
    args = parser.parse_args()
    args.output = args.output.resolve()
    frozen_source = prepare(args.reference.resolve(), args.output, args.trajectory_timeout_s)
    run_prepared(
        argparse.Namespace(output=args.output, mode="fixed", budget_s=2700, rollout_replicas=2),
        frozen_source,
    )
