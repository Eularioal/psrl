"""Preserve bounded comparison evidence and calculate policy timings."""

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path


def summarize(run, contracts, output):
    """Check completion and copy reproducible evidence without local launch paths."""
    evidence = json.loads((run / "run/evidence.json").read_text())
    controller = json.loads((run / "controller-status.json").read_text())
    cleanup = json.loads((run / "external-cleanup.json").read_text())
    checks = json.loads(contracts.read_text())
    plan = json.loads((run / "plan.json").read_text())
    if (
        evidence["status"] != "passed"
        or any(controller.values())
        or evidence["cleanup"]["remaining_containers"]
        or cleanup["remaining"]
        or checks["status"] != "passed"
    ):
        raise RuntimeError("The comparison or its cleanup did not pass.")
    policies = {}
    for policy in ("fresh", "restart", "split", "versioned", "blind"):
        blocks = [block for block in evidence["blocks"] if block["policy"] == policy]
        walls = [block["wall_s"] for block in blocks]
        policies[policy] = dict(
            blocks=len(blocks),
            wall_s=walls,
            mean_wall_s=statistics.mean(walls),
            builds_per_block=[block["builds"] for block in blocks],
            wrong_cases_per_block=[block["wrong"] for block in blocks],
        )
    versioned = policies["versioned"]["mean_wall_s"]
    split = policies["split"]["mean_wall_s"]
    faults = dict(evidence["faults"])
    for label in ("fault-race", "fault-retry"):
        lines = (run / f"run/{label}.log").read_text().splitlines()
        faults[label] = json.loads(
            next(line.removeprefix("STATE_RESULT ") for line in lines if line.startswith("STATE_RESULT "))
        )
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for script in sorted((run / "scripts").glob("*.py")):
        source_hashes[script.name] = hashlib.sha256(script.read_bytes()).hexdigest()
        shutil.copyfile(script, output / f"{script.name}.txt")
    provenance = dict(
        baseline=evidence["baseline"],
        main_commit=plan["main_commit"],
        main_workspace="Research scripts and documentation are local uncommitted files.",
        baseline_source_sha256=evidence["baseline_source_sha256"],
        backend_patches=[],
        import_isolation="Only empty package initializers were supplied around the exact baseline backend files.",
        image=evidence["image"],
        model_revision=evidence["model_revision"],
        frozen_script_sha256=source_hashes,
        scope=evidence["scope"],
        actual_container_resources=evidence["actual_container_resources"],
        controller=controller,
        cleanup=cleanup,
        active_budget_s=plan["active_budget_s"],
        outer_budget_s=plan["outer_budget_s"],
        policy_order=plan["policies"],
    )
    summary = dict(
        status="passed",
        policies=policies,
        versioned_minus_split_s=versioned - split,
        versioned_minus_split_percent=100 * (versioned / split - 1),
        versioned_vs_fresh_wall_reduction_percent=100 * (1 - versioned / policies["fresh"]["mean_wall_s"]),
        versioned_vs_restart_wall_reduction_percent=100 * (1 - versioned / policies["restart"]["mean_wall_s"]),
        faults=faults,
        elapsed_s=evidence["elapsed_s"],
        timing_boundary=(
            "Each block includes all service builds, eight commands and service shutdown. "
            "Container setup and global warmup are excluded."
        ),
        limits=[
            "CPU inference with scripted edits. No GPU, original full benchmark, Agent trajectory or training step.",
            "Two blocks each for split and versioned in ABBA order. Other controls have one block.",
            "Loader length changes conservatively restart the service. A specialized adapter could avoid this reload.",
            "The incomplete dependency signal is injected. Dependencies are not automatically discovered.",
            "No scheduling, memory budget selection, cross-node movement or production state management was added.",
        ],
    )
    for name, value in {
        "model-comparison.json": evidence,
        "publication-checks.json": checks,
        "provenance.json": provenance,
        "summary.json": summary,
    }.items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(dict(policies=policies, difference_s=versioned - split), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.contracts, args.output)
