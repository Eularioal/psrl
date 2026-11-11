"""Summarize completed service preparation comparisons without allocating a GPU."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(root):
    """Require complete correctness and cleanup evidence before computing differences."""
    readiness = json.loads((root / "probe/evidence.json").read_text())
    performance = json.loads((root / "performance/evidence.json").read_text())
    if readiness["status"] != "passed" or readiness["reward"] not in {"1", "1.0"}:
        raise ValueError("The unchanged verifier must pass before a benefit comparison.")
    reward = json.loads((root / "probe/verifier/reward.json").read_text())
    if (reward["f2p_passed"], reward["p2p_passed"]) != (18, 8):
        raise ValueError("Expected the complete original correctness verifier.")
    if performance["status"] != "passed" or performance["cleanup_errors"]:
        raise ValueError("The comparison or cleanup did not complete successfully.")
    if any(s["used_mb"] >= 64 or s["pids"] for s in performance["final_gpu"].values()):
        raise ValueError("GPU release was not confirmed.")
    batches = performance["batches"]
    if [batch["prepared"] for batch in batches] != [False, True, True, False]:
        raise ValueError("Expected a complete A/B/B/A comparison.")
    rows = []
    for batch in batches:
        commands = [performance["commands"][f"{batch['batch']}-{name}"] for name in ("a", "b")]
        if not all(command["passed"] == 4 for command in commands):
            raise ValueError("Every command must pass the same four original tests.")
        if any(command["preparation"]["cuda_initialized"] for command in commands):
            raise ValueError("The CPU preparation initialized CUDA before activation.")
        for command in commands:
            preparation, reservation = command["preparation"], command["reservation"]
            if batch["prepared"]:
                if preparation["ready"] > reservation["requested_at"]:
                    raise ValueError("Preparation did not finish before requesting GPU admission.")
            elif preparation["started"] < reservation["admitted_at"]:
                raise ValueError("The baseline prepared dependencies before GPU admission.")
            if preparation["actual_gpu"].removeprefix("GPU-") != reservation["gpu_uuids"][0].removeprefix("GPU-"):
                raise ValueError("The executed GPU differs from the admitted GPU.")
        rows.append(
            dict(
                batch=batch["batch"],
                prepared=batch["prepared"],
                wall_s=batch["wall_s"],
                gpu_held_s=sum(
                    command["reservation"]["released_at"] - command["reservation"]["admitted_at"]
                    for command in commands
                ),
                cpu_prepare_s=sum(
                    command["preparation"]["ready"] - command["preparation"]["started"] for command in commands
                ),
                ready_wait_s=sum(
                    command["preparation"]["activated"] - command["preparation"]["ready"] for command in commands
                ),
                submitted_to_ready_s=sum(
                    command["preparation"]["ready"] - command["submitted"] for command in commands
                ),
            )
        )
    groups = {}
    for prepared, label in [(False, "baseline"), (True, "prepared")]:
        selected = [row for row in rows if row["prepared"] == prepared]
        groups[label] = dict(
            n=len(selected),
            samples_s=[row["wall_s"] for row in selected],
            mean_wall_s=statistics.mean(row["wall_s"] for row in selected),
            mean_gpu_held_s=statistics.mean(row["gpu_held_s"] for row in selected),
        )
    baseline, prepared = groups["baseline"], groups["prepared"]
    return dict(
        scope="One GPU, two sessions, one complete class per command. Two batches per strategy. No training.",
        group=performance["group"],
        groups=groups,
        wall_reduction_pct=100 * (1 - prepared["mean_wall_s"] / baseline["mean_wall_s"]),
        gpu_held_reduction_pct=100 * (1 - prepared["mean_gpu_held_s"] / baseline["mean_gpu_held_s"]),
        batches=rows,
        correctness_executions=sum(command["passed"] for command in performance["commands"].values()),
        original_verifier=dict(f2p_passed=18, p2p_passed=8, reward=reward["reward"]),
        cleanup=dict(errors=performance["cleanup_errors"], gpu_released=True),
        limitation="Small descriptive comparison. It does not establish algorithm novelty or stable training speedup.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
