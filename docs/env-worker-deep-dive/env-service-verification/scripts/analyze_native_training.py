"""Compare native science training with explicit completion and workload boundaries."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from plot_science_training_ab import read_arm


def summarize(root, mode, design):
    """Retain failed attempts and avoid extrapolating missing training steps."""
    data = read_arm(root, mode)
    attempts = [json.loads(p.read_text()) for p in (root / mode).glob("pool-*/attempts/*.json")]
    started_ids = {p.stem for p in (root / mode).glob("pool-*/slot-starts/*.json")}
    recorded_ids = {a["uid"] for a in attempts}
    accepted = [a for a in attempts if a["accepted"]]
    groups = Counter(a["parent_id"] for a in accepted)
    steps = data["steps"]
    launch = data["launch"]
    planned = design["steps_per_arm"]
    complete = (
        launch.get("exit_code") == 0
        and {int(s["step"]) for s in steps} == set(range(1, planned + 1))
        and len(groups) == planned * design["batch_size"]
        and all(v == design["rollout_n"] for v in groups.values())
    )
    turns, trajectory_tokens = [], []
    for a in accepted:
        paths = list((root / mode).glob(f"pool-*/harbor-jobs/{a['session_id']}/**/agent/trajectory.json"))
        if len(paths) == 1:
            trajectory = json.loads(paths[0].read_text())
            turns.append(sum(step["source"] == "agent" for step in trajectory["steps"]))
            trajectory_tokens.append(trajectory.get("final_metrics", {}).get("completion_tokens"))
    units = [u for sample in data["telemetry"] for u in sample.get("units", [])]
    outside = []
    for sample in data["telemetry"]:
        for line in sample.get("gpu", "").splitlines():
            parts = line.split(",")
            if len(parts) == 5 and int(parts[0]) not in design["model_gpus"] and float(parts[2]) > 64:
                outside.append({"time": sample["time"], "index": int(parts[0]), "used_mib": float(parts[2])})
    log_path = root / mode / "launcher.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    oom = [line for line in log.splitlines() if re.search(r"CUDA out of memory|torch\.OutOfMemoryError", line)]
    first = min((a["slot_started_at"] for a in attempts), default=None)
    gpu_work = {str(index): [] for index in design["model_gpus"]}
    for sample in data["telemetry"]:
        if first is None or sample["time"] < first:
            continue
        for line in sample.get("gpu", "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5 and parts[0] in gpu_work:
                gpu_work[parts[0]].append([float(parts[2]), float(parts[3])])
    return {
        "complete": complete,
        "completed_steps": len(steps),
        "exit_code": launch.get("exit_code"),
        "attempts": len(attempts),
        "started_slots": len(started_ids),
        "slots_without_final_record": sorted(started_ids - recorded_ids),
        "accepted": len(accepted),
        "rejected": len(attempts) - len(accepted),
        "rewards": [a["rewards"] for a in accepted],
        "exception_counts": dict(Counter(str(a["exception_type"]) for a in attempts)),
        "accepted_per_group": dict(groups),
        "all_no_pressure": all(a.get("pressure_injected") is False for a in attempts) if attempts else None,
        "oom_log_lines": oom,
        "training_log_exists": log_path.exists(),
        "agent_turns": turns,
        "trajectory_completion_tokens": trajectory_tokens,
        "steps": steps,
        "step_seconds": sum(s["timing_s/step"] for s in steps) if steps else None,
        "wait_for_gen_seconds": sum(s["timing_s/wait_for_gen"] for s in steps) if steps else None,
        "update_actor_seconds": sum(s["timing_s/update_actor"] for s in steps) if steps else None,
        "trained_tokens": sum(s["termination/trained_tokens"] for s in steps) if steps else None,
        "driver_seconds_including_cleanup": (
            launch["finished_at"] - launch["started_at"]
            if "finished_at" in launch and "started_at" in launch
            else None
        ),
        "before_first_slot_seconds": first - launch["started_at"] if first is not None else None,
        "other_gpu_activity": outside,
        "gpu_telemetry_errors": data["gpu_errors"],
        "model_gpu_after_first_slot": {
            index: {
                "samples": len(values),
                "peak_memory_mib": max((v[0] for v in values), default=None),
                "mean_sampled_util_percent": sum(v[1] for v in values) / len(values) if values else None,
            }
            for index, values in gpu_work.items()
        },
        "unit_snapshots": len(units),
        "unit_peak": {
            key: max((u[key] for u in units), default=None)
            for key in (
                "containers_running",
                "cpu_used",
                "memory_used_mb",
                "declared_gpu_memory_mb",
                "sessions_waiting",
                "gpu_commands_active",
            )
        },
        "physical_release": launch.get("gpu_release"),
    }


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    design = json.loads((args.root / "design.json").read_text())
    arms = {mode: summarize(args.root, mode, design) for mode in ("off", "on")}
    result = {"scope": design["scope"], "arms": arms, "comparison": {}}
    common_steps = 0
    step_maps = {mode: {int(s["step"]): s for s in arm["steps"]} for mode, arm in arms.items()}
    while all(common_steps + 1 in steps for steps in step_maps.values()):
        common_steps += 1
    paired = {}
    for mode, arm in arms.items():
        rows = [step_maps[mode][i] for i in range(1, common_steps + 1)]
        group_ids = range(common_steps * design["batch_size"])
        if any(arm["accepted_per_group"].get(str(i), 0) != design["rollout_n"] for i in group_ids):
            raise ValueError(f"Completed prefix has inconsistent accepted groups in {mode!r}.")
        paired[mode] = {
            "steps": rows,
            "accepted": sum(arm["accepted_per_group"].get(str(i), 0) for i in group_ids),
            "step_seconds": sum(s["timing_s/step"] for s in rows),
            "wait_for_gen_seconds": sum(s["timing_s/wait_for_gen"] for s in rows),
            "update_actor_seconds": sum(s["timing_s/update_actor"] for s in rows),
            "trained_tokens": sum(s["termination/trained_tokens"] for s in rows),
        }
    result["common_prefix"] = {
        "completed_steps_per_arm": common_steps,
        "scope": "Completed prefix only. Excludes unfinished steps and does not imply suite completion.",
        "arms": paired,
        "comparison": {
            key + "_reduction_percent": (1 - paired["on"][key] / paired["off"][key]) * 100
            for key in ("step_seconds", "wait_for_gen_seconds", "update_actor_seconds")
            if paired["off"][key]
        },
    }
    if all(a["complete"] for a in arms.values()):
        for key in (
            "step_seconds",
            "wait_for_gen_seconds",
            "update_actor_seconds",
            "driver_seconds_including_cleanup",
        ):
            a, b = (arms[m][key] for m in ("off", "on"))
            result["comparison"][key + "_reduction_percent"] = (1 - b / a) * 100 if a else None
    (args.root / "comparison.json").write_text(json.dumps(result, indent=2))
    if not common_steps:
        print("No paired completed steps. Performance comparison is unavailable.")
        return
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "step_seconds", "Completed training steps"),
        (axes[1], "wait_for_gen_seconds", "Waiting for rollout data"),
        (axes[2], "update_actor_seconds", "Parameter update"),
    ):
        for index, mode in enumerate(("off", "on")):
            seconds = paired[mode][key]
            axis.bar(index, seconds, color=["#bd6730", "#157f9c"][index])
            axis.text(index, seconds, f"{seconds:.1f}", ha="center", va="bottom")
        axis.set(title=title, xticks=[0, 1], xticklabels=["OFF", "ON"], ylabel="Seconds")
        axis.margins(y=0.15)
    figure.suptitle("Native CPU science tasks with real GPU training, no injected pressure")
    figure.supxlabel(
        f"Paired prefix: {common_steps} steps each. Full run: OFF {arms['off']['completed_steps']}, "
        f"ON {arms['on']['completed_steps']} of {design['steps_per_arm']}. One sequential pair.",
        fontsize=9,
    )
    for suffix in ("png", "svg"):
        figure.savefig(args.root / f"comparison.{suffix}", dpi=160)
    print(json.dumps(result["common_prefix"]["comparison"], indent=2))


if __name__ == "__main__":
    main()
