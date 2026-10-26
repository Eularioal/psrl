"""Summarize one matched science training pair from recorded attempts and telemetry."""

import argparse
import json
from collections import Counter
from pathlib import Path

from plot_science_training_ab import read_arm


def analyze(root, mode, design):
    data = read_arm(root, mode)
    launch = data["launch"]
    attempts = [json.loads(p.read_text()) for p in (root / mode).glob("pool-*/attempts/*.json")]
    accepted = [a for a in attempts if a["accepted"]]
    groups = Counter(a["parent_id"] for a in accepted)
    target = (
        len(groups) == design["batch_size"] * design["steps_per_arm"]
        and all(n == design["rollout_n"] for n in groups.values())
        and all(a["exception_type"] is None for a in accepted)
    )
    first = min((a["slot_started_at"] for a in attempts), default=None)
    last = max((a["finished_at"] for a in accepted), default=None) if target else None
    samples = data["telemetry"]
    units = [u for s in samples for u in s.get("units", [])]
    gpu_rows = []
    for sample in samples:
        for line in sample.get("gpu", "").splitlines():
            fields = line.split(",")
            if len(fields) == 5 and int(fields[0]) in design["environment_gpus"]:
                gpu_rows.append({"time": sample["time"], "mib": float(fields[2]), "util": float(fields[3])})
    phases = []
    for path in (root / mode).glob("pool-*/harbor-jobs/**/gpu-command-phase.json"):
        value = json.loads(path.read_text())
        if value:
            phases.append(value)
    successful_cuda = [a["pressure"] for a in accepted]
    events = sorted(
        [(p["allocated_at"], 1) for p in successful_cuda] + [(p["finished_at"], -1) for p in successful_cuda]
    )
    active = 0
    peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    attempt_events = sorted([(a["admitted_at"], 1) for a in attempts] + [(a["finished_at"], -1) for a in attempts])
    active_attempts = peak_attempts = 0
    for _, change in attempt_events:
        active_attempts += change
        peak_attempts = max(peak_attempts, active_attempts)
    model_metrics = []
    agent_turns = []
    for attempt in accepted:
        paths = list((root / mode).glob(f"pool-*/harbor-jobs/{attempt['session_id']}/**/agent/trajectory.json"))
        if len(paths) == 1:
            trajectory = json.loads(paths[0].read_text())
            model_metrics.append(trajectory.get("final_metrics", {}))
            agent_turns.append(sum(step["source"] == "agent" for step in trajectory["steps"]))
    versions = {}
    for version in sorted({str(a.get("version_tag")) for a in attempts}):
        current = [a for a in attempts if str(a.get("version_tag")) == version]
        completed = [a for a in current if a["accepted"]]
        versions[version] = {
            "attempts": len(current),
            "accepted": len(completed),
            "oom_attempts": sum(a["pressure"].get("status") == "cuda_oom" for a in current),
            "first_slot_at": min(a["slot_started_at"] for a in current),
            "last_accepted_at": max((a["finished_at"] for a in completed), default=None),
        }
    batches = {}
    for batch in sorted({int(a["parent_id"]) // design["batch_size"] for a in attempts}):
        current = [a for a in attempts if int(a["parent_id"]) // design["batch_size"] == batch]
        completed = [a for a in current if a["accepted"]]
        first_slot = min(a["slot_started_at"] for a in current)
        last_accepted = max((a["finished_at"] for a in completed), default=None)
        complete = len(completed) == design["batch_size"] * design["rollout_n"]
        batches[str(batch)] = {
            "attempts": len(current),
            "accepted": len(completed),
            "oom_attempts": sum(a["pressure"].get("status") == "cuda_oom" for a in current),
            "first_slot_at": first_slot,
            "last_accepted_at": last_accepted,
            "rollout_span_seconds": last_accepted - first_slot if complete else None,
            "complete": complete,
        }
    outside = []
    selected = {*design["model_gpus"], *design["environment_gpus"]}
    for sample in samples:
        for line in sample.get("gpu", "").splitlines():
            fields = line.split(",")
            if len(fields) == 5 and int(fields[0]) not in selected and float(fields[2]) > 64:
                outside.append({"time": sample["time"], "index": int(fields[0]), "used_mib": float(fields[2])})
    for batch in batches.values():
        batch["other_gpu_busy_observations_during_rollouts"] = sum(
            batch["first_slot_at"] <= item["time"] <= batch["last_accepted_at"]
            for item in outside
            if batch["last_accepted_at"] is not None
        )
    return (
        {
            "mode": mode,
            "exit_code": launch.get("exit_code"),
            "timed_out": launch.get("timed_out", False),
            "driver_seconds_including_cleanup": launch.get("finished_at", 0) - launch.get("started_at", 0),
            "attempts": len(attempts),
            "oom_attempts": sum(a["pressure"].get("status") == "cuda_oom" for a in attempts),
            "non_oom_rejected_attempts": sum(
                not a["accepted"] and a["pressure"].get("status") != "cuda_oom" for a in attempts
            ),
            "accepted": len(accepted),
            "accepted_per_group": dict(groups),
            "equal_completion_target_met": target,
            "first_slot_started_at": first,
            "before_first_slot_seconds": first - launch["started_at"] if first is not None else None,
            "sampling_seconds_including_retries": last - first if last is not None else None,
            "steps": data["steps"],
            "per_request_version_tag": versions,
            "per_input_batch": batches,
            "other_gpu_activity": outside,
            "sum_step_seconds": sum(s.get("timing_s/step", 0) for s in data["steps"]),
            "sum_wait_for_gen_seconds": sum(s.get("timing_s/wait_for_gen", 0) for s in data["steps"]),
            "sum_update_actor_seconds": sum(s.get("timing_s/update_actor", 0) for s in data["steps"]),
            "sampling_span_includes_interstep_training": design["steps_per_arm"] > 1,
            "repair_rewards_by_group": {
                group: [a["rewards"].get("reward_repair") for a in accepted if a["parent_id"] == group]
                for group in groups
            },
            "accepted_exception_types": dict(Counter(str(a["exception_type"]) for a in accepted)),
            "accepted_llm_completion_tokens": sum(m.get("total_completion_tokens", 0) for m in model_metrics),
            "accepted_llm_metric_records": len(model_metrics),
            "accepted_agent_turn_counts": agent_turns,
            "environment_gpu_peak_mib": max((g["mib"] for g in gpu_rows), default=None),
            "environment_gpu_peak_utilization_percent": max((g["util"] for g in gpu_rows), default=None),
            "peak_successful_cuda_phases": peak,
            "peak_active_finished_attempts": peak_attempts,
            "gpu_uuids": sorted({a["pressure"].get("gpu_uuid_hex") for a in accepted}),
            "peak_leases": max((u["containers_running"] for u in units), default=None),
            "peak_active_gpu_commands": max((u["gpu_commands_active"] for u in units), default=None),
            "peak_waiting_gpu_commands": max((u["gpu_commands_waiting"] for u in units), default=None),
            "four_leases_one_active_three_waiting_observed": any(
                u["containers_running"] == 4 and u["gpu_commands_active"] == 1 and u["gpu_commands_waiting"] == 3
                for u in units
            ),
            "budget_violations": sum(
                u["declared_gpu_memory_mb"] > u["gpu_memory_mb"]
                or u["containers_running"] > u["max_containers"]
                or u["cpu_used"] > u["cpu_quota"]
                or u["memory_used_mb"] > u["memory_mb"]
                for u in units
            )
            if units
            else None,
            "last_ledger": samples[-1].get("units") if samples else None,
            "command_phases": phases,
            "gpu_release": launch.get("gpu_release"),
            "telemetry_errors": data["gpu_errors"],
        },
        data,
        gpu_rows,
    )


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    design = json.loads((args.root / "design.json").read_text())
    records, series = {}, {}
    for mode in ("off", "command"):
        records[mode], data, rows = analyze(args.root, mode, design)
        series[mode] = (data, rows)
    off, on = records["off"], records["command"]
    records["comparison"] = {}
    common_steps = 0
    for step in range(1, design["steps_per_arm"] + 1):
        if all(
            any(int(s["step"]) == step for s in arm["steps"])
            and arm["per_input_batch"].get(str(step - 1), {}).get("complete", False)
            for arm in (off, on)
        ):
            common_steps = step
        else:
            break
    if common_steps:
        prefix = {
            "scope": "All commonly completed leading steps. Does not replace the planned completion target.",
            "steps": common_steps,
            "planned_steps": design["steps_per_arm"],
            "both_planned_runs_completed": all(
                x["exit_code"] == 0 and x["equal_completion_target_met"] for x in (off, on)
            ),
        }
        for mode, arm in (("off", off), ("command", on)):
            steps = [s for s in arm["steps"] if int(s["step"]) <= common_steps]
            batches = [arm["per_input_batch"][str(i)] for i in range(common_steps)]
            prefix[mode] = {
                "step_seconds": sum(s["timing_s/step"] for s in steps),
                "wait_for_gen_seconds": sum(s["timing_s/wait_for_gen"] for s in steps),
                "update_actor_seconds": sum(s["timing_s/update_actor"] for s in steps),
                "trained_tokens": sum(s["termination/trained_tokens"] for s in steps),
                "accepted": sum(b["accepted"] for b in batches),
                "attempts": sum(b["attempts"] for b in batches),
                "oom_attempts": sum(b["oom_attempts"] for b in batches),
            }
        for key in ("step_seconds", "wait_for_gen_seconds", "update_actor_seconds"):
            a, b = (prefix[mode][key] for mode in ("off", "command"))
            prefix[f"{key}_reduction_percent"] = (1 - b / a) * 100 if a else None
        records["comparison"]["matched_prefix"] = prefix
    if off["equal_completion_target_met"] and on["equal_completion_target_met"]:
        a, b = (x["sampling_seconds_including_retries"] for x in (off, on))
        records["comparison"]["sampling_time_reduction_percent"] = (1 - b / a) * 100
    if all(
        x["exit_code"] == 0
        and x["equal_completion_target_met"]
        and {int(s["step"]) for s in x["steps"]} == set(range(1, design["steps_per_arm"] + 1))
        for x in (off, on)
    ):
        a, b = (x["driver_seconds_including_cleanup"] for x in (off, on))
        records["comparison"]["driver_time_reduction_percent"] = (1 - b / a) * 100
        for key in ("sum_step_seconds", "sum_wait_for_gen_seconds", "sum_update_actor_seconds"):
            records["comparison"][f"{key}_reduction_percent"] = (1 - on[key] / off[key]) * 100 if off[key] else None
    (args.root / "comparison.json").write_text(json.dumps(records, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for mode, color, label in (("off", "#bd6730", "OFF"), ("command", "#157f9c", "ON")):
        data, rows = series[mode]
        start = records[mode]["first_slot_started_at"] or data["launch"]["started_at"]
        rows = [g for g in rows if g["time"] >= start]
        axes[0, 0].plot([g["time"] - start for g in rows], [g["mib"] / 1024 for g in rows], label=label, color=color)
    axes[0, 0].set(
        xlabel="Seconds from first rollout request",
        ylabel="Environment GPU GiB",
        title="Measured memory, single environment GPU",
    )
    axes[0, 0].legend()
    for ax, key, title, unit in (
        (
            axes[0, 1],
            "sampling_seconds_including_retries",
            "First to last rollout, includes interstep work",
            "Seconds",
        ),
        (axes[1, 0], "oom_attempts", "OOM attempts retained in audit", "Attempts"),
        (
            axes[1, 1],
            "driver_seconds_including_cleanup",
            "Driver elapsed including initialization and cleanup",
            "Seconds",
        ),
    ):
        values = [off[key], on[key]]
        for i, value in enumerate(values):
            if value is not None:
                ax.bar(i, value, color=["#bd6730", "#157f9c"][i])
                note = ""
                if key == "driver_seconds_including_cleanup":
                    arm = (off, on)[i]
                    stop = "Budget stop" if arm["timed_out"] else "Failed" if arm["exit_code"] else "Completed"
                    note = f"\n{stop}, {len(arm['steps'])} steps"
                ax.text(i, value, f"{value:.1f}{note}", ha="center", va="bottom")
        ax.set(xticks=[0, 1], xticklabels=["OFF", "ON"], title=title, ylabel=unit)
        ax.margins(y=0.15)
    fig.suptitle("Real Qwen3.5-4B / LAPS training + declared synthetic GPU memory pressure")
    interference = any(x["other_gpu_activity"] for x in (off, on))
    footnote = (
        "Other-GPU activity observed." if interference else "No other-GPU memory activity above 64 MiB observed."
    )
    fig.supxlabel(f"One paired run. {footnote} Timing is observational.", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(args.root / f"comparison.{suffix}", dpi=160)
    if design["steps_per_arm"] > 1:
        steps_fig, steps_axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for axis, key, title, unit in (
            (steps_axes[0, 0], "timing_s/step", "Complete training step", "Seconds"),
            (steps_axes[0, 1], "timing_s/wait_for_gen", "Trainer waiting for rollout data", "Seconds"),
            (steps_axes[1, 0], "timing_s/update_actor", "Actor update", "Seconds"),
            (steps_axes[1, 1], "termination/trained_tokens", "Valid trained response tokens", "Tokens"),
        ):
            for mode, color, label in (("off", "#bd6730", "OFF"), ("command", "#157f9c", "ON")):
                steps = [s for s in records[mode]["steps"] if key in s]
                axis.plot([s["step"] for s in steps], [s[key] for s in steps], "o-", color=color, label=label)
            axis.set(title=title, xlabel="Completed optimizer step", ylabel=unit)
            axis.grid(alpha=0.2)
            axis.legend()
        steps_fig.suptitle(f"Planned {design['steps_per_arm']}-step science training, completed steps shown")
        steps_fig.supxlabel(f"{footnote} Compare token workload alongside elapsed time.", fontsize=10)
        for suffix in ("png", "svg"):
            steps_fig.savefig(args.root / f"per-step.{suffix}", dpi=160)
    if common_steps:
        prefix = records["comparison"]["matched_prefix"]
        prefix_fig, prefix_axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for axis, key, title, divisor, unit in (
            (prefix_axes[0], "step_seconds", "Complete training steps", 60, "Minutes"),
            (prefix_axes[1], "wait_for_gen_seconds", "Trainer waiting for rollout data", 60, "Minutes"),
            (prefix_axes[2], "oom_attempts", "OOM attempts", 1, "Attempts"),
        ):
            for i, mode in enumerate(("off", "command")):
                value = prefix[mode][key] / divisor
                axis.bar(i, value, color=["#bd6730", "#157f9c"][i])
                axis.text(i, value, f"{value:.2f}", ha="center", va="bottom")
            axis.set(xticks=[0, 1], xticklabels=["OFF", "ON"], title=title, ylabel=unit)
            axis.margins(y=0.15)
        prefix_fig.suptitle(f"Common completed prefix: {common_steps} steps per arm")
        completion = "complete" if prefix["both_planned_runs_completed"] else "incomplete"
        prefix_fig.supxlabel(
            f"Planned {design['steps_per_arm']}-step pair was {completion}. See failure and interference evidence."
        )
        for suffix in ("png", "svg"):
            prefix_fig.savefig(args.root / f"matched-prefix.{suffix}", dpi=160)
    print(json.dumps({k: v for k, v in records.items() if k == "comparison"}, indent=2))


if __name__ == "__main__":
    main()
