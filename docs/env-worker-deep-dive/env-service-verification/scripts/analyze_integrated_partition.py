"""Summarize completed PSRL steps separately from environment and inference work."""

import argparse
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p95": ordered[math.ceil(0.95 * len(values)) - 1],
        "max": max(values),
        "sum": sum(values),
    }


def analyze(arm):
    status = json.loads((arm / "status.json").read_text())
    launcher = (arm / "launcher.log").read_text()
    archive = arm / "trainer-console.log"
    pids = set(re.findall(r"TaskRunner pid=(\d+)", launcher))
    logs = Path(status["ray_tmp"]) / "session_latest" / "logs"
    actor_logs = [path for pid in pids for path in logs.glob(f"worker-*-{pid}.out")]
    if actor_logs:
        archive.write_text("\n".join(path.read_text() for path in actor_logs))
    console = launcher + ("\n" + archive.read_text() if archive.exists() else "")
    steps = {}
    for line in console.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        match = re.search(r"\bstep:(\d+) - ", line)
        if match:
            steps[int(match[1])] = {
                key: float(value)
                for key, value in re.findall(r"(?: - )([A-Za-z0-9_./%-]+):(?:np\.[a-z0-9_]+\()?([-+0-9.eE]+)", line)
            }
    steps = dict(sorted(steps.items()))
    episodes = [json.loads(path.read_text()) for path in sorted(arm.glob("episodes/*/episode.json"))]
    commands = [command for episode in episodes for command in episode["commands"]]
    queries = [query for episode in episodes for query in episode["queries"]]
    events = {}
    peaks = {}
    command_concurrency = {}
    pending_controls = {}
    telemetry_errors = 0
    for line in (arm / "telemetry.jsonl").read_text().splitlines():
        sample = json.loads(line)
        telemetry_errors += bool(sample.get("telemetry_error"))
        for unit in sample.get("units", []):
            name = unit["unit_id"]
            command_concurrency[name] = max(command_concurrency.get(name, 0), unit["gpu_commands_active"])
        for unit, model in sample.get("models", {}).items():
            pending_controls[unit] = max(pending_controls.get(unit, 0), model["pending_operations"])
            for event in model["events"]:
                events[(unit, event["method"], event["started_at"])] = event
        for gpu, reading in sample.get("gpus", {}).items():
            peaks[gpu] = max(peaks.get(gpu, 0), reading["used_mb"])
    waits = [command["admission"]["admitted_at"] - command["admission"]["requested_at"] for command in commands]
    placements = {}
    for command in commands:
        unit = command["admission"]["unit_id"]
        placements[unit] = placements.get(unit, 0) + 1
    versions = set()
    for path in (arm / "psrl_logs").rglob("smg_routing_status.jsonl"):
        for line in path.read_text().splitlines():
            versions.update(worker["version_tag"] for worker in json.loads(line).get("workers", []))
    timings = {}
    for metrics in steps.values():
        for key, value in metrics.items():
            if key.startswith("timing_s/"):
                timings[key] = timings.get(key, 0) + value
    overlaps = {}
    clock_file = arm.parent / "clock-calibration.json"
    if clock_file.exists():
        offset = json.loads(clock_file.read_text())["wall_minus_monotonic"]
        for replica, unit in ((0, "gpu-2"), (1, "gpu-3")):
            intervals = [
                (row["result"]["prepared_monotonic_s"] + offset, row["result"]["finished_monotonic_s"] + offset)
                for row in commands
                if row["admission"]["unit_id"] == unit
            ]
            samples = [
                json.loads(line)
                for path in (arm / "psrl_logs").rglob(f"stats_r{replica}_dp0.jsonl")
                for line in path.read_text().splitlines()
            ]
            overlaps[unit] = sum(
                sample["num_running_reqs"] > 0
                and any(
                    start + 1 < datetime.fromisoformat(sample["ts"]).timestamp() < end - 1 for start, end in intervals
                )
                for sample in samples
            )
    return {
        "mode": arm.name,
        "status": {
            key: status.get(key)
            for key in ("status", "exit_code", "error", "physical_release_verified", "initialization_s")
        },
        "launch_to_cleanup_s": status.get("finished_at", status["started_at"]) - status["started_at"],
        "step_metric_sources": ["launcher.log"] + (["trainer-console.log"] if archive.exists() else []),
        "steps": steps,
        "timing_totals": timings,
        "smg_observed_model_versions": sorted(versions),
        "episode_status_counts": {
            state: sum(row["status"] == state for row in episodes)
            for state in sorted({row["status"] for row in episodes})
        },
        "episodes_started": len(episodes),
        "episodes_completed": sum(row["status"] == "complete" for row in episodes),
        "episode_errors": [row.get("error") for row in episodes if row["status"] == "failed"],
        "parsed_actions": sum(row.get("action_parsed", False) for row in episodes),
        "environment_updates_completed": sum(row["result"].get("updates", 0) for row in commands),
        "commands_completed": len(commands),
        "command_placements": placements,
        "environment_wait_s": distribution(waits),
        "environment_total_s": distribution([row["finished_at"] - row["started_at"] for row in commands]),
        "environment_compute_s": distribution([row["result"]["compute_s"] for row in commands]),
        "command_phases": {
            mode: {
                "count": len(selected),
                "lease_s": distribution(
                    [row["admission"]["command_finished_at"] - row["admission"]["admitted_at"] for row in selected]
                ),
                "compute_s": distribution([row["result"]["compute_s"] for row in selected]),
                "queue_s": distribution(
                    [row["admission"]["admitted_at"] - row["admission"]["requested_at"] for row in selected]
                ),
            }
            for mode in ("train", "verify")
            if (selected := [row for row in commands if row["mode"] == mode])
        },
        "queries_completed": len(queries),
        "query_latency_s": distribution([query["finished_at"] - query["started_at"] for query in queries]),
        "generated_tokens": sum((query.get("usage") or {}).get("completion_tokens", 0) for query in queries),
        "reward": distribution([row["reward"] for row in episodes if "reward" in row]),
        "actions": {
            str(rate): sum(row.get("learning_rate") == rate for row in episodes) for rate in (0.0003, 0.001, 0.003)
        },
        "model_gate_wait_s": distribution([event["gate_wait_s"] for event in events.values()]),
        "model_gate_events": [{"unit": key[0], **event} for key, event in events.items()],
        "peak_device_used_mib": peaks,
        "peak_active_gpu_commands": command_concurrency,
        "peak_pending_model_controls": pending_controls,
        "telemetry_rpc_error_samples": telemetry_errors,
        "inference_active_samples_during_shared_compute": overlaps,
    }


def plot_comparison(result, output):
    """Plot measured end-to-end phases only after both arms complete."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    names = ["Static pool", "Resident sharing"]
    colors = ["#b6bec9", "#397dc4", "#e9a442", "#d7dce3"]
    labels = ["Initialization", "Waiting for batch", "Training phases", "Other / cleanup"]
    for index, arm in enumerate(result["arms"]):
        timings = arm["timing_totals"]
        initialization = arm["status"]["initialization_s"]
        phases = [
            initialization,
            timings["timing_s/wait_for_gen"],
            timings["timing_s/step"] - timings["timing_s/wait_for_gen"],
            arm["launch_to_cleanup_s"] - initialization - timings["timing_s/step"],
        ]
        left = 0
        for value, color, label in zip(phases, colors, labels):
            axes[0].barh(index, value / 60, left=left, color=color, label=label if index == 0 else None)
            left += value / 60
        axes[0].text(left + 0.3, index, f"{left:.1f} min", va="center")
        steps = sorted(arm["steps"])
        axes[1].plot(steps, [arm["steps"][step]["timing_s/step"] for step in steps], marker="o", label=names[index])
    axes[0].set(yticks=[0, 1], yticklabels=names, xlabel="Minutes", title="Initialization through cleanup")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(arm["launch_to_cleanup_s"] for arm in result["arms"]) / 60 * 1.2)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, fontsize=8)
    axes[1].set(xlabel="PSRL training step", ylabel="Seconds", title="Per-step time includes batch waiting")
    axes[1].legend()
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    design = json.loads((args.result / "design.json").read_text())
    arms = [
        analyze(args.result / name) for name in ("fixed", "resident") if (args.result / name / "status.json").exists()
    ]
    complete = len(arms) == 2 and all(
        arm["status"]["exit_code"] == 0
        and arm["status"]["physical_release_verified"]
        and sorted(arm["steps"]) == list(range(1, design["steps_per_arm"] + 1))
        for arm in arms
    )
    result = {"paired_complete": complete, "steps_per_arm": design["steps_per_arm"], "arms": arms}
    if complete:
        fixed, resident = [arm["timing_totals"]["timing_s/step"] for arm in arms]
        result["training_step_time_reduction_fraction"] = 1 - resident / fixed
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.figure:
        if not complete:
            raise ValueError("A complete pair is required for the comparison figure.")
        plot_comparison(result, args.figure)


if __name__ == "__main__":
    main()
