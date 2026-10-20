"""Summarize measured control spans and plot equal-work probes before and after the fix."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(data):
    """Attribute successful admission to measured leaf spans, keeping residual time explicit."""
    if not data.get("passed") or data.get("error") or data.get("cleanup_errors"):
        raise ValueError("Expected a completed probe with successful cleanup.")
    trace = data["control_trace"]
    node_events = [event for node in trace["nodes"].values() for event in node["events"]]
    trials = []
    for trial in data["trials"]:
        events = {event["event"]: event for event in trial["events"]}
        first = events["A_return_joined"]["timing"]
        following = events["A_next_command_completed"]["timing"]
        other = events["B_completed"]["timing"]
        grant = next(
            event
            for event in trace["router"]["events"]
            if event["scope"] == following["request_id"]
            and event["label"] == "acquire"
            and event.get("status") == "granted"
        )
        spans = {
            "gpu_snapshot_s": 0.0,
            "device_identity_s": 0.0,
            "docker_inspect_s": 0.0,
            "device_permissions_s": 0.0,
        }
        for event in node_events:
            if not grant["started_at"] <= event["started_at"] <= following["admitted_at"]:
                continue
            label = event["label"]
            if label.startswith("subprocess:nvidia-smi") or label == "read_gpu_snapshot":
                spans["gpu_snapshot_s"] += event["duration_s"]
            elif label == "read_gpu_devices":
                spans["device_identity_s"] += event["duration_s"]
            elif label.startswith("subprocess:docker inspect") or label == "inspect_container":
                spans["docker_inspect_s"] += event["duration_s"]
            elif label == "set_access":
                spans["device_permissions_s"] += event["duration_s"]
        spans["other_control_s"] = grant["duration_s"] - sum(spans.values())
        trials.append(
            {
                "index": trial["index"],
                "mode": trial["mode"],
                "total_s": trial["duration_s"],
                "command_to_planning_s": events["model_start"]["time"] - first["command_finished_at"],
                "first_return_s": first["released_at"] - first["command_finished_at"],
                "A_next_request_to_admission_s": following["admitted_at"] - following["requested_at"],
                "A_next_after_budget_available_s": following["admitted_at"]
                - max(following["requested_at"], other["released_at"]),
                "successful_acquire_s": grant["duration_s"],
                "successful_acquire_spans": spans,
                "unit_lock_wait_s": sum(
                    event["duration_s"]
                    for event in trace["router"]["events"]
                    if event["scope"] == following["request_id"] and event["label"] == "unit_lock_wait"
                ),
            }
        )
    return {
        "scope": data["scope"],
        "trials": trials,
        "means": {
            mode: {
                key: statistics.mean(trial[key] for trial in trials if trial["mode"] == mode)
                for key in ("total_s", "A_next_after_budget_available_s", "command_to_planning_s")
            }
            for mode in ("sync", "async")
        },
    }


def main(root):
    report = {arm: summarize(json.loads((root / arm / "result.json").read_text())) for arm in ("before", "after")}
    (root / "analysis.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    colors = {"sync": "#607d8b", "async": "#00897b"}
    for ax, metric, title in zip(
        axes[:2],
        ("total_s", "A_next_after_budget_available_s"),
        ("Complete all three CUDA commands", "A next GPU: delay after budget is available"),
        strict=True,
    ):
        for index, arm in enumerate(("before", "after")):
            for offset, mode in ((-0.18, "sync"), (0.18, "async")):
                values = [trial[metric] for trial in report[arm]["trials"] if trial["mode"] == mode]
                value = statistics.mean(values)
                x = index + offset
                ax.bar(x, value, width=0.34, color=colors[mode], label=mode if index == 0 else None)
                ax.scatter([x] * len(values), values, color="black", s=15, zorder=3)
                ax.text(x, max(values) + 0.1, f"{value:.2f}", ha="center", fontsize=9)
        ax.set_xticks([0, 1], ["CLI queries", "Direct API queries"])
        ax.set_ylabel("Seconds")
        ax.set_title(title, fontsize=11)
        ax.legend()
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    labels = {
        "gpu_snapshot_s": "Live GPU snapshot",
        "device_identity_s": "GPU identity",
        "docker_inspect_s": "Container inspect",
        "device_permissions_s": "Device permissions",
        "other_control_s": "Other control / RPC",
    }
    for index, arm in enumerate(("before", "after")):
        left = 0
        for color_index, (key, label) in enumerate(labels.items()):
            value = statistics.mean(trial["successful_acquire_spans"][key] for trial in report[arm]["trials"])
            axes[2].barh(
                index,
                value,
                left=left,
                color=plt.get_cmap("tab10")(color_index),
                label=label if index == 0 else None,
            )
            left += value
        axes[2].text(left + 0.03, index, f"{left:.3f}s", va="center", fontsize=9)
    axes[2].set_yticks([0, 1], ["CLI", "Direct API"])
    axes[2].set_xlabel("Seconds, mean successful A next admission")
    axes[2].set_title("Measured control work, excluding capacity wait", fontsize=11)
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_xlim(0, axes[2].get_xlim()[1] * 1.25)
    fig.suptitle("Real Ray + Docker + CUDA, fixed 8s simulated planning, two rounds per mode and implementation")
    fig.savefig(root / "control-comparison.png", dpi=180)
    fig.savefig(root / "control-comparison.svg")
    plt.close(fig)
    for arm, summary in report.items():
        print(json.dumps({"implementation": arm, "means": summary["means"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
