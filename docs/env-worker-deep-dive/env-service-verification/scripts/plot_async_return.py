"""Plot measured return and model overlap without attributing model quality to scheduling."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(directory):
    data = json.loads((directory / "result.json").read_text())
    if not data.get("passed"):
        raise RuntimeError("The real experiment did not complete. No success chart can be drawn.")
    trials = data["trials"]
    summary = {}
    for mode in ("sync", "async"):
        group = [trial for trial in trials if trial["mode"] == mode]
        summary[mode] = {
            "samples": len(group),
            "mean_s": statistics.mean(trial["duration_s"] for trial in group),
            "durations_s": [trial["duration_s"] for trial in group],
            "overlap_count": sum(trial["model_overlaps_return"] for trial in group),
            "completion_tokens": [],
            "command_to_model_s": [],
            "return_check_s": [],
        }
        for trial in group:
            events = {event["event"]: event for event in trial["events"]}
            record = events["A_return_joined"]["timing"]
            summary[mode]["completion_tokens"].append(events["model_end"]["usage"]["completion_tokens"])
            summary[mode]["command_to_model_s"].append(events["model_start"]["time"] - record["command_finished_at"])
            summary[mode]["return_check_s"].append(record["released_at"] - record["command_finished_at"])
    summary["reduction_pct"] = 100 * (1 - summary["async"]["mean_s"] / summary["sync"]["mean_s"])
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for x, mode in enumerate(("sync", "async")):
        ax.bar(x, summary[mode]["mean_s"], color=("#607d8b", "#00897b")[x], width=0.55)
        ax.scatter([x] * len(summary[mode]["durations_s"]), summary[mode]["durations_s"], color="black", zorder=3)
        ax.text(x, summary[mode]["mean_s"], f"{summary[mode]['mean_s']:.2f}s", ha="center", va="bottom")
    ax.set_xticks([0, 1], ["Wait for return, then plan", "Plan while return is checked"])
    ax.set_ylabel("Seconds to finish A and B")
    planning = "simulated planning" if data.get("planning_mode") == "controlled_delay" else "vLLM"
    change = "lower" if summary["reduction_pct"] >= 0 else "higher"
    counts = f"n={summary['sync']['samples']} sync / {summary['async']['samples']} async"
    ax.set_title(
        f"Real CUDA + {planning}, equal work\n{abs(summary['reduction_pct']):.1f}% {change} mean time, {counts}"
    )
    fig.savefig(directory / "completion.png", dpi=180)
    fig.savefig(directory / "completion.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    colors = {"GPU command": "#1565c0", "Return check": "#ef6c00", planning: "#00897b"}
    for ax, mode in zip(axes, ("sync", "async"), strict=True):
        trial = next(t for t in trials if t["mode"] == mode)
        events = {event["event"]: event for event in trial["events"]}
        a = events["A_return_joined"]["timing"]
        b = events["B_completed"]["timing"]
        next_a = events["A_next_command_completed"]["timing"]
        origin = a["command_finished_at"]
        spans = [
            (0, a["command_finished_at"], a["released_at"], "Return check"),
            (1, events["model_start"]["time"], events["model_end"]["time"], planning),
            (2, b["admitted_at"], b["command_finished_at"], "GPU command"),
            (2, b["command_finished_at"], b["released_at"], "Return check"),
            (3, next_a["admitted_at"], next_a["command_finished_at"], "GPU command"),
            (3, next_a["command_finished_at"], next_a["released_at"], "Return check"),
        ]
        for row, start, end, kind in spans:
            ax.barh(row, end - start, left=start - origin, color=colors[kind], height=0.55)
        ax.axvline(a["released_at"] - origin, color="#ef6c00", linestyle=":")
        ax.set_yticks(range(4), ["A return", "A planning", "B backfill", "A next GPU"])
        ax.invert_yaxis()
        ax.set_title(f"{mode}: first measured round, actual timestamps")
        ax.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("Seconds after A's first CUDA command completes")
    for label, color in colors.items():
        axes[0].plot([], [], color=color, linewidth=6, label=label)
    axes[0].legend(loc="upper right")
    fig.savefig(directory / "overlap.png", dpi=180)
    fig.savefig(directory / "overlap.svg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
