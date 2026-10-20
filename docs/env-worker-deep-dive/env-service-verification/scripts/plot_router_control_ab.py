"""Summarize equal completed work, retries and measured command reservations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(directory):
    result = json.loads((directory / "result.json").read_text())
    attempts = result["attempts"]
    accepted = [x for x in attempts if x["accepted"]]
    phases = [(a, p) for a in attempts for p in a["agent"].get("phases", [])]
    reservations = [p["reservation"] for _, p in phases if "reservation" in p]
    telemetry = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    units = [u for t in telemetry for u in t.get("units", [])]
    running = [(a, p["cuda"]) for a, p in phases if p.get("cuda", {}).get("status") == "ok"]
    overlap = []
    for a, p in running:
        for previous, q in running:
            if a["session"] == previous["session"]:
                continue
            if q["finished_at"] <= p["allocated_at"] < previous["agent"]["finished_at"]:
                overlap.append(
                    {
                        "previous": previous["session"],
                        "next": a["session"],
                        "gap_s": p["allocated_at"] - q["finished_at"],
                    }
                )
    queued = [p["admitted_at"] - p["requested_at"] for p in reservations if "admitted_at" in p]
    returned = [p["released_at"] - p["command_finished_at"] for p in reservations if "released_at" in p]
    cpu_reuse = [
        {"gpu_session": a["session"], "cpu_session": other["session"], "allocated_at": p["allocated_at"]}
        for a, p in running
        for other in attempts
        if a["session"] != other["session"]
        and other["agent"].get("cpu_started_at", float("inf"))
        <= p["allocated_at"]
        < other["agent"].get("cpu_finished_at", float("-inf"))
    ]
    control = {}
    if (directory / "control-trace.json").exists():
        trace = json.loads((directory / "control-trace.json").read_text())
        events = [e for n in trace["nodes"].values() for e in n["events"]]
        for label in ("inspect_container", "read_gpu_devices", "read_gpu_snapshot"):
            values = [e["duration_s"] for e in events if e["label"] == label]
            control[label] = {"count": len(values), "mean_s": float(np.mean(values)), "max_s": max(values)}
    summary = {
        "passed": result.get("passed", False),
        "error": result.get("error"),
        "completed": len(accepted),
        "attempts": len(attempts),
        "oom_attempts": sum(x["agent"]["status"] == "cuda_oom" for x in attempts),
        "successful_cuda_commands": len(running),
        "accepted_cuda_commands": sum(len(x["agent"]["phases"]) for x in accepted),
        "work_duration_s": result.get("work_duration_s"),
        "startup_s": result.get("work_started_at", result["started_at"]) - result["started_at"],
        "total_s": result["finished_at"] - result["started_at"],
        "agent_preparation_mean_s": float(np.mean([x["agent"]["started_at"] - x["started_at"] for x in accepted]))
        if accepted
        else None,
        "post_agent_mean_s": float(np.mean([x["finished_at"] - x["agent"]["finished_at"] for x in accepted]))
        if accepted
        else None,
        "ray_initialized": result.get("ray_initialized"),
        "runner_source": result["runner_source"],
        "completions_per_task": {str(i): sum(x["task"] == i for x in accepted) for i in range(4)},
        "gpu_peak_mib": max((t["gpu"]["used_mib"] for t in telemetry if "gpu" in t), default=None),
        "queue_mean_s": float(np.mean(queued)) if queued else None,
        "queue_max_s": max(queued, default=None),
        "return_mean_s": float(np.mean(returned)) if returned else None,
        "return_max_s": max(returned, default=None),
        "cpu_overlaps_return_count": sum(
            x["agent"].get("cpu_started_at", float("inf"))
            < x["agent"]["phases"][0].get("reservation", {}).get("released_at", float("-inf"))
            for x in accepted
        ),
        "ledger_max_containers": max((u["containers_running"] for u in units), default=None),
        "ledger_max_active": max((u["gpu_commands_active"] for u in units), default=None),
        "ledger_max_waiting": max((u["gpu_commands_waiting"] for u in units), default=None),
        "ledger_idle_live_samples": sum(
            u["containers_running"] > 0 and u["declared_gpu_memory_mb"] == 0 for u in units
        ),
        "ledger_budget_violations": sum(u["declared_gpu_memory_mb"] > u["gpu_memory_mb"] for u in units),
        "reuse_before_previous_agent_end": overlap,
        "gpu_start_during_other_cpu_interval": cpu_reuse,
        "control_queries": control,
        "gpu_after": result.get("gpu_after"),
        "remaining_owned_containers": result.get("remaining_owned_containers"),
        "cleanup_errors": result.get("cleanup_errors"),
    }
    return summary, result, telemetry


def main(root):
    data = {name: summarize(root / name) for name in ("low-off", "low-on", "pressure-off", "pressure-on")}
    (root / "comparison.json").write_text(json.dumps({k: v[0] for k, v in data.items()}, indent=2))
    colors = {"off": "#b65c36", "on": "#267e9f"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i, load in enumerate(("low", "pressure")):
        ax = axes[0, i]
        for j, mode in enumerate(("off", "on")):
            summary = data[f"{load}-{mode}"][0]
            duration = summary["work_duration_s"]
            if duration is not None:
                ax.bar(j, duration, color=colors[mode])
                ax.bar(j, summary["total_s"] - duration, bottom=duration, color="#cccccc", hatch="//")
                ax.text(
                    j,
                    summary["total_s"],
                    f"{duration:.1f}s work\n{summary['total_s']:.1f}s total",
                    ha="center",
                    va="bottom",
                )
        ax.set_xticks([0, 1], ["OFF: 0eeead8", "ON: fixed"])
        ax.set_ylabel("Recorded seconds including retries")
        ax.set_title("1 GiB per command, 4 completions" if load == "low" else "50 GiB per command, 16 completions")
        ax.margins(y=0.25)
    ax = axes[1, 0]
    for mode in ("off", "on"):
        summary, result, _ = data[f"pressure-{mode}"]
        times = sorted(x["finished_at"] - result["work_started_at"] for x in result["attempts"] if x["accepted"])
        ax.step([0, *times], list(range(len(times) + 1)), where="post", color=colors[mode], label=mode.upper())
    ax.set(xlabel="Seconds from task submission", ylabel="Completed tasks", title="Equal useful work under pressure")
    ax.legend()
    ax = axes[1, 1]
    positions = np.arange(2)
    for i, mode in enumerate(("off", "on")):
        summary = data[f"pressure-{mode}"][0]
        values = [summary["attempts"], summary["oom_attempts"]]
        ax.bar(positions + (i - 0.5) * 0.35, values, width=0.35, color=colors[mode], label=mode.upper())
        for x, value in zip(positions + (i - 0.5) * 0.35, values, strict=True):
            ax.text(x, value, str(value), ha="center", va="bottom")
    ax.set_xticks(positions, ["All attempts", "OOM attempts"])
    ax.set_title("Failures do not count as completion")
    ax.legend()
    fig.suptitle(
        "One GPU, science runner + Harbor + CUDA, no LLM or training\n"
        "Gray: separate initialization and final cleanup (Python import time excluded)"
    )
    for suffix in ("png", "svg"):
        fig.savefig(root / f"completion.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True, sharex=True)
    for i, mode in enumerate(("off", "on")):
        _, result, telemetry = data[f"pressure-{mode}"]
        origin = result["work_started_at"]
        samples = [t for t in telemetry if "gpu" in t]
        axes[i].plot(
            [t["time"] - origin for t in samples],
            [t["gpu"]["used_mib"] / 1024 for t in samples],
            color=colors[mode],
            label="Measured application GPU memory",
        )
        if mode == "on":
            samples = [t for t in telemetry if t.get("units")]
            axes[i].plot(
                [t["time"] - origin for t in samples],
                [t["units"][0]["declared_gpu_memory_mb"] / 1024 for t in samples],
                linestyle="--",
                color="#424242",
                label="Reserved GPU budget",
            )
            for field, label in (
                ("containers_running", "Live sessions"),
                ("gpu_commands_active", "Active GPU commands"),
                ("gpu_commands_waiting", "Waiting GPU commands"),
            ):
                axes[2].step(
                    [t["time"] - origin for t in samples],
                    [t["units"][0][field] for t in samples],
                    where="post",
                    label=label,
                )
        axes[i].set(ylabel="GiB", title=mode.upper() + " pressure: memory occupancy, not compute utilization")
        axes[i].legend(loc="upper right")
    axes[2].set(ylabel="Count", xlabel="Seconds from task submission", title="ON ledger: sessions and GPU phases")
    axes[2].legend(loc="upper right")
    for suffix in ("png", "svg"):
        fig.savefig(root / f"resources.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
