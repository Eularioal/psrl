"""Plot observed CUDA intervals and GPU memory from the science training A/B run."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(root, mode):
    arm = root / mode
    launch = json.loads((arm / "launch.json").read_text())
    episodes = []
    for path in arm.glob("harbor-jobs/**/budget-agent.json"):
        record = json.loads(path.read_text())
        cuda = json.loads(record["command"]["stdout"].strip())
        episodes.append(
            {
                "start": cuda["allocation_start"],
                "held": cuda["held_from"],
                "end": cuda["released_at"],
                "cuda": cuda,
                "path": str(path.relative_to(root)),
            }
        )
    episodes.sort(key=lambda x: x["start"])
    samples = [json.loads(line) for line in (arm / "telemetry.jsonl").read_text().splitlines()]
    if launch.get("final_ledger"):
        samples.append(
            {
                "time": launch["finished_at"],
                "units": launch["final_ledger"],
                "source": "Final snapshot recorded in launch.json",
            }
        )
    log = re.sub(r"\x1b\[[0-9;]*m", "", (arm / "launcher.log").read_text())
    metric_lines = [line for line in log.splitlines() if "training/global_step:" in line]
    metrics = dict(re.findall(r"([\w/]+):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", "\n".join(metric_lines)))
    events = sorted([(x["start"], 1) for x in episodes] + [(x["end"], -1) for x in episodes])
    live = peak = 0
    for _, change in events:
        live += change
        peak = max(peak, live)
    unit_samples = [u for s in samples for u in s.get("units", [])]
    result = {
        "scope": launch.get("scope", "Science training comparison"),
        "training_attempted": "no model inference or training" not in launch.get("scope", ""),
        "exit_code": launch.get("exit_code"),
        "episodes": episodes,
        "completed_cuda_commands": sum(x["cuda"]["kernel_output"] == 42 for x in episodes),
        "max_simultaneous_cuda_contexts": peak,
        "gpu_work_window_s": max(x["end"] for x in episodes) - episodes[0]["start"] if episodes else None,
        "process_wall_s": launch["finished_at"] - launch["started_at"],
        "training_completed": "Training completed successfully!" in log,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "peak_gpu_mib": max(float(s["gpu"].split(",")[1]) for s in samples if s.get("gpu")),
        "max_ledger_leases": max((u["containers_running"] for u in unit_samples), default=None),
        "max_ledger_declared_mib": max((u["declared_gpu_memory_mb"] for u in unit_samples), default=None),
        "max_waiters": max((u["sessions_waiting"] for u in unit_samples), default=None),
        "ledger_samples_with_three_and_waiter": sum(
            u["containers_running"] == 3 and u["sessions_waiting"] > 0 for u in unit_samples
        ),
    }
    return result, samples


def main(root):
    data = {mode: collect(root, mode) for mode in ("off", "on")}
    summary = {mode: values[0] for mode, values in data.items()}
    training_attempted = any(r["training_attempted"] for r in summary.values())
    summary["interpretation"] = {
        "task": (
            "Fixed resource verification fixture in the real SciAccel training path"
            if training_attempted
            else "Real science runner and Harbor CUDA comparison, no inference or training"
        ),
        "control": "EnvRouter disabled retains the existing exclusive GPU gate",
        "unit_budget_gib": 55,
        "per_task_declared_gib": 14,
        "nominal_remaining_gib_at_three": 13,
        "limitation": "No claim of universal contention freedom, convergence, or vLLM/Docker same-card lending",
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 11), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=[1.4, 0.85, 1, 0.65])
    memory_ax = fig.add_subplot(grid[0, :])
    colors = {"off": "#a95d27", "on": "#176e9d"}
    labels = {"off": "Ledger OFF (exclusive GPU gate)", "on": "Ledger ON (55 GiB unit budget)"}
    duration = max(x[0]["gpu_work_window_s"] for x in data.values()) + 5
    for column, mode in enumerate(("off", "on")):
        result, samples = data[mode]
        zero = result["episodes"][0]["start"]
        points = [
            (s["time"] - zero, float(s["gpu"].split(",")[1]) / 1024)
            for s in samples
            if s.get("gpu") and -2 <= s["time"] - zero <= duration
        ]
        memory_ax.plot([p[0] for p in points], [p[1] for p in points], color=colors[mode], label=labels[mode])
        ax = fig.add_subplot(grid[2, column])
        for index, episode in enumerate(result["episodes"]):
            ax.barh(
                index, episode["end"] - episode["start"], left=episode["start"] - zero, height=0.55, color=colors[mode]
            )
        ax.set(
            title=labels[mode],
            xlabel="Seconds since first CUDA allocation",
            xlim=(-2, duration),
            yticks=range(len(result["episodes"])),
            yticklabels=[f"Task {i + 1}" for i in range(len(result["episodes"]))],
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    memory_ax.axhline(55, color="#ae3343", linestyle="--", label="Configured unit budget: 55 GiB")
    subtitle = "model GPUs are separate" if training_attempted else "no model processes"
    memory_ax.set(
        title=f"Observed environment GPU memory ({subtitle})",
        ylabel="GPU memory / GiB",
        xlabel="Seconds since first CUDA allocation in each run",
        xlim=(-2, duration),
        ylim=(0, 60),
    )
    memory_ax.grid(alpha=0.2)
    memory_ax.legend(loc="upper right", fontsize=9)
    ledger_ax = fig.add_subplot(grid[1, :])
    zero = summary["on"]["episodes"][0]["start"]
    ledger_points = [
        (s["time"] - zero, u["containers_running"], u["sessions_waiting"])
        for s in data["on"][1]
        for u in s.get("units", [])
        if -15 <= s["time"] - zero <= duration
    ]
    ledger_ax.step(
        [p[0] for p in ledger_points],
        [p[1] for p in ledger_points],
        where="post",
        color=colors["on"],
        label="Active leases (ON)",
    )
    ledger_ax.step(
        [p[0] for p in ledger_points],
        [p[2] for p in ledger_points],
        where="post",
        color="#ae3343",
        label="Waiting sessions (ON)",
    )
    ledger_ax.set(
        title="Budget admission observed in the real EnvRouter",
        ylabel="Count",
        xlabel="Seconds since first CUDA allocation (ON)",
        xlim=(-15, duration),
        ylim=(-0.15, 4.3),
        yticks=range(5),
    )
    ledger_ax.grid(alpha=0.2)
    ledger_ax.legend(loc="upper right", fontsize=9)
    text_ax = fig.add_subplot(grid[3, :])
    text_ax.axis("off")
    rows = []
    for mode in ("off", "on"):
        r = summary[mode]
        rows.append(
            [
                mode.upper(),
                str(r["completed_cuda_commands"]),
                str(r["max_simultaneous_cuda_contexts"]),
                f"{r['gpu_work_window_s']:.1f} s",
                str(r["metrics"].get("training/global_step", "unconfirmed")) if training_attempted else "Not run",
                str(r["exit_code"]),
            ]
        )
    table = text_ax.table(
        cellText=rows,
        colLabels=["Ledger", "CUDA passed", "Peak concurrent", "GPU work window", "Train step", "Exit"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    title = "Science training" if training_attempted else "Science / Harbor CUDA (no training)"
    fig.suptitle(f"{title}: 4 tasks, 14 GiB each", fontsize=16)
    fig.savefig(root / "comparison.png", dpi=160)
    fig.savefig(root / "comparison.svg")
    plt.close(fig)
    print(
        json.dumps(
            {
                mode: {key: value for key, value in summary[mode].items() if key not in ("episodes", "metrics")}
                for mode in ("off", "on")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
