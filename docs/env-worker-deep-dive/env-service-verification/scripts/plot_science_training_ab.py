"""Plot original science training and measured GPU data without inventing missing metrics."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"off": "#d47832", "on": "#187f9c"}
LABELS = {"off": "Router OFF", "on": "Router ON"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
METRIC = re.compile(r"(?:^| - )([\w/]+):(?:np\.\w+\()?\s*(" + NUMBER + r")", re.I)


def read_arm(root, mode):
    arm = root / mode
    launch = json.loads((arm / "launch.json").read_text()) if (arm / "launch.json").exists() else {}
    steps = {}
    log = arm / "launcher.log"
    if log.exists():
        for raw in log.read_text(errors="replace").splitlines():
            line = ANSI.sub("", raw)
            match = re.search(r"\bstep:(\d+) - ", line)
            if match and "actor/" in line:
                record = {key: float(value) for key, value in METRIC.findall(line[match.start() :])}
                steps[int(match.group(1))] = record
    rows = []
    gpu_errors = 0
    path = arm / "telemetry.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            gpu_errors += int("gpu_error" in sample)
            devices = []
            for gpu in sample.get("gpu", "").splitlines():
                fields = gpu.split(",")
                if len(fields) == 5 and int(fields[0]) in (0, 1, 2):
                    devices.append([int(fields[0]), float(fields[2]), float(fields[3]), float(fields[4])])
            if devices:
                sample["memory_mib"] = sum(x[1] for x in devices)
                sample["util_mean"] = np.mean([x[2] for x in devices]).item()
                sample["power_w"] = sum(x[3] for x in devices)
                rows.append(sample)
    return {"launch": launch, "steps": list(steps.values()), "telemetry": rows, "gpu_errors": gpu_errors}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def metric_panel(ax, arms, key, title, ylabel):
    count = 0
    for mode, data in arms.items():
        rows = [r for r in data["steps"] if key in r]
        if rows:
            ax.plot(
                [r["step"] for r in rows],
                [r[key] for r in rows],
                "o",
                color=COLORS[mode],
                label=LABELS[mode],
                linewidth=2,
                markersize=5,
                alpha=0.8,
                linestyle="--" if mode == "off" else "-",
            )
            count += 1
    style(ax, title, "Optimizer step", ylabel)
    observed = sorted({int(r["step"]) for d in arms.values() for r in d["steps"]})
    ax.set_xticks(observed or [1, 2])
    if count:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No completed training step", transform=ax.transAxes, ha="center")


def summarize(data):
    rows = data["telemetry"]
    steps = data["steps"]
    busy = [r for r in rows if r["memory_mib"] > 1024]
    units = [u for r in rows for u in r.get("units", [])]
    result = {
        "completed_steps": len(steps),
        "exit_code": data["launch"].get("exit_code"),
        "failed_gpu_samples": data["gpu_errors"],
        "timed_out": data["launch"].get("timed_out", False),
        "wall_seconds": data["launch"].get("finished_at", 0) - data["launch"].get("started_at", 0),
        "timed_training_seconds": sum(r.get("timing_s/step", 0) for r in steps),
        "mean_gpu_utilization_full_run": time_weighted_utilization(rows),
        "mean_gpu_utilization_model_loaded_window": time_weighted_utilization(busy),
        "total_processed_tokens": sum(r.get("perf/total_num_tokens", 0) for r in steps) if steps else None,
        "logged_trained_tokens_sum": sum(r.get("termination/trained_tokens", 0) for r in steps) if steps else None,
        "peak_total_gpu_memory_gib": max((r["memory_mib"] / 1024 for r in rows), default=0),
        "unit_snapshot_samples": len(units),
        "peak_leases": max((u["containers_running"] for u in units), default=None),
        "peak_waiting": max((u["sessions_waiting"] for u in units), default=None),
        "budget_violation_samples": (
            sum(
                u["cpu_used"] > u["cpu_quota"]
                or u["memory_used_mb"] > u["memory_mb"]
                or u["containers_running"] > u["max_containers"]
                for u in units
            )
            if units
            else None
        ),
        "steps": steps,
    }
    return result


def time_weighted_utilization(rows):
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]["util_mean"]
    times = np.array([r["time"] for r in rows])
    values = np.array([r["util_mean"] for r in rows])
    integral = np.sum((values[:-1] + values[1:]) * 0.5 * np.diff(times))
    return float(integral / (times[-1] - times[0]))


def plot(root):
    arms = {mode: read_arm(root, mode) for mode in ("off", "on")}
    design = json.loads((root / "design.json").read_text())
    planned_steps = design["steps_per_arm"]
    summary = {mode: summarize(data) for mode, data in arms.items()}
    for item in summary.values():
        item["both_arms_have_completed_steps"] = all(data["steps"] for data in arms.values())
        item["gpu_metrics_scope"] = "Device-wide counters include all processes. Not isolated process utilization."
    (root / "comparison.json").write_text(json.dumps(summary, indent=2))
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "#fafbfc", "axes.facecolor": "white"})
    if not all(data["steps"] for data in arms.values()):
        plot_incomplete(root, arms)
        print(json.dumps({m: {k: v for k, v in s.items() if k != "steps"} for m, s in summary.items()}, indent=2))
        return
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    panels = [
        ("actor/entropy", "Policy entropy", "nats / token"),
        ("actor/pg_loss", "Policy gradient loss", "loss"),
        ("actor/grad_norm", "Gradient norm", "L2 norm"),
        ("critic/score/mean", "Task reward", "mean score"),
        ("termination/masked_sample_fraction", "Masked training samples", "fraction"),
        ("timing_s/step", "Time per training step", "seconds"),
    ]
    for ax, (key, title, ylabel) in zip(axs.flat, panels):
        metric_panel(ax, arms, key, title, ylabel)
    fig.suptitle(
        "Real LAPS repair training | Qwen3.5-4B | identical task / model / concurrency budgets\n"
        f"{planned_steps} requested steps per arm. Short-run loss and entropy do not establish learning superiority.",
        fontsize=14,
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(root / f"training-curves.{extension}", dpi=170)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for mode, data in arms.items():
        records = data["steps"]
        if records:
            elapsed = np.cumsum([r["timing_s/step"] for r in records]) / 60
            axs[0, 0].plot(elapsed, [r["step"] for r in records], "o-", color=COLORS[mode], label=LABELS[mode])
        rows = data["telemetry"]
        if rows:
            t = (np.array([r["time"] for r in rows]) - rows[0]["time"]) / 60
            util = np.array([r["util_mean"] for r in rows])
            # Bin by real elapsed time because snapshot polling can take more than one second.
            bins = (t * 2).astype(int)
            centers = np.unique(bins)
            averaged = [util[bins == x].mean() for x in centers]
            axs[0, 1].plot((centers + 0.5) / 2, averaged, color=COLORS[mode], label=LABELS[mode])
            axs[0, 2].plot(t, [r["memory_mib"] / 1024 for r in rows], color=COLORS[mode], label=LABELS[mode])
        for key, marker, name in [
            ("actor/perf/max_memory_allocated_gb", "o", "allocated"),
            ("actor/perf/max_memory_reserved_gb", "s", "reserved"),
        ]:
            values = [r for r in records if key in r]
            if values:
                axs[1, 0].plot(
                    [r["step"] for r in values],
                    [r[key] for r in values],
                    marker + "-",
                    color=COLORS[mode],
                    label=f"{LABELS[mode]} {name}",
                    alpha=0.8,
                )
    units = [(r["time"], u) for r in arms["on"]["telemetry"] for u in r.get("units", [])]
    if units:
        t0 = arms["on"]["telemetry"][0]["time"]
        t = [(timestamp - t0) / 60 for timestamp, _ in units]
        axs[1, 1].step(
            t, [u["containers_running"] for _, u in units], where="post", label="Active leases", color=COLORS["on"]
        )
        axs[1, 1].step(
            t, [u["sessions_waiting"] for _, u in units], where="post", label="Waiting sessions", color="#855fa8"
        )
        axs[1, 1].axhline(4, color="#888888", ls=":", label="Configured limit")
    else:
        axs[1, 1].text(0.5, 0.5, "No Router snapshot", transform=axs[1, 1].transAxes, ha="center")
    for i, mode in enumerate(("off", "on")):
        value = summary[mode]["mean_gpu_utilization_model_loaded_window"]
        if value is not None:
            axs[1, 2].bar(i, value, color=COLORS[mode], width=0.55)
            axs[1, 2].text(i, value + 0.5, f"{value:.1f}%", ha="center")
    axs[1, 2].set_xticks([0, 1], [LABELS["off"], LABELS["on"]])
    titles = [
        ("Completed updates vs measured step time", "Cumulative step time (min)", "Optimizer steps"),
        ("GPU utilization: mean of 3 GPUs", "Elapsed from launcher (min)", "30-second mean (%)"),
        ("Device memory used: sum of 3 GPUs", "Elapsed from launcher (min)", "GiB"),
        ("Training allocator peaks (not fragmentation)", "Optimizer step", "GiB, logged worker aggregate"),
        ("Router ON admission accounting", "Elapsed from launcher (min)", "Sessions"),
        ("Mean utilization while GPU memory > 1 GiB", "", "Mean of 3 GPUs (%)"),
    ]
    for ax, (title, xlabel, ylabel) in zip(axs.flat, titles):
        style(ax, title, xlabel, ylabel)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    axs[0, 1].set_ylim(0, 100)
    axs[1, 2].set_ylim(0, 100)
    axs[1, 0].set_xticks(sorted({int(r["step"]) for d in arms.values() for r in d["steps"]}))
    fig.suptitle(
        "Real science training resource comparison\nCPU task containers: this run does not measure GPU-container packing or allocator fragmentation.",
        fontsize=14,
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(root / f"resource-comparison.{extension}", dpi=170)
    plt.close(fig)
    print(json.dumps({m: {k: v for k, v in s.items() if k != "steps"} for m, s in summary.items()}, indent=2))


def plot_incomplete(root, arms):
    """Plot actual progress without assigning a historical failure to a new run."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for mode, data in arms.items():
        rows = data["telemetry"]
        if rows:
            start = rows[0]["time"]
            axs[0].plot(
                [(r["time"] - start) / 60 for r in rows],
                [r["memory_mib"] / 1024 for r in rows],
                color=COLORS[mode],
                label=LABELS[mode],
                linewidth=2,
            )
    style(
        axs[0], "Measured memory: sum of 3 experiment GPUs", "Elapsed from launcher (min)", "Device memory used (GiB)"
    )
    if axs[0].get_legend_handles_labels()[0]:
        axs[0].legend(frameon=False)
    details = []
    for mode, data in arms.items():
        launch = data["launch"]
        log_path = root / mode / "launcher.log"
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        if launch.get("timed_out"):
            status = "Time budget reached"
        elif launch.get("harness_error"):
            status = "Harness / preflight failure; see launch.json"
        elif "CUDA out of memory" in log:
            status = "CUDA out of memory"
        elif launch.get("exit_code") == 0:
            status = "Exited successfully"
        elif launch.get("exit_code") is not None:
            status = "Exited with code " + str(launch["exit_code"])
        elif launch:
            status = "No recorded final exit status"
        else:
            status = "Not started"
        details.append(f"{LABELS[mode]}: {len(data['steps'])} completed step(s)\n{status}")
    axs[1].axis("off")
    axs[1].text(
        0.03,
        0.92,
        "\n\n".join(details) + "\n\nPaired training comparison is incomplete.\nMissing metrics are not zero.",
        transform=axs[1].transAxes,
        va="top",
        fontsize=12,
    )
    fig.suptitle(
        "Real science training: incomplete paired result\n"
        "Observed progress does not establish Router performance superiority.",
        fontsize=14,
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(root / f"failure-evidence.{extension}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    plot(parser.parse_args().results)
