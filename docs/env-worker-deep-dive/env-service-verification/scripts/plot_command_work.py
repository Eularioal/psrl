"""Plot command reservations and equal completion without equating memory with compute."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_science_fixed_work as fixed

LABELS = {"off": "0eeead8 + runtime fixes", "command": "Router command reservations"}
COLORS = {"off": "#ce7136", "command": "#168775"}


def audit(root, mode):
    fixed.LABELS.update(LABELS)
    data, result = fixed.audit(root, mode)
    units = [u for row in data["telemetry"] for u in row.get("units", [])]
    turns = []
    completion_tokens = []
    for record in result["records"]:
        if not record["accepted"]:
            continue
        attempt = root / record["record_path"]
        folder = attempt.parent.parent / "harbor-jobs" / record["session_id"]
        paths = list(folder.glob("*/*/agent/trajectory.json"))
        if len(paths) == 1:
            trajectory = json.loads(paths[0].read_text())
            turns.append(sum(step["source"] == "agent" for step in trajectory["steps"]))
            completion_tokens.append(trajectory["final_metrics"]["total_completion_tokens"])
    result["accepted_agent_turns_histogram"] = dict(Counter(turns))
    result["accepted_agent_turns_total"] = sum(turns)
    result["harbor_reported_completion_tokens"] = sum(completion_tokens)
    accepted = sorted((r for r in result["records"] if r["accepted"]), key=lambda r: r["pressure"]["allocated_at"])
    pairs = []
    neighbors = []
    for gpu in {r["pressure"]["gpu_uuid_hex"] for r in accepted}:
        on_gpu = [r for r in accepted if r["pressure"]["gpu_uuid_hex"] == gpu]
        neighbors.extend(zip(on_gpu, on_gpu[1:]))
    for a, b in neighbors:
        if (
            a["pressure"]["gpu_uuid_hex"] == b["pressure"]["gpu_uuid_hex"]
            and a["pressure"]["finished_at"] <= b["pressure"]["allocated_at"] < a["finished_at"]
        ):
            pairs.append(
                {
                    "a_uid": a["uid"],
                    "b_uid": b["uid"],
                    "a_cuda_end": a["pressure"]["finished_at"],
                    "b_cuda_start": b["pressure"]["allocated_at"],
                    "a_task_end": a["finished_at"],
                }
            )
    result["same_gpu_reuse_before_previous_task_end"] = pairs
    if mode == "command":
        result["accounting_mismatches"] = sum(
            u["declared_gpu_memory_mb"] != u["gpu_commands_active"] * 52224
            or u["containers_running"] != len(u["sessions"])
            for u in units
        )
        result["peak_gpu_commands_active"] = max((u["gpu_commands_active"] for u in units), default=0)
        result["peak_gpu_commands_waiting"] = max((u["gpu_commands_waiting"] for u in units), default=0)
        result["host_leases_with_zero_gpu_charge_samples"] = sum(
            u["containers_running"] > 0 and u["declared_gpu_memory_mb"] == 0 for u in units
        )
        phases = [
            json.loads(p.read_text()) for p in (root / mode).glob("pool-*/harbor-jobs/**/gpu-command-phase.json")
        ]
        phases = [p for p in phases if p and "released_at" in p]
        result["completed_gpu_command_records"] = len(phases)
        result["gpu_queue_seconds"] = [p["admitted_at"] - p["requested_at"] for p in phases]
        result["gpu_return_seconds"] = [p["released_at"] - p["command_finished_at"] for p in phases]
    return result, data


def save(fig, root, name):
    for extension in ("png", "svg"):
        fig.savefig(root / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot(root, micro):
    results, data = {}, {}
    for mode in LABELS:
        results[mode], data[mode] = audit(root, mode)
    (root / "comparison.json").write_text(json.dumps(results, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for mode, r in results.items():
        start = r["first_slot_started_at"]
        accepted = [x for x in r["records"] if x["accepted"]]
        if start is not None:
            axs[0, 0].step(
                [0] + [(x["finished_at"] - start) / 60 for x in accepted],
                list(range(len(accepted) + 1)),
                where="post",
                label=LABELS[mode],
                color=COLORS[mode],
            )
    axs[0, 0].set(title="Completed rollouts, OOM excluded", xlabel="Minutes since first request", ylabel="Rollouts")
    axs[0, 0].axhline(16, color="gray", ls=":")
    axs[0, 0].legend(fontsize=8)
    specs = [
        ("sampling_seconds_including_retries", "Time to all 16 accepted rollouts", "Seconds incl. retries"),
        ("cuda_oom_attempts", "Failed CUDA allocations", "Attempts"),
        ("wall_seconds", "Observed wall time, OFF stopped at budget", "Seconds"),
        ("accepted_agent_turns_total", "Completed model turns in accepted rollouts", "Turns"),
        ("accepted_mean_reward_repair", "Science repair score", "Mean score"),
    ]
    for ax, (key, title, ylabel) in zip(list(axs.flat)[1:], specs):
        for i, mode in enumerate(LABELS):
            value = results[mode].get(key)
            if value is not None:
                ax.bar(i, value, color=COLORS[mode])
                label = f"{value:.2f}"
                if key == "wall_seconds" and results[mode]["timed_out"]:
                    label += "\nBudget cutoff"
                ax.text(i, value, label, ha="center", va="bottom")
        ax.set(title=title, ylabel=ylabel, xticks=[0, 1], xticklabels=["Old OFF", "Command ON"])
        ax.margins(y=0.22)
    for ax in axs.flat:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Qwen3.5-4B, 4 science tasks x 4 rollouts, 4 turns, target: 1 GRPO step\n"
        "Same 50 GiB / 45 s injected CUDA phase. Short-run metrics do not prove learning gains."
    )
    save(fig, root, "completion-comparison")

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, key, title in zip(
        axs,
        ["actor/entropy", "actor/pg_loss", "actor/grad_norm"],
        ["Policy entropy", "Policy gradient loss", "Gradient norm"],
    ):
        for i, mode in enumerate(LABELS):
            if data[mode]["steps"] and key in data[mode]["steps"][0]:
                value = data[mode]["steps"][0][key]
                ax.scatter(i, value, color=COLORS[mode], s=75)
                ax.annotate(f"{value:.5g}", (i, value), xytext=(0, 10), textcoords="offset points", ha="center")
            else:
                ax.text(i, 0.5, "Not completed", transform=ax.get_xaxis_transform(), ha="center", color="gray")
        ax.set(title=title, xticks=[0, 1], xticklabels=["Old OFF", "Command ON"], xlim=(-0.5, 1.5))
        ax.grid(alpha=0.2)
    fig.suptitle("Available optimizer metrics only. Missing points are not zero. No convergence claim.")
    save(fig, root, "training-metrics")

    fig, axs = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for mode in LABELS:
        start = results[mode]["first_slot_started_at"]
        if start is None:
            continue
        rows = [r for r in data[mode]["telemetry"] if r["time"] >= start]
        times = [(r["time"] - start) / 60 for r in rows]
        mem = [
            sum(float(g.split(",")[2]) / 1024 for g in r["gpu"].splitlines() if int(g.split(",")[0]) in (6, 7))
            for r in rows
        ]
        axs[0].plot(times, mem, color=COLORS[mode], label=LABELS[mode])
        if mode == "command":
            rows = [r for r in rows if r.get("units")]
            times = [(r["time"] - start) / 60 for r in rows]
            axs[1].step(
                times,
                [sum(u["containers_running"] for u in r["units"]) for r in rows],
                label="Live host leases",
                color="#425a99",
                where="post",
            )
            axs[1].step(
                times,
                [sum(u["gpu_commands_active"] for u in r["units"]) for r in rows],
                label="Active GPU phases",
                color=COLORS[mode],
                where="post",
            )
            axs[1].step(
                times,
                [sum(u["gpu_commands_waiting"] for u in r["units"]) for r in rows],
                label="Queued GPU commands",
                color="#d4a130",
                where="post",
            )
            axs[2].step(
                times,
                [sum(u["declared_gpu_memory_mb"] for u in r["units"]) / 1024 for r in rows],
                label="Charged GPU budget",
                color=COLORS[mode],
                where="post",
            )
            axs[2].step(
                times,
                [sum(u["containers_running"] for u in r["units"]) * 51 for r in rows],
                label="Counterfactual charge if each live lease retained 51 GiB",
                color="#888",
                ls=":",
                where="post",
            )
    for ax, title, ylabel in zip(
        axs,
        [
            "Environment GPU memory",
            "Command ON: state and GPU ownership",
            "Command ON: physical permission return releases quota",
        ],
        ["GiB", "Count", "GiB"],
    ):
        ax.set(title=title, xlabel="Minutes since first rollout request", ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Memory occupancy and admission budget, not compute utilization or allocator fragmentation")
    save(fig, root, "resource-timeline")

    if micro is not None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 5), constrained_layout=True)
        for ax, mode in zip(axs, ["episode", "command"]):
            d = json.loads((micro / f"{mode}.json").read_text())
            events = d["events"]
            for task, y in [("A", 1), ("B", 0)]:
                task_events = [e for e in events if e["task"] == task]
                for kind, end, color, label in [
                    ("gpu_granted", "cuda_finished", "#168775", "GPU command"),
                    ("cpu_gap_start", "cpu_gap_end", "#adb9ce", "CPU gap"),
                ]:
                    starts = [e["t"] for e in task_events if e["event"] == kind]
                    ends = [e["t"] for e in task_events if e["event"] == end]
                    ax.broken_barh([(a, b - a) for a, b in zip(starts, ends)], (y - 0.3, 0.6), facecolors=color)
            ax.set(
                title=f"{mode}: {d['duration_s']:.2f} s including permission checks and cleanup",
                yticks=[0, 1],
                yticklabels=["B", "A"],
                xlabel="Seconds",
            )
            ax.grid(alpha=0.2)
        fig.suptitle("Real GPU protocol probe: green = command, gray = CPU gap; two CUDA phases per task")
        save(fig, micro, "handoff-timeline")
    print(
        json.dumps(
            {mode: {k: v for k, v in r.items() if k not in ("records", "steps")} for mode, r in results.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--micro", type=Path)
    args = parser.parse_args()
    plot(args.root, args.micro)
