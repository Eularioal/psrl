"""Plot recorded CUDA pressure failures and actual training metrics, including missing steps."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_science_training_ab import COLORS, LABELS, metric_panel, read_arm, style, summarize


def inspect(root, mode):
    arm = root / mode
    data = read_arm(root, mode)
    summary = summarize(data)
    log = (arm / "launcher.log").read_text(errors="replace") if (arm / "launcher.log").exists() else ""
    sessions = {
        session: int(uid)
        for uid, session in re.findall(r"\[uid=(\d+)\] Starting Harbor episode: session=([^,]+)", log)
    }

    def request_uid(file):
        return next((sessions[part] for part in file.parts if part in sessions), None)

    probes = []
    trials = []
    for file in arm.rglob("gpu-pressure.json"):
        record = json.loads(file.read_text())
        record["source"] = str(file.relative_to(root))
        record["request_uid"] = request_uid(file)
        probes.append(record)
    for file in arm.rglob("result.json"):
        record = json.loads(file.read_text())
        if "trial_name" in record:
            trials.append(
                {
                    "source": str(file.relative_to(root)),
                    "request_uid": request_uid(file),
                    "task_name": record["task_name"],
                    "exception": record.get("exception_info"),
                    "verifier": record.get("verifier_result"),
                }
            )
    summary["trainable_tokens_per_training_second"] = (
        summary["logged_trained_tokens_sum"] / summary["timed_training_seconds"]
        if summary["timed_training_seconds"]
        else None
    )
    summary.update(
        pressure_phases=len(probes),
        cuda_ok=sum(p["status"] == "ok" for p in probes),
        cuda_oom=sum(p["status"] == "cuda_oom" for p in probes),
        probes=probes,
        trials=trials,
    )
    units = [u for r in data["telemetry"] for u in r.get("units", [])]
    summary["gpu_budget_violation_samples"] = (
        sum(u["declared_gpu_memory_mb"] > u["gpu_memory_mb"] for u in units) if units else None
    )
    summary["ledger_accounting_mismatch_samples"] = (
        sum(
            u["declared_gpu_memory_mb"] != u["containers_running"] * 52224
            or u["cpu_used"] != u["containers_running"] * 4
            or u["memory_used_mb"] != u["containers_running"] * 8192
            or len(u["sessions"]) != u["containers_running"]
            for u in units
        )
        if units
        else None
    )
    summary["peak_aggregate_leases"] = (
        max((sum(u["containers_running"] for u in r.get("units", [])) for r in data["telemetry"]), default=0)
        if units
        else None
    )
    summary["peak_declared_gpu_memory_mib_per_unit"] = max((u["declared_gpu_memory_mb"] for u in units), default=None)
    summary["peak_waiting"] = max((u["sessions_waiting"] for u in units), default=None)
    summary["ledger_unique_sessions"] = len({session for u in units for session in u["sessions"]}) if units else None
    summary["full_repair_successes"] = sum(
        t["verifier"] is not None and t["verifier"].get("rewards", {}).get("reward_repair") == 1 for t in trials
    )
    summary["nonzero_policy_gradient_steps"] = sum(r.get("actor/grad_norm", 0) > 0 for r in data["steps"])
    summary["gpu_metrics_scope"] = (
        "Device counters include all processes; pressure phase holds memory, not sustained compute."
    )
    summary["mean_reward_repair"] = (
        float(
            np.mean(
                [
                    t["verifier"]["rewards"]["reward_repair"]
                    for t in trials
                    if t["verifier"] and "reward_repair" in t["verifier"].get("rewards", {})
                ]
            )
        )
        if any(t["verifier"] and "reward_repair" in t["verifier"].get("rewards", {}) for t in trials)
        else None
    )
    return data, summary


def compare_matched(root, arms, summaries):
    horizon = min(len(d["steps"]) for d in arms.values())
    matched = {"steps_per_arm": horizon, "episodes_per_arm": horizon * 4, "arms": {}}
    for mode, data in arms.items():
        records = data["steps"][:horizon]
        probes = [
            p for p in summaries[mode]["probes"] if p["request_uid"] is not None and p["request_uid"] < horizon * 4
        ]
        trials = [
            t for t in summaries[mode]["trials"] if t["request_uid"] is not None and t["request_uid"] < horizon * 4
        ]
        seconds = sum(r.get("timing_s/step", 0) for r in records)
        tokens = sum(r.get("termination/trained_tokens", 0) for r in records)
        rewards = [(t["verifier"] or {}).get("rewards", {}).get("reward_repair") for t in trials]
        matched["arms"][mode] = {
            "steps": records,
            "cuda_phases": len(probes),
            "cuda_oom": sum(p["status"] == "cuda_oom" for p in probes),
            "cuda_ok": sum(p["status"] == "ok" for p in probes),
            "trained_tokens": tokens,
            "step_seconds": seconds,
            "trainable_tokens_per_second": tokens / seconds if seconds else None,
            "mean_reward_repair": float(np.mean([r for r in rewards if r is not None])) if rewards else None,
            "full_repair_successes": sum(r == 1 for r in rewards),
            "nonzero_policy_gradient_steps": sum(r.get("actor/grad_norm", 0) > 0 for r in records),
        }
    (root / "matched-comparison.json").write_text(json.dumps(matched, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        ("cuda_oom", "Actual CUDA OOM", "episodes"),
        ("trained_tokens", "Trainable response tokens", "tokens"),
        ("step_seconds", "Total step wall time (lower is faster)", "seconds"),
        ("full_repair_successes", "Original science repairs scoring 1", "episodes"),
        ("trainable_tokens_per_second", "Trainable tokens / step wall time", "tokens / second"),
        ("nonzero_policy_gradient_steps", "Steps with nonzero policy gradient", "steps"),
    ]
    for ax, (key, title, unit) in zip(axes.flat, panels):
        values = [matched["arms"][mode][key] for mode in ("off", "on")]
        ax.bar(["OFF", "ON"], values, color=[COLORS["off"], COLORS["on"]])
        for i, value in enumerate(values):
            ax.text(
                i,
                value,
                f"{value:,.1f}" if key in ("step_seconds", "trainable_tokens_per_second") else f"{value:,.0f}",
                ha="center",
                va="bottom",
            )
        style(ax, title, "Router setting", unit)
        ax.set_ylim(0, max(max(values) * 1.25, 1.25))
    fig.suptitle(
        f"Matched first {horizon} optimizer steps | {horizon * 4} episodes per arm\nON crashed in its next batch. This comparison does not establish end-to-end stability.",
        fontsize=13,
    )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(root / f"pressure-matched.{ext}", dpi=170)
    plt.close(fig)
    return matched


def plot_preflight(root):
    """Separate GPU-only proof from the model-training figures."""
    root = root / "preflight"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, mode in enumerate(("off", "on")):
        records = [json.loads(f.read_text()) for f in (root / mode).rglob("gpu-pressure.json")]
        axes[0].bar(
            i - 0.18,
            sum(r["status"] == "ok" for r in records),
            0.34,
            color="#187f9c",
            label="CUDA allocation OK" if i == 0 else None,
        )
        axes[0].bar(
            i + 0.18,
            sum(r["status"] == "cuda_oom" for r in records),
            0.34,
            color="#d47832",
            label="Actual CUDA OOM" if i == 0 else None,
        )
        if not records:
            continue
        start = min(r["started_at"] for r in records)
        for j, r in enumerate(sorted(records, key=lambda r: r["started_at"])):
            y = i * 5 + j
            color = "#187f9c" if r["status"] == "ok" else "#d47832"
            axes[1].barh(y, max(0.4, r["finished_at"] - r["started_at"]), left=r["started_at"] - start, color=color)
            axes[1].text(r["finished_at"] - start + 1, y, r["status"], va="center", fontsize=8)
    axes[0].set_xticks([0, 1], ["Router OFF", "Router ON"])
    axes[0].set_ylabel("CUDA phases")
    axes[0].set_ylim(0, 5)
    axes[0].legend()
    axes[0].set_title("Real Harbor CUDA preflight: 4 requests / arm")
    axes[1].set_yticks(range(9), ["OFF 1", "OFF 2", "OFF 3", "OFF 4", "", "ON 1", "ON 2", "ON 3", "ON 4"])
    axes[1].set_xlabel("Seconds from first pressure command in each arm")
    axes[1].set_title("Recorded pressure phases: queued ON succeeds")
    fig.suptitle(
        "Two task pools, same two GPUs, 50 GiB requested per container\nCUDA preflight only; no model inference or optimizer steps",
        fontsize=12,
    )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(root / f"cuda-preflight.{ext}", dpi=160)
    plt.close(fig)


def plot(root):
    plot_preflight(root)
    arms, summary = {}, {}
    for mode in ("off", "on"):
        arms[mode], summary[mode] = inspect(root, mode)
    (root / "pressure-comparison.json").write_text(json.dumps(summary, indent=2))
    compare_matched(root, arms, summary)
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "#fafbfc"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, (key, title, label) in zip(
        axes.flat,
        [
            ("actor/entropy", "Policy entropy (not a quality score)", "nats / token"),
            ("actor/pg_loss", "Policy gradient loss (not directly quality)", "loss"),
            ("actor/grad_norm", "Gradient norm", "L2 norm"),
            ("critic/score/mean", "Training reward", "score"),
            ("termination/trained_tokens", "Trainable response tokens", "tokens"),
            ("timing_s/step", "Elapsed time per optimizer step", "seconds"),
        ],
    ):
        metric_panel(ax, arms, key, title, label)
    fig.suptitle(
        "LAPS repair + identical injected 50 GiB CUDA phase | two independent task pools\nOFF completed 4 steps. ON stopped after step 2 due to worker segfault. Missing steps are not filled.",
        fontsize=13,
    )
    for ax in axes.flat:
        ax.axvline(2.5, color="#666666", linestyle=":", alpha=0.5)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(root / f"pressure-training-curves.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = ["OFF", "ON"]
    for i, mode in enumerate(("off", "on")):
        s = summary[mode]
        color = COLORS[mode]
        axes[0, 0].bar(i - 0.16, s["cuda_ok"], width=0.3, color=color, label=LABELS[mode] + " OK")
        axes[0, 0].bar(
            i + 0.16, s["cuda_oom"], width=0.3, color=color, hatch="///", alpha=0.45, label=LABELS[mode] + " CUDA OOM"
        )
        rate = s["trainable_tokens_per_training_second"]
        if rate is not None:
            axes[0, 1].bar(i, rate, color=color)
            axes[0, 1].text(i, rate, f"{rate:.1f}\n{s['completed_steps']} steps", ha="center", va="bottom", fontsize=9)
        rows = arms[mode]["telemetry"]
        if rows:
            t0 = rows[0]["time"]
            for index, ls in ((6, "-"), (7, "--")):
                points = []
                for row in rows:
                    for line in row.get("gpu", "").splitlines():
                        fields = line.split(",")
                        if len(fields) == 5 and int(fields[0]) == index:
                            points.append(((row["time"] - t0) / 60, float(fields[2]) / 1024))
                if points:
                    x, y = zip(*points)
                    axes[1, 0].plot(x, y, ls, color=color, alpha=0.8, label=f"{mode.upper()} GPU {index}")
            if mode == "on":
                x = [(r["time"] - t0) / 60 for r in rows if "units" in r]
                y = [sum(u["containers_running"] for u in r["units"]) for r in rows if "units" in r]
                q = [max((u["sessions_waiting"] for u in r["units"]), default=0) for r in rows if "units" in r]
                axes[1, 1].plot(x, y, label="ON active leases", color=color)
                axes[1, 1].plot(x, q, label="ON waiting sessions (node queue)", color="#7b5ca7")
    style(axes[0, 0], "CUDA phases: actual results", "Router setting", "count")
    axes[0, 0].set_xticks([0, 1], labels)
    axes[0, 0].legend(fontsize=8)
    style(axes[0, 1], "Trainable tokens / total step wall time", "Router setting", "tokens / second")
    axes[0, 1].set_xticks([0, 1], labels)
    style(axes[1, 0], "Environment GPU residency (not compute utilization)", "minutes from arm start", "GiB")
    axes[1, 0].legend(fontsize=8)
    style(axes[1, 1], "Unified ledger: bounded admission and waiting", "minutes from ON start", "sessions")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("All recorded data | OFF: 4 steps / 16 phases; ON: 2 steps / 8 phases, then segfault", fontsize=13)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(root / f"pressure-resources.{ext}", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {m: {k: v for k, v in s.items() if k not in ("probes", "trials", "steps")} for m, s in summary.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    plot(p.parse_args().root)
