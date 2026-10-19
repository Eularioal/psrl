"""Audit equal-completion science runs; never count OOM attempts as rollouts."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from plot_science_training_ab import read_arm, summarize

LABELS = {"off": "0eeead8 baseline + common fixes", "on": "Current Router ON"}
COLORS = {"off": "#cf7735", "on": "#187f9c"}


def audit(root, mode):
    data = read_arm(root, mode)
    records = []
    for path in (root / mode).glob("pool-*/attempts/*.json"):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        record["record_path"] = str(path.relative_to(root))
        records.append(record)
    records.sort(key=lambda r: r["finished_at"])
    accepted = [r for r in records if r["accepted"]]
    groups = Counter(r["parent_id"] for r in accepted)
    unique_slots = {r["uid"] for r in accepted}
    equal_target = (
        len(accepted) == len(unique_slots) == 16
        and set(groups) == {"0", "1", "2", "3"}
        and set(groups.values()) == {4}
    )
    step = data["steps"][0] if len(data["steps"]) == 1 else {}
    trained_target = (
        step.get("group/count") == 4
        and step.get("termination/masked_sample_fraction") == 0
        and step.get("termination/trained_tokens", 0) > 0
    )
    start = min((r["slot_started_at"] for r in records), default=None)
    target_time = max((r["finished_at"] for r in accepted), default=None) if equal_target else None
    result = summarize(data)
    result.update(
        {
            "label": LABELS[mode],
            "attempts": len(records),
            "accepted_rollouts": len(accepted),
            "accepted_per_group": dict(groups),
            "unique_accepted_slots": len(unique_slots),
            "cuda_oom_attempts": sum(r["pressure"].get("status") == "cuda_oom" for r in records),
            "fixed_completion_target_met": equal_target,
            "first_slot_started_at": start,
            "target_reached_at": target_time,
            "sampling_seconds_including_retries": target_time - start if target_time else None,
            "four_training_groups_without_masked_samples": trained_target,
            "full_acceptance": equal_target and trained_target and data["launch"].get("exit_code") == 0,
            "accepted_mean_reward_repair": sum(r["rewards"].get("reward_repair", 0) for r in accepted) / len(accepted)
            if accepted
            else None,
            "accepted_mean_shaped_reward": sum(r["rewards"].get("reward", 0) for r in accepted) / len(accepted)
            if accepted
            else None,
            "accepted_shaped_rewards_by_group": {
                group: [r["rewards"].get("reward", 0) for r in accepted if r["parent_id"] == group]
                for group in sorted(groups)
            },
            "accepted_cuda_gpu_uuids": dict(Counter(r["pressure"].get("gpu_uuid_hex") for r in accepted)),
            "accepted_full_repairs": sum(r["rewards"].get("reward_repair", 0) == 1 for r in accepted),
            "accepted_exception_types": dict(Counter(str(r["exception_type"]) for r in accepted)),
            "records": records,
        }
    )
    if accepted:
        result["mean_seconds_after_cuda_exit_until_attempt_end"] = sum(
            r["finished_at"] - r["pressure"]["finished_at"] for r in accepted
        ) / len(accepted)
        result["mean_seconds_in_cuda_allocation_phase"] = sum(
            r["pressure"]["finished_at"] - r["pressure"]["allocated_at"] for r in accepted
        ) / len(accepted)
        result["mean_seconds_active_attempt"] = sum(r["finished_at"] - r["admitted_at"] for r in accepted) / len(
            accepted
        )
    # Existing summarize() checks CPU/host memory/container budgets; include GPU memory.
    units = [u for sample in data["telemetry"] for u in sample.get("units", [])]
    result["gpu_budget_violations"] = (
        sum(u["declared_gpu_memory_mb"] > u["gpu_memory_mb"] for u in units) if units else None
    )
    snapshots = [sample["units"] for sample in data["telemetry"] if "units" in sample]
    result["peak_total_leases"] = max((sum(u["containers_running"] for u in row) for row in snapshots), default=None)
    # The same node waiter count is repeated in each unit, so do not add it.
    result["peak_node_waiters"] = max((u["sessions_waiting"] for u in units), default=None)
    result["accounting_mismatches"] = (
        sum(
            u["declared_gpu_memory_mb"] != len(u["sessions"]) * 52224 or u["containers_running"] != len(u["sessions"])
            for u in units
        )
        if units
        else None
    )
    result["last_observed_declared_gpu_memory_mb"] = (
        sum(u["declared_gpu_memory_mb"] for u in snapshots[-1]) if snapshots else None
    )
    # Sampled idle reservation is not allocator fragmentation. Exclude long gaps
    # rather than claiming the last observation remained true during missing data.
    observed, reserved, idle_wait = 0.0, 0.0, 0.0
    rows = data["telemetry"]
    for previous, following in zip(rows, rows[1:]):
        dt = following["time"] - previous["time"]
        if not 0 < dt <= 10 or not previous.get("units"):
            continue
        devices = {
            int(parts[0]): float(parts[2])
            for line in previous.get("gpu", "").splitlines()
            if len(parts := line.split(",")) == 5
        }
        if not {6, 7}.issubset(devices):
            continue
        observed += dt
        has_reservations = any(u["declared_gpu_memory_mb"] > 0 for u in previous["units"])
        has_waiters = any(u["sessions_waiting"] > 0 for u in previous["units"])
        if has_reservations:
            reserved += dt
            if has_waiters and devices[6] <= 16 and devices[7] <= 16:
                idle_wait += dt
    result["reservation_observation_seconds_max_gap_10s"] = observed if units else None
    result["reservation_active_seconds_observed"] = reserved if units else None
    result["both_environment_gpus_idle_while_reserved_and_waiting_seconds"] = idle_wait if units else None
    result["idle_with_waiters_fraction_of_observed_reservation_time"] = idle_wait / reserved if reserved else None
    return data, result


def plot(root):
    data, results = {}, {}
    for mode in ("off", "on"):
        data[mode], results[mode] = audit(root, mode)
    (root / "fixed-work-comparison.json").write_text(json.dumps(results, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(
        "Fixed target: 4 science tasks x 4 graded non-OOM rollouts + one GRPO update\n"
        "Historical baseline vs current Router; one run per arm, not convergence evidence"
    )
    ax = axes[0, 0]
    for mode, result in results.items():
        start = result["first_slot_started_at"]
        done = [r for r in result["records"] if r["accepted"]]
        if start is not None:
            ax.step(
                [0] + [(r["finished_at"] - start) / 60 for r in done],
                [0] + list(range(1, len(done) + 1)),
                where="post",
                color=COLORS[mode],
                label=LABELS[mode],
            )
    ax.axhline(16, color="gray", linestyle="--", linewidth=1)
    ax.set(
        title="Accepted rollout progress (OOM excluded)",
        xlabel="Minutes since first slot request",
        ylabel="Accepted rollouts",
        ylim=(0, 17),
    )
    ax.legend(fontsize=7)
    specs = [
        (axes[0, 1], "cuda_oom_attempts", "OOM attempts, all counted", "Attempts"),
        (
            axes[0, 2],
            "sampling_seconds_including_retries",
            "Time to all 16 accepted rollouts",
            "Seconds incl. OOM retries",
        ),
        (axes[1, 0], "logged_trained_tokens_sum", "Mask-valid training tokens", "Tokens"),
        (axes[1, 1], "accepted_mean_reward_repair", "Original repair reward", "Mean reward"),
        (axes[1, 2], "wall_seconds", "Full arm wall time", "Seconds incl. init and cleanup"),
    ]
    for ax, key, title, ylabel in specs:
        for i, mode in enumerate(("off", "on")):
            value = results[mode].get(key)
            if value is not None and value >= 0:
                ax.bar(i, value, color=COLORS[mode])
                ax.annotate(
                    f"{value:.2f}" if isinstance(value, float) else str(value),
                    (i, value),
                    ha="center",
                    xytext=(0, 3),
                    textcoords="offset points",
                )
            else:
                ax.text(i, 0, "Not completed", ha="center", rotation=25)
        ax.set(title=title, ylabel=ylabel, xticks=[0, 1], xticklabels=["0eeead8", "Router ON"])
        ax.margins(y=0.2)
    for ax in axes.flat:
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(root / "fixed-work-outcomes.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig.suptitle("One actual GRPO update: different samples; no convergence claim")
    for ax, key, title in zip(
        axes, ("actor/entropy", "actor/loss", "actor/grad_norm"), ("Entropy", "Policy loss", "Gradient norm")
    ):
        for i, mode in enumerate(("off", "on")):
            steps = data[mode]["steps"]
            if steps and key in steps[-1]:
                value = steps[-1][key]
                ax.bar(i, value, color=COLORS[mode])
                ax.annotate(f"{value:.5g}", (i, value), ha="center", xytext=(0, 5), textcoords="offset points")
            else:
                ax.text(i, 0, "No completed step", ha="center", rotation=25)
        ax.set(title=title, xticks=[0, 1], xticklabels=["0eeead8", "Router ON"])
        ax.margins(y=0.25)
    fig.savefig(root / "fixed-work-training.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    for ax, mode in zip(axes, ("off", "on")):
        rows = data[mode]["telemetry"]
        if not rows:
            continue
        start = rows[0]["time"]
        xs, gpu6, gpu7, reserved = [], [], [], []
        for row in rows:
            devices = {
                int(parts[0]): float(parts[2]) / 1024
                for line in row.get("gpu", "").splitlines()
                if len(parts := line.split(",")) == 5
            }
            if 6 not in devices or 7 not in devices:
                continue
            xs.append((row["time"] - start) / 60)
            gpu6.append(devices[6])
            gpu7.append(devices[7])
            reserved.append(
                sum(u["declared_gpu_memory_mb"] for u in row.get("units", [])) / 1024
                if "units" in row
                else float("nan")
            )
        ax.plot(xs, gpu6, label="GPU 6 physical memory", linewidth=1)
        ax.plot(xs, gpu7, label="GPU 7 physical memory", linewidth=1)
        if mode == "on":
            ax.plot(xs, [a + b for a, b in zip(gpu6, gpu7)], label="Total physical memory (both GPUs)", linewidth=1)
            ax.plot(xs, reserved, label="Total declared memory (both units)", linestyle="--", linewidth=1)
        ax.set(title=LABELS[mode], xlabel="Minutes since arm telemetry started", ylabel="GiB")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Environment GPU placement and reservation duration\nDeclared memory is admission accounting, not physical allocation or fragmentation"
    )
    fig.savefig(root / "fixed-work-environment-memory.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {mode: {k: v for k, v in r.items() if k not in ("records", "steps")} for mode, r in results.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
