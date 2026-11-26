"""Audit a completed capacity comparison and draw observed command timelines."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def summarize(values):
    return {"count": len(values), "mean": statistics.mean(values), "max": max(values)} if values else None


def partition_sensitivity(root):
    """Vary assumed inference latency without asserting an unmeasured GPU configuration."""
    from analyze_environment_capacity import simulate_batch

    selected = json.loads((root / "capacity-selection.json").read_text())
    profiles = {
        key: {int(cap): value for cap, value in points.items()} for key, points in selected["profiles"].items()
    }
    return {
        "scope": "Hypothetical fixed inference latency. Not an actual alternative-partition prediction or GPU run.",
        "rows": [
            {
                "environment_gpus": devices,
                "inference_gpus": 4 - devices,
                "assumed_request_s": latency,
                "batch_estimates_s": {
                    str(cap): simulate_batch(profiles, cap, groups=6, model_request_s=latency, devices=devices)
                    for cap in (1, 2, 4, 5)
                },
            }
            for devices in (1, 2, 3)
            for latency in (6, 12, 30, 60)
        ],
        "limitations": [
            "Isolated per-GPU profiles omit host-wide contention.",
            "Inference latency is a sensitivity input, not inferred from the engine count.",
            "No cross-batch prefetch, weight synchronization or active sharing is modeled.",
        ],
    }


def read_arm(root, name):
    episodes = [json.loads(path.read_text()) for path in sorted((root / name).glob("episodes/*/episode.json"))]
    commands = [command for episode in episodes for command in episode["commands"]]
    first_verify = min(command["admission"]["admitted_at"] for command in commands if command["mode"] == "verify")
    initial_trains = [
        command
        for command in commands
        if command["mode"] == "train" and command["admission"]["admitted_at"] < first_verify
    ]
    phases = defaultdict(list)
    for command in commands:
        result = command["result"]
        category = "verify" if command["mode"] == "verify" else "long" if result["updates"] > 448 else "short"
        pool = "shared" if command["admission"]["unit_id"] in ("gpu-2", "gpu-3") else "dedicated"
        phases[(pool, category)].append(command)
    hashes = defaultdict(set)
    for episode in episodes:
        for command in episode["commands"]:
            if command["mode"] == "train":
                key = f"{command['result']['updates']}:{episode['learning_rate']}"
                hashes[key].add(command["result"]["tensor_sha256"])
    summary = {
        "initial_trains_admitted_before_first_verify": len(initial_trains),
        "episodes_started_before_first_verify": sum(episode["started_at"] < first_verify for episode in episodes),
        "first_verify_admission_after_first_episode_s": first_verify - min(e["started_at"] for e in episodes),
        "source_paths": [json.loads(value) for value in sorted({json.dumps(e["source_paths"]) for e in episodes})],
        "uid_counts": dict(Counter(e["uid"] for e in episodes)),
        "learning_rate_by_uid": {e["uid"]: e["learning_rate"] for e in episodes},
        "reward_by_uid": {e["uid"]: e["reward"] for e in episodes},
        "adapter_hashes_by_updates_and_rate": {key: sorted(value) for key, value in hashes.items()},
        "command_phases": {
            f"{pool}:{category}": {
                "preparation_s": summarize([c["result"]["preparation_s"] for c in rows]),
                "compute_s": summarize([c["result"]["compute_s"] for c in rows]),
                "holding_s": summarize([c["admission"]["released_at"] - c["admission"]["admitted_at"] for c in rows]),
            }
            for (pool, category), rows in sorted(phases.items())
        },
    }
    return summary, episodes, commands


def plot_timeline(arms, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"short": "#377eb8", "long": "#1f4666", "verify": "#e6a23c"}
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    horizon = max(
        max(episode["finished_at"] for episode in episodes) - min(episode["started_at"] for episode in episodes)
        for _, episodes, _, _ in arms
    )
    for axis, (name, episodes, commands, events) in zip(axes, arms):
        origin = min(episode["started_at"] for episode in episodes)
        for index, unit in enumerate(("gpu-2", "gpu-3", "gpu-4", "gpu-5")):
            ends = []
            rows = sorted(
                (command for command in commands if command["admission"]["unit_id"] == unit),
                key=lambda c: c["admission"]["admitted_at"],
            )
            for command in rows:
                start, end = [command["admission"][key] for key in ("admitted_at", "released_at")]
                lane = next((i for i, value in enumerate(ends) if value <= start), len(ends))
                if lane == len(ends):
                    ends.append(end)
                else:
                    ends[lane] = end
                category = (
                    "verify"
                    if command["mode"] == "verify"
                    else "long"
                    if command["result"]["updates"] > 448
                    else "short"
                )
                axis.broken_barh(
                    [(start - origin, end - start)], (index - 0.35 + lane * 0.16, 0.14), color=colors[category]
                )
            for event in events:
                if event["unit"] == unit and event["gate_wait_s"] > 1:
                    axis.broken_barh(
                        [(event["started_at"] - origin, event["gate_wait_s"])],
                        (index + 0.32, 0.08),
                        color="#c84d4d",
                    )
        axis.set(
            title="Static pool" if name == "fixed" else "Resident sharing",
            xlabel="Seconds from first trajectory start",
            yticks=range(4),
            yticklabels=["GPU 2 (model)", "GPU 3 (model)", "GPU 4 (env)", "GPU 5 (env)"],
            xlim=(0, horizon * 1.03),
        )
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(
        handles=[Patch(color=color, label=label) for label, color in colors.items()]
        + [Patch(color="#c84d4d", label="model control waits")],
        loc="lower left",
        bbox_to_anchor=(0, 1.12),
        ncol=4,
    )
    figure.savefig(output)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--integrated-name", default="integrated-retry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    analysis = json.loads((args.root / "analysis.json").read_text())
    if not analysis["paired_complete"]:
        raise ValueError("The paired comparison must finish before analysis.")
    integrated = args.root / args.integrated_name
    manifest = json.loads((integrated / "source-manifest.json").read_text())
    changed = [
        name
        for name, expected in manifest.items()
        if hashlib.sha256((integrated / "source" / name).read_bytes()).hexdigest() != expected
    ]
    result = {"frozen_source_files_checked": len(manifest), "frozen_source_changes": changed, "arms": {}}
    figures = []
    for arm in analysis["arms"]:
        summary, episodes, commands = read_arm(integrated, arm["mode"])
        result["arms"][arm["mode"]] = summary
        figures.append((arm["mode"], episodes, commands, arm["model_gate_events"]))
    fixed, resident = (result["arms"][name] for name in ("fixed", "resident"))
    result["same_learning_rate_by_uid"] = fixed["learning_rate_by_uid"] == resident["learning_rate_by_uid"]
    result["same_reward_by_uid"] = fixed["reward_by_uid"] == resident["reward_by_uid"]
    result["same_adapter_hashes_by_work_and_rate"] = (
        fixed["adapter_hashes_by_updates_and_rate"] == resident["adapter_hashes_by_updates_and_rate"]
    )
    fixed_hashes = fixed["adapter_hashes_by_updates_and_rate"]
    resident_hashes = resident["adapter_hashes_by_updates_and_rate"]
    common_conditions = sorted(set(fixed_hashes) & set(resident_hashes))
    result["common_parameter_conditions"] = common_conditions
    result["same_adapter_hashes_for_common_conditions"] = bool(common_conditions) and all(
        fixed_hashes[key] == resident_hashes[key] for key in common_conditions
    )
    result["parameter_conditions_only_in_fixed"] = sorted(set(fixed_hashes) - set(resident_hashes))
    result["parameter_conditions_only_in_resident"] = sorted(set(resident_hashes) - set(fixed_hashes))
    result["scope"] = "Observed trace audit. Placement-conditioned phase averages do not prove causal interference."
    result["static_partition_sensitivity"] = partition_sensitivity(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.figure:
        plot_timeline(figures, args.figure)


if __name__ == "__main__":
    main()
