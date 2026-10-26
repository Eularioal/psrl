"""Summarize the bounded OFF/ON/ON/OFF comparison without launching GPU work."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_science_tail import summarize

ARMS = ("p1-off", "p1-on", "p2-on", "p2-off")


def describe(values):
    return {
        "count": len(values),
        "mean_s": statistics.mean(values),
        "median_s": statistics.median(values),
        "max_s": max(values),
    }


def analyze(root):
    arms = {}
    uuid = None
    for name in ARMS:
        folder = root / name
        result = json.loads((folder / "result.json").read_text())
        tail = summarize(folder)
        assert len(result["attempts"]) == 8 and len(tail["phases"]) == 16, "Every arm must complete equal work."
        assert not result["cleanup_errors"] and not result["remaining_owned_containers"]
        current_uuid = result["gpu_before"]["uuid"]
        uuid = current_uuid if uuid is None else uuid
        assert current_uuid == uuid, "All arms must execute on the same physical GPU."
        task = [item["finished_at"] - item["started_at"] for item in result["attempts"]]
        command = [phase["call_s"] for phase in tail["phases"]]
        cuda = [phase["script_s"] for phase in tail["phases"]]
        host = [phase["call_s"] - phase["script_s"] for phase in tail["phases"]]
        active = []
        violations = []
        for line in (folder / "telemetry.jsonl").read_text().splitlines():
            item = json.loads(line)
            for unit in item.get("units", []):
                active.append(unit["gpu_commands_active"])
                if unit["declared_gpu_memory_mb"] > unit["gpu_memory_mb"]:
                    violations.append(item["time"])
        assert not violations, "The declared command budget must remain within capacity."
        assert active or name.endswith("off"), "ON must have actual ledger observations."
        arms[name] = {
            "work_s": result["work_duration_s"],
            "waves_s": tail["waves_s"],
            "setup_s": result["work_started_at"] - result["started_at"],
            "preparation_s": result["preparation_finished_at"] - result["preparation_started_at"],
            "cleanup_s": result["finished_at"] - result["work_finished_at"],
            "total_s": result["finished_at"] - result["started_at"],
            "task": describe(task),
            "command": describe(command),
            "cuda": describe(cuda),
            "command_host_remainder": describe(host),
            "task_samples_s": task,
            "command_samples_s": command,
            "cuda_samples_s": cuda,
            "work_loop_lag_max_s": tail["work_loop_lag_max_s"],
            "stack_categories": {key: len(value) for key, value in tail["work_stack_samples"].items()},
            "ledger_samples": len(active),
            "ledger_max_active_commands": max(active) if active else None,
            "ledger_violations": len(violations),
            "runner_source": result["runner_source"],
        }
    pairs = []
    for pair in ("p1", "p2"):
        off, on = arms[f"{pair}-off"], arms[f"{pair}-on"]
        pairs.append(
            {
                "pair": pair,
                "off_s": off["work_s"],
                "on_s": on["work_s"],
                "on_delta_pct": 100 * (on["work_s"] / off["work_s"] - 1),
            }
        )
    pooled = {}
    for mode in ("off", "on"):
        selected = [arms[name] for name in ARMS if name.endswith(mode)]
        pooled[mode] = {
            "work_s": sum(item["work_s"] for item in selected),
            "total_s": sum(item["total_s"] for item in selected),
            **{
                key: describe([v for item in selected for v in item[f"{key}_samples_s"]])
                for key in ("task", "command", "cuda")
            },
        }
    summary = {
        "scope": "Two order-balanced pairs, descriptive results, no confidence interval",
        "gpu_uuid": uuid,
        "arms": arms,
        "pairs": pairs,
        "pooled": pooled,
    }
    (root / "paired-analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot(root, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"off": "#7c8da8", "on": "#198f83"}
    fig, grid = plt.subplots(2, 2, figsize=(11, 7.6), layout="constrained")
    axes = grid.flat
    for ax, metric, title in zip(
        list(axes)[:2],
        ("work_s", "total_s"),
        ("Work phase: complete 8 tasks", "Driver total: setup + work + cleanup"),
        strict=True,
    ):
        for mode, shift in (("off", -0.18), ("on", 0.18)):
            values = [data["arms"][f"p{index}-{mode}"][metric] for index in (1, 2)]
            bars = ax.bar(
                [index + shift for index in (1, 2)], values, width=0.34, label=mode.upper(), color=colors[mode]
            )
            ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(
            title=title,
            xticks=[1, 2],
            xticklabels=["Pair 1: OFF then ON", "Pair 2: ON then OFF"],
            ylabel="Seconds",
        )
        ax.legend()
    for ax, metric, title in zip(
        list(grid.flat)[2:], ("task", "command"), ("Per-task latency", "Per-GPU-command latency"), strict=True
    ):
        for index, mode in enumerate(("off", "on")):
            values = [v for name in ARMS if name.endswith(mode) for v in data["arms"][name][f"{metric}_samples_s"]]
            offsets = [((i % 8) - 3.5) * 0.025 for i in range(len(values))]
            ax.scatter([index + offset for offset in offsets], values, color=colors[mode], alpha=0.65, s=19)
            mean = statistics.mean(values)
            ax.hlines(mean, index - 0.28, index + 0.28, color="black", linewidth=2)
            ax.annotate(f"Mean {mean:.3f}", (index, mean), xytext=(0, 7), textcoords="offset points", ha="center")
        ax.set(title=title, xticks=[0, 1], xticklabels=["OFF", "ON"], ylabel="Seconds", xlim=(-0.5, 1.5))
    for ax in grid.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.12)
    fig.suptitle(
        "Low-load paired experiment: fixed historical OFF vs latest Router ON\n"
        "Same GPU, image, task workload and instrumentation. No model training."
    )
    fig.savefig(root / "paired-latency.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    plot(args.root, result)
    print(json.dumps({"pairs": result["pairs"], "pooled": result["pooled"]}, indent=2))
