"""Render measured placement comparisons with individual repeats and descriptive ranges."""

import argparse
import json
import statistics
from pathlib import Path

CASES = ("default", "short", "boundary", "long")
LABELS = ("Fixed old runner", "Affinity", "Always copy", "Predictive")
COLORS = ("#737373", "#4477AA", "#AA66AA", "#228833")


def group(cells, case_id, label):
    return [
        cell
        for cell in cells
        if cell["case_id"] == case_id
        and cell["phase"] == "core"
        and cell["backend"] == ("baseline" if label == 0 else "native")
        and cell["strategy"] == ("affinity" if label < 2 else ("always_copy" if label == 2 else "predictive"))
    ]


def measured_bar(ax, position, values, color, width, label=None):
    mean = statistics.mean(values)
    ax.bar(position, mean, width=width, color=color, alpha=0.85, label=label)
    ax.errorbar(
        position, mean, yerr=[[mean - min(values)], [max(values) - mean]], color="#222222", capsize=3, linewidth=1
    )
    offset = [(index - (len(values) - 1) / 2) * width / 5 for index in range(len(values))]
    ax.scatter([position + value for value in offset], values, color="#222222", s=15, zorder=4)
    ax.annotate(
        f"n={len(values)}", (position, max(values)), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7
    )


def save_figure(fig, directory, stem):
    fig.savefig(directory / (stem + ".svg"), bbox_inches="tight")
    fig.savefig(directory / (stem + ".png"), dpi=180, bbox_inches="tight")


def render(data, directory, *, allow_incomplete=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    status = data["audit"]["status"]
    if status != "passed" and not (status == "incomplete" and allow_incomplete):
        raise ValueError("Publishing an incomplete audit requires explicit permission to show its audited subset.")
    cells = [cell for cell in data["cells"] if cell.get("audit", {}).get("passed")]
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.18
    for label in range(4):
        for case_index, case_id in enumerate(CASES):
            values = [cell["primary_makespan_s"] for cell in group(cells, case_id, label)]
            if values:
                measured_bar(
                    ax,
                    case_index + (label - 1.5) * width,
                    values,
                    COLORS[label],
                    width,
                    LABELS[label] if case_index == 0 else None,
                )
    ax.set_xticks(range(4), ["Default-sized A", "Short wait", "Near threshold", "Long wait"])
    ax.set_ylabel("Command group wall time (s)")
    ax.set_title("Fixed workload per case, completed audited counts shown")
    ax.margins(y=0.15)
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(
        0.125,
        -0.02,
        "Bars: mean. Points: runs. Whiskers: observed range, not confidence intervals. "
        + f"Matrix audit: {status}, {len(cells)}/{data['planned_cells']} cells audited.",
        fontsize=9,
    )
    save_figure(fig, directory, "warm-placement-core")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    phase_colors = ("#DDAA33", "#CC6677", "#4477AA", "#999999")
    phase_labels = ("Queue / channel wait", "Model preparation / load", "Compute wall time", "Other preparation")
    positions, names = [], []
    for case_index, case_id in enumerate(CASES):
        for label in range(1, 4):
            values = group(cells, case_id, label)
            if not values:
                continue
            position = case_index * 4 + label - 1
            positions.append(position)
            short_label = {1: "Affinity", 2: "Copy", 3: "Predict"}[label]
            names.append(short_label + "\n" + case_id + f"\nn={len(values)}")
            components = []
            for cell in values:
                wait = cell["target_wait_s"]
                load, compute = cell["target"]["load_s"], cell["target"]["compute_s"]
                components.append((wait, load, compute, max(0, cell["target_latency_s"] - wait - load - compute)))
            bottom = 0
            for phase in range(4):
                height = statistics.mean(value[phase] for value in components)
                ax.bar(
                    position,
                    height,
                    bottom=bottom,
                    color=phase_colors[phase],
                    width=0.8,
                    label=phase_labels[phase] if len(positions) == 1 else None,
                )
                bottom += height
    ax.set_xticks(positions, names, fontsize=7)
    ax.set_ylabel("Target arrival to service completion (s)")
    ax.set_title("Target latency decomposition, means of completed audited runs")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save_figure(fig, directory, "warm-placement-decomposition")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fault_cases = (("overpredict_short", "short"), ("stale_short", "short"), ("memory_long", "long"))
    for index, (fault, reference) in enumerate(fault_cases):
        observed = [cell for cell in cells if cell["case_id"] == fault]
        repeats = {cell["round"] for cell in observed}
        normal = [cell["primary_makespan_s"] for cell in group(cells, reference, 3) if cell["round"] in repeats]
        injected = [cell["primary_makespan_s"] for cell in observed]
        for shift, values, color, label in (
            (-0.18, normal, "#228833", "Matched normal case"),
            (0.18, injected, "#CC6677", "Injected condition"),
        ):
            if values:
                measured_bar(axes[0], index + shift, values, color, 0.34, label if index == 0 else None)
    axes[0].set_xticks(range(3), ["10x overestimate", "Stale report", "6 GiB cold budget"], fontsize=9)
    axes[0].set_ylabel("Command group wall time (s)")
    axes[0].set_title("Injected conditions use the same command workload")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].margins(y=0.15)
    pairs = [
        (cell["blocker_remaining_actual_s"], cell["progress_forecast"].get("estimate_s"))
        for cell in cells
        if cell["phase"] == "core"
    ]
    pairs = [(actual, estimate) for actual, estimate in pairs if estimate is not None]
    if pairs:
        axes[1].scatter([x[0] for x in pairs], [x[1] for x in pairs], s=20, alpha=0.65, color="#4477AA")
        maximum = max(max(pair) for pair in pairs) * 1.05
        axes[1].plot([0, maximum], [0, maximum], "--", color="#777777", linewidth=1)
    axes[1].set_xlabel("Measured blocker remaining (s)")
    axes[1].set_ylabel("Online forecast remaining (s)")
    axes[1].set_title("Core forecasts use only progress available at arrival")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, directory, "warm-placement-faults")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    render(json.loads(args.evidence.read_text()), args.output_dir, allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
