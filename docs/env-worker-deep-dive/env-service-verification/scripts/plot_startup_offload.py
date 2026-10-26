"""Plot equal-work startup offload results from the saved tail analysis."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(root):
    data = json.loads((root / "tail-analysis.json").read_text())
    arms = [data[name] for name in ("before", "after")]
    assert all(arm["passed"] for arm in arms), "Both equal-work arms must finish."
    assert arms[0]["attempts"] == arms[1]["attempts"], "Compare the same completed workload."
    panels = [
        ("Complete workload", [arm["work_s"] for arm in arms]),
        ("Maximum event-loop stall", [arm["work_loop_lag_max_s"] for arm in arms]),
        ("Mean CUDA script duration", [statistics.mean(p["script_s"] for p in arm["phases"]) for arm in arms]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, (title, values) in zip(axes, panels, strict=True):
        bars = ax.bar(["Before", "After"], values, color=["#7c8da8", "#198f83"], width=0.58)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=5)
        ax.set(title=title, ylabel="Seconds", ylim=(0, max(values) * 1.22))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Router ON: startup probes moved off the event loop\n8 tasks per arm, 4 concurrent, 16 CUDA commands")
    fig.savefig(root / "startup-offload.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
