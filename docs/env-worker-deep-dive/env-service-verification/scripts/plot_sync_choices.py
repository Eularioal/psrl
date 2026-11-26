"""Render a shared-time-axis teaching example for synchronization choices."""

import argparse
from pathlib import Path


def draw(path):
    """Keep the same remaining work and timing origin in all three examples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [
        ("Wait for A", [(0, 100), (100, 2), (102, 40)], 142),
        ("Allow overlap", [(0, 105), (0, 3), (3, 40)], 105),
        ("Cancel and restart A", [(2, 135), (0, 2), (2, 40)], 137),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    colors = ["#397dc4", "#e9a442", "#3c9b70"]
    labels = ["Environment A", "Weight sync", "Model work B"]
    for ax, (name, work, complete) in zip(axes, cases):
        for i, ((start, duration), color) in enumerate(zip(work, colors)):
            ax.barh(i, duration, left=start, color=color, height=0.65)
            ax.text(start + duration + 1, i, f"{start} to {start + duration} s", va="center", fontsize=9)
        ax.set(yticks=range(3), yticklabels=labels, xlim=(0, 165))
        ax.invert_yaxis()
        ax.set_title(f"{name}: both complete at {complete} s", loc="left", fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Seconds after the synchronization request")
    fig.suptitle("Teaching example: compare the same remaining work", fontsize=15, y=0.99)
    fig.text(0.5, 0.94, "At t=0: A has 100 s left. Sync alone takes 2 s. B needs 40 s after sync.", ha="center")
    fig.text(
        0.5,
        0.015,
        "Assumptions: shared A adds 5 s. Shared sync adds 1 s. Restart adds 30 s. "
        "Cancellation cost omitted. Illustrative only.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    draw(parser.parse_args().output)
