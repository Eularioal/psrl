"""Plot completion counts beside latency without treating failed training as faster work."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root):
    report = json.loads((root / "analysis.json").read_text())
    order = ["low-off", "low-on", "high-off", "high-on"]
    labels = ["Low / OFF", "Low / ON", "High / OFF", "High / ON"]
    arms = [report["arms"][key] for key in order]
    colors = ["#4878a8", "#39a878", "#4878a8", "#39a878"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    x = list(range(4))
    axes[0].bar(x, [arm["success"] for arm in arms], color=colors)
    axes[0].bar(
        x,
        [arm["submitted"] - arm["success"] for arm in arms],
        bottom=[arm["success"] for arm in arms],
        color="#db8656",
        hatch="///",
        label="Failed commands",
    )
    axes[0].set_ylim(0, 4.8)
    axes[0].set_ylabel("Commands")
    axes[0].set_title("Completed 10 training steps")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].bar(x, [arm["agent_s"] for arm in arms], color=colors)
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Agent start to all commands returned")
    axes[1].set_ylim(0, max(arm["agent_s"] for arm in arms) * 1.3)
    for i, arm in enumerate(arms):
        axes[0].text(i, 4.1, f"{arm['success']}/4 completed", ha="center", fontsize=8)
        note = f"{arm['agent_s']:.1f}s" + ("\nincomplete work" if arm["success"] != 4 else "")
        axes[1].text(i, arm["agent_s"] + 4, note, ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MLGym CIFAR: 4 commands, 10 steps each | High pressure leaves 1.5 GiB free")
    fig.savefig(root / "capacity-comparison.svg")
    fig.savefig(root / "capacity-comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
