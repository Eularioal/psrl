"""Draw the completed November 25 workload comparisons from audited JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parents[1]
OUT = BASE.parent / "experiments"
FONT = "/usr/share/fonts/google-droid/DroidSansFallback.ttf"
ARMS = ("episode_static", "command_static", "command_elastic")
LABELS = ("Episode 静态", "Command 静态", "完整方案")


def read(relative):
    return json.loads((BASE / "results" / relative / "comparison.json").read_text())["arms"]


def bars(ax, rows, maximum):
    for index, (name, label) in enumerate(zip(ARMS, LABELS)):
        row = rows[name]
        if not row.get("complete_workload"):
            ax.text(0, index, "本组未完成", va="center", color="#a34b34", fontsize=12)
            continue
        train = row["training_s"]
        rest = row["launch_to_cleanup_s"] - train
        ax.barh(index, rest, color="#c3cdd8", height=0.55)
        ax.barh(index, train, left=rest, color="#277caf", height=0.55)
        ax.text(rest + train + maximum * 0.015, index, f'{row["launch_to_cleanup_s"]:.2f}', va="center")
        ax.text(rest + train / 2, index, f"{train:.0f}", ha="center", va="center", color="white")
    ax.set_yticks(range(3), LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, maximum)
    ax.set_xlabel("完整作业时间（秒）")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e5eaf0")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=100)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def overview():
    """Compare each scene against its own Command-static timing."""
    result_root = BASE / "results"
    sources = (
        ("AIME · 1步先导", "partition-three-way-20261125/one-step-comparison.json"),
        ("AIME · 10步", "partition-three-way-20261125/command-static-10-retry/three-way-comparison.json"),
        ("高显存 · 1步先导", "diverse-workloads-20261125/high-memory/comparison.json"),
        ("视觉混合 · 1步先导", "diverse-workloads-20261125/vision-mix/comparison.json"),
    )
    rows = [json.loads((result_root / source).read_text())["arms"] for _, source in sources]
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.2), sharey=True)
    for ax, metric, title in zip(
        axes, ("training_s", "launch_to_cleanup_s"), ("累计训练时间", "启动至清理时间")
    ):
        ax.axvline(100, color="#637386", linestyle="--", linewidth=1)
        for y, row in enumerate(rows):
            baseline = row["command_static"][metric]
            for arm, offset, color, label in (
                ("episode_static", -0.14, "#b87926", "Episode 静态"),
                ("command_elastic", 0.14, "#187e72", "完整方案"),
            ):
                value = 100 * row[arm][metric] / baseline
                ax.plot([100, value], [y + offset] * 2, color=color, linewidth=2, alpha=0.65)
                ax.scatter(value, y + offset, s=55, color=color, label=label if y == 0 else None, zorder=3)
                ax.text(value + 2, y + offset, f"{value:.1f}", va="center", fontsize=11, color=color)
        ax.set_xlim(65, 148)
        ax.set_ylim(3.5, -0.6)
        ax.set_xticks([80, 100, 120, 140])
        ax.set_title(title, fontsize=14, pad=12)
        ax.grid(axis="y", color="#edf0f3")
        ax.tick_params(axis="y", length=0, pad=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(4), [label for label, _ in sources])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.12, 1.32), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.2, top=0.77, wspace=0.16)
    fig.text(0.25, 0.95, "各场景自己的 Command 静态＝100，越低越快", fontsize=13)
    fig.text(0.25, 0.12, "低于100为更快；高于100为更慢", fontsize=12)
    fig.text(0.03, 0.04, "单次顺序测量；单步为历史先导；AIME 的 Command 静态含独立补测", fontsize=11)
    save(fig, "partition-evidence-overview")


def main():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=FONT).get_name(),
        "font.size": 12,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    })
    overview()
    fig, ax = plt.subplots(figsize=(8, 3.7))
    comparison = BASE / "results/partition-three-way-20261125/command-static-10-retry/three-way-comparison.json"
    bars(ax, json.loads(comparison.read_text())["arms"], 5200)
    fig.text(0.23, 0.94, "八卡 AIME · 各 10 个 PSRL step", fontsize=15)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.77, bottom=0.24)
    fig.text(0.23, 0.85, "灰色为初始化及其余时间；蓝色为累计训练", fontsize=11)
    fig.text(0.03, 0.035, "200 条轨迹／组；Command 静态采用独立补跑；完整方案少用 16.77%", fontsize=11)
    save(fig, "partition-long-current")

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.9))
    for ax, scene, title in zip(
        axes, ("high-memory", "vision-mix"),
        ("高显存 AIME · 每组 1 步，12 条轨迹", "CIFAR-10＋COCO · 每组 1 步，20 条轨迹"),
    ):
        bars(ax, read(f"diverse-workloads-20261125/{scene}"), 1100)
        ax.set_title(title, loc="left", fontsize=14)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.89, bottom=0.17, hspace=0.9)
    fig.text(0.23, 0.96, "灰色为初始化及其余时间；蓝色为训练", fontsize=11)
    fig.text(0.03, 0.025, "每组一次；相对 Command 静态完整时间分别少 5.21%、多 4.62%", fontsize=11)
    save(fig, "partition-diversity-live")


if __name__ == "__main__":
    main()
