"""Plot all three scheduling arms in the memory and workload replay."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from plot_expanded_partition_batch import FONT, read, save_figure

LABELS = {
    "aime_resident_43": "AIME，推理 43 GiB",
    "aime_resident_62": "AIME，推理 62 GiB",
    "aime_resident_72": "AIME，推理 72 GiB",
    "larger_environment": "工具 24 GiB，推理 43",
    "larger_environment_and_resident": "工具 24 GiB，推理 62",
    "vision_4g": "CIFAR-10 短命令",
    "mixed_43": "AIME＋图像，43 GiB",
    "mixed_62": "AIME＋图像，62 GiB",
}


def plot(rows, output):
    """Show model memory and workload shape on the same completion-time axis."""
    font_manager.fontManager.addfont(FONT)
    name = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams.update({"font.family": name, "font.size": 11, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=100)
    fig.subplots_adjust(left=0.29, right=0.96, top=0.79, bottom=0.21)
    fig.suptitle("显存压力和任务组成会改变动态调度的收益", x=0.035, ha="left", fontsize=18)
    fig.text(0.035, 0.85, "2 张训练卡＋3 张推理卡＋3 张环境卡，CPU 回放的配对耗时中位数", fontsize=11)

    positions = list(range(len(rows)))
    series = (
        ("median_episode_s", "Episode 静态", "#a7b5c2"),
        ("median_command_static_s", "Command 静态", "#3577ab"),
        ("median_dynamic_s", "Command 动态", "#237a67"),
    )
    for offset, (metric, label, color) in zip((-0.24, 0, 0.24), series, strict=True):
        values = [row[metric] for row in rows]
        ax.barh([y + offset for y in positions], values, height=0.21, label=label, color=color)
    ax.set(
        yticks=positions, yticklabels=[LABELS[row["scenario"]["name"]] for row in rows], xlabel="收齐整批轨迹（秒）"
    )
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e1e7ec", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.44, 1.01), ncol=3, frameon=False)
    fig.text(0.035, 0.055, "工具预约 24 GiB 时按 1.4 倍耗时回放，共 12 条轨迹。其余场景为 20 条。", fontsize=11)
    fig.text(0.035, 0.025, "每个条件抽样 12 次，同一次抽样的三组使用同一批模型请求和工具命令。", fontsize=11)
    save_figure(fig, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(read(args.replay)["pairs"], args.output)
