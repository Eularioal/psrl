"""Explain the fixed episode baseline with an explicitly hypothetical timeline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def box(ax, x, y, width, height, color, label, text_color="#20354a"):
    """Draw a resource or time interval in the shared pixel coordinate system."""
    ax.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="white", linewidth=1.4))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", color=text_color, fontsize=12)


def main():
    """Keep device ownership and reservation duration visible in one screen."""
    plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 12, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(8, 6), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 800), ylim=(600, 0))
    ax.axis("off")
    ax.text(30, 35, "统一主基线：固定环境卡池，episode 全程持卡", fontsize=17, weight="bold")
    ax.text(30, 67, "① 卡的角色固定。只有绿色池接纳环境，满了就等。", fontsize=12)
    for x, title, color, labels in (
        (30, "两张训练卡", "#d7d4ec", ("训练 0", "训练 1")),
        (290, "两张推理卡", "#cce2f6", ("vLLM 0", "vLLM 1")),
        (550, "两张环境卡", "#b7e2d3", ("环境 0", "环境 1")),
    ):
        ax.text(x, 98, title, fontsize=12, weight="bold")
        for index, label in enumerate(labels):
            box(ax, x + index * 112, 112, 108, 47, color, label)
    ax.text(30, 187, "主基线不借推理余量，也不让 vLLM 休眠来腾卡。", fontsize=12)
    ax.text(30, 229, "② 同一个例子：只剩 1 个名额，A 在第 20 秒暂停，B 此时到达。", fontsize=12)
    ax.text(30, 256, "教学假设：A 算 20 秒、思考 30 秒、再算 20 秒，B 算 20 秒。", fontsize=11)
    left, scale = 175, 5.4
    for seconds in (0, 20, 40, 50, 70, 90, 100):
        x = left + seconds * scale
        ax.plot((x, x), (281, 474), color="#e1e7ee", linewidth=0.8, zorder=0)
        ax.text(x, 493, str(seconds), ha="center", fontsize=11)
    rows = (("episode · A", 286), ("episode · B", 332), ("command · A", 390), ("command · B", 436))
    for label, y in rows:
        ax.text(155, y + 16, label, ha="right", va="center", fontsize=11)
    for y in (286, 390):
        box(ax, left, y, 20 * scale, 33, "#287d68", "计算", "white")
        box(ax, left + 50 * scale, y, 20 * scale, 33, "#287d68", "计算", "white")
    box(ax, left + 20 * scale, 286, 30 * scale, 33, "#a6c9e4", "思考，仍持卡")
    box(ax, left + 20 * scale, 332, 50 * scale, 33, "#f6d59a", "B 等待 50 秒")
    box(ax, left + 70 * scale, 332, 20 * scale, 33, "#e49144", "B 计算")
    box(ax, left + 20 * scale, 390, 30 * scale, 33, "#eef3f8", "思考，已归还")
    box(ax, left + 20 * scale, 436, 20 * scale, 33, "#e49144", "B 计算")
    ax.text(left + 90 * scale + 8, 348, "90 秒完", fontsize=11, va="center")
    ax.text(left + 70 * scale + 8, 406, "70 秒完", fontsize=11, va="center")
    ax.text(725, 493, "秒", fontsize=11)
    ax.text(30, 534, "浅蓝表示为 A 保留 GPU 预算。深绿才是计算，两者含义不同。", fontsize=12)
    ax.text(30, 560, "command 释放经物理检查的 GPU 预算，CPU、主存和容器继续保留。", fontsize=11)
    ax.text(30, 584, "时间线为教学假设，省略交接成本。本图不报告实测加速或自动最优决策。", fontsize=11)
    output = Path(__file__).resolve().parents[2] / "experiments/partition-episode-baseline"
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
