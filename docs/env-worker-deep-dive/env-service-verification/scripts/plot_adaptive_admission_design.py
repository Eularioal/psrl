"""Render a teaching comparison of memory-safe admission and bounded deferral."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    """Compare equal remaining work under two explicitly assumed interference levels."""
    plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 11.5})
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), dpi=100)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.72, bottom=0.18, hspace=0.8)
    fig.suptitle("动态准入：显存放得下，是否现在就放第五条？", fontsize=16, y=0.98)
    fig.text(0.04, 0.89, "同一状态：A～D 正在运行，各剩 10 秒，E 到达，单独执行需 10 秒。", fontsize=11.5)
    fig.text(0.04, 0.83, "每条预约 16 GiB，五条共 80 GiB，均满足 90 GiB 预算。", fontsize=11.5)
    purple, blue, green = "#8b65b2", "#397dc4", "#2b9568"
    for ax, duration, title in zip(
        axes,
        (12, 25),
        ("① 轻竞争假设：第五条加入后，五条各需 12 秒", "② 强竞争假设：仅改变干扰程度，五条各需 25 秒"),
    ):
        ax.set_title(title, loc="left", fontsize=12, pad=9)
        ax.barh(1, duration, color=purple, height=0.48)
        ax.text(duration / 2, 1, f"A～E 并行：{duration} 秒", ha="center", va="center", color="white")
        ax.barh(0, 10, color=blue, height=0.48)
        ax.barh(0, 10, left=10, color=green, height=0.48)
        ax.text(5, 0, "A～D：10 秒", ha="center", va="center", color="white")
        ax.text(15, 0, "E：10 秒", ha="center", va="center", color="white")
        ax.text(20.4, 0, "20 秒", va="center")
        ax.set_yticks([1, 0], ["立即准入 E", "E 等待归还"])
        ax.set_xlim(0, 27)
        ax.set_ylim(-0.45, 1.45)
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[-1].set_xlabel("从 E 到达开始计时（秒），终点是五条命令全部完成")
    fig.text(0.04, 0.045, "教学假设，非 GPU 实测。默认放满，只有等待能可靠缩短完成时间时才暂缓。", fontsize=11)
    output = Path(__file__).resolve().parents[2] / "experiments" / "adaptive-admission-design"
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
