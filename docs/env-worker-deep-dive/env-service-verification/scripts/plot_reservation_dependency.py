"""Render a fixed-work teaching example; no GPU or experimental measurements."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    font = FontProperties(fname="/usr/share/fonts/google-droid/DroidSansFallback.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.77, bottom=0.19, hspace=0.70)
    fig.text(0.06, 0.95, "教学假设：立即填满 GPU，为什么可能更晚完成训练？", fontsize=15, weight="bold")
    fig.text(0.06, 0.90, "相同工作：A 再等 6 秒返回、评分用 4 秒；B 已就绪、命令用 30 秒。", fontsize=11)
    fig.text(0.06, 0.85, "仅一份环境容量可用。A 补齐批次 1，B 补齐批次 2；每次训练 20 秒。", fontsize=11)
    colors = {"A": "#2783ba", "B": "#ed9942", "train": "#8f72b3", "hold": "#cbd5e1"}
    schedules = [
        ("① 命令结束立即归还 → B 先运行", [(0, 30, "B"), (30, 4, "A")], [34, 54], 74),
        ("② 为 A 短暂保留 → 提早补齐当前批次", [(0, 6, "hold"), (6, 4, "A"), (10, 30, "B")], [10, 40], 60),
    ]
    for ax, (title, env, train, finish) in zip(axes, schedules):
        ax.set_title(title, loc="left", fontsize=12, pad=18)
        ax.set_ylim(-0.5, 1.6)
        ax.set_xlim(0, 80)
        ax.set_yticks([0, 1], ["训练卡", "环境卡"])
        ax.set_xticks(range(0, 81, 10))
        ax.grid(axis="x", color="#e4e9ee", zorder=0)
        for start, width, label in env:
            ax.barh(1, width, left=start, height=0.52, color=colors[label], zorder=3)
            if label == "B":
                ax.text(start + width / 2, 1, "B：30 秒", va="center", ha="center", fontsize=11)
            else:
                ax.text(start + width / 2, 1.36, "留" if label == "hold" else "A", va="center", ha="center", fontsize=11)
        for step, start in enumerate(train, 1):
            ax.barh(0, 20, left=start, height=0.52, color=colors["train"], zorder=3)
            ax.text(start + 10, 0, f"训练 {step}", color="white", va="center", ha="center", fontsize=11)
        ax.axvline(finish, color="#334155", linestyle="--", linewidth=1)
        ax.text(finish - 0.8, 0.65, f"{finish} 秒结束", ha="right", fontsize=11, color="#334155")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("从 A 上一条命令结束开始计时（秒）", fontsize=11)
    fig.text(0.06, 0.08, "环境最后完成：34 → 40 秒；两步训练完成：74 → 60 秒，反而提前 14 秒。", fontsize=11)
    fig.text(0.06, 0.03, "假设：固定批次、不能替换 A、独立训练卡、无重载开销；非实测或已实现策略。", fontsize=10)
    target = Path(__file__).resolve().parents[2] / "experiments" / "reservation-batch-dependency"
    for suffix in (".png", ".svg"):
        fig.savefig(target.with_suffix(suffix), dpi=100, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
