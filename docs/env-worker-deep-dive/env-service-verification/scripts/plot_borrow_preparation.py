"""Draw a controlled teaching example of delayed admission before GPU borrowing."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    """Compare identical commands and expose both the winner and the delayed command."""
    plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 12})
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), dpi=100)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.77, bottom=0.25, hspace=0.65)
    policies = [
        ("现有：观察期仍接纳 B，确认借卡后等 B 结束", [12, 101, 101], 131),
        ("设想：观察期暂缓新命令，确认后一起进入整卡", [33, 33, 33], 119),
    ]
    for ax, (title, starts, complete) in zip(axes, policies):
        ax.axvspan(0, 30, color="#f6ebc9", zorder=0)
        ax.axvline(30, color="#a17b23", linestyle="--", linewidth=1)
        for y, (start, duration) in enumerate(zip(starts, (86, 30, 30))):
            if start > 12:
                ax.barh(y, start - 12, left=12, color="#d6dce3", height=0.64)
            ax.barh(y, duration, left=start, color="#337cac" if y == 0 else "#398a70", height=0.64)
            ax.text(
                start + duration / 2,
                y,
                f"{start} → {start + duration}",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
            )
        ax.axvline(complete, color="#b84a40", linewidth=1)
        ax.text(complete - 1, -0.65, f"全部完成：{complete} 秒", ha="right", color="#b84a40", fontsize=11)
        ax.set(
            yticks=[0, 1, 2],
            yticklabels=["命令 B", "命令 C", "命令 D"],
            xlim=(0, 150),
            xticks=[0, 30, 60, 90, 120, 150],
            ylim=(2.6, -0.85),
        )
        ax.set_title(title, fontsize=12, loc="left", pad=20)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("从开始观察低推理负载计时（秒）")
    fig.suptitle("提前准备借卡：B 晚结束，C／D 早结束，整批可能更快", fontsize=15, y=0.97)
    fig.text(
        0.04,
        0.86,
        "教学假设：三条命令都在第 12 秒就绪。模型常驻时可放 1 条，整卡可放 3 条。\n"
        "黄色为 30 秒观察期，灰色为排队。确认借卡后转换需 3 秒。",
        fontsize=11,
    )
    fig.text(
        0.04,
        0.035,
        "同一工作量：B 执行 86 秒，C／D 各执行 30 秒，独立环境卡一直忙。\n"
        "B 晚 21 秒，C／D 各早 68 秒，整批早 12 秒。耗时固定是教学假设。\n"
        "观察期 vLLM 继续服务，未取消已运行命令。下图策略尚未实现、未实测。",
        fontsize=11,
    )
    output = Path(__file__).resolve().parents[2] / "experiments/borrow-preparation-example"
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
