"""Render measured synchronization results and the prelaunch partition protocol."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "experiments"
RESULTS = BASE / "env-service-verification/results"
plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 12, "axes.unicode_minus": False})


def save(fig, stem):
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=100)
    plt.close(fig)


def results():
    """Separate initial setup, fixed work, and the observed overlap lifetime."""
    data = json.loads((RESULTS / "sync-overlap-20261122/analysis.json").read_text())
    fig = plt.figure(figsize=(8, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])
    ax = fig.add_subplot(grid[0])
    for index, name in enumerate(("wait", "overlap")):
        row = data[name]
        values = [row["status"]["initialization_s"], row["timing_totals"]["timing_s/step"]]
        values.append(row["launch_to_cleanup_s"] - sum(values))
        left = 0
        for value, color, label in zip(
            values, ("#cad3df", "#267ea8", "#e7ac50"), ("初始化", "四步训练", "其余与清理")
        ):
            ax.barh(index, value, left=left, color=color, label=label if index == 0 else None, height=0.48)
            if value > 100:
                ax.text(
                    left + value / 2,
                    index,
                    f"{value:.1f}s",
                    ha="center",
                    va="center",
                    color="white" if label == "四步训练" else "#1f2937",
                )
            left += value
        ax.text(left + 15, index, f"{left:.1f}s", va="center", fontsize=11)
    ax.set(
        yticks=[0, 1], yticklabels=["等待环境归还", "允许同步重叠"], xlim=(0, 1990), xlabel="从各组启动起计时（秒）"
    )
    ax.invert_yaxis()
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, fontsize=11, frameon=False)
    ax.set_title("实测：四步训练慢了 4.62%，启动至清理却快了 3.95%", pad=42, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = fig.add_subplot(grid[1])
    ax2.axis("off")
    ax2.text(0, 0.94, "为何不能把门禁等待下降直接当作训练收益？", fontsize=13, weight="bold")
    rows = [
        ("请求 P95", "35.34 → 17.34 秒", "尾延迟改善"),
        ("环境平均排队", "80.91 → 99.30 秒", "另一路等待增加"),
        ("初始化", "780.96 → 669.98 秒", "影响总时长，需单列"),
    ]
    for y, (label, number, reason) in zip((0.68, 0.44, 0.20), rows):
        ax2.text(0, y, label)
        ax2.text(0.29, y, number, color="#176d96")
        ax2.text(0.67, y, reason)
    fig.supxlabel("两组各 4 step / 48 轨迹；并行组有额外 GPU 进程占用，时间差不等于因果效果。", fontsize=10)
    save(fig, "sync-overlap-results")


def timeline():
    """Show that the measured transfer overlaps a command but precedes its compute."""
    evidence = json.loads((RESULTS / "sync-overlap-20261122/mechanism-audit.json").read_text())
    event = evidence["arms"]["overlap"]["overlapping_transfers"][0]
    remaining = event["remaining_lease_after_transfer_s"] + event["transfer_s"]
    lower, upper = event["estimated_compute_start_after_transfer_s_bounds"]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.barh(1, remaining, color="#b7d8ef", height=0.35)
    ax.barh(0, event["transfer_s"], color="#e69a36", height=0.35)
    ax.axvspan(lower + event["transfer_s"], upper + event["transfer_s"], color="#78ad88", alpha=0.4)
    ax.annotate(
        "真实 NIXL 拉取约 0.262 秒\n完成时，环境还要约 21 秒才归还",
        xy=(0.26, 0),
        xytext=(3, -0.6),
        arrowprops={"arrowstyle": "->"},
        fontsize=12,
    )
    ax.text(2, 1, "环境命令仍持卡", va="center", fontsize=12)
    ax.text(11.1, 1.65, "计算开始的推算区间\n同步后约 9.1～18.5 秒", fontsize=12, ha="center")
    ax.text(21.7, 1, "归还", va="center", fontsize=11)
    ax.set(
        yticks=[0, 1],
        yticklabels=["权重传输", "环境评分命令"],
        xlim=(-0.5, 25),
        ylim=(-1, 2.3),
        xlabel="从权重拉取开始计时（秒）",
    )
    ax.set_title("实测边界：这次重叠发生在环境准备阶段", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("区间由两种时钟及命令边界推算，假设同机时钟偏移稳定；未证明 GPU 内核同时执行。", fontsize=10)
    save(fig, "sync-overlap-timeline")


def design():
    """Show allocation ownership and the unchanged workload across planned arms."""
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    ax.set(xlim=(0, 8), ylim=(0, 5.5))
    ax.axis("off")
    ax.text(0, 5.2, "先确定常驻分区，再决定是否借用推理卡余量", fontsize=15, weight="bold")
    ax.text(0, 4.75, "共同条件：另有 2 张训练卡；下面只画 4 张环境／推理卡。", fontsize=12)
    for y, label, roles in (
        (3.9, "3+1", ["环境", "环境", "环境", "vLLM"]),
        (2.9, "1+3", ["环境", "vLLM", "vLLM", "vLLM"]),
    ):
        ax.text(0, y + 0.25, label, fontsize=14, va="center")
        for i, role in enumerate(roles):
            x = 1 + 1.7 * i
            ax.add_patch(
                Rectangle(
                    (x, y), 1.45, 0.65, facecolor="#d0e7f2" if role == "环境" else "#f9dfb7", edgecolor="#627487"
                )
            )
            ax.text(x + 0.725, y + 0.42, role, ha="center", fontsize=12)
            if role == "环境":
                for slot in range(4):
                    ax.add_patch(Rectangle((x + 0.12 + slot * 0.31, y + 0.08), 0.24, 0.13, facecolor="#267ea8"))
            else:
                ax.add_patch(Rectangle((x + 0.12, y + 0.08), 0.8, 0.13, facecolor="#d58a26"))
                ax.add_patch(Rectangle((x + 0.96, y + 0.08), 0.34, 0.13, facecolor="white", edgecolor="#627487"))
    ax.text(0, 2.35, "静态：独立环境卡满了就等。动态：再评估 vLLM 卡的安全余量。", fontsize=12)
    ax.text(0, 1.9, "四组预先固定的验证，均为真实 PSRL 四步训练：", fontsize=12, weight="bold")
    rows = [
        ("① 3+1 静态，上限 4", "同窗口基线"),
        ("② 3+1 动态，上限 4", "检验共享增量，vLLM 保持常驻"),
        ("③ 3+1 静态，上限 5", "检验预测器推荐的更高容量"),
        ("④ 1+3 静态，上限 4", "检验缺失分区及预测排序"),
    ]
    for y, (label, reason) in zip((1.5, 1.1, 0.7, 0.3), rows):
        ax.text(0, y, label, fontsize=12)
        ax.text(3.65, y, reason, fontsize=12, color="#3a5e70")
    save(fig, "partition-prelaunch-design")


if __name__ == "__main__":
    results()
    timeline()
    design()
