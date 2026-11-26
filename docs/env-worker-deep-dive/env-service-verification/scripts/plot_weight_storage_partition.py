"""Explain live weight destinations and the measured static partition bottleneck."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "experiments"
plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 12, "axes.unicode_minus": False})


def save(fig, name):
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=100)
    plt.close(fig)


def memory():
    fig, ax = plt.subplots(figsize=(8, 5.6), layout="constrained")
    ax.set(xlim=(0, 8), ylim=(0, 5.6))
    ax.axis("off")

    def box(x, y, width, height, color, label):
        ax.add_patch(
            FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#607487")
        )
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=12)

    ax.text(0.1, 5.32, "在线权重同步：改 vLLM 的显存参数，环境参数是另一份", fontsize=14)
    box(0.15, 3.65, 1.95, 1.15, "#f7dfb9", "训练端 GPU\n优化器更新\n得到新策略参数")
    box(2.65, 3.65, 1.95, 1.15, "#e8dff4", "参数服务器\n主存中的\n策略参数缓冲区")
    box(5.15, 1.8, 2.65, 3.1, "#f2f5f8", "")
    ax.text(6.475, 4.64, "同一张共享 GPU", ha="center", fontsize=12)
    box(5.4, 3.3, 2.15, 1.02, "#b8d7ee", "vLLM 策略参数\n旧版本 → 新版本\n生成暂停后写入")
    box(5.4, 2.03, 2.15, 0.95, "#c7e5d3", "环境模型参数\n独立张量与存储\n此次同步不写这里")
    for start, end in (((2.15, 4.2), (2.59, 4.2)), ((4.65, 4.0), (5.34, 3.85))):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2, "color": "#267ea8"})
    ax.text(4.76, 4.3, "NIXL", fontsize=10)
    box(
        0.15,
        1.95,
        4.45,
        1.03,
        "#e6e8eb",
        "磁盘模型文件 / checkpoint\n用于初次加载、保存与恢复\n本次在线同步不先写磁盘再让 vLLM 重载",
    )
    ax.text(0.15, 1.25, "独立存储解决读写冲突：改策略参数，不会把环境参数一起改掉。", fontsize=12)
    ax.text(0.15, 0.78, "同卡仍竞争容量与带宽：能正确共存，不保证一起执行更快。", fontsize=12)
    ax.text(0.15, 0.28, "本实验：策略 Qwen2.5-3B，环境 Qwen3-4B；nixl_cpu 路径。", fontsize=11)
    save(fig, "weight-sync-live-storage")


def partition():
    data = json.loads(
        (BASE / "env-service-verification/results/sync-followup-20261119/three-way-summary.json").read_text()
    )
    rows = [data[0], data[2]]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.7), layout="constrained", gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    for index, row in enumerate(rows):
        total = row["timing_totals"]["timing_s/step"]
        wait = row["timing_totals"]["timing_s/wait_for_gen"]
        ax.barh(index, wait, color="#438eb4", height=0.43, label="训练器等待批次就绪" if index == 0 else None)
        ax.barh(
            index,
            total - wait,
            left=wait,
            color="#dfa04b",
            height=0.43,
            label="更新等其余步骤" if index == 0 else None,
        )
        ax.text(wait / 2, index, f"等待 {wait:.1f} 秒", ha="center", va="center", color="white")
        ax.text(total + 12, index, f"总计 {total:.1f} 秒", va="center", fontsize=11)
    ax.set(
        yticks=[0, 1],
        yticklabels=["2 环境＋2 推理", "3 环境＋1 推理"],
        xlim=(0, 1600),
        xlabel="四步训练累计时间（秒），不含初始化与清理",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=2, frameon=False, fontsize=11)
    ax.set_title("历史静态池对照：约 352 秒的差额出现在等待批次这一段", fontsize=13, pad=37)
    lower = axes[1]
    lower.axis("off")
    lower.text(0, 0.9, "多给环境一张卡，少给推理一个副本，实际改变了什么？", fontsize=12)
    for y, label, value, reason in (
        (0.64, "环境名额", "8 → 12", "每张卡上限仍为 4，名额增加 50%"),
        (0.39, "命令平均排队", "157.68 → 81.22 秒", "环境等待明显减少"),
        (0.14, "请求平均延迟", "6.36 → 6.39 秒", "一个推理副本未表现出明显退化"),
    ):
        lower.text(0, y, label, fontsize=12)
        lower.text(0.25, y, value, fontsize=12, color="#176e97")
        lower.text(0.56, y, reason, fontsize=11)
    fig.supxlabel("各 4 step / 48 轨迹 / 96 命令 / 144 请求；顺序单次、生成动作有差异。", fontsize=10)
    save(fig, "static-partition-bottleneck")


if __name__ == "__main__":
    memory()
    partition()
