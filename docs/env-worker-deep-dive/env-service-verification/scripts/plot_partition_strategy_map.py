"""Explain resident sharing versus replica lending and plot the completed 3+1 pair."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "experiments"
plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 12, "axes.unicode_minus": False})


def save(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=100)
    plt.close(fig)


def strategies():
    """Keep four physical GPUs fixed while showing different ownership layouts."""
    fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
    ax.set(xlim=(0, 8), ylim=(0, 6))
    ax.axis("off")
    ax.text(0.05, 5.72, "两种“动态”不同：借推理卡余量，还是腾出整张卡？", fontsize=14)
    ax.text(0.05, 5.3, "同样四张可分配卡；完整训练实验另有两张固定训练卡。", fontsize=11)
    layouts = [
        (4.3, "2+2 常驻共享", ["env", "env", "shared", "shared"], "两个模型都继续占卡，环境只用安全余量。"),
        (2.85, "2+2 → 3+1 借卡", ["env", "env", "env", "model"], "请求集中到最后一张，第三张副本排空并休眠后借出。"),
        (1.4, "3+1 静态", ["env", "env", "env", "model"], "从启动就如此分配，没有运行中排空、休眠和归还。"),
    ]
    for y, name, roles, caption in layouts:
        ax.text(0.05, y + 0.8, name, fontsize=12)
        for index, role in enumerate(roles):
            x = 0.12 + index * 1.96
            ax.add_patch(Rectangle((x, y), 1.75, 0.62, facecolor="#eef2f5", edgecolor="#627487"))
            if role == "env":
                ax.add_patch(Rectangle((x + 0.04, y + 0.04), 1.67, 0.54, facecolor="#a5cfe5"))
                ax.text(x + 0.875, y + 0.31, "独立环境", ha="center", va="center", fontsize=12)
            elif role == "shared":
                ax.add_patch(Rectangle((x + 0.04, y + 0.30), 1.67, 0.28, facecolor="#f3c785"))
                ax.add_patch(Rectangle((x + 0.04, y + 0.04), 1.67, 0.23, facecolor="#a5cfe5"))
                ax.text(x + 0.875, y + 0.44, "vLLM 常驻", ha="center", va="center", fontsize=11)
                ax.text(x + 0.875, y + 0.155, "环境余量", ha="center", va="center", fontsize=10)
            else:
                ax.add_patch(Rectangle((x + 0.04, y + 0.04), 1.67, 0.54, facecolor="#f3c785"))
                ax.text(x + 0.875, y + 0.31, "vLLM", ha="center", va="center", fontsize=12)
        ax.text(0.05, y - 0.29, caption, fontsize=11)
    ax.text(0.05, 0.48, "整卡借用已有组件实验；完整 PSRL 多步实验目前使用常驻余量共享。", fontsize=11)
    ax.text(0.05, 0.08, "色块表示资源角色，不按显存比例绘制；借出卡仍有少量驱动占用。", fontsize=10)
    save(fig, "partition-strategy-distinction")


def paired():
    root = BASE / "env-service-verification/results/partition-matrix-20261123"
    rows = [
        json.loads((root / name / "analysis.json").read_text()) for name in ("three-env-static", "three-env-dynamic")
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), layout="constrained", gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axes[0]
    for index, row in enumerate(rows):
        init = row["status"]["initialization_s"]
        step = row["timing_totals"]["timing_s/step"]
        extra = row["launch_to_cleanup_s"] - init - step
        left = 0
        for width, color, label in zip(
            (init, step, extra), ("#cad3df", "#438eb4", "#dfa04b"), ("初始化", "四步训练", "其余与清理")
        ):
            ax.barh(index, width, left=left, height=0.45, color=color, label=label if index == 0 else None)
            if width > 100:
                ax.text(
                    left + width / 2,
                    index,
                    f"{width:.1f}s",
                    ha="center",
                    va="center",
                    color="white" if label == "四步训练" else "#263340",
                )
            left += width
        ax.text(left + 12, index, f"{left:.1f}s", va="center", fontsize=11)
    ax.set(
        yticks=[0, 1], yticklabels=["3+1 静态 cap4", "3+1 常驻共享"], xlim=(0, 2020), xlabel="各组从启动开始计时（秒）"
    )
    ax.invert_yaxis()
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=3, frameon=False, fontsize=11)
    ax.set_title("新 3+1 配对：训练计时缩短 6.89%，完整启动至清理缩短 0.25%", fontsize=12, pad=38)
    ax.spines[["top", "right"]].set_visible(False)
    a = axes[1]
    a.axis("off")
    a.text(0, 0.88, "这轮没有收缩副本：唯一 vLLM 保持运行，9 条命令使用其余量。", fontsize=12)
    a.text(0, 0.61, "环境平均排队：87.22 → 66.57 秒；请求 P95：16.82 → 14.57 秒。", fontsize=12)
    a.text(0, 0.34, "两组均 4 step / 48 轨迹 / 96 命令 / 144 请求；资源释放已确认。", fontsize=11)
    a.text(0, 0.08, "顺序单次、动作有差异；cap5 强静态仍在执行，尚非全局最优比较。", fontsize=11)
    save(fig, "three-env-resident-pair")


if __name__ == "__main__":
    strategies()
    paired()
