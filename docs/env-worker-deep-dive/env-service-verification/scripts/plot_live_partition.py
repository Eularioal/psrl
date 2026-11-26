"""Draw the integrated partition lifecycle and its independent ownership layers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 11, "axes.unicode_minus": False})
fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
ax.set(xlim=(0, 8), ylim=(0, 6))
ax.axis("off")
ax.text(0.05, 5.75, "先调整整卡比例，再使用常驻推理余量", fontsize=15)
ax.text(0.05, 5.4, "四张可分配卡，另有两张固定训练卡；代码已接入，实机验证待完成。")
for y, title, labels, colors in (
    (
        4.25,
        "① 初始 2+2：环境持续拥堵，单副本容量仍足够",
        ["环境", "环境", "vLLM A", "vLLM B"],
        ["#a9d2e5"] * 2 + ["#f4c888"] * 2,
    ),
    (
        2.7,
        "② A 排空并休眠后为 3+1：先用三张环境卡，仍拥堵再共享 B",
        ["环境", "环境", "借来的环境卡", "B 常驻\n＋环境余量"],
        ["#a9d2e5"] * 3 + ["#c9d7a0"],
    ),
):
    ax.text(0.05, y + 0.75, title)
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 0.1 + i * 1.98
        ax.add_patch(Rectangle((x, y), 1.78, 0.6, facecolor=color, edgecolor="#60768a"))
        ax.text(x + 0.89, y + 0.3, label, ha="center", va="center", fontsize=12)
ax.text(0.05, 3.85, "停止给 A 派新请求 → 等在途请求结束 → 休眠 → 核实显存释放")
ax.text(0.05, 2.3, "A 不处理请求也不参与常规同步；B 继续生成并更新策略权重。")
ax.text(0.05, 1.85, "③ 推理需求持续回升：归还 A 的卡，刷新版本后恢复服务", fontsize=12)
steps = [
    (0.1, 1.6, "关闭新环境准入", "#a9d2e5"),
    (2.0, 1.6, "等已有命令归还", "#e8b6a9"),
    (3.9, 1.6, "唤醒＋拉取新权重", "#f4c888"),
    (5.8, 1.6, "公布版本＋接请求", "#b6d7ba"),
]
for x, w, label, color in steps:
    ax.add_patch(Rectangle((x, 1.15), w, 0.45, facecolor=color, edgecolor="#60768a"))
    ax.text(x + w / 2, 1.375, label, ha="center", va="center", fontsize=10)
ax.text(0.05, 0.72, "主机账本不搬家：原容器、CPU、主存、文件保留；逐命令重新选择 GPU。")
ax.text(0.05, 0.3, "同步重叠只放行常驻模型的受控权重更新；休眠与唤醒仍等待物理归还。")
out = Path(__file__).resolve().parents[2] / "experiments"
for suffix in ("png", "svg"):
    fig.savefig(out / f"live-partition-lifecycle.{suffix}", dpi=100)
plt.close(fig)
