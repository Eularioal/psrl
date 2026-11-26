"""Explain startup stages with archived worker registration failures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def main():
    """Separate process creation, Ray registration and actor readiness."""
    base = Path(__file__).resolve().parents[1]
    evidence = json.loads((base / "results/ray-startup-review-20261125/evidence.json").read_text())
    plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 12, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(8, 6), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 800), ylim=(600, 0))
    ax.axis("off")
    ax.text(25, 32, "Ray 启动排障：进程存在、注册成功、Actor 就绪是三件事", fontsize=16)
    ax.text(25, 65, "先定位失败阶段，再解释原因。下方时间线为11月23日真实日志。", fontsize=12)
    stages = (
        (25, "#dbe6f4", "① Driver 配置", "Hydra 查找配置", "缺data/data在此失败"),
        (220, "#f4e4cf", "② Python 启动", "导入Ray及其依赖", "慢文件查询可能在此"),
        (415, "#d5eae3", "③ 向Raylet注册", "回报worker已就绪", "超时后活进程也会被杀"),
        (610, "#e5dff0", "④ Actor 业务", "导入业务／运行初始化", "缺模块、gRPC冲突在此"),
    )
    for index, (x, color, title, line1, line2) in enumerate(stages):
        ax.add_patch(Rectangle((x, 88), 170, 108, facecolor=color, edgecolor="white"))
        for y, text in ((112, title), (148, line1), (177, line2)):
            ax.text(x + 8, y, text, fontsize=11 if y != 112 else 12)
        if index < 3:
            ax.add_patch(FancyArrowPatch((x + 171, 140), (x + 194, 140), arrowstyle="-|>", mutation_scale=14))
    ax.text(25, 229, "注册失败时间线：t＝0为15:32:13.426，三个worker各等30秒", fontsize=12)
    left, scale = 170, 5.35
    for value in (0, 30, 60, 90):
        x = left + value * scale
        ax.plot((x, x), (248, 443), color="#dce3eb", linewidth=0.8, zorder=0)
        ax.text(x, 464, str(value), ha="center", fontsize=11)
    for index, row in enumerate(evidence["retries_20261123"]):
        y = 255 + index * 48
        start = left + row["start_offset_s"] * scale
        width = row["duration_s"] * scale
        ax.text(150, y + 18, f"进程{index + 1} · {row['pid']}", ha="right", va="center", fontsize=11)
        ax.add_patch(Rectangle((start, y), width, 34, facecolor="#bc5b55"))
        ax.text(start + width / 2, y + 17, "活着，30秒未注册", ha="center", va="center", fontsize=11, color="white")
        ax.text(start + width + 4, y + 17, "终止", va="center", fontsize=11, color="#8e352e")
    row = evidence["successful_fourth_worker"]
    y = 399
    start = left + row["start_offset_s"] * scale
    width = row["core_worker_constructed_after_s"] * scale
    ax.text(150, y + 18, f"进程4 · {row['pid']}", ha="right", va="center", fontsize=11)
    ax.add_patch(Rectangle((start, y), width, 34, facecolor="#2e8a70"))
    ax.text(start + width + 7, y + 17, "CoreWorker", va="center", fontsize=11, color="#21644f")
    ax.text(680, 464, "秒", fontsize=11)
    ax.text(25, 497, "第4个约8.19秒进入CoreWorker，约87.75秒后Actor创建完成。", fontsize=12)
    ax.text(25, 534, "红条证实注册超时，不说明具体卡在哪个import。失败进程没有留下Python栈。", fontsize=11)
    ax.text(25, 566, "11月24日另一个worker的9.4秒栈停在文件查询，它未超时，不能替代历史失败栈。", fontsize=11)
    output = base.parent / "experiments/ray-startup-stages"
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
