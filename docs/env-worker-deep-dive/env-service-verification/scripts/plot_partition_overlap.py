"""Explain staged capacity admission during a optional model retirement."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    """Draw the same GPU before, during and after a hypothetical handover."""
    plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 11})
    fig = plt.figure(figsize=(8, 5.8), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 800), ylim=(580, 0))
    ax.axis("off")
    ax.text(30, 32, "借卡过渡：环境继续跑，容量确认后再扩大", fontsize=17)
    ax.text(30, 62, "可选实现，CPU 回归通过。教学假设：可用预算 90 GiB，每条环境命令预约 16 GiB。")
    colors = {"model": "#8ab3df", "margin": "#f2c57c", "env": "#8dceb1", "free": "#e4e8ed"}
    rows = [
        (
            100,
            "① 决定借卡：关闭该副本的新模型请求，环境命令 B 已经在跑",
            [(40, "vLLM 40", "model"), (16, "预留 16", "margin"), (16, "B 16", "env"), (18, "可用 18", "free")],
            "仍按模型常驻预算判断：90 - 40 - 16 - 16 = 18 GiB，可再接一条 C。",
        ),
        (
            230,
            "② 模型请求排空中：B 继续执行，C 使用已有余量进入",
            [
                (40, "vLLM 40", "model"),
                (16, "预留 16", "margin"),
                (16, "B 16", "env"),
                (16, "C 16", "env"),
                (2, "", "free"),
            ],
            "只剩 2 GiB，D、E、F 继续等待；不能提前使用模型尚未释放的 40 GiB。",
        ),
        (
            360,
            "③ 模型请求完成且休眠释放已核验：保留 B、C，再接入 D、E、F",
            [(4, "4", "margin"), *[(16, f"{name} 16", "env") for name in "BCDEF"], (6, "6", "free")],
            "模型退出后扩大预算：5 × 16 + 4 = 84 GiB，剩余 6 GiB；B、C 不需先退出。",
        ),
    ]
    for y, title, blocks, caption in rows:
        ax.text(30, y, title, fontsize=12)
        x = 40
        for size, label, kind in blocks:
            width = size * 8
            ax.add_patch(Rectangle((x, y + 18), width, 45, facecolor=colors[kind], edgecolor="white"))
            ax.text(x + width / 2, y + 41, label, ha="center", va="center")
            x += width
        ax.text(30, y + 88, caption, fontsize=11)
    for x, (kind, label) in zip(
        (35, 220, 405, 590),
        (("model", "模型实际占用"), ("margin", "安全预留"), ("env", "环境预约"), ("free", "可用预算")),
        strict=True,
    ):
        ax.add_patch(Rectangle((x, 484), 20, 17, facecolor=colors[kind]))
        ax.text(x + 28, 498, label)
    ax.text(30, 534, "睡眠阶段预留 4 GiB 仅为教学假设，实际需校准。旧环境预约不能在切换时清零。")
    ax.text(30, 559, "显存可容纳并不保证速度更快：新增命令可能与推理尾部、权重拷贝竞争计算或带宽。")
    output = Path(__file__).resolve().parents[2] / "experiments/partition-overlap-proposal"
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix))
    plt.close(fig)


if __name__ == "__main__":
    main()
