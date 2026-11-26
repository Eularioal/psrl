"""Render the current Env execution layout and its four reading destinations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def canvas(height, title, subtitle):
    """Use screen pixels so text remains readable at the embedded size."""
    fig = plt.figure(figsize=(8, height / 100), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 800), ylim=(height, 0))
    ax.axis("off")
    ax.text(25, 34, title, fontsize=17, color="#17324d")
    ax.text(25, 64, subtitle, fontsize=11, color="#445b70")
    return fig, ax


def rect(ax, x, y, width, height, color, label="", size=12):
    """Show a resource region with a centered explanation."""
    ax.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="white", linewidth=1.5))
    if label:
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=size, color="#17324d")


def arrow(ax, start, end):
    """Keep arrows limited to calls and data movement."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, color="#52697d"))


def save(fig, name):
    """Retain both the screen raster and editable vector."""
    output = Path(__file__).resolve().parents[2] / "experiments" / name
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=100)
    plt.close(fig)


def layout():
    """Explain model routing, command placement and retained host resources."""
    fig, ax = canvas(
        600,
        "Env 现状：会话留在节点，GPU 按协议申请与归还",
        "本图为显式启用后的单卡 direct 命令池。共享与整卡调整均需另外配置。",
    )
    ax.text(25, 98, "① 同一条 Agent 轨迹，交替请求模型和执行工具", fontsize=12)
    rect(ax, 25, 115, 350, 85, "#e5eef9")
    rect(ax, 425, 115, 350, 85, "#e4f1eb")
    ax.text(43, 142, "模型请求 → SMG → vLLM 副本", fontsize=12)
    ax.text(43, 178, "返回回答，Agent 决定下一条操作", fontsize=11)
    ax.text(442, 142, "GPU 工具 → EnvRouter＋资源账本", fontsize=12)
    ax.text(442, 178, "检查预算、占用、权限与物理归还", fontsize=11)
    rect(ax, 25, 219, 750, 217, "#f2f5f8")
    ax.text(42, 247, "② 容器在原节点，下一条命令可重新选择同机候选 GPU", fontsize=12)
    rect(ax, 42, 269, 215, 148, "#dfe3ed")
    ax.text(58, 295, "会话资源持续保留", fontsize=12)
    for y, label in ((308, "容器与文件"), (349, "CPU 配额与主存")):
        rect(ax, 57, y, 185, 34, "#c7cfe1", label, 11)
    arrow(ax, (259, 344), (291, 344))
    for x, title in ((297, "独立环境卡 1"), (457, "独立环境卡 2")):
        ax.text(x, 288, title, fontsize=11)
        rect(ax, x, 301, 145, 86, "#bddfce")
        rect(ax, x + 8, 310, 129, 30, "#79b69c", "活动命令预约", 11)
        rect(ax, x + 8, 348, 129, 30, "#eaf5ee", "可用容量", 11)
    ax.text(617, 288, "显式共享推理卡", fontsize=11)
    rect(ax, 617, 301, 140, 86, "#cce0f4")
    rect(ax, 625, 309, 124, 34, "#99bfe1", "vLLM 常驻", 11)
    rect(ax, 625, 351, 124, 28, "#b8dfcd", "安全余量", 11)
    ax.text(297, 414, "先查独立卡，全部不满足才查推理余量", fontsize=11)
    ax.text(25, 466, "可选整卡调整：推理排空 → 休眠 → 卡给环境。借出已实测，恢复未触发。", fontsize=11)
    ax.text(25, 497, "③ 轨迹完成后先入队，满足题目组和版本条件才训练", fontsize=12)
    rect(ax, 25, 510, 200, 40, "#ecdfbc", "完整轨迹＋评分", 11)
    rect(ax, 280, 510, 235, 40, "#d9d1eb", "成批进行策略训练", 11)
    rect(ax, 570, 510, 205, 40, "#cce0f4", "新权重发给推理", 11)
    arrow(ax, (228, 530), (277, 530))
    arrow(ax, (518, 530), (567, 530))
    ax.text(25, 582, "Env Service 默认关闭。不搬迁运行中 CUDA 进程，不自动借用训练卡。", fontsize=11)
    save(fig, "env-current-layout")


def reading():
    """Connect each reader question to its maintained primary document."""
    fig, ax = canvas(
        560, "回顾 Env：先读四处，按问题深入", "现状、效果、下一步分别维护，实验报告负责给出具体对照和原始证据。"
    )
    items = (
        (
            25,
            85,
            "#e4eef9",
            "01 目前怎样工作",
            "Env 架构：现状与完整执行流程",
            ("谁申请、谁执行、谁确认归还", "容器／CPU／主存与 GPU 怎样分账", "已实现、默认关闭、接入限制"),
        ),
        (
            415,
            85,
            "#e3f1e9",
            "02 相比基线有什么优势",
            "实验总览：调度收益与证据范围",
            ("主基线：episode 静态池满了排队", "历史 command 与旧提交分别标注", "完整作业是否更快，失败为何发生"),
        ),
        (
            25,
            300,
            "#f7ebd8",
            "03 接下来可以做什么",
            "调度研究：研究全景与证据详解",
            ("补齐共同协议的配对归因", "再看批次依赖、准备阶段与同步", "每个方向给出边界和验收目标"),
        ),
        (
            415,
            300,
            "#ece7f6",
            "04 具体做过哪些实验",
            "分区九轮实验＋实验目录",
            ("每轮问题、实际配置与相同工作量", "训练时间和初始化清理分开报告", "跳到 JSON、日志与复现脚本"),
        ),
    )
    for x, y, color, title, document, lines in items:
        rect(ax, x, y, 360, 190, color)
        ax.text(x + 16, y + 30, title, fontsize=14, color="#17324d")
        ax.text(x + 16, y + 60, document, fontsize=11, color="#355b7b")
        for i, line in enumerate(lines):
            ax.text(x + 16, y + 98 + i * 31, line, fontsize=11, color="#263f51")
    ax.text(25, 530, "Episode 补测已完成；训练与完整作业分开比，单次测量仍需注明边界。", fontsize=12, color="#8e4828")
    save(fig, "env-reading-guide")


def main():
    """Render documentation only without accessing GPU services."""
    plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 12, "svg.fonttype": "none"})
    layout()
    reading()


if __name__ == "__main__":
    main()
