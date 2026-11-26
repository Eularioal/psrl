"""Plot handover measurements, the observation window, and the dependency comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/handover-rerun-20261125"
FIGURES = ROOT.parent / "experiments"
BLUE = "#386eaa"
TEAL = "#17806a"
RED = "#aa493d"
SLATE = "#304159"


def save(fig, name):
    """Save the screen version and a matching editable figure."""
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=100, facecolor="white")
    plt.close(fig)


def imports():
    """Compare elapsed time with CPU time using the exact same package bytes."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(left=0.265, right=0.94, top=0.77, bottom=0.26)
    fig.text(0.06, 0.91, "同一份 Transformers，文件放在哪会影响导入吗？", fontsize=17, color=SLATE)
    fig.text(0.06, 0.83, "私有节点实测，每次启动新 Python 进程。只移动 2434 个 Transformers 文件。", fontsize=11)
    rows = [
        ("共享目录 · 第一次", "shared-first", BLUE),
        ("节点本地 · 第一次", "local-first", TEAL),
        ("节点本地 · 第二次", "local-second", TEAL),
        ("共享目录 · 第二次", "shared-second", BLUE),
    ]
    for y, (_, filename, color) in enumerate(rows):
        d = json.loads((RUN / f"transformers-{filename}.json").read_text())
        wall, cpu = d["import_wall_s"], d["cpu_s"]
        ax.barh(y, wall, color=color, height=0.45, zorder=3)
        ax.text(wall + 0.15, y, f"{wall:.2f} s", va="center", fontsize=11, color=color)
        ax.scatter(cpu, y, color="#222d3d", edgecolors="white", marker="D", s=42, zorder=4)
    ax.set(xlim=(0, 10.3), yticks=list(range(4)), yticklabels=[row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("秒")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.grid(axis="x", alpha=0.24, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.06, 0.115, "色条是等待到导入完成的时间，黑色菱形是各线程累计的 CPU 时间。", fontsize=11)
    fig.text(0.06, 0.056, "两种位置的 CPU 时间均约 7.5 秒。多线程 CPU 时间可以超过实际经过时间。", fontsize=11)
    save(fig, "handover-import-profile")


def windows():
    """Show how fresh observations and repeated old statistics differ on the same time axis."""
    d = json.loads((RUN / "historical-window-reconstruction.json").read_text())
    best = d["longest_low_window"]
    start = best["start_s"]
    end = best["end_s"]
    following = next(gap for gap in d["decision_gaps"] if gap["start_s"] == end)
    valid = [t - start for t in d["decision_times_s"] if start <= t <= following["end_s"]]
    elapsed = end - start
    max_age = d["max_gap_s"]
    required = d["policy_window_s"]
    next_sample = following["end_s"] - start
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=0.225, right=0.94, top=0.74, bottom=0.28)
    fig.text(0.06, 0.92, "为什么观察了 24 秒，最后还是没借卡？", fontsize=17, color=SLATE)
    fig.text(0.06, 0.835, f"旧组实测。横轴 0 是本次低负载窗口起点，首条轨迹已开始 {start:.2f} 秒。", fontsize=11)
    ax.broken_barh([(0, elapsed)], (1.765, 0.47), color="#d9ebf7")
    ax.broken_barh([(elapsed, next_sample - elapsed)], (1.765, 0.47), color="#f3d9d5")
    ax.scatter(valid, [2] * len(valid), s=30, color=BLUE, zorder=4)
    ax.plot([0, next_sample], [1, 1], color=TEAL, linewidth=5, solid_capstyle="round")
    ax.text(2, 0.68, "窗口附近 SMG 仍持续记录，排队为 0", fontsize=10, color=TEAL)
    ax.axvline(required, linestyle="--", color=RED, linewidth=1.3)
    ax.text(required + 0.8, 2.57, f"{required} 秒借卡门槛", color=RED, fontsize=11)
    ax.text(elapsed - 1.1, 2.30, f"{elapsed:.2f} 秒", color=BLUE, ha="right", fontsize=10.5)
    ax.text(next_sample, 1.52, f"{next_sample:.2f} 秒", ha="right", color=BLUE, fontsize=10.5)
    ax.set(xlim=(-1.1, 48), ylim=(0.42, 2.93), yticks=[1, 2], yticklabels=["独立网关日志", "控制器有效样本"])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("从该次观察窗口起点经过的秒数")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.06,
        0.122,
        f"两个有效样本相隔 {following['duration_s']:.2f} 秒，超过 {max_age} 秒间隔上限，观察窗口清零。",
        fontsize=11,
    )
    fig.text(0.06, 0.057, "新逻辑每轮查一次当前请求数。引擎本轮答复为 0 时，零负载证据可以继续累积。", fontsize=11)
    save(fig, "handover-window-reset")


def pair():
    """Show the measured GPU handover and achieved workload separately."""
    comparison_path = RUN / "comparison.json"
    if not comparison_path.exists():
        return
    comparison = json.loads(comparison_path.read_text())
    arms = comparison["arms"]
    if any(not value["physical_release_verified"] or not value["transitions"] for value in arms.values()):
        return
    fig, (timing, completed) = plt.subplots(2, 1, figsize=(8, 5.7), height_ratios=(2, 3))
    fig.subplots_adjust(left=0.23, right=0.87, top=0.80, bottom=0.13, hspace=0.96)
    gated, overlap = [arm["transitions"][0]["last_transition_event_after_decision_s"] for arm in arms.values()]
    title = f"GPU2 交接耗时由 {gated:.2f} 秒降至 {overlap:.2f} 秒"
    fig.text(0.06, 0.935, title, fontsize=16, color=SLATE)
    fig.text(0.06, 0.865, "真实 PSRL 配对。每组计划两步训练，初始均为两张环境卡与两张推理卡。", fontsize=11)
    names = ("等待交接", "重叠交接")
    timing.set_title("从决定借出到新环境命令可以进卡（秒）", loc="left", fontsize=12)
    for row, arm in enumerate(arms.values()):
        switch = arm["transitions"][0]
        total = switch["last_transition_event_after_decision_s"]
        pause = next(event for event in switch["controls"] if event["method"] == "pause_generation")
        wait = pause["gate_wait_s"]
        timing.barh(row, wait, color=RED, height=0.41, label="等旧命令归还" if row == 0 else None)
        timing.barh(row, total - wait, left=wait, color=BLUE, height=0.41, label="其他借卡动作" if row == 0 else None)
        if wait >= 15:
            timing.text(wait / 2, row, f"等待 {wait:.2f} s", color="white", ha="center", va="center")
        timing.text(total + 1.5, row, f"{total:.2f} s", va="center", fontsize=10)
    timing.set(yticks=(0, 1), yticklabels=names, xlim=(0, max(gated, overlap) * 1.15))
    timing.invert_yaxis()
    timing.legend(loc="upper left", bbox_to_anchor=(-0.025, -0.28), ncol=2, frameon=False, fontsize=10)
    completed.set_title("两组实际完成的工作量（占计划比例）", loc="left", fontsize=12)
    for row, (key, planned) in enumerate((("steps", 2), ("episodes", 24), ("commands", 48))):
        for offset, arm, color in zip((-0.19, 0.19), arms.values(), (SLATE, TEAL), strict=True):
            value = arm[key]
            fraction = value / planned
            completed.barh(row + offset, fraction, height=0.28, color=color)
            completed.text(fraction + 0.025, row + offset, f"{value}/{planned}", va="center", fontsize=10, color=color)
    completed.set(yticks=(0, 1, 2), yticklabels=("训练 step", "轨迹", "GPU 命令"), xlim=(0, 1.15))
    completed.set_xticks((0, 0.25, 0.5, 0.75, 1), ("0%", "25%", "50%", "75%", "100%"))
    completed.invert_yaxis()
    for ax in (timing, completed):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.23)
    status = "两组均完成" if comparison["same_completed_workload"] else "重叠组在凑齐首批轨迹前退出。"
    fig.text(0.06, 0.039, f"深蓝为等待组，绿色为重叠组。{status}", fontsize=11)
    save(fig, "handover-overlap-timings")


def completed_handover():
    """Compare the GPU2 timelines from two runs that completed the same workload."""
    retry = ROOT / "results/handover-timeout-20261125"
    comparison = json.loads((retry / "comparison.json").read_text())
    arms = comparison["arms"]
    workload = arms["handover_wait"]["workload"]
    if workload != arms["handover_overlap"]["workload"] or not comparison["physical_release_verified"]:
        raise ValueError("The figure requires equal completed work and verified resource cleanup")
    steps, episodes, commands, requests, _ = workload
    waiting = json.loads((RUN / "handover-timing.json").read_text())["gated"]["transitions"][0]
    overlap = json.loads((retry / "handover-timing.json").read_text())["transitions"][0]
    pause = next(event for event in waiting["controls"] if event["method"] == "pause_generation")
    old_command = overlap["tracked_sleep_owners"][0]
    old_return = old_command["returned_relative_to_decision_s"]
    shared_sample = overlap["old_and_new_ownership_sample"]
    shared_at = shared_sample["at"] - overlap["decision_at"]
    shared_count = shared_sample["new_sleeping_commands"]
    wait_end = pause["started_at"] - waiting["decision_at"] + pause["gate_wait_s"]
    axis_end = 120

    fig, ax = plt.subplots(figsize=(8, 3.7))
    fig.subplots_adjust(left=0.21, right=0.94, top=0.76, bottom=0.21)
    fig.text(0.055, 0.91, "同样跑完两步，GPU2 何时向新环境命令开放？", fontsize=17, color=SLATE)
    fig.text(
        0.055,
        0.815,
        f"两组各完成 {steps} 步训练、{episodes} 条轨迹、{commands} 条 GPU 命令、{requests} 次模型请求",
        fontsize=11,
        color=SLATE,
    )
    for y, transition in ((2.35, waiting), (1.04, overlap)):
        available = transition["last_transition_event_after_decision_s"]
        ax.barh(y, available, height=0.37, color=BLUE, zorder=2)
        ax.barh(y, axis_end - available, left=available, height=0.37, color="#d5ebe5", zorder=2)
        ax.scatter(available, y, color=TEAL, edgecolor="white", linewidth=0.9, s=65, zorder=4)
        ax.text(available + 1.7, y + 0.28, f"{available:.2f} 秒开放", color=TEAL, fontsize=11, va="bottom")
    ax.barh(2.35, wait_end, height=0.37, color="#f1d8cf", zorder=3)
    ax.text(2, 2.35, f"等待原命令归还 {pause['gate_wait_s']:.2f} 秒", color=RED, va="center")
    ax.text(shared_at + 1.5, 1.04, f"{shared_at:.2f} 秒时已有 {shared_count} 条新命令获准", color=TEAL, va="center")
    ax.hlines(0.48, 0, old_return, color=RED, linewidth=1.65, zorder=3)
    ax.scatter(old_return, 0.48, color=RED, s=25, zorder=4)
    ax.text(2, 0.32, f"原命令继续持卡，于 {old_return:.2f} 秒归还", color=RED, va="top")

    ax.set(
        xlim=(0, axis_end + 3),
        ylim=(-0.05, 3.05),
        yticks=(2.35, 1.04),
        yticklabels=("等待交接", "重叠补跑"),
        xlabel="从决定借出 GPU2 起经过的秒数",
    )
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#adb9c6")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#d8dfe7")
    fig.text(0.055, 0.04, "蓝段为模型退场和显存核验，浅绿段为容量已开放。两组各运行一次，时限有差异。", fontsize=10)
    save(fig, "handover-completed-work")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-only", action="store_true", help="Render the completed-workload GPU2 timeline")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 11, "svg.fonttype": "none"})
    if args.completed_only:
        completed_handover()
    else:
        imports()
        windows()
        pair()
