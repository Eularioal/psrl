"""Plot the completed two-step command partition comparison from recorded evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_startup_repeat(base):
    """Compare the completed repeat with its original elastic reference."""
    root = base / "results/elastic-startup-20261124/elastic"
    if not (root / "analysis.json").exists():
        return
    state = json.loads((root / "supervisor.json").read_text())
    if state.get("status") != "complete" or not state.get("physical_release_verified"):
        return
    current = json.loads((root / "analysis.json").read_text())
    previous = json.loads((base / "results/live-partition-20261123/analysis.json").read_text())["elastic"]
    fig, ax = plt.subplots(figsize=(8, 4.3), dpi=100)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.75, bottom=0.36)
    for y, row in enumerate((previous, current)):
        init = row["status"]["initialization_s"]
        training = row["timing_totals"]["timing_s/step"]
        total = row["launch_to_cleanup_s"]
        ax.barh(y, init, color="#c3ced9", label="初始化" if y == 0 else None)
        ax.barh(y, training, left=init, color="#347cb7", label="两步训练" if y == 0 else None)
        ax.barh(
            y, total - init - training, left=init + training, color="#e6a148", label="其余与清理" if y == 0 else None
        )
        ax.text(init / 2, y, f"{init:.1f}", ha="center", va="center")
        ax.text(init + training / 2, y, f"{training:.1f}", ha="center", va="center", color="white")
        ax.text(total + 15, y, f"{total:.1f}", va="center")
    ax.set(
        yticks=[0, 1],
        yticklabels=["原整卡组（已借卡）", "启动复测（未借卡）"],
        xlabel="本组 runner 开始至物理清理完成（秒）",
    )
    ax.set_xlim(0, max(previous["launch_to_cleanup_s"], current["launch_to_cleanup_s"]) * 1.16)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.3), ncol=3, frameon=False, fontsize=11)
    fig.suptitle("整卡组复测：启动缩短，本轮未触发借卡", fontsize=16, y=0.96)
    fig.text(0.03, 0.83, "两组均为 2 步、24 轨迹、48 GPU 命令、72 模型请求，初始 2 环境＋2 推理。", fontsize=11)
    fig.text(0.03, 0.09, "单次顺序复测，保留全部启动与清理耗时；差异仍可能受缓存、生成和负载影响。", fontsize=11)
    fig.text(0.03, 0.04, "本轮仅常驻共享实际生效；模型排空期间继续准入的提案尚未实现或实测。", fontsize=11)
    output = base.parent / "experiments/elastic-startup-repeat"
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix))
    plt.close(fig)


def plot_completed_episode(base, current=False):
    """Include the longer episode deadline only after equal work completes."""
    path = base / "results/episode-static-1200-20261124/analysis.json"
    if not path.exists():
        return
    episode = json.loads(path.read_text())
    if not episode.get("complete_workload"):
        return
    reference = json.loads((base / "results/live-partition-20261123/analysis.json").read_text())
    rows = [episode, reference["static"], reference["resident"]]
    labels = ["Episode 静态池", "Command 静态池", "Command 常驻共享"]
    if current:
        for path, label in (
            ("handover-rerun-20261125/gated", "整卡借用（等旧命令）"),
            ("handover-timeout-20261125/overlap", "整卡借用（重叠）"),
        ):
            arm = base / "results" / path
            if not (arm / "analysis.json").exists():
                continue
            supervisor = json.loads((arm / "supervisor.json").read_text())
            result = json.loads((arm / "analysis.json").read_text())
            if (
                supervisor["status"] == "complete"
                and supervisor.get("physical_release_verified")
                and len(result["steps"]) == 2
                and result["episodes_completed"] == 24
                and result["commands_completed"] == 48
                and result["queries_completed"] == 72
            ):
                rows.append(result)
                labels.append(label)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    fig.subplots_adjust(left=0.28, right=0.96, top=0.78, bottom=0.28)
    for y, row in enumerate(rows):
        init = row["status"]["initialization_s"]
        training = row["timing_totals"]["timing_s/step"]
        total = row["launch_to_cleanup_s"]
        ax.barh(y, init, color="#c3ced9", label="初始化" if y == 0 else None)
        ax.barh(y, training, left=init, color="#347cb7", label="两步训练" if y == 0 else None)
        ax.barh(
            y, total - init - training, left=init + training, color="#e6a148", label="其余与清理" if y == 0 else None
        )
        ax.text(init / 2, y, f"{init:.1f}", ha="center", va="center")
        ax.text(init + training / 2, y, f"{training:.1f}", ha="center", va="center", color="white")
        ax.text(total + 12, y, f"{total:.1f}", va="center", fontsize=11)
    ax.set(
        yticks=list(range(len(rows))),
        yticklabels=labels,
        xlim=(0, max(r["launch_to_cleanup_s"] for r in rows) * 1.15),
        xlabel="本组 runner 开始至物理清理完成（秒）",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.36, -0.18), ncol=3, frameon=False, fontsize=11)
    title = "完成相同任务后，哪个方案用时更短？" if current else "Episode 补测：训练收益与完整作业耗时分别比较"
    fig.suptitle(title, fontsize=15, y=0.97)
    fig.text(0.04, 0.86, "每组 2 步、24 轨迹、48 GPU 命令、72 模型请求，均已完成并物理释放。", fontsize=11)
    fig.text(
        0.04,
        0.04,
        (
            "各组为不同时间运行的单次观测。代码快照、启动阶段和超时设置存在差异。\n"
            "训练耗时包含等待轨迹。完整用时还包括初始化与实际资源清理。"
            if current
            else "单次顺序测量。Episode 单轨迹上限 1200 秒，历史 Command 为 720 秒。\n"
            "同任务与资源预算，生成 token 略有差异，完整耗时没有扣除初始化波动。"
        ),
        fontsize=11,
    )
    name = "partition-current-comparison" if current else "episode-partition-comparison"
    output = base.parent / "experiments" / name
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def plot_episode(base):
    """Explain the observed reservation queue with three real trajectories."""
    root = base / "results/episode-static-20261123/episode/fixed"
    episodes = {
        row["uid"]: row for path in root.glob("episodes/*/episode.json") if (row := json.loads(path.read_text()))
    }
    if not episodes:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.68, bottom=0.30)
    for index, uid in enumerate(("11", "16", "12")):
        row = episodes[uid]
        start = row["started_at"]
        first_reply = row["queries"][0]["finished_at"] - start
        granted = row["lease_granted_at"] - start
        finished = row["finished_at"] - start
        ax.barh(
            index,
            granted - first_reply,
            left=first_reply,
            color="#e7b762",
            height=0.62,
            label="等待环境准入" if index == 0 else None,
        )
        ax.barh(
            index,
            finished - granted,
            left=granted,
            color="#adc8e4",
            height=0.62,
            label="持续持有 GPU 预算" if index == 0 else None,
        )
        for command in row["commands"]:
            left = command["started_at"] - start
            width = command["finished_at"] - command["started_at"]
            ax.barh(
                index,
                width,
                left=left,
                color="#338477",
                height=0.34,
                label="已完成的 GPU 命令" if index == 0 and command["mode"] == "train" else None,
            )
            ax.text(
                left + width / 2,
                index,
                "适配" if command["mode"] == "train" else "验证",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
            )
        if index:
            ax.text(
                (first_reply + granted) / 2,
                index,
                f"等待 {granted - first_reply:.1f} 秒",
                ha="center",
                va="center",
                fontsize=11,
            )
        else:
            ax.text(finished + 10, index, f"{finished:.1f} 秒完成", va="center", fontsize=11)
    ax.axvline(720, color="#ba4040", linestyle="--")
    ax.text(712, -0.53, "720 秒上限", ha="right", color="#ba4040", fontsize=11)
    ax.set(
        yticks=[0, 1, 2],
        yticklabels=["轨迹 11", "轨迹 16", "轨迹 12"],
        xlim=(0, 790),
        xlabel="从各自轨迹开始计时（秒，三者开始时刻相差不足 1 秒）",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.47, -0.22), ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Episode 基线：后排任务等到拿卡，剩余时间已不足", fontsize=16, y=0.96)
    fig.text(
        0.04,
        0.80,
        "两张环境卡各最多持有 5 个会话，24 条轨迹分成约 10／10／4 三批。\n"
        "下图各取一条真实轨迹，均在 GPU4，浅蓝空隙也持续保留显存预算。",
        fontsize=12,
    )
    fig.text(
        0.04,
        0.035,
        "轨迹 12 等待 541.7 秒后准入，适配完成，验证期间达到 720 秒上限。\n"
        "超时后继续清理至约 732.5 秒。整组只完成 1／2 训练 step，不计算两步加速比。",
        fontsize=11,
    )
    output = base.parent / "experiments/episode-partition-queue"
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def plot_startup(base):
    """Show measured startup intervals and the location of registration failures."""
    data = json.loads((base / "results/live-partition-20261123/startup-audit.json").read_text())
    stages = ["start", "task_process_started", "ps_manager", "coordinator", "weights_ready", "first_episode"]
    labels = ["主控进程启动前", "至参数服务就绪", "至推理协调器就绪", "至初始权重就绪", "至首条轨迹"]
    colors = ["#b55c42", "#507fad", "#39a38b", "#d4a235", "#8a73b0"]
    fig, ax = plt.subplots(figsize=(8, 5.1), dpi=100)
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.36, top=0.70)
    for y, name in enumerate(("static", "elastic")):
        values = data[name]["elapsed_from_start_s"]
        for index, (start, end) in enumerate(zip(stages, stages[1:])):
            width = values[end] - values[start]
            ax.barh(y, width, left=values[start], color=colors[index], label=labels[index] if y == 0 else None)
            ax.text(values[start] + width / 2, y, f"{width:.1f}", ha="center", va="center", color="white")
        ax.text(values["first_episode"] + 8, y, f"{values['first_episode']:.1f}", va="center", fontsize=11)
    ax.set(
        yticks=[0, 1],
        yticklabels=["command 静态", "整卡＋共享"],
        xlim=(0, 1090),
        xlabel="启动至首条轨迹实际开始（秒）",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.46, -0.34), ncol=2, frameon=False, fontsize=11)
    fig.suptitle("额外初始化耗时主要出现在主控进程启动前", fontsize=16, y=0.96)
    fig.text(
        0.04,
        0.84,
        "首段增加 172.5 秒，其中有三次约 30 秒的 Ray 进程注册超时。\n"
        "成功的主控进程启动后，剩余各段合计再增加约 59.9 秒。",
        fontsize=12,
    )
    fig.text(
        0.04,
        0.025,
        "整卡借出在首条轨迹之后约 314 秒才发起，不在上图初始化区间。\n"
        "历史日志分段实测，不等于隔离后的因果成本。轮询边界另有不足 2 秒偏差。",
        fontsize=11,
    )
    output = base.parent / "experiments/live-partition-startup"
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main():
    """Expose startup and training costs with the same timing boundary for each arm."""
    base = Path(__file__).resolve().parents[1]
    data = json.loads((base / "results/live-partition-20261123/analysis.json").read_text())
    plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 4.7), dpi=100)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.76, bottom=0.27)
    labels = ["command 静态池", "常驻余量共享", "整卡调整＋共享"]
    for y, name in enumerate(("static", "resident", "elastic")):
        row = data[name]
        init = row["status"]["initialization_s"]
        training = row["timing_totals"]["timing_s/step"]
        tail = row["launch_to_cleanup_s"] - init - training
        ax.barh(y, init, color="#c3ced9", label="初始化" if y == 0 else None)
        ax.barh(y, training, left=init, color="#347cb7", label="两步训练" if y == 0 else None)
        ax.barh(y, tail, left=init + training, color="#e6a148", label="其余与清理" if y == 0 else None)
        ax.text(init / 2, y, f"{init:.1f}", ha="center", va="center")
        ax.text(init + training / 2, y, f"{training:.2f} 秒", ha="center", va="center", color="white")
        ax.text(row["launch_to_cleanup_s"] + 12, y, f"{row['launch_to_cleanup_s']:.1f}", va="center")
    ax.set_yticks(range(3), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1730)
    ax.set_xlabel("本组启动至物理清理完成（秒）")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, frameon=False)
    fig.suptitle("整卡调整缩短训练，初始化增加了完整耗时", fontsize=16, y=0.96)
    fig.text(0.04, 0.84, "每组两步、24 轨迹、48 GPU 命令、72 模型请求，初始 2 环境＋2 推理。", fontsize=11)
    fig.text(0.04, 0.04, "2026-11-23 单次实测。静态组也使用 command，episode 超时组不参与此图排名。", fontsize=11)
    output = base.parent / "experiments/live-partition-results"
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
    plot_startup(base)
    plot_episode(base)
    plot_completed_episode(base)
    plot_completed_episode(base, current=True)
    plot_startup_repeat(base)


if __name__ == "__main__":
    main()
