"""Archive vision startup boundaries and render the timing and preparation diagrams."""

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/gpu-control-startup-20261126"
SCENE = BASE / "results/partition-short-2-20261126/vision-mix"
LABELS = [
    "预检与Ray",
    "Driver至Router就绪",
    "Router就绪至广播完成",
    "广播后至采样管理器开始创建",
    "管理器创建至首轨迹",
]
COLORS = ["#8da9c4", "#76b7b2", "#4c78a8", "#e49d49", "#9b85bc"]


def timestamp(line):
    value = re.search(r"2026-11-26 \d\d:\d\d:\d\d,\d{3}", line).group()
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()


def audit():
    evidence = {
        "scope": "Historical vision two-step runs. Boundaries locate delay, not exclusive component CPU time.",
        "arms": {},
    }
    for arm, mode, router_pid, manager_pid in (
        ("command_static", "fixed", 530669, 530699),
        ("command_elastic", "resident", 689659, 689625),
    ):
        root = SCENE / arm / mode
        status = json.loads((root / "status.json").read_text())
        logs = Path("/tmp") / status["tag"] / "session_latest/logs"
        lines = []

        def core(pid, needle, logs=logs, lines=lines):
            path = next(logs.glob(f"python-core-worker-*_{pid}.log"))
            line = next(line for line in path.read_text().splitlines() if needle in line)
            lines.append({"file": str(path), "line": line})
            return timestamp(line)

        router_start = core(router_pid, "Creating actor")
        router_ready = core(router_pid, "Actor creation task finished")
        manager_start = core(manager_pid, "Creating actor")
        manager_ready = core(manager_pid, "Actor creation task finished")
        line = next(
            line
            for line in (root / "launcher.log").read_text().splitlines()
            if "Broadcast initialization complete" in line and "INFO:2026" in line
        )
        lines.append({"file": str(root / "launcher.log"), "line": line})
        broadcast = timestamp(line)
        events = {event["stage"]: event["at"] for event in status["startup_events"]}
        boundaries = [
            status["started_at"],
            events["driver_launched"],
            router_ready,
            broadcast,
            manager_start,
            events["first_episode_observed"],
        ]
        err = next(logs.glob(f"worker-*{router_pid}.err")).read_text()
        (OUT / f"{arm}-router.err").write_text(err)
        evidence["arms"][arm] = {
            "boundaries_epoch": boundaries,
            "stages_s": [right - left for left, right in zip(boundaries, boundaries[1:])],
            "router_construction_s": router_ready - router_start,
            "manager_construction_s": manager_ready - manager_start,
            "ray_worker_registration_timeouts": (logs / "raylet.out").read_text().count("have not registered within"),
            "sources": lines,
        }
    evidence["stage_labels"] = LABELS
    evidence["difference_s"] = [
        b - a
        for a, b in zip(
            evidence["arms"]["command_static"]["stages_s"],
            evidence["arms"]["command_elastic"]["stages_s"],
        )
    ]
    (OUT / "historical-stages.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    return evidence


def figures(evidence):
    plt.rcParams.update({"font.family": "Droid Sans Fallback", "font.size": 12, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8, 5.3), dpi=100)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.79, bottom=0.46)
    for y, (key, label) in enumerate((("command_static", "Command静态"), ("command_elastic", "完整方案"))):
        left = 0
        for value, color in zip(evidence["arms"][key]["stages_s"], COLORS):
            ax.barh(y, value, left=left, height=0.5, color=color)
            if value > 38:
                ax.text(left + value / 2, y, f"{value:.1f}", ha="center", va="center", color="white", fontsize=12)
            left += value
        ax.text(left + 5, y, f"{left:.1f}s", va="center", fontsize=11)
    ax.set(
        yticks=[0, 1], yticklabels=["Command静态", "完整方案"], xlim=(0, 645), xlabel="从实验组入口开始累计计时（秒）"
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("视觉两步：额外101.85秒落在哪些启动阶段？", fontsize=16, y=0.96)
    fig.text(0.05, 0.87, "历史实测：同工作量，顺序各一次。完整方案没有发生整卡借用。", fontsize=12)
    fig.legend(
        handles=[Patch(color=color, label=label) for color, label in zip(COLORS, LABELS)],
        loc="lower left",
        bbox_to_anchor=(0.04, 0.16),
        frameon=False,
        fontsize=12,
    )
    fig.text(0.05, 0.12, "橙段：1.83 → 35.29秒。完整组此时出现Router首次Torch加载日志。", fontsize=12)
    fig.text(0.05, 0.055, "Router自身创建仅0.78 / 1.64秒。阶段差不能全部归因为动态调度。", fontsize=12)
    save(fig, "gpu-control-startup-stages")

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), dpi=100, sharex=True)
    fig.subplots_adjust(left=0.19, right=0.94, top=0.80, bottom=0.20, hspace=0.6)
    for ax, title, control_start, activation in (
        (axes[0], "原来：等权重准备好，才连接控制服务", 40, 70),
        (axes[1], "修改后：先准备控制连接，安全检查仍在最后", 0, 40),
    ):
        ax.barh(1, 40, left=0, height=0.55, color=COLORS[2])
        ax.barh(0, 30, left=control_start, height=0.55, color=COLORS[3])
        ax.barh(0, 2, left=activation, height=0.55, color="#42846b")
        ax.text(20, 1, "权重准备40秒", color="white", ha="center", va="center")
        ax.text(control_start + 15, 0, "连接／导入30秒", ha="center", va="center")
        ax.axvline(activation + 2, color="#42846b", linestyle="--")
        ax.text(activation + 3, 0.5, f"{activation + 2}秒\n可准入", va="center", fontsize=11)
        ax.set(yticks=[0, 1], yticklabels=["Router CPU", "模型进程"], xlim=(0, 83), ylim=(-0.6, 1.6), title=title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("共同起点：模型进程已创建，接下来准备初始权重（秒）")
    fig.suptitle("缩短初始化：重叠独立工作，保留准入检查", fontsize=16, y=0.96)
    fig.text(0.05, 0.87, "教学假设：权重40秒、控制连接30秒、安全检查2秒。不是实测提速。", fontsize=12)
    fig.text(0.05, 0.08, "72 → 42秒：等待取两条并行路径中更长的一条，再做2秒检查。", fontsize=12)
    fig.text(0.05, 0.035, "连接准备期间共享卡始终阻塞。没有提前休眠vLLM，也没有提交环境命令。", fontsize=12)
    save(fig, "gpu-control-startup-overlap")


def save(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(BASE.parent / "experiments" / f"{name}.{suffix}", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "historical-stages.json"
    figures(json.loads(path.read_text()) if path.exists() else audit())
