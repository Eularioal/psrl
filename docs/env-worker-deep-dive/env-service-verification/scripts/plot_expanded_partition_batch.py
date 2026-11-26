"""Audit equal work and draw the real pair and simulated scale comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_integrated_partition import distribution
from capacity_validation_pipeline import save
from matplotlib import font_manager

COLORS = {"startup": "#c4ceda", "step": "#2874ae", "cleanup": "#e69b3d", "faster": "#197967", "slower": "#b44d32"}
FONT = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")


def read(path):
    return json.loads(path.read_text())


def live_arm(root, arm, mode, protocol):
    """Check workload, placement and resource ownership from completed live records."""
    folder = root / arm
    result = read(folder / "analysis.json")
    supervisor = read(folder / "supervisor.json")
    expected = protocol["expected"]
    work = dict(
        steps=len(result["steps"]),
        trajectories=result["episodes_completed"],
        commands=result["commands_completed"],
        requests=result["queries_completed"],
        environment_updates=result["environment_updates_completed"],
    )
    if work != expected or supervisor["status"] != "complete" or supervisor["source_changes"]:
        raise ValueError(f"Arm {arm!r} did not complete the frozen workload.")
    release = read(folder / "final-resource-check.json")["gpus"]
    if not supervisor["physical_release_verified"] or set(release) != set(map(str, range(8))):
        raise ValueError(f"Arm {arm!r} has incomplete release evidence.")
    task_hash = hashlib.sha256((folder / "tasks.parquet").read_bytes()).hexdigest()
    if task_hash != protocol["tasks_sha256"]:
        raise ValueError(f"Arm {arm!r} used a different task table.")
    records = [read(path) for path in sorted((folder / mode).glob("episodes/*/episode.json"))]
    if len(records) != expected["trajectories"] or any(row["status"] != "complete" for row in records):
        raise ValueError(f"Arm {arm!r} has extra or incomplete trajectories.")
    reservations = {row["gpu_reservation"] for row in records}
    expected_reservation = {"episode"} if mode == "fixed" else {"command"}
    if reservations != expected_reservation:
        raise ValueError(f"Arm {arm!r} used unexpected GPU reservation modes.")
    timings = result["timing_totals"]
    first_start = min(row["started_at"] for row in records)
    handover = read(folder / "handover.json") if mode == "resident" else {"transitions": []}
    return {
        "work": work,
        "arm": arm,
        "gpu_reservation": next(iter(reservations)),
        "trajectories_s": max(row["finished_at"] for row in records) - first_start,
        "first_reply_to_first_gpu_s": distribution(
            [row["commands"][0]["admission"]["admitted_at"] - row["queries"][0]["finished_at"] for row in records]
        ),
        "initialization_s": result["status"]["initialization_s"],
        "step_s": timings["timing_s/step"],
        "wait_for_gen_s": timings["timing_s/wait_for_gen"],
        "policy_update_s": timings["timing_s/update_actor"],
        "job_s": result["launch_to_cleanup_s"],
        "generated_tokens": result["generated_tokens"],
        "query_p95_s": result["query_latency_s"]["p95"],
        "command_placements": result["command_placements"],
        "handover": handover["transitions"],
        "peak_active_gpu_commands": result["peak_active_gpu_commands"],
        "peak_device_used_mib": result["peak_device_used_mib"],
        "environment_wait_s": result["environment_wait_s"],
        "smg_observed_model_versions": result["smg_observed_model_versions"],
        "grad_norm": next(iter(result["steps"].values())).get("actor/grad_norm"),
        "release_verified": True,
    }


def live_comparison(root, baseline, dynamic):
    """Audit both arms before calculating speed differences."""
    protocol = read(root / "protocol.json")
    episode = live_arm(root, baseline, "fixed", protocol)
    elastic = live_arm(root, dynamic, "resident", protocol)
    if set(episode["command_placements"]) - {f"gpu-{i}" for i in protocol["gpu_layout"]["dedicated_environment"]}:
        raise ValueError("Baseline commands left the dedicated environment pool.")
    changed = ["trajectories_s", "step_s", "job_s", "wait_for_gen_s", "query_p95_s", "initialization_s"]
    return {
        "protocol": protocol,
        "episode_static": episode,
        "command_elastic": elastic,
        "dynamic_change_pct": {key: 100 * (elastic[key] / episode[key] - 1) for key in changed},
    }


def save_figure(figure, path):
    for suffix in (".png", ".svg"):
        figure.savefig(path.with_suffix(suffix), facecolor="white")
    plt.close(figure)


def plot_live(data, output):
    """Use the same whole-job time boundary for both arms."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    fig.subplots_adjust(left=0.23, right=0.93, top=0.73, bottom=0.31)
    fig.suptitle("八张 H20 上的单批次实机对照", x=0.04, ha="left", fontsize=19)
    fig.text(0.04, 0.84, "2 训练＋3 推理＋3 环境，Qwen3-8B，AIME 2025 子集", fontsize=12)
    arms = [data["episode_static"], data["command_elastic"]]
    x_max = max(arm["job_s"] for arm in arms)
    for y, arm in enumerate(arms):
        startup, step, total = (arm[key] for key in ("initialization_s", "step_s", "job_s"))
        for length, start, color, label in (
            (startup, 0, COLORS["startup"], "初始化"),
            (step, startup, COLORS["step"], "一个训练 step"),
            (total - startup - step, startup + step, COLORS["cleanup"], "其余与清理"),
        ):
            ax.barh(y, length, left=start, color=color, height=0.46, label=label if y == 0 else None)
        ax.text(startup / 2, y, f"{startup:.1f}", ha="center", va="center", fontsize=12)
        ax.text(startup + step / 2, y, f"{step:.1f}", ha="center", va="center", color="white", fontsize=12)
        ax.text(total + 0.02 * x_max, y, f"{total:.1f}", va="center", fontsize=12)
    ax.set(
        yticks=[0, 1],
        yticklabels=["Episode 静态池", "Command 动态"],
        xlabel="各组作业入口至物理清理（秒）",
        xlim=(0, x_max * 1.17),
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.36), ncol=3, frameon=False)
    fig.text(0.04, 0.055, "每组 1 步、20 条完整轨迹、40 条环境 GPU 命令、60 次模型请求。", fontsize=11)
    save_figure(fig, output)


def plot_replay(rows, output):
    """Show paired change and the range across matched input samples."""
    labels = [
        "6 卡，3B，GSM8K",
        "8 卡，3B，GSM8K",
        "8 卡，8B，GSM8K",
        "8 卡，8B，AIME",
        "8 卡，8B，CIFAR-10",
        "12 卡，8B，AIME",
        "12 卡，40 轨迹",
        "8 卡，40 轨迹",
        "8 卡，请求耗时 ×3",
        "8 卡，共卡减速 ×2",
    ]
    fig, ax = plt.subplots(figsize=(8, 5.9), dpi=100)
    fig.subplots_adjust(left=0.27, right=0.91, top=0.79, bottom=0.20)
    fig.suptitle("相同轨迹下的十组场景回放", x=0.04, ha="left", fontsize=19)
    fig.text(0.04, 0.86, "基础批次 20 条轨迹，最后四组为 40 条轨迹的 AIME 场景。", fontsize=11)
    values = [row["median_paired_change_pct"] for row in rows]
    for i, (row, value) in enumerate(zip(rows, values, strict=True)):
        low, high = row["min_max_paired_change_pct"]
        color = COLORS["faster"] if value < 0 else COLORS["slower"] if value > 0 else "#677686"
        ax.hlines(i, low, high, color=color, alpha=0.65, linewidth=2)
        ax.scatter(value, i, color=color, s=46, zorder=3)
    ax.axvline(0, color="#7f8d9c", linewidth=1)
    ax.grid(axis="x", color="#e5e9ed", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="动态方案相对静态 Episode 的批次用时变化（%）")
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(0.04, 0.07, "点是 12 次配对的中位数，线是最小至最大。负值表示动态方案提前收齐。", fontsize=11)
    fig.text(0.04, 0.035, "时钟从第一轮模型请求开始，计至最后一条轨迹完成。", fontsize=11)
    save_figure(fig, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--episode-arm", default="episode_static")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()
    font_manager.fontManager.addfont(FONT)
    font_name = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams.update({"font.family": font_name, "font.size": 12, "axes.unicode_minus": False})
    if not args.skip_live:
        results = live_comparison(args.root, args.episode_arm, "command_elastic")
        save(args.root / "live-comparison.json", results)
        plot_live(results, args.figures / "expanded-partition-live")
    replay = args.replay or args.root / "replay.json"
    plot_replay(read(replay)["pairs"], args.figures / "expanded-partition-replay")
