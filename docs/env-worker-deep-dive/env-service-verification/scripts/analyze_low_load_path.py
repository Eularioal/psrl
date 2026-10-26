"""Locate low-load latency in saved runs without starting Ray, Docker or CUDA."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


def stats(values):
    return {"count": len(values), "mean_s": statistics.mean(values), "max_s": max(values)}


def phase_parts(phase):
    """Disjoint intervals, unlike nested router spans which must not be added together."""
    cuda = phase["cuda"]
    reservation = phase.get("reservation")
    admitted = reservation["admitted_at"] if reservation else phase["requested_at"]
    parts = {
        "before_reservation": reservation["requested_at"] - phase["requested_at"] if reservation else 0,
        "request_to_grant": reservation["admitted_at"] - reservation["requested_at"] if reservation else 0,
        "grant_to_script": cuda["started_at"] - admitted,
        "script_to_allocation": cuda["allocated_at"] - cuda["started_at"],
        "allocation_to_script_end": cuda["finished_at"] - cuda["allocated_at"],
        "script_end_to_call_end": phase["finished_at"] - cuda["finished_at"],
    }
    assert all(value >= 0 for value in parts.values()), "Phase timestamps must be monotonic."
    assert abs(sum(parts.values()) - (phase["finished_at"] - phase["requested_at"])) < 1e-6, (
        "Intervals must sum to the phase duration."
    )
    return parts


def summarize(directory):
    record = json.loads((directory / "result.json").read_text())
    assert record["passed"] and len(record["attempts"]) == 4, "Expected four successful attempts."
    tasks = []
    for attempt in record["attempts"]:
        assert attempt["accepted"], "All compared tasks must succeed."
        agent = attempt["agent"]
        # Diagram 1 covers session preparation, 2-4 cover commands, and 5 covers scoring and cleanup.
        tasks.append(
            {
                "task": attempt["task"],
                "launch_offset": attempt["started_at"] - record["work_started_at"],
                "pre_agent": agent["started_at"] - attempt["started_at"],
                "agent": agent["finished_at"] - agent["started_at"],
                "post_agent": attempt["finished_at"] - agent["finished_at"],
                "end_offset": attempt["finished_at"] - record["work_started_at"],
                "cpu_interval": agent["cpu_finished_at"] - agent["cpu_started_at"],
                "phases": [phase_parts(phase) for phase in agent["phases"]],
            }
        )
    critical = max(tasks, key=lambda task: task["end_offset"])
    assert (
        abs(
            sum(critical[key] for key in ("launch_offset", "pre_agent", "agent", "post_agent"))
            - critical["end_offset"]
        )
        < 1e-6
    )
    result = {
        "work_s": record["work_duration_s"],
        "preparation_and_work_s": record["work_finished_at"] - record["preparation_started_at"],
        "total_s": record["finished_at"] - record["started_at"],
        "tasks": sorted(tasks, key=lambda task: task["task"]),
        "last_finishing_task": critical,
        "task_means": {
            key: statistics.mean(task[key] for task in tasks)
            for key in ("pre_agent", "agent", "post_agent", "cpu_interval")
        },
        "command_intervals": {
            key: stats([phase[key] for task in tasks for phase in task["phases"]]) for key in tasks[0]["phases"][0]
        },
    }
    trials = [json.loads(path.read_text()) for path in (directory / "jobs").glob("*/*/*/result.json")]
    assert len(trials) == 4, "Expected four Harbor trial records."
    result["harbor_stage_s"] = {}
    for key in ("environment_setup", "agent_execution", "verifier"):
        durations = [
            (
                datetime.fromisoformat(trial[key]["finished_at"]) - datetime.fromisoformat(trial[key]["started_at"])
            ).total_seconds()
            for trial in trials
        ]
        result["harbor_stage_s"][key] = stats(durations)
    trace_path = directory / "control-trace.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        events = trace["router"]["events"] + [e for node in trace["nodes"].values() for e in node["events"]]
        result["trace_spans_not_additive"] = {
            label: stats([event["duration_s"] for event in events if event["label"] == label])
            for label in sorted({event["label"] for event in events})
        }
        reservations = [phase["reservation"] for attempt in record["attempts"] for phase in attempt["agent"]["phases"]]
        result["command_end_to_confirmed_return"] = stats(
            [reservation["released_at"] - reservation["command_finished_at"] for reservation in reservations]
        )
    return result


def plot(root, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    names = ["low-off", "low-on-overlap"]
    for row, name in enumerate(names):
        task = data[name]["last_finishing_task"]
        left = task["launch_offset"]
        for key, color in zip(("pre_agent", "agent", "post_agent"), ("#9baab5", "#267e9f", "#dca447"), strict=True):
            value = task[key]
            axes[0].barh(row, value, left=left, color=color, label=key if row == 0 else None)
            axes[0].text(left + value / 2, row, f"{value:.2f}s", ha="center", va="center")
            left += value
        axes[0].text(left + 0.3, row, f"{left:.2f}s")
    axes[0].set_yticks([0, 1], ["OFF last task (0)", "ON last task (1)"])
    axes[0].set(
        title="Observed last-finishing task: wall time is not the sum of four tasks",
        xlabel="Seconds from task batch start",
        xlim=(0, 49),
    )
    axes[0].legend(loc="lower right", fontsize=8)
    colors = ["#aaaabb", "#e95858", "#8e65a3", "#58a394", "#267e9f", "#dca447"]
    for row, name in enumerate(names):
        parts = data[name]["last_finishing_task"]["phases"][0]
        left = 0
        for (key, value), color in zip(parts.items(), colors, strict=True):
            axes[1].barh(row, value, left=left, color=color, label=key if row == 1 else None)
            if value > 0.5:
                axes[1].text(left + value / 2, row, f"{value:.2f}s", ha="center", va="center", color="white")
            left += value
    axes[1].set_yticks([0, 1], ["OFF first command", "ON first command"])
    axes[1].set(
        title="ON outlier: grant took 0.039s; most time was after grant",
        xlabel="Seconds within GPU call",
        xlim=(0, 12),
    )
    axes[1].legend(loc="lower right", fontsize=8, ncol=2)
    fig.suptitle(
        "Saved private-node run: real Docker/CUDA, one observation per arm\n"
        "Timing locates delay. It does not identify the cause inside Docker or the driver."
    )
    for suffix in ("png", "svg"):
        fig.savefig(root / f"low-load-breakdown.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--plot", action="store_true", help="Optional matplotlib figure. JSON analysis uses only stdlib."
    )
    args = parser.parse_args()
    data = {
        name: summarize(args.root / name) for name in ("low-off", "low-on-lazy", "low-on-serial", "low-on-overlap")
    }
    (args.root / "low-load-breakdown.json").write_text(json.dumps(data, indent=2) + "\n")
    if args.plot:
        plot(args.root, data)
    for name, result in data.items():
        print(name, "work_s=", round(result["work_s"], 3), "task_means=", result["task_means"])
        print(
            "last_task=",
            result["last_finishing_task"]["task"],
            "first_command=",
            result["last_finishing_task"]["phases"][0],
        )


if __name__ == "__main__":
    main()
