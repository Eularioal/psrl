"""Audit private node comparisons and distinguish preparation overlap from moving a timer."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_router_control_ab import summarize

ARMS = ("low-off", "low-on-lazy", "low-on-serial", "low-on-overlap", "pressure-off", "pressure-on")


def main(root):
    data = {name: summarize(root / name) for name in ARMS}
    for name, (summary, record, telemetry) in data.items():
        summary["preparation_and_work_s"] = record["work_finished_at"] - record["preparation_started_at"]
        summary["prewarm_s"] = (
            record["prewarm_finished_at"] - record["prewarm_started_at"] if "prewarm_finished_at" in record else None
        )
        summary["node_start_s"] = None
        if "on" in name:
            trace = json.loads((root / name / "control-trace.json").read_text())
            starts = [event for event in trace["router"]["events"] if event["label"] == "_start_node"]
            assert len(starts) == 1, "Prewarming and admission must share one node service."
            summary["node_start_s"] = starts[0]["duration_s"]
            demand = 51 * 1024 if name.startswith("pressure") else 2 * 1024
            units = [unit for sample in telemetry for unit in sample.get("units", [])]
            assert all(unit["declared_gpu_memory_mb"] == unit["gpu_commands_active"] * demand for unit in units)
            assert not summary["ledger_budget_violations"]
            assert all(
                unit["containers_running"] == unit["gpu_commands_active"] == 0 for unit in record["final_ledger"]
            )
        assert summary["passed"] and summary["completed"] == 4 and summary["accepted_cuda_commands"] == 8
        assert summary["completions_per_task"] == dict.fromkeys(map(str, range(4)), 1)
        assert not summary["cleanup_errors"] and not summary["remaining_owned_containers"]
        assert not summary["gpu_after"]["pids"] and summary["gpu_after"]["used_mib"] < 64
        for attempt in record["attempts"]:
            for phase in attempt["agent"].get("phases", []):
                gpu_uuid = record["gpu_before"]["uuid"].removeprefix("GPU-").replace("-", "")
                assert phase["cuda"]["gpu_uuid_hex"] == gpu_uuid
                expected_bytes = (50 if name.startswith("pressure") else 1) * 1024**3
                assert phase["cuda"]["requested_bytes"] == expected_bytes
            if attempt["accepted"]:
                assert attempt["result"]["reward"] == 1
                assert all(phase["cuda"]["kernel_output"] == 42 for phase in attempt["agent"]["phases"])
    assert len({record["gpu_before"]["uuid"] for _, record, _ in data.values()}) == 1
    (root / "comparison.json").write_text(json.dumps({name: values[0] for name, values in data.items()}, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = ["#b65c36", "#7f8c8d", "#dbab3d", "#267e9f"]
    low = ARMS[:4]
    for index, name in enumerate(low):
        summary, record, _ = data[name]
        prep = record["work_started_at"] - record["preparation_started_at"]
        axes[0, 0].bar(index, prep, color="#cccccc")
        axes[0, 0].bar(index, summary["work_duration_s"], bottom=prep, color=colors[index])
        axes[0, 0].text(
            index, summary["preparation_and_work_s"] + 0.5, f"{summary['preparation_and_work_s']:.2f}s", ha="center"
        )
    axes[0, 0].set_xticks(range(4), ["OFF", "ON lazy", "ON serial", "ON overlap"])
    axes[0, 0].set(title="Low load: equal preparation + 4 tasks", ylabel="Seconds (gray = preparation)")
    axes[0, 0].margins(y=0.15)
    for index, name in enumerate(low[1:]):
        _, record, _ = data[name]
        origin = record["preparation_started_at"]
        trace = json.loads((root / name / "control-trace.json").read_text())
        start = next(event for event in trace["router"]["events"] if event["label"] == "_start_node")
        spans = [
            (record["independent_preparation_started_at"] - origin, 8, "#cccccc"),
            (start["started_at"] - origin, start["duration_s"], "#267e9f"),
        ]
        for row, (begin, duration, color) in enumerate(spans):
            axes[0, 1].broken_barh([(begin, duration)], (index * 3 + row, 0.8), facecolors=color)
    axes[0, 1].set_yticks([0.9, 3.9, 6.9], ["ON lazy", "ON serial", "ON overlap"])
    axes[0, 1].set(title="Gray: fixed preparation. Blue: real node startup", xlabel="Seconds from preparation start")
    for index, name in enumerate(ARMS[4:]):
        summary = data[name][0]
        color = "#b65c36" if name.endswith("off") else "#267e9f"
        axes[1, 0].bar(index, summary["preparation_and_work_s"], color=color)
        axes[1, 0].text(
            index,
            summary["preparation_and_work_s"] + 1,
            f"{summary['preparation_and_work_s']:.2f}s\n{summary['oom_attempts']} OOM",
            ha="center",
        )
        _, record, _ = data[name]
        origin = record["preparation_started_at"]
        times = sorted(item["finished_at"] - origin for item in record["attempts"] if item["accepted"])
        axes[1, 1].step([0, *times], range(5), where="post", color=color, label=name)
    axes[1, 0].set_xticks([0, 1], ["OFF: 0eeead8", "ON: overlap"])
    axes[1, 0].set(title="50 GiB commands: 4 successes, retries included", ylabel="Preparation + work seconds")
    axes[1, 0].margins(y=0.25)
    axes[1, 1].set(title="Pressure: equal completed work", xlabel="Seconds from preparation start", ylabel="Successes")
    axes[1, 1].legend()
    fig.suptitle(
        "Real science runner, Docker and CUDA. Fixed 8s preparation represents model setup.\n"
        "No model or training. One run per arm. Ray initialization reported separately in JSON."
    )
    for suffix in ("png", "svg"):
        fig.savefig(root / f"prewarm-comparison.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    end = max(data[name][0]["preparation_and_work_s"] for name in ARMS[4:])
    for ax, name in zip(axes, ARMS[4:], strict=True):
        _, record, telemetry = data[name]
        origin = record["preparation_started_at"]
        samples = [sample for sample in telemetry if "gpu" in sample]
        ax.plot(
            [sample["time"] - origin for sample in samples],
            [sample["gpu"]["used_mib"] / 1024 for sample in samples],
            label="Physical application memory",
        )
        if name.endswith("on"):
            samples = [sample for sample in telemetry if sample.get("units")]
            ax.step(
                [sample["time"] - origin for sample in samples],
                [sample["units"][0]["declared_gpu_memory_mb"] / 1024 for sample in samples],
                where="post",
                label="Declared active command budget",
            )
        ax.set(title=name, ylabel="GiB", ylim=(0, 100), xlim=(0, end + 2))
        ax.legend(loc="upper right")
    axes[1].set_xlabel("Seconds from preparation start (same scale)")
    fig.suptitle("GPU memory and command budget. Memory occupancy is not compute utilization.")
    for suffix in ("png", "svg"):
        fig.savefig(root / f"resource-timeline.{suffix}", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                name: {
                    key: values[0][key]
                    for key in (
                        "preparation_and_work_s",
                        "work_duration_s",
                        "total_s",
                        "attempts",
                        "oom_attempts",
                        "node_start_s",
                    )
                }
                for name, values in data.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
