"""Audit equal work and plot cold and warm batches from the NVML connection experiment."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(path):
    result = json.loads((path / "result.json").read_text())
    trace = json.loads((path / "control-trace.json").read_text())
    telemetry = [json.loads(line) for line in (path / "telemetry.jsonl").read_text().splitlines()]
    spans = defaultdict(list)
    for actor in [trace["router"], *trace["nodes"].values()]:
        for event in actor["events"]:
            spans[event["label"]].append(event["duration_s"])
    attempts = result["attempts"]
    phases = [phase for attempt in attempts for phase in attempt["agent"]["phases"]]
    units = [unit for row in telemetry for unit in row.get("units", [])]
    assert result["passed"] and len(attempts) == 8
    assert all(attempt["accepted"] and attempt["result"]["reward"] == 1 for attempt in attempts)
    assert len(phases) == 16 and all(phase["cuda"]["kernel_output"] == 42 for phase in phases)
    assert all(phase["cuda"]["requested_bytes"] == 1024**3 for phase in phases)
    assert not result["cleanup_errors"] and not result["remaining_owned_containers"]
    assert not result["gpu_after"]["pids"]
    assert all(unit["declared_gpu_memory_mb"] <= unit["gpu_memory_mb"] for unit in units)
    assert all(unit["declared_gpu_memory_mb"] == unit["gpu_commands_active"] * 2048 for unit in units)
    assert all(unit["containers_running"] == unit["gpu_commands_active"] == 0 for unit in result["final_ledger"])
    return {
        "successful_tasks": len(attempts),
        "successful_commands": len(phases),
        "oom": sum(phase["cuda"]["status"] == "oom" for phase in phases),
        "work_duration_s": result["work_duration_s"],
        "waves_s": [wave["duration_s"] for wave in result["waves"]],
        "startup_s": result["work_started_at"] - result["started_at"],
        "gpu_uuid": result["gpu_after"]["uuid"],
        "gpu_final_used_mib": result["gpu_after"]["used_mib"],
        "max_active_commands": max(unit["gpu_commands_active"] for unit in units),
        "request_to_admitted_mean_s": statistics.mean(
            phase["reservation"]["admitted_at"] - phase["reservation"]["requested_at"] for phase in phases
        ),
        "command_end_to_return_mean_s": statistics.mean(
            phase["reservation"]["released_at"] - phase["reservation"]["command_finished_at"] for phase in phases
        ),
        "queries": {
            label: {"count": len(values), "mean_s": statistics.mean(values), "max_s": max(values)}
            for label, values in spans.items()
        },
    }


def main(root):
    data = {name: summarize(root / name) for name in ("before", "after")}
    assert data["before"]["gpu_uuid"] == data["after"]["gpu_uuid"]
    (root / "comparison.json").write_text(json.dumps(data, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"before": "#b65c36", "after": "#267e9f"}
    for offset, name in enumerate(("before", "after")):
        x = [index + (offset - 0.5) * 0.34 for index in range(2)]
        bars = axes[0].bar(x, data[name]["waves_s"], width=0.34, color=colors[name], label=name.title())
        axes[0].bar_label(bars, fmt="%.2f s", padding=3)
        labels = ["read_gpu_devices", "read_gpu_snapshot", "_return"]
        x = [index + (offset - 0.5) * 0.34 for index in range(3)]
        values = [data[name]["queries"][label]["mean_s"] * 1000 for label in labels]
        bars = axes[1].bar(x, values, width=0.34, color=colors[name], label=name.title())
        axes[1].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].set_xticks([0, 1], ["First batch: cold service", "Second batch: reused service"])
    axes[0].set(ylabel="Task wall time (seconds)", ylim=(0, 65), title="4 completed tasks per batch")
    axes[0].legend()
    axes[1].set_xticks([0, 1, 2], ["GPU identity", "Live GPU state", "Return protocol"])
    axes[1].set(yscale="log", ylabel="Mean milliseconds (log scale)", ylim=(1, 1200))
    axes[1].set_title("Includes first query and lock waits")
    fig.suptitle(
        "ON before / after: persistent NVML connection, fresh readings\n"
        "Real science runner + CUDA, no LLM or training. One run per arm."
    )
    for suffix in ("png", "svg"):
        fig.savefig(root / f"low-overhead.{suffix}", dpi=160)
    plt.close(fig)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
