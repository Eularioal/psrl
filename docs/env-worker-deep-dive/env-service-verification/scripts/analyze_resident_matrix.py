"""
Compare completed short training arms without substituting historical or missing steps.
"""

import argparse
import json
from pathlib import Path

from analyze_single_gpu_training import analyze


def main(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries, records, series = {}, {}, {}
    for pressure in ("low", "high"):
        directory = root / pressure
        design = json.loads((directory / "design.json").read_text())
        records[pressure], summaries[pressure] = {}, {}
        for mode in ("off", "command"):
            record, data, gpu = analyze(directory, mode, design)
            records[pressure][mode] = record
            complete = (
                record["exit_code"] == 0
                and record["equal_completion_target_met"]
                and len(record["steps"]) == design["steps_per_arm"]
            )
            summary = {
                "complete": complete,
                "exit_code": record["exit_code"],
                "steps": len(record["steps"]),
                "accepted": record["accepted"],
                "attempts": record["attempts"],
                "oom_attempts": record["oom_attempts"],
                "sampling_seconds": record["sampling_seconds_including_retries"],
                "step_seconds": record["sum_step_seconds"] if complete else None,
                "wait_for_gen_seconds": record["sum_wait_for_gen_seconds"] if complete else None,
                "update_actor_seconds": record["sum_update_actor_seconds"] if complete else None,
                "before_first_slot_seconds": record["before_first_slot_seconds"],
                "driver_seconds": record["driver_seconds_including_cleanup"],
                "trained_tokens": sum(s.get("termination/trained_tokens", 0) for s in record["steps"]),
                "agent_turn_counts": record["accepted_agent_turn_counts"],
                "environment_gpu_peak_mib": record["environment_gpu_peak_mib"],
                "peak_active_gpu_commands": record["peak_active_gpu_commands"],
                "peak_waiting_gpu_commands": record["peak_waiting_gpu_commands"],
                "peak_successful_cuda_phases": record["peak_successful_cuda_phases"],
                "command_admission_wait_seconds": [
                    p["admitted_at"] - p["requested_at"]
                    for p in record["command_phases"]
                    if "admitted_at" in p and "requested_at" in p
                ],
                "resident_pressure": data["launch"].get("resident_pressure"),
                "resident_cleanup": data["launch"].get("resident_cleanup"),
                "gpu_release": record["gpu_release"],
                "repair_rewards_by_group": record["repair_rewards_by_group"],
                "telemetry_errors": record["telemetry_errors"],
            }
            summaries[pressure][mode] = summary
            first = record["first_slot_started_at"]
            series[pressure, mode] = [
                (g["time"] - first, g["mib"] / 1024) for g in gpu if first is not None and g["time"] >= first
            ]
        off, on = summaries[pressure]["off"], summaries[pressure]["command"]
        paired = off["complete"] and on["complete"]
        summaries[pressure]["paired_complete"] = paired
        summaries[pressure]["step_time_reduction_percent"] = (
            100 * (1 - on["step_seconds"] / off["step_seconds"]) if paired else None
        )
    (root / "analysis.json").write_text(json.dumps(records, indent=2))
    (root / "summary.json").write_text(json.dumps(summaries, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"off": "#d9822b", "command": "#2878b5"}
    for row, pressure in enumerate(("low", "high")):
        for index, mode in enumerate(("off", "command")):
            summary = summaries[pressure][mode]
            label = "OFF baseline" if mode == "off" else "ON EnvRouter"
            seconds = summary["step_seconds"]
            if seconds is not None:
                axes[row, 0].bar(index, seconds, color=colors[mode])
                axes[row, 0].text(
                    index, seconds, f"{seconds:.1f}s\nOOM: {summary['oom_attempts']}", ha="center", va="bottom"
                )
            else:
                axes[row, 0].text(index, 0, "Incomplete", ha="center")
            values = series[pressure, mode]
            if values:
                x, y = zip(*values, strict=True)
                axes[row, 1].plot(x, y, color=colors[mode], label=label)
        axes[row, 0].set_xticks([0, 1], ["OFF", "ON"])
        axes[row, 0].set_ylabel("Training step seconds")
        axes[row, 0].set_title(f"{pressure.title()} resident memory pressure")
        axes[row, 0].margins(y=0.3)
        axes[row, 1].set_xlabel("Seconds after first rollout request")
        axes[row, 1].set_ylabel("Environment GPU memory (GiB)")
        axes[row, 1].axhline(90, color="gray", linestyle=":", label="Unit admission budget")
        axes[row, 1].legend()
    fig.suptitle("One PSRL update per arm, identical 14 GiB task phase\nResident allocation: low 0 GiB, high 70 GiB")
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    fig.savefig(root / "comparison.svg")
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
