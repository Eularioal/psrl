"""Summarize complete command records and observed GPU concurrency without filling gaps."""

import argparse
import json
from pathlib import Path


def analyze(root):
    summary = {}
    for mode in ("off", "on"):
        arm = root / mode
        driver = json.loads((arm / "result.json").read_text())
        paths = list((arm / "jobs").rglob("natural-gpu.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one agent record for {mode!r}, found {len(paths)!r}.")
        agent = json.loads(paths[0].read_text())
        commands = agent["commands"]
        workloads = []
        for command in commands:
            lines = (command.get("stdout") or "").splitlines()
            matches = [x.removeprefix("NATURAL_GPU_RESULT=") for x in lines if x.startswith("NATURAL_GPU_RESULT=")]
            if len(matches) == 1:
                workloads.append(json.loads(matches[0]))
        telemetry = [json.loads(x) for x in (arm / "telemetry.jsonl").read_text().splitlines()]
        gpu = [x["gpu"] for x in telemetry if "gpu" in x]
        good = [x for x in workloads if x["status"] == "ok" and x["steps"] == 10]
        summary[mode] = {
            "submitted": len(commands),
            "returned": sum("return_code" in x for x in commands),
            "completed_training": len(good),
            "missing_training_records": agent["spec"]["count"] - len(workloads),
            "expected": agent["spec"]["count"],
            "complete": len(good) == agent["spec"]["count"],
            "oom": sum("out of memory" in json.dumps(x).lower() for x in workloads),
            "agent_seconds": agent["finished_at"] - agent["started_at"],
            "peak_memory_mib": max(x["used_mib"] for x in gpu),
            "peak_compute_processes": max(len(x["pids"]) for x in gpu),
            "peak_utilization_gpu": max(x["utilization_gpu"] for x in gpu),
            "samples": len(telemetry),
            "telemetry_errors": sum("error" in x for x in telemetry),
            "command_wall_seconds": [x["finished_at"] - x["started_at"] for x in good],
            "import_and_data_seconds": [x["cuda_started_at"] - x["started_at"] for x in good],
            "training_seconds": [x["cuda_finished_at"] - x["cuda_started_at"] for x in good],
            "training_loss": [x["log_history"][-1]["train_loss"] for x in good],
            "parameter_sums": [x["parameter_sum"] for x in good],
            "gpu_after": driver["gpu_after"],
            "driver_error": driver.get("error"),
            "episode_result": driver.get("result"),
        }
    if all(summary[x]["complete"] for x in ("off", "on")):
        summary["on_agent_time_change_percent"] = 100 * (
            summary["on"]["agent_seconds"] / summary["off"]["agent_seconds"] - 1
        )
    if (root / "early-stop.json").exists():
        stop = json.loads((root / "early-stop.json").read_text())
        summary["early_stop"] = {key: value for key, value in stop.items() if key != "commands_before_stop"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(analyze(parser.parse_args().root), indent=2))
