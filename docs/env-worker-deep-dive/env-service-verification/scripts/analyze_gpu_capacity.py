"""Compare completed training and OOM outcomes under measured GPU capacity pressure."""

import argparse
import json
from pathlib import Path


def analyze(root):
    matrix = json.loads((root / "matrix.json").read_text())
    reports = {}
    for arm in matrix["arms"]:
        name = arm["arm"]
        directory = root / name
        driver = json.loads((directory / "result.json").read_text())
        paths = list(directory.rglob("natural-gpu.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one complete agent record for {name!r}.")
        agent = json.loads(paths[0].read_text())
        commands = []
        for item in agent["commands"]:
            matches = [
                line.partition("=")[2]
                for line in (item.get("stdout") or "").splitlines()
                if line.startswith("NATURAL_GPU_RESULT=")
            ]
            work = json.loads(matches[0]) if len(matches) == 1 else None
            commands.append(
                {
                    "index": item["index"],
                    "return_code": item.get("return_code"),
                    "work": work,
                    "error": item.get("error"),
                    "oom": "out of memory" in json.dumps(item).lower(),
                    "library_allocation_failed": "cublas_status_alloc_failed" in json.dumps(item).lower(),
                }
            )
        good = [x["work"] for x in commands if x["work"] and x["work"]["status"] == "ok" and x["work"]["steps"] == 10]
        samples = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
        observed = [sample["gpu"] for sample in samples if "gpu" in sample]
        holder = driver.get("pressure_holder")
        excluded = {holder["gpu_pid"]} if holder else set()
        if holder and not all(holder["gpu_pid"] in sample["pids"] for sample in observed):
            raise ValueError(f"The pressure holder was not present throughout {name!r}.")
        baseline = driver["gpu_before"]["used_mib"]
        report = {
            "submitted": len(commands),
            "success": len(good),
            "oom": sum(x["oom"] for x in commands),
            "library_allocation_failed": sum(x["library_allocation_failed"] for x in commands),
            "other_failures": sum(
                not x["oom"] and not x["library_allocation_failed"] and (not x["work"] or x["work"]["status"] != "ok")
                for x in commands
            ),
            "agent_s": agent["finished_at"] - agent["started_at"],
            "episode_s": driver["work_finished_at"] - driver["work_started_at"],
            "driver_and_release_s": arm["finished_at"] - arm["started_at"],
            "training_process_peak": max(len(set(x["pids"]) - excluded) for x in observed),
            "incremental_memory_peak_mib": max(x["used_mib"] - baseline for x in observed),
            "total_memory_peak_mib": max(x["used_mib"] for x in observed),
            "telemetry_missing": sum("gpu" not in x for x in samples),
            "pressure_free_mib": holder["free_mib"] if holder else None,
            "pressure_used_mib": holder["used_mib"] if holder else 0,
            "pressure_present_throughout": bool(holder),
            "successful_loss": [x["log_history"][-1]["train_loss"] for x in good],
            "successful_parameter_sums": [x["parameter_sum"] for x in good],
            "commands": commands,
        }
        reports[name] = report
    comparison = {}
    for level in ("low", "high"):
        off, on = reports[f"{level}-off"], reports[f"{level}-on"]
        complete = off["success"] == on["success"] == off["submitted"] == on["submitted"]
        comparison[level] = {
            "equal_complete_work": complete,
            "on_agent_time_change_pct": 100 * (on["agent_s"] / off["agent_s"] - 1) if complete else None,
            "on_episode_time_change_pct": 100 * (on["episode_s"] / off["episode_s"] - 1) if complete else None,
            "on_driver_time_change_pct": (
                100 * (on["driver_and_release_s"] / off["driver_and_release_s"] - 1) if complete else None
            ),
            "scope": "Latency ratio is withheld when successful work differs.",
        }
    return {"arms": reports, "comparison": comparison, "cleanup": matrix["gpu_after"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = analyze(args.root)
    (args.root / "analysis.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "arms": {k: {n: v for n, v in x.items() if n != "commands"} for k, x in report["arms"].items()},
                "comparison": report["comparison"],
            },
            indent=2,
        )
    )
