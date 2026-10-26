"""Compare completed ON command styles and retain failed preparation evidence."""

import argparse
import json
from pathlib import Path


def peak_overlap(intervals):
    events = [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals]
    count = peak = 0
    for _, change in sorted(events):
        count += change
        peak = max(peak, count)
    return peak


def summarize(directory):
    result = json.loads((directory / "result.json").read_text())
    paths = list(directory.rglob("natural-gpu.json"))
    if len(paths) != 1:
        raise ValueError("Each arm must contain exactly one agent record.")
    agent = json.loads(paths[0].read_text())
    records = []
    for command in agent["commands"]:
        for line in command.get("stdout", "").splitlines():
            if line.startswith("NATURAL_GPU_RESULT="):
                record = json.loads(line.partition("=")[2])
                record["reservation"] = command.get("reservation", {})
                records.append(record)
    samples = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    observed = [sample["gpu"] for sample in samples if "gpu" in sample]
    completed = [item for item in records if item["status"] == "ok" and item.get("steps") == 10]
    return {
        "success": len(completed),
        "commands": len(agent["commands"]),
        "agent_s": agent["finished_at"] - agent["started_at"],
        "episode_s": result["work_finished_at"] - result["work_started_at"],
        "cpu_prepare_peak": peak_overlap([(item["started_at"], item["prepared_at"]) for item in records]),
        "gpu_process_peak": max((len(item["pids"]) for item in observed), default=None),
        "gpu_memory_peak_mib": max((item["used_mib"] for item in observed), default=None),
        "gpu_utilization_peak_pct": max((item["utilization_gpu"] for item in observed), default=None),
        "missing_samples": sum("gpu" not in sample for sample in samples),
        "gpu_after": result["gpu_after"],
        "final_ledger": result.get("final_ledger"),
        "records": records,
    }


def analyze(root):
    arms = {name: summarize(root / name) for name in ("on-whole", "on-prepared", "on-prepared-safe-imports")}
    before, after = arms["on-whole"], arms["on-prepared-safe-imports"]
    if before["success"] != before["commands"] or after["success"] != before["commands"]:
        raise ValueError("Performance comparison requires equal successful work.")
    comparison = {
        "scope": "ON whole command vs ON prepared with safe imports, both use isolated metadata fix",
        "agent_time_reduction_pct": (1 - after["agent_s"] / before["agent_s"]) * 100,
        "episode_time_reduction_pct": (1 - after["episode_s"] / before["episode_s"]) * 100,
        "parameter_sums_identical": len({item["parameter_sum"] for arm in (before, after) for item in arm["records"]})
        == 1,
    }
    return {"arms": arms, "comparison": comparison}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = analyze(args.root)
    (args.root / "analysis.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "arms": {
                    k: {n: v for n, v in x.items() if n not in ("records", "final_ledger")}
                    for k, x in report["arms"].items()
                },
                "comparison": report["comparison"],
            },
            indent=2,
        )
    )
