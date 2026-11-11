"""Compare fixed preparation windows using completed GPU correctness evidence."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(root):
    runs = []
    for path in sorted(root.glob("*/evidence.json")):
        data = json.loads(path.read_text())
        if data["status"] != "passed" or "final_gpu" not in data:
            raise ValueError(f"Incomplete experiment: {path!s}.")
        commands = []
        for name, command in data["commands"].items():
            if "_turn" not in name:
                continue
            prep = json.loads((path.parent / f"{name}.preparation.json").read_text())
            grant = command["reservation"]
            commands.append(
                dict(
                    name=name,
                    submitted=command["requested"],
                    ready=prep["ready"],
                    prepare_started=prep["started"],
                    admitted=grant["admitted_at"],
                    released=grant["released_at"],
                    finished=command["finished"],
                    cpu_prepare_s=prep["ready"] - prep["started"],
                    ready_wait_s=prep["activated"] - prep["ready"],
                    maxrss_kb=prep["maxrss_kb"],
                    gpu_held_s=grant["released_at"] - grant["admitted_at"],
                    gpu_indices=grant["gpu_indices"],
                    passed=sum(p["passed"] for p in command["probes"]),
                    cuda_initialized_before_grant=prep["cuda_initialized"],
                )
            )
        assert len(commands) == 6 and all(c["passed"] == 6 for c in commands), (
            "Expected six complete correctness commands."
        )
        assert not data["cleanup_errors"], "Cleanup must succeed."
        assert all(not s["pids"] and s["used_mb"] < 64 for s in data["final_gpu"].values()), "GPU cleanup incomplete."
        runs.append(
            dict(
                run=path.parent.name,
                window=data["preparation_window"],
                batch_s=data["batch_s"],
                elapsed_s=data["elapsed_s"],
                commands=commands,
                gpu_held_s=sum(c["gpu_held_s"] for c in commands),
                cpu_prepare_s=sum(c["cpu_prepare_s"] for c in commands),
                ready_memory_proxy_mib_s=sum(c["maxrss_kb"] / 1024 * c["ready_wait_s"] for c in commands),
                passed=sum(p["passed"] for c in data["commands"].values() for p in c["probes"]),
                cleanup_errors=data["cleanup_errors"],
            )
        )
    groups = {}
    for window in sorted({r["window"] for r in runs}):
        selected = [r for r in runs if r["window"] == window]
        groups[str(window)] = {
            key: statistics.mean(r[key] for r in selected)
            for key in ["batch_s", "gpu_held_s", "cpu_prepare_s", "ready_memory_proxy_mib_s"]
        }
        groups[str(window)]["batch_samples_s"] = [r["batch_s"] for r in selected]
        groups[str(window)]["n"] = len(selected)
    return dict(
        scope="Three sessions, two prepared correctness commands each, two GPUs. Window 0 has no extra limit.",
        memory_metric="Maximum per-process RSS times ready waiting is a proxy, not sampled total physical RAM.",
        groups=groups,
        runs=runs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["groups"], indent=2))
