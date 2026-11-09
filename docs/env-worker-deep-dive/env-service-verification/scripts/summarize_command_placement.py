"""Summarize completed placement experiments without rerunning GPU work."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(path):
    data = json.loads(path.read_text())
    if data.get("status") != "passed" or "final_gpu" not in data:
        raise ValueError(f"Experiment is incomplete or failed: {path!s}.")
    commands = {}
    for name, command in data["commands"].items():
        reservation = command["reservation"]
        probes = command["probes"]
        commands[name] = dict(
            wall_s=command["wall_s"],
            queue_s=reservation["admitted_at"] - reservation["requested_at"],
            held_s=reservation["released_at"] - reservation["admitted_at"],
            start=reservation["requested_at"],
            admitted=reservation["admitted_at"],
            finished=command["finished"],
            released=reservation["released_at"],
            gpu_indices=reservation["gpu_indices"],
            passed=sum(p["passed"] for p in probes),
            import_s=sum(p["import_s"] for p in probes),
            cuda_init_s=sum(p["cuda_init_s"] for p in probes),
            pytest_s=sum(p["pytest_s"] for p in probes),
            cache_hits=sum(e["cache_hit"] for p in probes for e in p["compiler_events"]),
        )
    preparation = path.parent / "a_measured.preparation.json"
    fields = ("cpu_used", "memory_used_mb", "leases_active")
    result = dict(
        run=path.parent.name,
        mode=data["mode"],
        prepared_a=data.get("prepared_a", False),
        status=data["status"],
        batch_s=data["batch_s"],
        elapsed_s=data["elapsed_s"],
        commands=commands,
        passed=sum(c["passed"] for c in commands.values()),
        cleanup_errors=data["cleanup_errors"],
        final_gpu={
            i: {"used_mb": s["used_mb"], "process_count": len(s["pids"])} for i, s in data["final_gpu"].items()
        },
        host_before=[{k: s[k] for k in fields} for s in data["host_before"]],
        host_after=[{k: s[k] for k in fields} for s in data["host_after"]],
        queued_cancellation=data["queued_cancellation"],
    )
    if preparation.exists():
        prep = json.loads(preparation.read_text())
        result["preparation"] = {
            "cpu_prepare_s": prep["ready"] - prep["started"],
            "ready_wait_s": prep["activated"] - prep["ready"],
            "maxrss_kb": prep["maxrss_kb"],
            "cuda_initialized": prep["cuda_initialized"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [summarize(path) for root in args.runs for path in sorted(root.glob("*/evidence.json"))]
    assert runs, "No completed experiment was supplied."
    groups = {}
    for mode, prepared in sorted({(r["mode"], r["prepared_a"]) for r in runs}):
        group = [r for r in runs if (r["mode"], r["prepared_a"]) == (mode, prepared)]
        groups[f"{mode}-prepared-{prepared}"] = {
            "n": len(group),
            "mean_batch_s": statistics.mean(r["batch_s"] for r in group),
            "mean_a_wall_s": statistics.mean(r["commands"]["a_measured"]["wall_s"] for r in group),
            "mean_a_held_s": statistics.mean(r["commands"]["a_measured"]["held_s"] for r in group),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                scope="Controlled correctness replay, not full training or historical OFF.", groups=groups, runs=runs
            ),
            indent=2,
        )
    )
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
