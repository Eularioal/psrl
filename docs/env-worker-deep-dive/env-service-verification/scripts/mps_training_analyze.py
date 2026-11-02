"""Summarize fixed training work and separately inspect CUDA profiling windows."""

import argparse
import json
import statistics
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def trace_intervals(path):
    trace = json.loads(path.read_text())
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel" and e.get("ph") == "X"]
    # Independent traces need an explicit shared clock before their timestamps can be compared.
    if "baseTimeNanoseconds" not in trace:
        return {"kernel_count": len(kernels), "error": "Trace has no explicit timestamp base."}, []
    base = trace["baseTimeNanoseconds"] / 1000
    intervals = merge([(base + e["ts"], base + e["ts"] + e["dur"]) for e in kernels])
    return {"kernel_count": len(kernels), "base_time_ns": trace["baseTimeNanoseconds"]}, intervals


def overlap(left, right):
    i = j = 0
    total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize(root):
    results = []
    failures = []
    records = json.loads((root / "summary.json").read_text())
    completed = {(r["iteration"], r["mode"]) for r in records}
    for path in sorted(root.glob("*/result.json")):
        record = json.loads(path.read_text())
        if record["passed"]:
            continue
        errors = [job["error"] for job in record.get("jobs", []) if job.get("error")]
        failures.append({"directory": path.parent.name, "error": "; ".join(errors) or record.get("error", "")[:500]})
        workloads = [path.parent / str(i) / "result.json" for i in range(2)]
        if (record["iteration"], record["mode"]) in completed or not all(p.exists() for p in workloads):
            continue
        record["workloads"] = [json.loads(p.read_text()) for p in workloads]
        if not all("train_end" in w for w in record["workloads"]):
            continue
        record["training_span_s"] = max(w["train_end"] for w in record["workloads"]) - min(
            w["train_start"] for w in record["workloads"]
        )
        record["training_seconds"] = [w["train_end"] - w["train_start"] for w in record["workloads"]]
        records.append(record)
    for record in records:
        arm = root / f"{record['iteration']}-{record['mode']}"
        samples = [json.loads(line) for line in (arm / "gpu.jsonl").read_text().splitlines()]
        row = {
            "mode": record["mode"],
            "iteration": record["iteration"],
            "passed": record["passed"],
            "training_span_s": record["training_span_s"],
            "training_seconds": record["training_seconds"],
            "container_setup_s": record["containers_ready"] - record["started"],
            "prepare_s": record["prepared"] - record["containers_ready"],
            "wall_seconds": record["wall_seconds"],
            "peak_device_used_mb": max(gpu["used_mb"] for s in samples for gpu in s.get("gpu", {}).values()),
            "missing_samples": sum("missing" in s for s in samples),
            "loss_after_10": [r["loss_after_10"] for r in record["workloads"]],
            "measured_steps": [len([s for s in r["steps"] if 3 <= s["step"] < 13]) for r in record["workloads"]],
            "profile_errors": [r.get("profile_error") for r in record["workloads"]],
            "mps_clients_verified": "mps_clients" in record,
            "job_states": [j["state"] for j in record["jobs"]],
            "job_errors": [j.get("error") for j in record["jobs"]],
            "physical_resources_empty": not record["remaining"]
            and all(not g["pids"] and g["used_mb"] < 64 for g in record["after"].values()),
            "cleanup_verified": not record.get("cleanup_errors")
            and not record["remaining"]
            and all(not g["pids"] and g["used_mb"] < 64 for g in record["after"].values()),
        }
        paths = [arm / str(i) / "trace.json" for i in range(2)]
        if all(p.exists() for p in paths):
            parsed = [trace_intervals(p) for p in paths]
            row["trace"] = {"clients": [p[0] for p in parsed]}
            left, right = [p[1] for p in parsed]
            if left and right:
                union = merge(left + right)
                active = sum(end - start for start, end in union)
                together = overlap(left, right)
                row["trace"].update(
                    kernel_overlap_ms=together / 1000,
                    kernel_union_ms=active / 1000,
                    overlap_fraction_of_union=together / active,
                )
        results.append(row)
    medians = {
        mode: statistics.median(r["training_span_s"] for r in results if r["mode"] == mode and r["passed"])
        for mode in {r["mode"] for r in results if r["passed"]}
    }
    metadata = json.loads((root / "metadata.json").read_text())
    provenance = {key: metadata[key] for key in ("image", "workload_sha256")}
    snapshot = root.parent / "source/provenance.json"
    if snapshot.exists():
        frozen = json.loads(snapshot.read_text())
        provenance["git_head_before_workspace_patches"] = frozen["git_head"]
        provenance["env_service_source_sha256"] = {
            name: value for name, value in frozen["files"].items() if name.startswith("psrl/workers/env_service/")
        }
    return {
        "provenance": provenance,
        "scope": "Fixed autoresearch updates, not native LAPS or historical OFF.",
        "arms": results,
        "medians_of_complete_arms": medians,
        "failures": failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.root), indent=2) + "\n")
