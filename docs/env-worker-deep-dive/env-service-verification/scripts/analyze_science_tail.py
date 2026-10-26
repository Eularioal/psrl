"""Summarize instrumented science runs without starting any external services."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def stats(values):
    return {"count": len(values), "mean_s": statistics.mean(values), "max_s": max(values)}


def summarize(directory):
    result = json.loads((directory / "result.json").read_text())
    trace = json.loads((directory / "execution-trace.json").read_text())
    assert result["passed"], "Expected a completed bounded experiment."
    calls = []
    phases = []
    for attempt in result["attempts"]:
        assert attempt["accepted"], "Only successful equal-work attempts are expected."
        for phase in attempt["agent"]["phases"]:
            cuda = phase["cuda"]
            assert cuda["kernel_output"] == 42, "The CUDA readback must match the written value."
            wall = cuda["finished_at"] - cuda["started_at"]
            monotonic = cuda["finished_mono"] - cuda["started_mono"]
            record = {
                "task": attempt["index"],
                "phase": phase["phase"],
                "session": attempt["session"],
                "call_s": phase["finished_at"] - phase["requested_at"],
                "script_s": monotonic,
                "clock_difference_s": wall - monotonic,
                "cpu_before": cuda["cpu_before"],
                "cpu_after": cuda["cpu_after"],
                "script_started_mono": cuda["started_mono"],
            }
            request = phase.get("reservation")
            record["grant_to_script_s"] = cuda["started_at"] - (
                request["admitted_at"] if request else phase["requested_at"]
            )
            record["request_to_grant_s"] = request["admitted_at"] - request["requested_at"] if request else None
            phases.append(record)
            calls.extend({**call, "task": attempt["index"], "phase": phase["phase"]} for call in cuda["calls"])
    durations = defaultdict(list)
    for call in calls:
        durations[call["name"]].append(call["duration_s"])
    spans = defaultdict(list)
    for event in trace["events"]:
        spans[event["label"]].append(event["duration_s"])
    work_lags = [
        event["lag_s"]
        for event in trace["events"]
        if event["label"] == "loop_tick"
        and result["work_started_at"] <= event["started_at"] <= result["work_finished_at"]
    ]
    stack_groups = defaultdict(list)
    for sample in trace.get("stack_samples", []):
        if not result["work_started_at"] <= sample["time"] <= result["work_finished_at"]:
            continue
        frames = sample["frames"]
        if frames[0]["function"] == "_execute_child":
            category = "process_creation"
        elif "pynvml" in frames[0]["file"]:
            category = "nvml"
        elif any(frame["function"] in ("_detect_daemon_os", "_egress_control_kernel_support") for frame in frames):
            category = "synchronous_docker_probe"
        else:
            category = "other"
        stack_groups[category].append(sample)
    return {
        "passed": result["passed"],
        "attempts": len(result["attempts"]),
        "waves_s": [wave["duration_s"] for wave in result["waves"]],
        "work_s": result["work_duration_s"],
        "cuda_calls": {name: stats(values) for name, values in durations.items()},
        "cuda_slowest_calls": sorted(calls, key=lambda call: call["duration_s"], reverse=True)[:25],
        "phases": phases,
        "host_spans_not_additive": {name: stats(values) for name, values in spans.items()},
        "loop_lags": [event for event in trace["events"] if event["label"] == "loop_tick"],
        "work_loop_lag_max_s": max(work_lags, default=0),
        "work_stack_samples": dict(stack_groups),
        "gpu_after": result["gpu_after"],
        "cleanup_errors": result["cleanup_errors"],
        "remaining_owned_containers": result["remaining_owned_containers"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--arms", nargs="+", default=["on", "off"])
    args = parser.parse_args()
    data = {arm: summarize(args.root / arm) for arm in args.arms}
    (args.root / "tail-analysis.json").write_text(json.dumps(data, indent=2) + "\n")
    for arm, record in data.items():
        print(arm, "waves", record["waves_s"])
        print("CUDA calls", json.dumps(record["cuda_calls"], indent=2))
        print("Host spans", json.dumps(record["host_spans_not_additive"], indent=2))
        print("Slow phases", sorted(record["phases"], key=lambda phase: phase["call_s"], reverse=True)[:2])
