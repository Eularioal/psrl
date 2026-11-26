"""Summarize complete feedback trials with matched work and measured return delays."""

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    """Return the observed nearest rank without interpolating a small sample."""
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(evidence):
    """Validate completed work before comparing task and request latency."""
    if evidence["status"] != "passed" or not evidence.get("physical_release_confirmed"):
        raise ValueError("Completed work and physical resource release are required.")
    args = evidence["args"]
    trials = []
    for row in evidence["trials"]:
        if len(row["commands"]) != 2 * args["jobs"] or len(row["episodes"]) != args["jobs"]:
            raise ValueError("An episode did not complete.")
        if len(row["queries"]) != 3 * args["jobs"] + args["burst_requests"]:
            raise ValueError("Query counts differ.")
        if any(len(q["result"]["token_ids"]) != args["tokens"] for q in row["queries"]):
            raise ValueError("Generated token counts differ.")
        burst = [q for q in row["queries"] if q["phase"] == "burst"]
        feedback = [q for q in row["queries"] if q["phase"].startswith("after-")]
        restoration = []
        pending = {}
        for event in row["events"]:
            if event["action"] == "restore_start":
                pending[event["device"]] = event["at_s"]
            if event["action"] == "wake_ack":
                restoration.append(event["at_s"] - pending.pop(event["device"]))
        origin = row["events"][0]["at"] - row["events"][0]["at_s"]
        overlapping_queries = 0
        for command in row["commands"]:
            if int(command["device"]) not in args["gpus"][2:]:
                continue
            result = command["result"]
            begin = result["prepared_monotonic_s"] - origin
            end = result["finished_monotonic_s"] - origin
            overlapping_queries += sum(
                query["device"] == command["device"]
                and max(query["at_s"] + query["wait_s"], begin) < min(query["at_s"] + query["latency_s"], end)
                for query in row["queries"]
            )
        trials.append(
            dict(
                id=row["id"],
                total_with_restore_s=row["total_with_restore_s"],
                all_episode_results_ready_s=row["all_episode_results_ready_s"],
                mean_env_wait_s=statistics.mean(c["wait_s"] for c in row["commands"]),
                mean_burst_latency_s=statistics.mean(q["latency_s"] for q in burst),
                p95_burst_latency_s=percentile([q["latency_s"] for q in burst], 0.95),
                mean_feedback_latency_s=statistics.mean(q["latency_s"] for q in feedback),
                mean_burst_queue_s=statistics.mean(q["wait_s"] for q in burst),
                model_gpu_commands=sum(int(c["device"]) in args["gpus"][2:] for c in row["commands"]),
                restoration_waits_s=restoration,
                transition_calls_s=sum(t["duration_s"] for t in row["transitions"]),
                model_gpu_compute_overlapping_queries=overlapping_queries,
            )
        )
    return dict(
        status=evidence["status"],
        initialization_s=evidence["initialization_s"],
        inference_calibration=evidence["inference_calibration"],
        inference_limit=evidence["inference_limit"],
        trials=trials,
        task_outputs_identical=evidence["task_outputs_identical"],
        physical_release_confirmed=True,
        boundary=evidence["boundary"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.evidence.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
