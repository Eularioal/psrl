"""Summarize matched fixed-work partition trials without claiming unmeasured saturation."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(evidence):
    """Require completed work, exact task outputs, fixed tokens, and physical cleanup."""
    if evidence["status"] != "passed" or not evidence.get("physical_release_confirmed"):
        raise ValueError("Only completed and physically released experiments can be summarized.")
    args = evidence["args"]
    summaries = []
    for trial in evidence["trials"]:
        if len(trial["commands"]) != 2 * args["jobs"] or len(trial["queries"]) != args["requests"]:
            raise ValueError("A trial has incomplete work.")
        if any(len(q["result"]["token_ids"]) != args["tokens"] for q in trial["queries"]):
            raise ValueError("Inference workloads differ in token count.")
        if trial["policy"] == "fixed" and any(int(c["device"]) in args["gpus"][2:] for c in trial["commands"]):
            raise ValueError("The fixed baseline used a model GPU.")
        summaries.append(
            {
                "id": trial["id"],
                "policy": trial["policy"],
                "work_done_s": trial["work_done_s"],
                "total_with_restore_s": trial["total_with_restore_s"],
                "mean_env_wait_s": statistics.mean(c["wait_s"] for c in trial["commands"]),
                "mean_inference_latency_s": statistics.mean(q["latency_s"] for q in trial["queries"]),
                "max_inference_latency_s": max(q["latency_s"] for q in trial["queries"]),
                "borrowed_commands": sum(int(c["device"]) in args["gpus"][2:] for c in trial["commands"]),
                "sleep_wake_s": sum(e["duration_s"] for e in trial["transitions"]),
            }
        )
    grouped = {
        policy: statistics.mean(t["total_with_restore_s"] for t in summaries if t["policy"] == policy)
        for policy in ("fixed", "dynamic")
    }
    query_outputs = [{q["index"]: q["result"]["token_ids"] for q in trial["queries"]} for trial in evidence["trials"]]
    return {
        "trials": summaries,
        "mean_total_s": grouped,
        "dynamic_reduction_fraction": 1 - grouped["dynamic"] / grouped["fixed"],
        "tensor_hashes": evidence["tensor_hashes"],
        "scores": evidence["scores"],
        "inference_outputs_identical": all(outputs == query_outputs[0] for outputs in query_outputs),
        "initialization_s": evidence["initialization_s"],
        "elapsed_including_validation_and_cleanup_s": evidence["elapsed_s"],
        "physical_release_confirmed": True,
        "boundary": evidence["boundary"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.evidence.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
