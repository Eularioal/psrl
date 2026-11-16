"""Compare paired real verifier scores and complete timing boundaries without a GPU."""

import argparse
import json
import math
from pathlib import Path


def main():
    """Reject missing groups or candidate contamination before presenting any savings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.run / "status.json").read_text())
    if status["status"] != "complete":
        raise RuntimeError("Only a completed bounded study can be summarized as a comparison.")
    names = ["native-A1", "native-B", "native-A2", "process-reload", "resident-base"]
    workers = {name: json.loads((args.run / f"{name}.json").read_text()) for name in names}
    if any(worker["status"] != "passed" for worker in workers.values()):
        raise RuntimeError("Every worker must pass before interpreting performance.")
    all_rows = [row for worker in workers.values() for row in worker["rows"]]
    if len(all_rows) != 9:
        raise RuntimeError("The declared comparison requires exactly nine verifier requests.")
    identities = {}
    for row in all_rows:
        candidate = row["candidate_tensor_id"]
        identities.setdefault(candidate, []).append(row)
        if row["applied_norm_tensor_id"] != candidate:
            raise RuntimeError("A verifier model did not contain its declared candidate tensor.")
    if len(identities) != 2 or sorted(map(len, identities.values())) != [3, 6]:
        raise RuntimeError("Real A and B must be distinct candidate tensors with the declared request counts.")
    references = {}
    for row in all_rows:
        key = row["request_id"]
        if key in references:
            for metric in ("answer_nll", "reward"):
                if not math.isclose(row["metrics"][metric], references[key][metric], rel_tol=1e-5, abs_tol=1e-7):
                    raise RuntimeError("Scores changed across execution strategies for an identical request.")
        else:
            references[key] = row["metrics"]
    if len(references) != 2:
        raise RuntimeError("A and B must retain the same complete input identities across strategies.")
    wall = {item["label"]: item["container_wall_s"] for item in status["records"]}
    group_names = {"new_process": names[:3], "process_reload": [names[3]], "resident_base": [names[4]]}
    groups = {}
    for group, members in group_names.items():
        rows = [row for name in members for row in workers[name]["rows"]]
        phases = {key: sum(row["phases"][key] for row in rows) for key in rows[0]["phases"]}
        groups[group] = {
            "container_wall_s": sum(wall[name] for name in members),
            "worker_wall_s": sum(workers[name]["total_wall_s"] for name in members),
            "setup_s": sum(workers[name]["setup_s"] for name in members),
            "identity_s": sum(row["identity_s"] for row in rows),
            "execute_s": sum(row["execute_s"] for row in rows),
            "execute_other_s": sum(row["execute_s"] for row in rows) - sum(phases.values()),
            "request_wall_s": sum(row["total_wall_s"] for row in rows),
            "request_other_s": sum(row["total_wall_s"] - row["execute_s"] - row["identity_s"] for row in rows),
            "phases": phases,
            "model_hits": sum(row["cache_hits"]["model"] for row in rows),
            "requests": [
                {
                    "request_wall_s": row["total_wall_s"],
                    "identity_s": row["identity_s"],
                    "execute_s": row["execute_s"],
                    "cache_hits": row["cache_hits"],
                }
                for row in rows
            ],
        }
    resident = groups["resident_base"]["container_wall_s"]
    cleanup = status["gpu_after"]
    released = (
        cleanup["exit_code"] == 0
        and not cleanup["processes"]
        and cleanup["memory_exit_code"] == 0
        and cleanup["memory_used_mib"] <= 16
    )
    if not released:
        raise RuntimeError("Physical GPU cleanup has not been verified.")
    report = {
        "status": "passed",
        "groups": groups,
        "scores": references,
        "candidate_tensor_counts": {key: len(rows) for key, rows in identities.items()},
        "resident_observed_savings_vs_new_process_s": groups["new_process"]["container_wall_s"] - resident,
        "resident_observed_savings_vs_process_reload_s": groups["process_reload"]["container_wall_s"] - resident,
        "gpu_released": released,
        "limits": [
            "One fixed order pass, without statistical stability or a scheduling workload claim.",
            "All three strategies use identical content audits around unchanged local execute code.",
            "The local protocol changes only final normalization parameters and grades four rows per request.",
            "No original PostTrain full fine tuning, PSRL training, EnvRouter ON/OFF or queueing comparison.",
            "Read only study inputs are required. Content hashes do not provide atomic multi file publication.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
