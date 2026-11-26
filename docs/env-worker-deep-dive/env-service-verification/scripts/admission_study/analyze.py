"""Summarize completed paired blocks without dropping failed tasks or queue time."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def summarize(path, treatment="pool"):
    raw = path.read_bytes()
    evidence = json.loads(raw)
    blocks = evidence["blocks"]
    policies = evidence.get("policies", [])
    expected = len(policies) if len(policies) > 2 else 4
    if evidence.get("error") or len(blocks) != expected or any("makespan_s" not in b for b in blocks):
        raise ValueError("Require every planned block and retain incomplete evidence separately.")
    if len(blocks) % 2 or any(
        {b["policy"] for b in blocks[i : i + 2]} != {"static", treatment} for i in range(0, len(blocks), 2)
    ):
        raise ValueError("Each adjacent pair must contain the control and treatment.")
    result = {
        "baseline": evidence["baseline"],
        "scope": evidence["scope"],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "runner_sha256": evidence["runner_sha256"],
        "baseline_patch_sha256": hashlib.sha256(evidence["patch"].encode()).hexdigest(),
        "budget_s": evidence["budget_s"],
        "elapsed_s": evidence["elapsed_s"],
        "treatment": treatment,
        "blocks": [],
        "families": {},
        "physical_release": all(not g["pids"] and g["used_mb"] < 128 for g in evidence["after"]),
    }
    for block in blocks:
        result["blocks"].append(
            {
                "index": block["index"],
                "policy": block["policy"],
                "makespan_s": block["makespan_s"],
                "tasks": [
                    {
                        k: row[k]
                        for k in (
                            "family",
                            "wait_s",
                            "service_s",
                            "completion_s",
                            "passed",
                            "result",
                            "remaining_containers",
                        )
                    }
                    for row in block["tasks"]
                ],
            }
        )
    for family in [row["family"] for row in blocks[0]["tasks"]]:
        observations = {b["index"]: next(row for row in b["tasks"] if row["family"] == family) for b in blocks}
        control_indices = [b["index"] for b in blocks if b["policy"] == "static"]
        treatment_indices = [b["index"] for b in blocks if b["policy"] == treatment]
        pairs = [
            (
                next(b["index"] for b in pair if b["policy"] == "static"),
                next(b["index"] for b in pair if b["policy"] == treatment),
            )
            for i in range(0, len(blocks), 2)
            for pair in [blocks[i : i + 2]]
        ]
        static = mean(observations[i]["completion_s"] for i in control_indices)
        pooled = mean(observations[i]["completion_s"] for i in treatment_indices)
        result["families"][family] = dict(
            static_s=static,
            treatment_s=pooled,
            reduction_pct=100 * (1 - pooled / static),
            paired_reduction_pct=[
                100 * (1 - observations[p]["completion_s"] / observations[s]["completion_s"]) for s, p in pairs
            ],
            valid=all(row["passed"] and not row["remaining_containers"] for row in observations.values()),
            rewards=[observations[i]["result"]["rewards"] for i in range(len(blocks))],
        )
    static = mean(b["makespan_s"] for b in blocks if b["policy"] == "static")
    pooled = mean(b["makespan_s"] for b in blocks if b["policy"] == treatment)
    result["total"] = dict(static_s=static, treatment_s=pooled, reduction_pct=100 * (1 - pooled / static))
    result["families_improved_10pct"] = sum(
        row["valid"] and row["reduction_pct"] >= 10 for row in result["families"].values()
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--treatment", choices=("pool", "local"), default="pool")
    args = parser.parse_args()
    result = summarize(args.input, args.treatment)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("total", "families_improved_10pct", "physical_release")}))
