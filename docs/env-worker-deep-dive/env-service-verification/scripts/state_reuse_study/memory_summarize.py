"""Audit bounded state scheduling evidence and compare with the strongest measured controls."""

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from memory_policy import POLICIES


def summarize(root):
    """Keep real wall time, calibrated replay and correctness checks distinct."""
    study = json.loads((root / "study-run/study.json").read_text())
    profile = json.loads((root / "calibrated-profile.json").read_text())
    replay = json.loads((root / "replay-calibrated.json").read_text())
    groups = collections.defaultdict(list)
    for row in study["blocks"]:
        groups[row["scenario"]["name"]].append(row)
    output = dict(
        status=study["status"],
        stop_reason="budget_reached" if "budget was reached" in study.get("error", "") else study.get("error"),
        scope=study["scope"],
        setup_s=study["setup_s"],
        scenarios={},
        replay={},
        audit=dict(commands=0, numerical_checks=0, version_checks=0, payload_budget_checks=0, equal_work_scenarios=0),
    )
    for name, rows in groups.items():
        outputs = []
        for row in rows:
            index = next(i for i, scene in enumerate(study["plan"]["scenarios"]) if scene["name"] == name)
            block = root / "study-run" / f"block-{index}-{row['policy']}"
            logical = []
            expected = {}
            for trace in row["trace"]:
                key = trace["family"], trace["seed"]
                reference = study["references"][f"{key[0]}:{key[1]}"]["value"]
                target = reference * (1 + trace["turn"] * 0.1) + trace["session"] * 10 + trace["turn"]
                if abs(trace["result"] - target) > max(1e-4, abs(target) * 1e-5):
                    raise RuntimeError("The independent formula audit failed.")
                if trace["payload_bytes"] > int(row["scenario"]["budget_gib"] * 1024**3):
                    raise RuntimeError("The independent memory budget audit failed.")
                identity = trace["session"], trace["turn"], trace["seed"]
                if identity in expected:
                    raise RuntimeError("The same logical command was executed more than once.")
                accepted = json.loads((block / f"answer-{identity[0]}-{identity[1]}-{identity[2]}.json").read_text())
                snapshot = block / f"session-{identity[0]}" / "versions" / accepted["snapshot"]
                files = {file: (snapshot / file).read_text() for file in ("helper.py", "scorer.py", "loader.json")}
                actual_code = hashlib.sha256(
                    json.dumps({file: files[file] for file in ("helper.py", "scorer.py")}, sort_keys=True).encode()
                ).hexdigest()
                actual_model = hashlib.sha256(
                    json.dumps(json.loads(files["loader.json"]), sort_keys=True).encode()
                ).hexdigest()
                if actual_code != trace["code_identity"] or actual_model != trace["model_identity"]:
                    raise RuntimeError("The published version disagreed with its immutable source snapshot.")
                if accepted["value"] != trace["result"] or accepted["generation"] != trace["turn"] + 1:
                    raise RuntimeError("The accepted result or generation disagreed with the completed command.")
                output["audit"]["version_checks"] += 1
                expected[identity] = trace["family"]
                logical.append((*identity, trace["family"], trace["result"]))
            if len(expected) != 24:
                raise RuntimeError("A policy did not finish the fixed 24 command workload.")
            outputs.append(sorted(logical))
            output["audit"]["commands"] += len(logical)
            output["audit"]["numerical_checks"] += len(logical)
            output["audit"]["payload_budget_checks"] += len(logical)
        if any(values != outputs[0] for values in outputs):
            raise RuntimeError("A scenario changed its logical inputs or outputs between policies.")
        output["audit"]["equal_work_scenarios"] += 1
        strategies = {}
        for row in rows:
            modes = collections.Counter(trace["mode"] for trace in row["trace"])
            outliers = [
                dict(
                    session=t["session"],
                    turn=t["turn"],
                    family=t["family"],
                    mode=t["mode"],
                    total_s=t["total_s"],
                    load_gpu_s=t["load_gpu_s"],
                    compute_s=t["compute_s"],
                    teardown_s=t["total_s"] - t["load_gpu_s"] - t["compute_s"],
                )
                for t in row["trace"]
                if t["total_s"] > 2
            ]
            strategies[row["policy"]] = dict(
                wall_s=row["wall_s"],
                mean_feedback_s=row["mean_feedback_s"],
                gpu_command_s=sum(t["total_s"] for t in row["trace"]),
                preparation_s=sum(t["prepare_s"] for t in row["trace"]),
                eviction_s=sum(t["evict_s"] for t in row["trace"]),
                planning_cpu_s=row["planning_cpu_s"],
                idle_s=row["idle_s"],
                preparations=row["preparations"],
                hits=row["cache_hits"],
                modes=dict(modes),
                peak_payload_bytes=row["peak_payload_bytes"],
                peak_private_dirty_kb=max(t["memory"]["Private_Dirty_kb"] for t in row["trace"]),
                long_command_diagnostics=outliers,
            )
        controls = [p for p in strategies if p != "feedback-dp"]
        comparison = {}
        if "feedback-dp" in strategies:
            for metric in ("wall_s", "mean_feedback_s"):
                best = min(controls, key=lambda p: strategies[p][metric])
                comparison[metric] = dict(
                    best_control=best,
                    relative_improvement=1 - strategies["feedback-dp"][metric] / strategies[best][metric],
                )
        output["scenarios"][name] = dict(
            strategies=strategies, comparison=comparison, all_policies_present=set(strategies) == set(POLICIES)
        )
    for budget in (0.8, 1.3, 3.0, 5.0):
        rows = [row for row in replay["scenarios"] if row["budget_gib"] == budget]
        metrics = {}
        for metric in ("makespan_s", "mean_feedback_s"):
            values = [
                1
                - row["results"]["feedback-dp"][metric]
                / min(result[metric] for policy, result in row["results"].items() if policy != "feedback-dp")
                for row in rows
            ]
            metrics[metric] = dict(
                count=len(values),
                mean_relative_improvement=statistics.mean(values),
                min_relative_improvement=min(values),
                max_relative_improvement=max(values),
                wins_above_one_percent=sum(value > 0.01 for value in values),
                losses_above_one_percent=sum(value < -0.01 for value in values),
            )
        output["replay"][str(budget)] = metrics
    output["profile"] = dict(payload_bytes=profile["payload_bytes"], eviction_s=profile["evict_s"])
    output["audit"]["status"] = "passed"
    output["incomplete_blocks"] = [
        dict(
            name=path.name,
            published_partial_commands=len(list(path.glob("answer-*.json"))),
            included_in_performance=False,
        )
        for path in (root / "study-run").glob("block-*")
        if not any(
            path.name == f"block-{i}-{row['policy']}"
            for i, scene in enumerate(study["plan"]["scenarios"])
            for row in study["blocks"]
            if scene["name"] == row["scenario"]["name"]
        )
    ]
    output["inputs_sha256"] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in ("study-run/study.json", "calibrated-profile.json", "replay-calibrated.json")
    }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], audit=result["audit"])))
