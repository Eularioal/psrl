"""Summarize paired GPU measurements without requiring CUDA or third party packages."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)] if ordered else None


def analyze(data):
    before = data["references"]["before"]
    after = data["references"].get("after", {})
    references = {}
    for case, candidates in before.items():
        first = {name: median(v) for name, v in candidates.items()}
        last = {name: median(v) for name, v in after.get(case, {}).items()}
        pooled = {name: median(v + after.get(case, {}).get(name, [])) for name, v in candidates.items()}
        order = sorted(pooled, key=pooled.get)
        gap = pooled[order[1]] / pooled[order[0]] - 1
        drift = max(abs(last[name] / first[name] - 1) for name in last) if last else None
        references[case] = {
            "latency_ms": pooled,
            "before_ms": first,
            "after_ms": last,
            "best": order[0],
            "gap": gap,
            "max_drift": drift,
            "stable_separated": bool(last) and min(first, key=first.get) == min(last, key=last.get) and gap > 0.05,
        }
    records = []
    for group in data["groups"]:
        errors, regrets = [], []
        stable_errors = stable_total = 0
        winners = {}
        for case, candidates in group["cases"].items():
            observed = {name: median(v) for name, v in candidates.items()}
            reference = references[case]
            clean = reference["latency_ms"]
            winner = min(observed, key=observed.get)
            regret = clean[winner] / min(clean.values()) - 1
            errors.extend(abs(observed[name] / clean[name] - 1) for name in observed)
            regrets.append(regret)
            winners[case] = {"chosen": winner, "reference_best": reference["best"], "regret": regret}
            if reference["stable_separated"]:
                stable_total += 1
                stable_errors += int(winner != reference["best"])
        records.append(
            {
                **{
                    k: group[k]
                    for k in [
                        "block",
                        "pressure",
                        "size",
                        "repeat",
                        "policy",
                        "feedback_s",
                        "makespan_s",
                        "evaluations",
                        "switch_wait_s",
                        "protected_entries",
                        "correctness_s",
                        "background_active_at_samples",
                    ]
                },
                "candidate_errors": errors,
                "winner_regrets": regrets,
                "stable_errors": stable_errors,
                "stable_total": stable_total,
                "winners": winners,
                "background_units": sum(x["units"] for x in group["background"]),
            }
        )
    buckets = defaultdict(list)
    for row in records:
        buckets[(row["pressure"], row["policy"])].append(row)
    aggregate = []
    for (pressure, policy), rows in sorted(buckets.items()):
        errors = [x for r in rows for x in r["candidate_errors"]]
        regrets = [x for r in rows for x in r["winner_regrets"]]
        aggregate.append(
            {
                "pressure": pressure,
                "policy": policy,
                "groups": len(rows),
                "feedback_s": sum(r["feedback_s"] for r in rows),
                "makespan_s": sum(r["makespan_s"] for r in rows),
                "evaluations": sum(r["evaluations"] for r in rows),
                "switch_wait_s": sum(r["switch_wait_s"] for r in rows),
                "protected_entries": sum(r["protected_entries"] for r in rows),
                "latency_error_median": median(errors),
                "latency_error_p95": percentile(errors, 0.95),
                "winner_regret_median": median(regrets),
                "winner_regret_max": max(regrets),
                "winner_regret_over_5pct": sum(x > 0.05 for x in regrets),
                "selections": len(regrets),
                "stable_wrong": sum(r["stable_errors"] for r in rows),
                "stable_total": sum(r["stable_total"] for r in rows),
                "active_fraction": sum(r["background_active_at_samples"] for r in rows)
                / sum(r["evaluations"] for r in rows),
            }
        )
    by_block = defaultdict(dict)
    for row in records:
        by_block[row["block"]][row["policy"]] = row
    comparisons = []
    for baseline in ["shared", "exclusive", "per_call", "group5", "group2"]:
        for policy in ["group5", "group2", "adaptive"]:
            if baseline == policy:
                continue
            for pressure in ["none", "memory4", "compute4", "mixed8"]:
                pairs = [
                    b for b in by_block.values() if baseline in b and policy in b and b[policy]["pressure"] == pressure
                ]
                if not pairs:
                    continue
                ratios = [b[policy]["makespan_s"] / b[baseline]["makespan_s"] for b in pairs]
                comparisons.append(
                    {
                        "pressure": pressure,
                        "baseline": baseline,
                        "policy": policy,
                        "pairs": len(pairs),
                        "makespan_ratio_paired_median": median(ratios),
                        "makespan_ratio_range": [min(ratios), max(ratios)],
                        "faster_pairs": sum(r < 1 for r in ratios),
                    }
                )
    return {
        "complete": data["passed"] and bool(after),
        "reference_note": "Clean measurements before and after are empirical references, not exact ground truth.",
        "reference_cases": references,
        "groups": records,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "portability_passed": sum(x["passed"] for x in data["portability"]),
        "portability_total": len(data["portability"]),
        "total_evaluations": sum(r["evaluations"] for r in records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: result[k] for k in ["complete", "total_evaluations", "portability_passed", "portability_total"]}
        )
    )
    for row in result["aggregate"]:
        print(
            f"{row['pressure']:9} {row['policy']:9} total={row['makespan_s']:8.2f}s "
            f"feedback={row['feedback_s']:8.2f}s median_error={row['latency_error_median']:.1%} "
            f"p95_error={row['latency_error_p95']:.1%} loss>5%={row['winner_regret_over_5pct']}/{row['selections']}"
        )
