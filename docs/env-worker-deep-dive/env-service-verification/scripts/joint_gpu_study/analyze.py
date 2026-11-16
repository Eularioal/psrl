"""Audit and publish bounded two-GPU command evidence without host identities."""

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

POLICIES = ("fifo", "lpt", "affinity", "joint")
MODELS = ("Qwen3-1.7B-Base", "Qwen3-4B-Base", "SmolLM3-3B-Base")
FIELDS = ("norm_sha256", "data_sha256", "loss", "losses", "optimizer_updates")
WORKER_VARIATION = {
    "recorded_at_start_sha256": "cb17e04d08bbfd65f6b6f546abb8f30a20e7521d54c0a2b2c41ebd9c8a02d0a5",
    "final_workspace_sha256": "ed7fa4c22f10f4eb3fb9d6f476f3706db1f24d55d0fa5fc8b73f963d5cf0c946",
    "change": "torch.cuda.reset_peak_memory_stats() moved from immediately before compute to command start.",
    "training_core": "Unchanged. The saved start source and final source differ only by this moved call.",
    "per_command_attribution": "Not recorded. Worker instrumentation revision cannot be assigned to every command.",
    "limitation": "Memory peaks reliably cover training, but loading transients are not uniformly covered.",
}


def mean(values):
    return statistics.mean(values) if values else None


def reduction(reference, candidate):
    return 100 * (reference - candidate) / reference if reference else None


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)] if values else None


def data_name(value):
    return value.removeprefix("/assets/")


def redact(value):
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"GPU-[0-9a-fA-F-]+", "redacted-device", value)
        value = re.sub(r"(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)", "redacted-host", value)
        return re.sub(r"(?<!\w)/(?:[^\s,'\"\]\[()]+)", "redacted-path", value)
    return value


def expected_ids(workload):
    if workload == "balanced":
        return {f"{model}-{data}" for model, data in itertools.product(MODELS, ("aime2025", "gsm8k"))}
    return {
        *(f"{MODELS[0]}-{data}" for data in ("aime2025", "gsm8k", "humaneval")),
        *(f"{MODELS[1]}-{data}" for data in ("aime2025", "gsm8k")),
        f"{MODELS[2]}-aime2025",
    }


def group_metrics(block):
    results = [row["result"] for row in block["rows"]]
    phase = {key: sum(result[key] for result in results) for key in ("load_s", "compute_s", "evict_s", "total_s")}
    phase["unattributed_worker_s"] = phase["total_s"] - sum(phase[key] for key in ("load_s", "compute_s", "evict_s"))
    return {
        "wall_s": block["wall_s"],
        "planning_s": block["planning_s"],
        "load_count": sum(not result["cache_hit"] for result in results),
        "cache_hits": sum(result["cache_hit"] for result in results),
        "phase_seconds_sum": phase,
        "final_clear_seconds_sum": sum(row["clear_s"] for row in block["final_clears"]),
        "peak_allocated_mb": {
            f"gpu{index}": max(
                (row["result"]["peak_allocated_mb"] for row in block["rows"] if row["gpu"] == index), default=0
            )
            for index in range(2)
        },
    }


def analyze(raw):
    errors, identities, seen = [], {}, set()
    blocks = raw.get("blocks", [])
    expected = {
        (budget, workload, repeat, policy)
        for budget in (10240, 24576)
        for workload in ("balanced", "skewed")
        for repeat in range(2 if budget == 10240 else 1)
        for policy in POLICIES
    }
    records, predictor = [], defaultdict(list)

    def check_result(result, model, data, context, budget):
        if result.get("status") != "passed":
            errors.append(f"Failed command in {context}.")
        if result.get("optimizer_updates") != 2:
            errors.append(f"Incorrect optimizer update count in {context}.")
        for key in FIELDS:
            if key not in result:
                errors.append(f"Missing {key!r} in {context}.")
        identity = tuple(json.dumps(result.get(key), sort_keys=True) for key in FIELDS)
        identity_key = model, data
        if identity_key in identities and identities[identity_key] != identity:
            errors.append(f"Exact numerical or identity mismatch for {model!r}, {data_name(data)!r} in {context}.")
        identities.setdefault(identity_key, identity)
        for key in ("load_s", "compute_s", "evict_s", "total_s", "peak_allocated_mb", "loss"):
            value = result.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"Nonfinite or missing {key!r} in {context}.")
        if result.get("peak_allocated_mb", math.inf) > budget:
            errors.append(f"Observed allocated memory exceeded its budget in {context}.")
        if result.get("cache_bytes", math.inf) > (budget - 1024) * 1024**2:
            errors.append(f"Model states exceeded the budget after activation reserve in {context}.")

    for budget_key, models in raw.get("profiles", {}).items():
        budget = int(budget_key)
        for model, profile in models.items():
            samples = profile.get("samples", [])
            if len(samples) != 4:
                errors.append(f"Expected four profile commands for {model!r}, budget {budget}.")
            for index, sample in enumerate(samples):
                check_result(
                    sample, model, "/assets/data/aime2025/public/rows.json", f"profile {model} {index}", budget
                )

    for index, block in enumerate(blocks):
        key = block["budget_mb"], block["workload"], block["repeat"], block["policy"]
        if key in seen or key not in expected:
            errors.append(f"Duplicate or unexpected block {key!r}.")
        seen.add(key)
        rows = block["rows"]
        ids = [row["job"]["id"] for row in rows]
        if len(ids) != 6 or set(ids) != expected_ids(block["workload"]):
            errors.append(f"Wrong submitted job set in block {index}.")
        for row in rows:
            job, result = row["job"], row["result"]
            if result.get("id") != job["id"] or result.get("model") != job["model"]:
                errors.append(f"Mismatched returned command identity in block {index}.")
            check_result(result, job["model"], job["data"], f"block {index} job {job['id']}", block["budget_mb"])
            profile = raw["profiles"][str(block["budget_mb"])][job["model"]]
            estimate = profile["compute_s"]
            predictor[(block["budget_mb"], job["model"], data_name(job["data"]), "compute")].append(
                (estimate, result["compute_s"])
            )
            if not result["cache_hit"]:
                predictor[(block["budget_mb"], job["model"], data_name(job["data"]), "load")].append(
                    (profile["load_s"], result["load_s"])
                )
        if len(block.get("final_clears", [])) != 2 or any(
            item.get("cache_keys") or item.get("cache_bytes") for item in block.get("final_clears", [])
        ):
            errors.append(f"Final cache cleanup was incomplete in block {index}.")
        commands = [
            dict(
                job=dict(id=row["job"]["id"], model=row["job"]["model"], data=data_name(row["job"]["data"])),
                gpu=f"gpu{row['gpu']}",
                submitted_s=row["submitted_s"],
                finished_s=row["finished_s"],
                result=row["result"],
            )
            for row in rows
        ]
        records.append(
            dict(
                key=dict(zip(("budget_mb", "workload", "repeat", "policy"), key)),
                commands=commands,
                **group_metrics(block),
            )
        )
    missing = expected - seen
    if missing:
        errors.append(f"Missing {len(missing)} expected complete blocks.")
    if raw.get("owned_containers") != "":
        errors.append("Physical container cleanup is missing or incomplete.")
    if len(raw.get("devices", [])) != 2:
        errors.append("Expected two selected physical devices for cleanup verification.")
    device_after = {}
    for alias, device in enumerate(raw.get("devices", [])):
        row = next(
            (item.split(", ") for item in raw.get("after", "").splitlines() if item.startswith(f"{device['index']},")),
            None,
        )
        if row is None:
            errors.append(f"Final GPU reading is missing for gpu{alias}.")
        else:
            device_after[f"gpu{alias}"] = float(row[2])
            if float(row[2]) > 32:
                errors.append(f"Final device memory remains occupied for gpu{alias}.")
    grouped, pairs = defaultdict(list), []
    paired = defaultdict(dict)
    for record in records:
        key = record["key"]
        grouped[(key["budget_mb"], key["workload"], key["policy"])].append(record)
        paired[(key["budget_mb"], key["workload"], key["repeat"])][key["policy"]] = record
    summary = []
    for (budget, workload, policy), rows in grouped.items():
        summary.append(
            dict(
                budget_mb=budget,
                workload=workload,
                policy=policy,
                repeats=len(rows),
                wall_s=[row["wall_s"] for row in rows],
                mean_wall_s=mean([row["wall_s"] for row in rows]),
                mean_load_count=mean([row["load_count"] for row in rows]),
                mean_planning_s=mean([row["planning_s"] for row in rows]),
                mean_phase_seconds_sum={
                    name: mean([row["phase_seconds_sum"][name] for row in rows])
                    for name in rows[0]["phase_seconds_sum"]
                },
                max_peak_allocated_mb={
                    name: max(row["peak_allocated_mb"][name] for row in rows) for name in ("gpu0", "gpu1")
                },
            )
        )
    for (budget, workload, repeat), policies in paired.items():
        if set(policies) != set(POLICIES):
            continue
        best = min(("fifo", "lpt", "affinity"), key=lambda policy: policies[policy]["wall_s"])
        pairs.append(
            dict(
                budget_mb=budget,
                workload=workload,
                repeat=repeat,
                best_simple_policy=best,
                joint_reduction_vs_best_simple_pct=reduction(policies[best]["wall_s"], policies["joint"]["wall_s"]),
                affinity_reduction_vs_fifo_pct=reduction(policies["fifo"]["wall_s"], policies["affinity"]["wall_s"]),
                wall_s={name: row["wall_s"] for name, row in policies.items()},
            )
        )
    prediction_errors = []
    for (budget, model, data, phase), values in predictor.items():
        relative = [abs(actual - estimate) / estimate for estimate, actual in values if estimate]
        prediction_errors.append(
            dict(
                budget_mb=budget,
                model=model,
                data=data,
                phase=phase,
                count=len(values),
                estimate_s=values[0][0],
                mean_observed_s=mean([actual for _, actual in values]),
                mean_absolute_error_s=mean([abs(actual - estimate) for estimate, actual in values]),
                mean_absolute_relative_error=mean(relative),
                p95_absolute_relative_error=percentile(relative, 0.95),
            )
        )
    status = "passed" if raw.get("status") == "passed" and not errors else raw.get("status", "incomplete")
    if status == "passed" and errors:
        status = "failed"
    if raw.get("status") == "passed" and errors:
        status = "failed"
    completed_errors = [error for error in errors if not error.endswith("expected complete blocks.")]
    timed_out = "TimeoutError" in raw.get("error", "") and "budget" in raw.get("error", "").lower()
    audit_status = "failed"
    if status == "passed":
        audit_status = "complete_experiment_validated"
    elif blocks and not completed_errors and missing and timed_out:
        audit_status = "partial_complete_blocks_validated"
    return redact(
        dict(
            status=status,
            audit_status=audit_status,
            raw_status=raw.get("status"),
            scope=raw.get("scope"),
            image=raw.get("image"),
            image_id=raw.get("image_id"),
            error=raw.get("error"),
            source_hashes={Path(key).name: value for key, value in raw.get("source_hashes", {}).items()},
            source_variation=(
                dict(worker_py=WORKER_VARIATION)
                if raw.get("source_hashes", {}).get("worker.py") == WORKER_VARIATION["recorded_at_start_sha256"]
                else {}
            ),
            analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            validation=dict(
                expected_blocks=24,
                complete_blocks=len(blocks),
                expected_timed_commands=144,
                timed_commands=sum(len(block["rows"]) for block in blocks),
                timed_optimizer_updates=sum(
                    row["result"].get("optimizer_updates", 0) for block in blocks for row in block["rows"]
                ),
                profile_commands=sum(
                    len(profile.get("samples", []))
                    for models in raw.get("profiles", {}).values()
                    for profile in models.values()
                ),
                exact_identity_groups=len(identities),
                complete_four_policy_pairs=len(pairs),
                incomplete_comparison_groups=[
                    dict(
                        budget_mb=budget,
                        workload=workload,
                        repeat=repeat,
                        present_policies=sorted(policies),
                        missing_policies=sorted(set(POLICIES) - set(policies)),
                    )
                    for (budget, workload, repeat), policies in paired.items()
                    if set(policies) != set(POLICIES)
                ],
                errors=errors,
                completed_block_errors=completed_errors,
                missing_blocks=[list(key) for key in sorted(missing)],
            ),
            profiles=raw.get("profiles", {}),
            summary=summary,
            paired_comparisons=pairs,
            block_metrics=records,
            prediction_errors=prediction_errors,
            elapsed_s=raw.get("elapsed_s"),
            budget_s=raw.get("budget_s"),
            physical_cleanup=dict(
                owned_containers_empty=raw.get("owned_containers") == "", final_gpu_used_mb=device_after
            ),
            limits=[
                "Independent workers. No EnvRouter, Agent rollout, historical baseline, or official benchmark score.",
                "Profiles use AIME only. GSM8K and HumanEval compute costs are not measured before dispatch.",
                "The 10 GiB budget simulates pressure on an H20. The 24 GiB arm has capacity for all model states.",
                "All main blocks start with empty GPU caches. The 24 GiB arm is not a fully prewarmed control.",
                "The optional fully warm control and original-program crosscheck were not run after budget expiry.",
                "Only two pressure repeats and one all-fit repeat. No statistical significance claim.",
                "The shared node CPU and storage were not exclusive. Other users' jobs can affect timing variance.",
                "Worker peak-memory instrumentation changed midrun; loading transients are not uniformly covered.",
                "Best-simple pair comparison is a hindsight lower envelope, not a deployable policy.",
                "Two optimizer updates tune only the final normalization vector. This does not train the PSRL policy.",
                "Phase sums measure worker work. They are not additive components of parallel wall time.",
                "Only fully recorded blocks enter comparisons. An aborted block can contain unreported partial work.",
            ],
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], **result["validation"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
