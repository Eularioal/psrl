"""Audit every planned long placement comparison and export public paired results."""

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import analyze as short_audit

BASELINE = short_audit.BASELINE
METRICS = ("primary_makespan_s", "target_latency_s", "tail_makespan_s", "server_work_makespan_s", "closed_cycle_s")
CORE_SOURCES = ("warm_router_study/server.py", "warm_router_study/long_study.py")
POLICIES = ("baseline", "off", "on")
BLOCK_LABELS = {f"{scenario}-{policy}" for scenario in ("long", "short") for policy in POLICIES}
NUMERIC_BLOCK_FIELDS = (
    *METRICS,
    "target_wait_s",
    "blocker_remaining_actual_s",
    "decision_overhead_s",
    "cold_clear_s",
    "cold_clear_rpc_s",
    "target_rpc_wall_s",
    "blocker_rpc_wall_s",
    "driver_join_both_s",
)
COMMAND_TIMINGS = (
    "core_compute_s",
    "progress_write_s",
    "progress_publish_s",
    "progress_publish_count",
    "progress_every",
    "wall_chunk_count",
    "wall_chunk_total_s",
    "progress_write_total_s",
)
TELEMETRY_FIELDS = {
    "gpu_util_pct",
    "gpu_utilization_pct",
    "utilization_gpu_pct",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "clock_sm_mhz",
    "clock_memory_mhz",
    "power_w",
    "temperature_c",
    "gpu_memory_used_mb",
    "memory_used_mb",
    "cpu_usage_usec",
    "cpu_throttled_usec",
    "cpu_nr_periods",
    "cpu_nr_throttled",
    "usage_usec",
    "user_usec",
    "system_usec",
    "nr_periods",
    "nr_throttled",
    "throttled_usec",
    "samples",
    "count",
    "minimum",
    "maximum",
    "mean",
    "median",
    "p10",
    "p90",
    "p95",
    "start",
    "end",
    "delta",
    "duration_s",
    "cpu",
    "cpu_stat",
    "gpu",
    "gpus",
    "devices",
    "hot",
    "cold",
    "start_cpu",
    "end_cpu",
    "cpu_throttling",
}
CPU_COUNTERS = ("usage_usec", "user_usec", "system_usec", "nr_periods", "nr_throttled", "throttled_usec")
HOST_LIMITS = (
    "NanoCpus",
    "CpuPeriod",
    "CpuQuota",
    "CpuShares",
    "Memory",
    "MemorySwap",
    "MemoryReservation",
    "ShmSize",
)
SAMPLE_STATUSES = {"ok", "partial", "unavailable", "timeout", "missing", "failed", "invalid"}


def number(value):
    """Keep finite measurements and reject booleans or missing samples."""
    return short_audit.number(value)


def numeric_fields(value, keys):
    return short_audit.take_numbers(value if isinstance(value, dict) else {}, keys)


def digest(value):
    return short_audit.digest(value)


def normalize_label(value):
    if value in POLICIES:
        return f"long-{value}"
    return value if isinstance(value, str) and value in BLOCK_LABELS else None


def distribution(values):
    samples = [sample for sample in values if number(sample) is not None]
    if not samples:
        return None
    ordered = sorted(samples)
    return dict(
        count=len(samples),
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=statistics.mean(samples),
        median=statistics.median(samples),
        sum=math.fsum(samples),
    )


def telemetry(value):
    """Keep named numeric telemetry without identifiers, paths or free text."""
    if number(value) is not None:
        return value
    if isinstance(value, list):
        return [sample for item in value if (sample := telemetry(item)) is not None]
    if isinstance(value, dict):
        return {
            key: sample
            for key, item in value.items()
            if key in TELEMETRY_FIELDS and (sample := telemetry(item)) is not None
        }
    return None


def host_limits(value):
    """Keep measured numeric limits and explicitly permitted isolation settings."""
    value = value if isinstance(value, dict) else {}
    result = numeric_fields(value, HOST_LIMITS)
    if value.get("IpcMode") in {"private", "host", "none", "shareable"}:
        result["IpcMode"] = value["IpcMode"]
    for key in ("CpusetCpus", "CpusetMems"):
        setting = value.get(key)
        if isinstance(setting, str) and (not setting or re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", setting)):
            result[key] = setting
    return result


def inspection(raw):
    """Match private container identities internally and export numeric CPU deltas."""
    before, after = raw.get("container_inspection_before"), raw.get("container_inspection_after")
    if not isinstance(before, dict) and not isinstance(after, dict):
        return None
    before, after = before or {}, after or {}
    initial, final = before.get("containers", []), after.get("containers", [])
    final_by_id = {value.get("id"): value for value in final if isinstance(value, dict) and value.get("id")}
    result = dict(
        before_status=before.get("status") if before.get("status") in SAMPLE_STATUSES else "unknown",
        after_status=after.get("status") if after.get("status") in SAMPLE_STATUSES else "unknown",
        before_container_count=len(initial),
        after_container_count=len(final),
        containers=[],
        interval_includes_profiles=True,
        per_block_attribution_available=False,
    )
    for ordinal, first in enumerate(initial, start=1):
        if not isinstance(first, dict):
            continue
        last = final_by_id.get(first.get("id"), {})
        first_stat, last_stat = first.get("cpu_stat", {}), last.get("cpu_stat", {})
        first_values = numeric_fields(first_stat.get("values"), CPU_COUNTERS)
        last_values = numeric_fields(last_stat.get("values"), CPU_COUNTERS)
        delta = {
            key: last_values[key] - value
            for key, value in first_values.items()
            if key in last_values and last_values[key] >= value
        }
        sample = dict(
            container_ordinal=ordinal,
            same_container_observed_after=bool(last),
            before_cpu_status=first_stat.get("status") if first_stat.get("status") in SAMPLE_STATUSES else "unknown",
            after_cpu_status=last_stat.get("status") if last_stat.get("status") in SAMPLE_STATUSES else "unknown",
            host_config_before=host_limits(first.get("host_config")),
            host_config_after=host_limits(last.get("host_config")),
            cpu_before=first_values,
            cpu_after=last_values,
            cpu_delta=delta,
        )
        began, ended = number(first_stat.get("observed_unix_s")), number(last_stat.get("observed_unix_s"))
        duration = ended - began if began is not None and ended is not None and ended > began else None
        sample["observation_interval_s"] = duration
        sample["average_cpu_cores_used"] = (
            delta["usage_usec"] / (duration * 1e6) if duration and "usage_usec" in delta else None
        )
        periods, throttled = delta.get("nr_periods"), delta.get("nr_throttled")
        sample["throttled_period_fraction"] = throttled / periods if periods and throttled is not None else None
        result["containers"].append(sample)
    return result


def command(value):
    result = short_audit.command(value)
    if result is None:
        return None
    result.update(numeric_fields(value, COMMAND_TIMINGS))
    for field in ("wall_step_s", "wall_chunk_s", "chunk_wall_s", "core_step_s"):
        if isinstance(value.get(field), list):
            result[field + "_summary"] = distribution(value[field])
    if isinstance(value.get("chunk_steps"), list):
        result["chunk_steps_summary"] = distribution(value["chunk_steps"])
    samples = value.get("step_s", value.get("core_step_s", []))
    measured = distribution(samples) if isinstance(samples, list) else None
    if measured and number(value.get("compute_s")) is not None:
        result["sampled_step_sum_s"] = measured["sum"]
        result["compute_outside_sampled_steps_s"] = value["compute_s"] - measured["sum"]
    return result


def forecast(value):
    """Select numerical forecast fields without changing their measured meaning."""
    if not isinstance(value, dict):
        return None
    result = numeric_fields(
        value,
        (
            "point_s",
            "estimate_s",
            "lower_s",
            "upper_s",
            "median_step_s",
            "samples",
            "completed_steps",
            "total_steps",
            "sample_count",
            "remaining_steps",
            "per_step_estimate_s",
            "per_step_lower_s",
            "per_step_upper_s",
            "sample_chunks",
            "next_report_upper_s",
        ),
    )
    interval = short_audit.interval(value)
    if interval is not None:
        result["remaining"] = interval
    per_step = short_audit.interval(dict(lower_s=value.get("per_step_lower_s"), upper_s=value.get("per_step_upper_s")))
    if per_step is not None:
        result["per_step"] = per_step
    for key in ("remaining", "per_step", "duration"):
        interval = short_audit.interval(value.get(key))
        if interval is not None:
            result[key] = interval
    if value.get("status") in {"complete", "estimated", "insufficient_history", "overrun", "stale", "unknown"}:
        result["status"] = value["status"]
    result["statistical_guarantee"] = False
    return result


def block(value, run_ordinal, round_id, backend):
    policy = "baseline" if backend == "baseline" else ("on" if value.get("enabled") is True else "off")
    if value.get("policy") in POLICIES:
        policy = value["policy"]
    target, blocker = command(value.get("target")), command(value.get("blocker"))
    result = dict(
        run_ordinal=run_ordinal,
        round=round_id,
        policy=policy,
        scenario=value.get("scenario") if value.get("scenario") in {"long", "short"} else "unknown",
        complete=bool(target and blocker and target.get("status") == "passed" and blocker.get("status") == "passed"),
        target=target,
        blocker=blocker,
        selected_device="cold" if value.get("selected") == 1 else "hot",
        decision=short_audit.safe_decision(value.get("decision")),
    )
    result["label"] = normalize_label(value.get("label")) or f"{result['scenario']}-{policy}"
    result.update(numeric_fields(value, NUMERIC_BLOCK_FIELDS))
    if value.get("decision", {}).get("mode") == "median_point_with_2s_margin":
        result["decision"]["mode"] = "median_point_with_2s_margin"
    progress = value.get("observed_progress", {})
    result["observed_progress"] = numeric_fields(
        progress, ("completed_steps", "total_steps", "progress_every", "compute_elapsed_s")
    )
    for field in ("wall_chunk_s", "chunk_steps"):
        if isinstance(progress.get(field), list):
            result["observed_progress"][field + "_summary"] = distribution(progress[field])
    prediction = forecast(value.get("progress_estimate"))
    result["remaining_forecast"] = prediction
    interval = prediction.get("remaining") if prediction else None
    actual = number(value.get("blocker_remaining_actual_s"))
    result["residual_covered"] = (
        interval["lower_s"] <= actual <= interval["upper_s"] if interval is not None and actual is not None else None
    )
    estimate = number(prediction.get("estimate_s", prediction.get("point_s"))) if prediction else None
    if estimate is not None and actual is not None:
        result["residual_prediction_error_s"] = estimate - actual
        result["residual_absolute_error_s"] = abs(estimate - actual)
        result["residual_relative_error_pct"] = 100 * (estimate - actual) / actual if actual > 0 else None
    if isinstance(value.get("telemetry"), dict):
        result["telemetry"] = telemetry(value["telemetry"])
    return result


def frozen_module(raw, folder, name, field):
    recorded = digest(raw.get(field))
    source = folder / name
    actual = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    return dict(
        recorded_sha256=recorded, frozen_file_sha256=actual, passed=bool(recorded and actual and recorded == actual)
    )


def safe_protocol(raw):
    value = raw.get("protocol", {})
    if not isinstance(value, dict):
        value = {}
    result = numeric_fields(
        value,
        (
            "target_steps",
            "blocker_steps",
            "profile_steps",
            "public_examples",
            "arrival_after_completed_steps",
            "progress_every",
            "switch_margin_s",
        ),
    )
    for field in ("target_model", "blocker_model"):
        result[field] = value.get(field) if value.get(field) in short_audit.MODEL_NAMES else None
    if isinstance(value.get("token_limits"), list) and all(number(x) is not None for x in value["token_limits"]):
        result["token_limits"] = value["token_limits"]
    result["official_benchmark_score"] = False
    return result


def sources(raw):
    return {
        key: value
        for key, value in raw.get("source_hashes", {}).items()
        if re.fullmatch(r"warm_router_study/[a-z_]+\.py|joint_gpu_study/worker\.py", key) and digest(value)
    }


def runtime(raw):
    image = raw.get("image_id", "")
    versions = []
    for value in raw.get("service_ready", []):
        versions.append(
            {
                key: version
                for key in ("torch", "transformers")
                if isinstance(version := value.get(key), str)
                and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[+a-zA-Z][0-9a-zA-Z.]*)?", version)
            }
        )
    return dict(
        image_sha256=digest(image.removeprefix("sha256:")) if isinstance(image, str) else None,
        library_versions=versions,
    )


def load_run(root, entry, ordinal):
    """Retain a planned run even when its raw record was never created."""
    backend = entry.get("backend") if entry.get("backend") in {"router", "baseline"} else "unknown"
    round_id = entry.get("round") if isinstance(entry.get("round"), int) else None
    result = dict(
        run_ordinal=ordinal,
        round=round_id,
        backend=backend,
        raw_status="not_started",
        finished=False,
        raw_error_present=False,
        blocks=[],
    )
    labels = entry.get("planned_labels", [])
    result["planned_labels"] = [normalize_label(label) for label in labels] if isinstance(labels, list) else []
    if backend == "baseline":
        result["planned_labels"] = [
            "long-baseline" if label == "long-off" else label for label in result["planned_labels"]
        ]
    try:
        path = (root / entry["path"]).resolve()
        path.relative_to(root)
    except (KeyError, TypeError, ValueError):
        result["load_error"] = "invalid_run_path"
        return result
    if not path.is_file():
        return result
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError):
        result["load_error"] = "unreadable_or_incomplete_raw_record"
        return result
    if not isinstance(raw, dict):
        result["load_error"] = "raw_record_is_not_an_object"
        return result
    status = raw.get("status")
    result.update(
        raw_status=status if status in {"starting", "running", "passed", "failed"} else "unknown",
        finished=number(raw.get("finished")) is not None,
        raw_status_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_error_present=bool(raw.get("error")),
        git_commit=short_audit.digest(raw.get("git_commit"), 40),
        protocol=safe_protocol(raw),
        source_hashes=sources(raw),
        container_runtime=runtime(raw),
        placement_policy=frozen_module(raw, path.parent, "placement-policy.py", "placement_policy_sha256"),
        step_predictor=frozen_module(raw, path.parent, "step-predictor.py", "step_predictor_sha256"),
        cleanup=short_audit.cleanup(raw, router=backend == "router"),
        timing_s=numeric_fields(raw, ("setup_s", "elapsed_s", "profile_s")),
        budgets=numeric_fields(raw, ("budget_s", "cleanup_budget_s", "per_gpu_reserved_mb")),
        blocks=[block(value, ordinal, round_id, backend) for value in raw.get("blocks", [])],
        profiles=[command(value) for value in raw.get("profiles", [])],
    )
    result["completion_reason"] = {"passed": "completed", "failed": "unclassified_failure"}.get(status, "in_progress")
    if status == "failed" and "Insufficient fixed budget for another long block." in str(raw.get("error", "")):
        result["completion_reason"] = "long_block_entry_budget_guard"
    result["counts"] = dict(
        blocks=len(result["blocks"]),
        complete_blocks=sum(v["complete"] for v in result["blocks"]),
        profiles=len(result["profiles"]),
        requests=len(raw.get("requests", [])),
        optimizer_updates=sum(c.get("optimizer_updates", 0) for c in result["profiles"] if c)
        + sum(
            b[role].get("optimizer_updates", 0) for b in result["blocks"] for role in ("target", "blocker") if b[role]
        ),
    )
    if isinstance(raw.get("predictor"), dict):
        result["predictor"] = numeric_fields(raw["predictor"], ("execution_estimate_s", "rebuild_estimate_s"))
        if raw["predictor"].get("decision_mode") == "median_point_with_2s_margin":
            result["predictor"]["decision_mode"] = "median_point_with_2s_margin"
    result["planned_labels_match"] = (
        bool(result["planned_labels"])
        and all(result["planned_labels"])
        and result["planned_labels"] == [value["label"] for value in result["blocks"]]
    )
    if backend == "baseline":
        patch = path.parent / "baseline.patch"
        result.update(
            fixed_commit_matches=result["git_commit"] == BASELINE,
            runner_sha256=digest(raw.get("runner_sha256")),
            runner_worktree_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest() if patch.is_file() else None,
        )
    if isinstance(raw.get("telemetry"), dict):
        result["telemetry"] = telemetry(raw["telemetry"])
    result["container_cpu_inspection"] = inspection(raw)
    return result


def registered_bytes(root, relative):
    try:
        path = (root / relative).resolve()
        path.relative_to(root)
        return path.read_bytes()
    except (OSError, TypeError, ValueError):
        return None


def startup_repair(root, manifest):
    """Retain the zero-work failed attempt and verify the narrow startup repair."""
    repair = manifest.get("startup_repair")
    if not isinstance(repair, dict):
        return dict(present=False, passed=True, archived_failed_attempts=[])
    raw_bytes = registered_bytes(root, repair.get("archived_path"))
    patch = registered_bytes(root, repair.get("patch_path"))
    preflight_bytes = registered_bytes(root, "startup-repair-preflight.json")
    try:
        raw = json.loads(raw_bytes) if raw_bytes else {}
        preflight = json.loads(preflight_bytes) if preflight_bytes else {}
    except (ValueError, TypeError):
        return dict(present=True, passed=False, load_error="startup_repair_evidence_unreadable")
    counts = {field: len(raw.get(field) or []) for field in ("profiles", "blocks", "requests", "service_ready")}
    no_work = not any(counts.values())
    no_created_resources = not any(
        raw.get(field) for field in ("units", "leases", "containers", "ledger_after_cleanup")
    )
    source_name = "warm_router_study/experiment.py"
    archive = Path(repair.get("archived_path", "")).parent
    old_source = registered_bytes(root, archive / "source-snapshot" / source_name)
    new_source = registered_bytes(root, Path("source-snapshot") / source_name)

    def source_digest(value):
        return hashlib.sha256(value).hexdigest() if value is not None else None

    patch_lines = patch.decode().splitlines() if patch else []
    added = [
        line[1:].strip()
        for line in patch_lines
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    ]
    removed = [
        line[1:].strip()
        for line in patch_lines
        if line.startswith("-") and not line.startswith("---") and line[1:].strip()
    ]
    expected_added = [
        "@ray.remote",
        "def worker_origins():",
        "import importlib.util",
        "return {",
        "name: importlib.util.find_spec(name).origin",
        'for name in ("psrl", "verl", "transfer_queue", "harbor")',
        "}",
        'record["ray_worker_sources"] = await worker_origins.remote()',
    ]
    checked, failed_at, resumed = (
        number(preflight.get("checked_at")),
        number(raw.get("finished")),
        number(manifest.get("resumed_at")),
    )
    checks = dict(
        exactly_one_resume=manifest.get("resume_count") == 1,
        archived_failure_finished=raw.get("status") == "failed" and failed_at is not None,
        observed_no_training_work=no_work and no_created_resources,
        declared_no_training_work=repair.get("no_training_work") is True,
        known_module_discovery_failure="long_study_backend" in str(raw.get("error", "")),
        preflight_matches_failed_owner=bool(raw.get("tag")) and raw.get("tag") == preflight.get("tag"),
        preflight_after_failure_before_resume=checked is not None
        and failed_at is not None
        and resumed is not None
        and failed_at <= checked <= resumed,
        physical_cleanup_verified=short_audit.cleanup(preflight, router=False)["passed"],
        changed_source_is_preflight=repair.get("changed_source") == source_name,
        patch_scope_is_preflight_only=added == expected_added
        and removed == ['record["ray_worker_sources"] = await ray.remote(origins).remote()'],
        patch_hash_matches=bool(patch) and source_digest(patch) == digest(repair.get("patch_sha256")),
        original_source_hash_matches=bool(old_source)
        and source_digest(old_source) == digest(repair.get("original_sha256")),
        repaired_source_hash_matches=bool(new_source)
        and source_digest(new_source) == digest(repair.get("repaired_sha256")),
    )
    return dict(
        present=True,
        passed=all(checks.values()),
        scope="dependency_discovery_preflight_only",
        checks=checks,
        original_sha256=digest(repair.get("original_sha256")),
        repaired_sha256=digest(repair.get("repaired_sha256")),
        patch_sha256=source_digest(patch),
        preflight_sha256=source_digest(preflight_bytes),
        archived_failed_attempts=[
            dict(
                raw_status="failed" if raw.get("status") == "failed" else "unknown",
                raw_sha256=source_digest(raw_bytes),
                no_training_work=no_work,
                observed_counts=counts,
                raw_cleanup=short_audit.cleanup(raw, router=True),
                fresh_physical_cleanup=short_audit.cleanup(preflight, router=False),
            )
        ],
        discarded_measured_blocks=counts["blocks"],
    )


def compare_commands(left, right):
    return short_audit.compare_commands(left, right)


def matching_block(runs, round_id, policy):
    matches = [
        value
        for run in runs
        for value in run["blocks"]
        if value["round"] == round_id and value["policy"] == policy and value["scenario"] == "long"
    ]
    return matches[0] if len(matches) == 1 else None


def compare_pair(on, control, round_id):
    complete = bool(on and control and on["complete"] and control["complete"])
    checks = {
        role: compare_commands(on.get(role) if on else None, control.get(role) if control else None)
        for role in ("target", "blocker")
    }
    metrics = {}
    for name in METRICS:
        a, b = (on or {}).get(name), (control or {}).get(name)
        if number(a) is not None and number(b) is not None and a >= 0 and b > 0:
            metrics[name] = dict(on_s=a, control_s=b, improvement_pct=100 * (1 - a / b))
    return dict(
        round=round_id,
        on_run_ordinal=on.get("run_ordinal") if on else None,
        control_run_ordinal=control.get("run_ordinal") if control else None,
        complete=complete,
        outputs=checks,
        work_equivalent=complete and all(check["passed"] for check in checks.values()),
        metrics=metrics,
    )


def paired_summary(pairs, metric):
    values = [value["metrics"][metric] for value in pairs if metric in value["metrics"]]
    on, control = [value["on_s"] for value in values], [value["control_s"] for value in values]
    changes = [value["improvement_pct"] for value in values]
    return dict(
        n_pairs=len(values),
        n_audited_pairs=sum(
            metric in pair["metrics"] and pair.get("evidence_audit", {}).get("status") == "passed" for pair in pairs
        ),
        planned_pairs=len(pairs),
        complete=len(values) == len(pairs),
        all_samples_work_equivalent=all(value["work_equivalent"] for value in pairs),
        all_available_pair_audits_passed=bool(values)
        and all(
            pair.get("evidence_audit", {}).get("status") == "passed" for pair in pairs if metric in pair["metrics"]
        ),
        wins=sum(value > 0 for value in changes),
        ties=sum(value == 0 for value in changes),
        losses=sum(value < 0 for value in changes),
        on_s=distribution(on),
        control_s=distribution(control),
        paired_improvement_pct=distribution(changes),
        aggregate_improvement_pct=100 * (1 - math.fsum(on) / math.fsum(control)) if control else None,
        statistical_significance_claimed=False,
    )


def prediction_summary(blocks):
    """Describe every observed forecast without promising calibrated coverage."""
    result = {}
    for policy in POLICIES:
        selected = [value for value in blocks if value["scenario"] == "long" and value["policy"] == policy]
        coverage = [value["residual_covered"] for value in selected if value["residual_covered"] is not None]
        result[policy] = dict(
            blocks=len(selected),
            comparable_intervals=len(coverage),
            intervals_covering_actual=sum(coverage),
            interval_coverage_fraction=sum(coverage) / len(coverage) if coverage else None,
            signed_error_estimate_minus_actual_s=distribution(
                value.get("residual_prediction_error_s") for value in selected
            ),
            absolute_error_s=distribution(value.get("residual_absolute_error_s") for value in selected),
            relative_error_pct=distribution(value.get("residual_relative_error_pct") for value in selected),
            statistical_guarantee=False,
        )
    return result


def identity(runs):
    loaded = [run for run in runs if "container_runtime" in run]
    images = [run["container_runtime"]["image_sha256"] for run in loaded]
    versions = [version for run in loaded for version in run["container_runtime"]["library_versions"]]
    policies = [run["placement_policy"]["recorded_sha256"] for run in loaded]
    predictors = [run["step_predictor"]["recorded_sha256"] for run in loaded]
    protocols = [run["protocol"] for run in loaded]
    source_checks = {}
    for key in CORE_SOURCES:
        hashes = [run["source_hashes"].get(key) for run in loaded]
        source_checks[key] = bool(hashes) and all(hashes) and len(set(hashes)) == 1
    return dict(
        all_planned_runs_loaded=len(loaded) == len(runs),
        same_image=bool(images) and all(images) and len(set(images)) == 1,
        same_library_versions=len(versions) == 2 * len(runs)
        and all(set(version) == {"torch", "transformers"} and version == versions[0] for version in versions),
        same_placement_policy=bool(policies) and all(policies) and len(set(policies)) == 1,
        same_step_predictor=bool(predictors) and all(predictors) and len(set(predictors)) == 1,
        same_protocol=bool(protocols) and all(value == protocols[0] for value in protocols),
        same_core_sources=source_checks,
    )


def pair_evidence(pair, runs, manifest, repair):
    """Audit completed comparisons independently from the unfinished suite plan."""
    indexed = {run["run_ordinal"]: run for run in runs}
    selected = [indexed.get(pair.get(key)) for key in ("on_run_ordinal", "control_run_ordinal")]
    available = [run for run in selected if run is not None]
    runtime_checks = identity(available)
    source_hashes = sources(manifest)
    modules = manifest.get("module_hashes", {})
    checks = dict(
        both_runs_finished_passed=len(available) == 2
        and all(run["finished"] and run["raw_status"] == "passed" for run in available),
        both_physical_cleanups_passed=len(available) == 2
        and all(run.get("cleanup", {}).get("passed") for run in available),
        preregistered_block_order_matches=len(available) == 2
        and all(run.get("planned_labels_match") for run in available),
        work_and_outputs_equal=pair["work_equivalent"],
        primary_measurement_present="primary_makespan_s" in pair["metrics"],
        same_runtime=len(available) == 2
        and all(value for key, value in runtime_checks.items() if key != "same_core_sources")
        and all(runtime_checks["same_core_sources"].values()),
        registered_core_sources=len(available) == 2
        and all(
            source_hashes.get(key)
            and all(run.get("source_hashes", {}).get(key) == source_hashes[key] for run in available)
            for key in CORE_SOURCES
        ),
        frozen_modules_verified=len(available) == 2
        and all(
            run.get(field, {}).get("passed") and run[field]["recorded_sha256"] == modules.get(module)
            for run in available
            for module, field in (("placement-policy", "placement_policy"), ("step-predictor", "step_predictor"))
        ),
        fixed_baseline_correct=all(
            run.get("fixed_commit_matches") for run in available if run["backend"] == "baseline"
        ),
        startup_repair_verified=repair["passed"],
    )
    terminal = len(available) == 2 and all(run["finished"] for run in available)
    return dict(status="passed" if all(checks.values()) else ("failed" if terminal else "incomplete"), checks=checks)


def analyze_suite(root):
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("runs", [])
    rounds = manifest.get("rounds", manifest.get("plan", {}).get("rounds", 3))
    if not isinstance(rounds, int) or not 1 <= rounds <= 100:
        raise ValueError("The manifest must declare a bounded positive round count.")
    runs = [load_run(root, entry, ordinal) for ordinal, entry in enumerate(entries, start=1)]
    repair = startup_repair(root, manifest)
    comparisons = {}
    for label, policy in (("on_vs_off", "off"), ("on_vs_fixed_baseline", "baseline")):
        pairs = [
            compare_pair(matching_block(runs, index, "on"), matching_block(runs, index, policy), index)
            for index in range(1, rounds + 1)
        ]
        for pair in pairs:
            pair["evidence_audit"] = pair_evidence(pair, runs, manifest, repair)
        comparisons[label] = dict(pairs=pairs, summaries={metric: paired_summary(pairs, metric) for metric in METRICS})
    blocks = [value for run in runs for value in run["blocks"]]
    total_updates = sum(run.get("counts", {}).get("optimizer_updates", 0) for run in runs)
    profile_updates = sum(
        command.get("optimizer_updates", 0) for run in runs for command in run.get("profiles", []) if command
    )
    protocol = safe_protocol(manifest)
    expected_updates = rounds * 3 * (protocol.get("target_steps", 1536) + protocol.get("blocker_steps", 2048))
    expected_updates += rounds * 2 * 3 * protocol.get("profile_steps", 128)
    manifest_sources = sources(manifest)
    manifest_modules = manifest.get("module_hashes", {})
    frozen_sources_match = all(
        bool(manifest_sources.get(key))
        and all(run.get("source_hashes", {}).get(key) == manifest_sources[key] for run in runs)
        for key in CORE_SOURCES
    )
    modules_match = all(
        bool(digest(manifest_modules.get(module)))
        and all(run.get(field, {}).get("recorded_sha256") == manifest_modules[module] for run in runs)
        for module, field in (("placement-policy", "placement_policy"), ("step-predictor", "step_predictor"))
    )
    identity_checks = identity(runs)
    observed = [run for run in runs if "raw_status_sha256" in run]
    observed_identity = identity(observed)
    observed_runtime_matches = all(value for key, value in observed_identity.items() if key != "same_core_sources")
    observed_runtime_matches = observed_runtime_matches and all(observed_identity["same_core_sources"].values())
    observed_baselines = [run for run in observed if run["backend"] == "baseline"]
    checks = dict(
        all_planned_runs_finished=bool(runs) and all(run["finished"] for run in runs),
        all_raw_runs_passed=bool(runs) and all(run["raw_status"] == "passed" for run in runs),
        planned_labels_match=bool(runs) and all(run.get("planned_labels_match") for run in runs),
        startup_repair_verified=repair["passed"],
        frozen_sources_match_registered_suite=frozen_sources_match and modules_match,
        optimizer_update_count_matches_plan=total_updates == expected_updates,
        all_resources_released=bool(runs) and all(run.get("cleanup", {}).get("passed") for run in runs),
        frozen_modules_verified=bool(runs)
        and all(
            run.get("placement_policy", {}).get("passed") and run.get("step_predictor", {}).get("passed")
            for run in runs
        ),
        fixed_baseline_matches=any(run["backend"] == "baseline" for run in runs)
        and all(run.get("fixed_commit_matches") for run in runs if run["backend"] == "baseline"),
        expected_long_blocks=len([value for value in blocks if value["scenario"] == "long"]) == rounds * 3,
        same_work_and_outputs=all(
            pair["work_equivalent"] for comparison in comparisons.values() for pair in comparison["pairs"]
        ),
        all_primary_measurements_present=all(
            comparison["summaries"]["primary_makespan_s"]["complete"] for comparison in comparisons.values()
        ),
        same_runtime=all(value for key, value in identity_checks.items() if key != "same_core_sources")
        and all(identity_checks["same_core_sources"].values()),
    )
    passed = all(checks.values())
    terminal = bool(runs) and all(run["raw_status"] in {"passed", "failed"} for run in runs)
    began, resumed = number(manifest.get("started")), number(manifest.get("resumed_at"))
    before_resume = resumed - began if began is not None and resumed is not None and resumed >= began else None
    budget = number(manifest.get("budget_s", manifest.get("plan", {}).get("budget_s")))
    result = dict(
        schema_version=1,
        scope="Repeated controlled PostTrain placement through real PSRL execution backends.",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_helper_sha256=hashlib.sha256(Path(short_audit.__file__).read_bytes()).hexdigest(),
        required_fixed_baseline=BASELINE,
        planned_rounds=rounds,
        planned_long_blocks=rounds * 3,
        work_counts=dict(
            expected_optimizer_updates=expected_updates,
            observed_optimizer_updates=total_updates,
            observed_profile_optimizer_updates=profile_updates,
            observed_block_optimizer_updates=total_updates - profile_updates,
        ),
        registered_protocol=protocol,
        suite_budget_s=budget,
        suite_execution=dict(
            status=manifest.get("status")
            if manifest.get("status") in {"preparing", "running", "passed", "incomplete"}
            else "unknown",
            elapsed_s=number(manifest.get("elapsed_s")),
            registered_retry_count=number(manifest.get("plan", {}).get("retry_count")),
            startup_resume_count=number(manifest.get("resume_count", 0)),
            wall_s_consumed_before_resume=before_resume,
            original_budget_remaining_at_resume_s=max(0, budget - before_resume)
            if budget is not None and before_resume is not None
            else None,
        ),
        primary_metric="primary_makespan_s",
        runs=runs,
        comparisons=comparisons,
        startup_repair=repair,
        prediction_audit=prediction_summary(blocks),
        audit=dict(
            status="passed" if passed else ("failed" if terminal else "incomplete"),
            checks=checks,
            identity=identity_checks,
            raw_statuses_preserved=True,
            all_planned_runs_exported=True,
            completed_pair_counts={
                key: value["summaries"]["primary_makespan_s"]["n_audited_pairs"] for key, value in comparisons.items()
            },
            all_available_pair_evidence_passed=all(
                value["summaries"]["primary_makespan_s"]["all_available_pair_audits_passed"]
                for value in comparisons.values()
            ),
            observed_scope=dict(
                started_runs=len(observed),
                measured_runs=sum(bool(run["blocks"]) for run in observed),
                complete_long_blocks=sum(value["scenario"] == "long" and value["complete"] for value in blocks),
                all_started_runs_finished=bool(observed) and all(run["finished"] for run in observed),
                all_started_resources_released=bool(observed) and all(run["cleanup"]["passed"] for run in observed),
                all_observed_fixed_baselines_match=bool(observed_baselines)
                and all(run.get("fixed_commit_matches") for run in observed_baselines),
                same_observed_runtime=bool(observed) and observed_runtime_matches,
                all_observed_frozen_modules_verified=bool(observed)
                and all(run["placement_policy"]["passed"] and run["step_predictor"]["passed"] for run in observed),
            ),
        ),
        limitations=[
            "The fixed-step public-example extension is not the default or complete official benchmark.",
            "Three rounds were planned. Only observed pairs are summarized, without a statistical significance claim.",
            "The fixed-step forecasts do not establish arbitrary-command predictability.",
            "Observation-channel changes are shared by every group and are not a replication-policy benefit.",
            "The primary boundary includes driver submission and joining both requests, excluding setup and profiles.",
            "Subsequent cache clearing is reported separately and permanent replicas remain reserved until cleanup.",
            "Runtime GPU utilization and clocks were not sampled. Idle pre/post snapshots establish release only.",
            "Missing CPU throttle counters remain unavailable, rather than being interpreted as zero throttling.",
        ],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_suite(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                audit=result["audit"]["status"],
                planned_blocks=result["planned_long_blocks"],
                actual_blocks=sum(len(run["blocks"]) for run in result["runs"]),
            )
        )
    )
    if result["audit"]["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
