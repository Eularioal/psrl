"""Audit every preregistered placement cell and export descriptive paired evidence."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import analyze_long as base

number = base.number
numeric_fields = base.numeric_fields
digest = base.digest
distribution = base.distribution
BASELINE = base.BASELINE
STRATEGIES = {"affinity", "always_copy", "predictive"}
CORE_CASES = {"default", "short", "boundary", "long"}
FAULT_CASES = {"overpredict_short", "stale_short", "memory_long"}
MODULES = ("warm_dispatch.py", "step_predictor.py", "warm_placement.py")
CORE_SOURCES = ("warm_router_study/matrix.py", "warm_router_study/server.py", "warm_router_study/client.py")
METRICS = ("primary_makespan_s", "closed_cycle_s", "target_latency_s")
CELL_FIELDS = ("round", "target_steps", "blocker_steps", "forecast_scale", "report_age_offset_s", "cold_budget_mb")
REASONS = {
    "no_declared_capacity",
    "known_cold_copy_requested",
    "predictive_margin_met",
    "unknown_hot_completion_affinity_fallback",
    "affinity_retained",
    "deadline_uncertain",
    "no_feasible_hot_replica",
    "predicted_deadline_miss",
    "cold_estimate_unavailable",
}
GPU_FIELDS = ("utilization_gpu_pct", "memory_used_mb", "clock_sm_mhz", "power_draw_w", "query_wall_s")
CPU_SOURCES = {"cgroup_v2", "cgroup_v1_cpu", "cgroup_v1_cpu_cpuacct"}


def backend(value):
    return "native" if value in {"native", "router"} else ("baseline" if value == "baseline" else "unknown")


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,95}", value):
        return None
    return None if re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", value) else value


def sha(value):
    return hashlib.sha256(value).hexdigest() if value is not None else None


def read_json(root, relative):
    content = base.registered_bytes(root, relative)
    try:
        value = json.loads(content) if content is not None else None
    except (ValueError, TypeError):
        value = None
    return value if isinstance(value, dict) else None, sha(content)


def estimate(value):
    if number(value) is not None:
        return dict(point_s=value, statistical_guarantee=False)
    if not isinstance(value, dict):
        return None
    result = numeric_fields(value, ("point_s", "estimate_s", "lower_s", "upper_s"))
    result["interval"] = base.short_audit.interval(value.get("interval", value))
    result["statistical_guarantee"] = False
    return result


def cpu_inspection(raw):
    result = base.inspection(raw)
    if result is None:
        return None
    before = [
        value
        for value in (raw.get("container_inspection_before") or {}).get("containers", [])
        if isinstance(value, dict)
    ]
    after = {
        value.get("id"): value
        for value in (raw.get("container_inspection_after") or {}).get("containers", [])
        if isinstance(value, dict)
    }
    for public, first in zip(result["containers"], before, strict=True):
        last = after.get(first.get("id"), {})
        first_stat, last_stat = first.get("cpu_stat", {}), last.get("cpu_stat", {})
        first_source, last_source = first_stat.get("source"), last_stat.get("source")
        public["before_counter_source"] = first_source if first_source in CPU_SOURCES else None
        public["after_counter_source"] = last_source if last_source in CPU_SOURCES else None
        matches = first_source in CPU_SOURCES and first_source == last_source
        public["counter_sources_match"] = matches
        for stage, stat in (("before", first_stat), ("after", last_stat)):
            public["cpu_" + stage].update(numeric_fields(stat.get("values"), ("throttled_time",)))
        if matches:
            a = public["cpu_before"].get("throttled_time")
            b = public["cpu_after"].get("throttled_time")
            if a is not None and b is not None and b >= a:
                public["cpu_delta"]["throttled_time"] = b - a
        else:
            public.update(cpu_delta={}, average_cpu_cores_used=None, throttled_period_fraction=None)
        public["counter_units"] = dict(
            throttled_time="nanoseconds",
            throttled_usec="microseconds",
            usage_usec="microseconds",
            nr_periods="count",
            nr_throttled="count",
        )
    result["complete"] = bool(result["containers"]) and all(
        value["same_container_observed_after"]
        and value["before_cpu_status"] == "ok"
        and value["after_cpu_status"] == "ok"
        and value["counter_sources_match"]
        for value in result["containers"]
    )
    return result


def auxiliary_cpu_observation(root, runs, cells):
    snapshot, raw_sha = read_json(root, "native-cpu-near-end.json")
    result = dict(
        status="missing" if raw_sha is None else "invalid",
        raw_sha256=raw_sha,
        scope="after all native core, during fault checks, before cleanup",
        is_final_inspection=False,
        interval_includes_fault_work=True,
        affects_cell_audit=False,
    )
    if snapshot is None:
        return result
    native = [run for run in runs if run["backend"] == "native"]
    core = [cell for cell in cells if cell["backend"] == "native" and cell["phase"] == "core"]
    observed_at = number(snapshot.get("observed_unix_s"))
    complete_before_snapshot = sum(
        cell.get("work_complete") is True
        and number(cell.get("_finished_unix_s")) is not None
        and observed_at is not None
        and cell["_finished_unix_s"] <= observed_at
        for cell in core
    )
    reported = number(snapshot.get("completed_core_cells"))
    count_matches = bool(core) and reported == len(core) == complete_before_snapshot
    inspection = cpu_inspection(
        dict(
            container_inspection_before=native[0].get("_container_inspection_before") if len(native) == 1 else None,
            container_inspection_after=snapshot,
        )
    )
    result.update(
        status="ok" if count_matches and inspection and inspection["complete"] else "partial",
        completed_core_cells_reported=reported,
        completed_core_cells_before_snapshot=complete_before_snapshot,
        planned_native_core_cells=len(core),
        completed_core_cells_verified=count_matches,
        cpu_observation=inspection,
    )
    return result


def gpu_samples(root):
    content = base.registered_bytes(root, "gpu-samples.jsonl")
    records, invalid = [], 0
    for line in content.splitlines() if content else []:
        try:
            value = json.loads(line)
        except ValueError:
            invalid += 1
            continue
        if isinstance(value, dict) and number(value.get("time_unix_s")) is not None:
            records.append(value)
    return records, dict(
        raw_sha256=sha(content), records=len(records), invalid_json_lines=invalid, missing_samples_are_zero=False
    )


def gpu_summary(samples):
    valid = [sample for sample in samples if sample.get("status") == "ok"]
    return dict(
        samples=len(samples),
        ok_samples=len(valid),
        unavailable_samples=len(samples) - len(valid),
        measurements={key: distribution(sample.get(key) for sample in valid) for key in GPU_FIELDS},
    )


def attach_gpu(cells, runs, samples):
    indexed = {run["run_ordinal"]: run for run in runs}
    for cell in cells:
        run = indexed.get(cell.get("run_ordinal"), {})
        start, end = cell.get("_started_unix_s"), cell.get("_finished_unix_s")
        if number(start) is None or number(end) is None:
            continue
        devices = run.get("_gpu_indices", [])
        by_role = {}
        for role in ("hot", "cold"):
            ordinal = cell.get(role + "_device")
            physical = devices[ordinal] if isinstance(ordinal, int) and 0 <= ordinal < len(devices) else None
            selected = [
                sample
                for sample in samples
                if sample.get("index") == physical and start <= sample["time_unix_s"] <= end
            ]
            by_role[role] = gpu_summary(selected)
        cell["gpu_telemetry"] = dict(window_s=end - start, by_role=by_role, nominal_sampling_interval_s=2)


def decision(value):
    value = value if isinstance(value, dict) else {}
    result = dict(
        action=value.get("action") if value.get("action") in {"reuse", "wait", "rebuild", "defer", "reject"} else None,
        device_id=value.get("device_id") if value.get("device_id") in {"hot", "cold"} else None,
        strategy=value.get("strategy") if value.get("strategy") in STRATEGIES else None,
        mode=value.get("mode") if value.get("mode") in {"point", "interval"} else None,
        reason_code=value.get("reason_code") if value.get("reason_code") in REASONS else None,
        statistical_guarantee=False,
        estimates=[],
        **numeric_fields(value, ("switch_margin_s",)),
    )
    for item in value.get("estimates", []):
        if not isinstance(item, dict) or item.get("device_id") not in {"hot", "cold"}:
            continue
        result["estimates"].append(
            dict(
                device_id=item["device_id"],
                resident=item.get("resident") is True,
                capacity_fits=item.get("capacity_fits") is True,
                **numeric_fields(item, ("required_gpu_bytes", "available_gpu_bytes")),
                **{key: estimate(item.get(key)) for key in ("waiting", "rebuild", "execution", "completion")},
            )
        )
    return result


def states(values):
    result = []
    for index, value in enumerate(values if isinstance(values, list) else []):
        if not isinstance(value, dict):
            continue
        models = value.get("cache_keys", value.get("resident_models", []))
        if isinstance(models, str) and models.startswith("frozenset(") and models.endswith(")"):
            try:
                models = ast.literal_eval(models[10:-1]) if models[10:-1] else []
            except (ValueError, SyntaxError):
                models = []
        if not isinstance(models, (list, tuple, set, frozenset)):
            models = []
        result.append(
            dict(
                role=value.get("device_id")
                if value.get("device_id") in {"hot", "cold"}
                else ("hot" if index == 0 else "cold"),
                resident_models=[model for model in models if model in base.short_audit.MODEL_NAMES],
                **numeric_fields(value, ("cache_bytes", "budget_bytes", "workspace_bytes", "available_gpu_bytes")),
                active_remaining=estimate(value.get("active_remaining")),
                busy=value.get("busy") if isinstance(value.get("busy"), bool) else None,
            )
        )
    return result


def plan_cell(value, ordinal):
    result = dict(
        cell_ordinal=ordinal,
        cell_id=identifier(value.get("cell_id")),
        backend=backend(value.get("backend")),
        case_id=value.get("case_id") if value.get("case_id") in CORE_CASES | FAULT_CASES else "unknown",
        strategy=value.get("strategy") if value.get("strategy") in STRATEGIES else "unknown",
        phase=value.get("phase") if value.get("phase") in {"core", "fault"} else "unknown",
        **numeric_fields(value, CELL_FIELDS),
    )
    if value.get("reference_case") in CORE_CASES:
        result["reference_case"] = value["reference_case"]
    return result


def observation(raw, run_ordinal):
    target, blocker = base.command(raw.get("target")), base.command(raw.get("blocker"))
    result = dict(
        **plan_cell(raw, 0),
        run_ordinal=run_ordinal,
        recorded_status=raw.get("status") if raw.get("status") in {"passed", "failed", "running"} else "unknown",
        target=target,
        blocker=blocker,
        selected_device=raw.get("selected_device") if raw.get("selected_device") in {"hot", "cold"} else None,
        decision=decision(raw.get("decision")),
        progress_forecast=base.forecast(raw.get("progress_forecast")),
        effective_forecast=base.forecast(raw.get("effective_forecast")),
        effective_active=estimate(raw.get("effective_active")),
        execution_estimate=estimate(raw.get("execution_estimate")),
        rebuild_estimate=estimate(raw.get("rebuild_estimate")),
        snapshots=states(raw.get("snapshots")),
        service_states=states(raw.get("service_states")),
        **numeric_fields(
            raw,
            (
                *METRICS,
                "target_wait_s",
                "blocker_remaining_actual_s",
                "driver_join_both_s",
                "decision_overhead_s",
                "cold_clear_s",
                "cold_clear_rpc_s",
                "actual_report_age_s",
            ),
        ),
    )
    result.pop("cell_ordinal")
    result.update(
        _started_unix_s=number(raw.get("started_unix_s")),
        _finished_unix_s=number(raw.get("finished_unix_s")),
        **numeric_fields(raw, ("hot_device", "cold_device")),
    )
    progress = raw.get("observed_progress", {})
    result["observed_progress"] = numeric_fields(
        progress, ("completed_steps", "total_steps", "progress_every", "compute_elapsed_s")
    )
    for key in ("chunk_steps", "wall_chunk_s"):
        if isinstance(progress.get(key), list):
            result["observed_progress"][key + "_summary"] = distribution(progress[key])
    admission = raw.get("server_admission", {})
    result["server_admission"] = dict(
        **numeric_fields(admission, ("required_bytes", "available_bytes")),
        admits=admission.get("admits") if isinstance(admission.get("admits"), bool) else None,
    )
    result["work_complete"] = bool(
        result["recorded_status"] == "passed"
        and target
        and blocker
        and all(item.get("status") == "passed" for item in (target, blocker))
    )
    result["timing_complete"] = all(number(result.get(key)) is not None and result[key] > 0 for key in METRICS)
    return result


def safe_hashes(value):
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str)
        and ".." not in key
        and re.fullmatch(r"(?:psrl|examples|warm_router_study|joint_gpu_study)/[a-zA-Z0-9_./-]+\.py", key)
        and digest(item)
    }


def run_record(root, entry, ordinal):
    raw, raw_sha = read_json(root, entry.get("path"))
    result = dict(
        run_ordinal=ordinal,
        backend=backend(entry.get("backend")),
        raw_status="not_started",
        finished=False,
        blocks=[],
        profiles=[],
    )
    if raw is None:
        result["raw_record_unreadable"] = raw_sha is not None
        return result
    result.update(
        raw_status=raw.get("status")
        if raw.get("status") in {"starting", "running", "passed", "failed", "incomplete"}
        else "unknown",
        finished=number(raw.get("finished")) is not None,
        raw_status_sha256=raw_sha,
        raw_error_present=bool(raw.get("error")),
        git_commit=base.short_audit.digest(raw.get("git_commit"), 40),
        worktree_dirty=bool(raw.get("git_status")),
        container_runtime=base.runtime(raw),
        protocol=base.safe_protocol(raw),
        source_hashes=safe_hashes(raw.get("source_hashes")),
        critical_psrl_hashes=safe_hashes(raw.get("critical_psrl_hashes")),
        policy_hashes={name: digest(raw.get("policy_hashes", {}).get(name)) for name in MODULES},
        cleanup=base.short_audit.cleanup(raw, router=result["backend"] == "native"),
        container_cpu_inspection=cpu_inspection(raw),
        timing_s=numeric_fields(raw, ("setup_s", "elapsed_s", "budget_s")),
        blocks=[observation(value, ordinal) for value in raw.get("blocks", [])],
        profiles=[
            dict(profile_ordinal=index, **numeric_fields(value, ("round", "device")), command=base.command(value))
            for index, value in enumerate(raw.get("profiles", []), start=1)
        ],
    )
    result["_gpu_indices"] = [int(value) for value in raw.get("gpu_before", {}) if re.fullmatch(r"\d+", str(value))]
    result["_container_inspection_before"] = raw.get("container_inspection_before")
    result["request_counts"] = dict(
        total=len(raw.get("requests", [])),
        failed=sum(item.get("status") == "failed" for item in raw.get("requests", [])),
        optimizer_updates=sum(item.get("result", {}).get("optimizer_updates", 0) for item in raw.get("requests", [])),
    )
    if result["backend"] == "baseline":
        patch = Path(entry["path"]).parent / "baseline.patch"
        result.update(
            fixed_commit_matches=result["git_commit"] == BASELINE,
            runner_sha256=digest(raw.get("runner_sha256")),
            runner_worktree_patch_sha256=sha(base.registered_bytes(root, patch)),
        )
    if "next registered cell" in str(raw.get("error", "")):
        result["stop_cause"] = "registered_cell_budget_guard"
    if identifier(raw.get("skipped_budget_cell")):
        result["skipped_budget_cell"] = raw["skipped_budget_cell"]
    return result


def policy_audit(root, manifest):
    directory = manifest.get("policy_dir", "policy")
    declared = manifest.get("policy_hashes", {})
    checks = {}
    for name in MODULES:
        actual = sha(base.registered_bytes(root, Path(directory) / name)) if isinstance(directory, str) else None
        recorded = digest(declared.get(name))
        checks[name] = dict(
            recorded_sha256=recorded, frozen_sha256=actual, passed=bool(recorded and actual and recorded == actual)
        )
    return dict(passed=all(value["passed"] for value in checks.values()), modules=checks)


def profile_audit(plans, runs, protocol):
    groups = sorted({(cell["backend"], cell.get("round")) for cell in plans if cell["phase"] == "core"})
    roles, steps = protocol.get("hot_device_by_round", []), protocol.get("profile_steps", 128)
    result = []
    for name, repeat in groups:
        observed = [
            value
            for run in runs
            if run["backend"] == name
            for value in run["profiles"]
            if value.get("round") == repeat
        ]
        hot = roles[repeat - 1] if isinstance(repeat, int) and 0 < repeat <= len(roles) else None
        expected = {(0, "Qwen3-4B-Base"), (1, "Qwen3-4B-Base"), (hot, "Qwen3-1.7B-Base")}
        actual = {(value.get("device"), (value.get("command") or {}).get("model")) for value in observed}
        checks = dict(
            three_distinct_registered_profiles=len(observed) == 3 and hot in {0, 1} and actual == expected,
            registered_update_counts=bool(observed)
            and all(
                (value.get("command") or {}).get("status") == "passed"
                and (value.get("command") or {}).get("steps") == steps
                and (value.get("command") or {}).get("optimizer_updates") == steps
                for value in observed
            ),
        )
        result.append(
            dict(
                backend=name, round=repeat, observed_profiles=len(observed), passed=all(checks.values()), checks=checks
            )
        )
    return dict(passed=bool(result) and all(value["passed"] for value in result), groups=result)


def attach_cells(plans, runs):
    cells = []
    for planned in plans:
        observed = [value for run in runs for value in run["blocks"] if value["cell_id"] == planned["cell_id"]]
        match = len(observed) == 1 and all(
            observed[0].get(key) == value for key, value in planned.items() if key != "cell_ordinal"
        )
        value = dict(planned, observations=observed, unique_observation=len(observed) == 1, plan_matches=match)
        value["work_complete"] = len(observed) == 1 and observed[0]["work_complete"]
        value["timing_complete"] = len(observed) == 1 and observed[0]["timing_complete"]
        if len(observed) == 1:
            value.update({key: item for key, item in observed[0].items() if key not in planned})
        cells.append(value)
    return cells


def output_check(left, right):
    result = base.compare_commands(left, right)
    result["fields"]["optimizer_updates"] = bool(
        left
        and right
        and number(left.get("optimizer_updates")) is not None
        and left["optimizer_updates"] == right.get("optimizer_updates")
    )
    result["passed"] = all(result["fields"].values())
    return result


def cell_checks(cell, indexed_runs, manifest, policies, source_files_verified):
    run = indexed_runs.get(cell.get("run_ordinal"), {})
    commands = [cell.get("target"), cell.get("blocker")]
    actual_steps_match = all(
        isinstance(value, dict)
        and value.get("steps") == cell.get(key)
        and value.get("optimizer_updates") == cell.get(key)
        for value, key in zip(commands, ("target_steps", "blocker_steps"), strict=True)
    )
    states_by_role = {value["role"]: value for value in cell.get("service_states", [])}
    cold_budget = states_by_role.get("cold", {}).get("budget_bytes")
    budgets_match = number(cell.get("cold_budget_mb")) is not None and cold_budget == cell["cold_budget_mb"] * 1024**2
    admission = cell.get("server_admission", {})
    decision_estimate = next(
        (value for value in cell.get("decision", {}).get("estimates", []) if value["device_id"] == "cold"), {}
    )
    capacity_matches = admission.get("admits") is not None and admission["admits"] == decision_estimate.get(
        "capacity_fits"
    )
    capacity_bytes_match = all(
        number(admission.get(source)) is not None and admission[source] == decision_estimate.get(target)
        for source, target in (("required_bytes", "required_gpu_bytes"), ("available_bytes", "available_gpu_bytes"))
    )
    roles = manifest.get("protocol", {}).get("hot_device_by_round", [])
    repeat = cell.get("round")
    hot = roles[repeat - 1] if isinstance(repeat, int) and 0 < repeat <= len(roles) else None
    image_id = manifest.get("image_id", "")
    image_sha = digest(image_id.removeprefix("sha256:")) if isinstance(image_id, str) else None
    declared_hashes = manifest.get("policy_hashes", {})
    declared_sources = safe_hashes(manifest.get("source_hashes"))
    return dict(
        unique_registered_observation=cell["unique_observation"] and cell["plan_matches"],
        work_and_timings_complete=cell["work_complete"] and cell["timing_complete"],
        actual_update_counts_match=actual_steps_match,
        run_finished=run.get("finished") is True,
        resources_released=run.get("cleanup", {}).get("passed") is True,
        fixed_baseline_correct=run.get("fixed_commit_matches") is True if run.get("backend") == "baseline" else True,
        policies_frozen=policies["passed"]
        and all(run.get("policy_hashes", {}).get(name) == declared_hashes.get(name) for name in MODULES),
        core_sources_registered=all(
            declared_sources.get(name) and run.get("source_hashes", {}).get(name) == declared_sources[name]
            for name in CORE_SOURCES
        ),
        frozen_core_script_files_verified=source_files_verified,
        registered_image_matches=bool(image_sha and run.get("container_runtime", {}).get("image_sha256") == image_sha),
        hot_cold_roles_match=hot in {0, 1} and cell.get("hot_device") == hot and cell.get("cold_device") == 1 - hot,
        decision_identity_matches=cell.get("decision", {}).get("device_id") == cell.get("selected_device")
        and cell.get("selected_device") in {"hot", "cold"}
        and cell.get("decision", {}).get("strategy") == cell.get("strategy"),
        decision_mode_and_margin_match=cell.get("decision", {}).get("mode")
        == manifest.get("protocol", {}).get("mode", "point")
        and cell.get("decision", {}).get("switch_margin_s")
        == manifest.get("protocol", {}).get("switch_margin_s", 2.0),
        service_budget_readback_matches=budgets_match,
        controller_and_server_admission_agree=capacity_matches,
        controller_and_server_admission_bytes_agree=capacity_bytes_match,
        selected_cold_only_if_admitted=cell.get("selected_device") != "cold" or admission.get("admits") is True,
    )


def pair(left, right, repeat, scope):
    checks = {role: output_check((left or {}).get(role), (right or {}).get(role)) for role in ("target", "blocker")}
    runtime_checks = {}
    for key in ("container_runtime", "protocol", "policy_hashes"):
        a, b = (left or {}).get("_run", {}).get(key), (right or {}).get("_run", {}).get(key)
        runtime_checks[key] = a is not None and b is not None and a == b
    valid = bool(left and right and left.get("audit", {}).get("passed") and right.get("audit", {}).get("passed"))
    valid = valid and all(value["passed"] for value in checks.values()) and all(runtime_checks.values())
    complete = all(
        cell
        and all(
            cell.get(key) is True for key in ("work_complete", "timing_complete", "unique_observation", "plan_matches")
        )
        for cell in (left, right)
    )
    measured = {}
    for metric in METRICS:
        a, b = number((left or {}).get(metric)), number((right or {}).get(metric))
        if complete and a is not None and a >= 0 and b is not None and b > 0:
            measured[metric] = dict(treatment_s=a, control_s=b, saved_s=b - a, improvement_pct=100 * (1 - a / b))
    return dict(
        round=repeat,
        scope=scope,
        treatment_cell=(left or {}).get("cell_id"),
        control_cell=(right or {}).get("cell_id"),
        complete=complete,
        passed=valid,
        output_checks=checks,
        runtime_checks=runtime_checks,
        metrics=measured,
    )


def summaries(pairs):
    result = {}
    for metric in METRICS:
        measured = [value["metrics"][metric] for value in pairs if metric in value["metrics"]]
        treated = [value["treatment_s"] for value in measured]
        controls = [value["control_s"] for value in measured]
        changes = [value["improvement_pct"] for value in measured]
        result[metric] = dict(
            planned_pairs=len(pairs),
            measured_pairs=len(measured),
            audited_pairs=sum(value["passed"] and metric in value["metrics"] for value in pairs),
            plan_complete=len(measured) == len(pairs),
            all_measured_pairs_audited=bool(measured)
            and all(value["passed"] for value in pairs if metric in value["metrics"]),
            treatment_s=distribution(treated),
            control_s=distribution(controls),
            saved_s=distribution(value["saved_s"] for value in measured),
            paired_improvement_pct=distribution(changes),
            aggregate_improvement_pct=100 * (1 - sum(treated) / sum(controls)) if controls else None,
            wins=sum(value > 0 for value in changes),
            ties=sum(value == 0 for value in changes),
            losses=sum(value < 0 for value in changes),
            statistical_significance_claimed=False,
        )
    return result


def comparisons(cells):
    result = {}
    indexed = {}
    for cell in cells:
        key = (cell["backend"], cell.get("round"), cell["case_id"], cell["strategy"])
        indexed.setdefault(key, []).append(cell)

    def get(key):
        values = indexed.get(key, [])
        return values[0] if len(values) == 1 else None

    for case in sorted({cell["case_id"] for cell in cells if cell["phase"] == "core"}):
        rounds = sorted({cell["round"] for cell in cells if cell["case_id"] == case})
        for label, treatment, control in (
            ("predictive_vs_affinity", ("native", "predictive"), ("native", "affinity")),
            ("predictive_vs_always_copy", ("native", "predictive"), ("native", "always_copy")),
            ("always_copy_vs_affinity", ("native", "always_copy"), ("native", "affinity")),
            ("native_affinity_vs_fixed_baseline", ("native", "affinity"), ("baseline", "affinity")),
            ("predictive_vs_fixed_baseline", ("native", "predictive"), ("baseline", "affinity")),
        ):
            pairs = [
                pair(
                    get((treatment[0], repeat, case, treatment[1])),
                    get((control[0], repeat, case, control[1])),
                    repeat,
                    "same_workload_and_capacity",
                )
                for repeat in rounds
            ]
            result[f"core_{case}_{label}"] = dict(phase="core", case_id=case, pairs=pairs, summaries=summaries(pairs))
    for fault in sorted({cell["case_id"] for cell in cells if cell["phase"] == "fault"}):
        selected = [cell for cell in cells if cell["case_id"] == fault]
        reference = selected[0].get("reference_case", "long" if fault == "memory_long" else "short")
        scope = (
            "same_workload_with_declared_service_budget_change"
            if fault == "memory_long"
            else "same_workload_with_forecast_fault"
        )
        for control in ("predictive", "affinity"):
            pairs = [
                pair(cell, get(("native", cell["round"], reference, control)), cell["round"], scope)
                for cell in selected
            ]
            result[f"fault_{fault}_vs_{control}"] = dict(
                phase="fault", case_id=fault, pairs=pairs, summaries=summaries(pairs)
            )
    return result


def mixed_comparisons(comparisons_by_case):
    result = {}
    for name, cases in (
        ("all_core_equal_case_counts", sorted(CORE_CASES)),
        ("long_short_equal_case_counts", ["long", "short"]),
    ):
        for label in ("predictive_vs_affinity", "predictive_vs_always_copy", "predictive_vs_fixed_baseline"):
            groups = [comparisons_by_case.get(f"core_{case}_{label}") for case in cases]
            if not all(groups):
                continue
            repeats = sorted({pair["round"] for group in groups for pair in group["pairs"]})
            pairs = []
            for repeat in repeats:
                components = [
                    next((pair for pair in group["pairs"] if pair["round"] == repeat), None) for group in groups
                ]
                metrics = {}
                for metric in METRICS:
                    if all(part and metric in part["metrics"] for part in components):
                        treated = sum(part["metrics"][metric]["treatment_s"] for part in components)
                        control = sum(part["metrics"][metric]["control_s"] for part in components)
                        metrics[metric] = dict(
                            treatment_s=treated,
                            control_s=control,
                            saved_s=control - treated,
                            improvement_pct=100 * (1 - treated / control),
                        )
                pairs.append(
                    dict(
                        round=repeat,
                        passed=all(part and part["passed"] for part in components),
                        component_cases=cases,
                        metrics=metrics,
                    )
                )
            result[f"{name}_{label}"] = dict(
                case_weights={case: 1 for case in cases}, pairs=pairs, summaries=summaries(pairs)
            )
    return result


def forecast_errors(cells):
    groups = {}
    for cell in cells:
        label = f"normal_{cell['backend']}_core" if cell["phase"] == "core" else f"fault_{cell['case_id']}"
        groups.setdefault(label, []).append(cell)
    result = {}
    for label, selected in groups.items():
        errors = {name: [] for name in ("base", "effective")}
        covered = []
        comparable = 0
        for cell in selected:
            if not cell.get("work_complete") or not cell.get("timing_complete"):
                continue
            actual = number(cell.get("blocker_remaining_actual_s"))
            if actual is None:
                continue
            comparable += 1
            initial = cell.get("progress_forecast") or {}
            effective = cell.get("effective_active") or {}
            for name, point in (
                ("base", number(initial.get("estimate_s"))),
                ("effective", number(effective.get("point_s"))),
            ):
                if point is not None:
                    errors[name].append(
                        dict(
                            signed_s=point - actual,
                            absolute_s=abs(point - actual),
                            absolute_relative_pct=100 * abs(point - actual) / actual if actual > 0 else None,
                        )
                    )
            interval = initial.get("remaining")
            if interval is not None:
                covered.append(interval["lower_s"] <= actual <= interval["upper_s"])
        result[label] = dict(
            planned_cells=len(selected),
            actual_residuals=comparable,
            empirical_intervals=len(covered),
            empirical_intervals_covering_actual=sum(covered),
            empirical_coverage_fraction=sum(covered) / len(covered) if covered else None,
            estimates={
                name: dict(
                    count=len(values),
                    unknown=comparable - len(values),
                    signed_error_s=distribution(value["signed_s"] for value in values),
                    absolute_error_s=distribution(value["absolute_s"] for value in values),
                    absolute_relative_error_pct=distribution(value["absolute_relative_pct"] for value in values),
                )
                for name, values in errors.items()
            },
            statistical_guarantee=False,
        )
    return result


def offline_controls(cells):
    results = []
    for repeat, case in sorted(
        {(cell["round"], cell["case_id"]) for cell in cells if cell["backend"] == "native" and cell["phase"] == "core"}
    ):
        selected = {
            cell["strategy"]: cell
            for cell in cells
            if cell["backend"] == "native" and cell["round"] == repeat and cell["case_id"] == case
        }
        if not all(
            strategy in selected
            and selected[strategy].get("work_complete")
            and selected[strategy].get("timing_complete")
            and number(selected[strategy].get("primary_makespan_s")) is not None
            for strategy in STRATEGIES
        ):
            continue
        costs = {strategy: selected[strategy]["primary_makespan_s"] for strategy in STRATEGIES}
        best = min(costs["affinity"], costs["always_copy"])
        results.append(
            dict(
                round=repeat,
                case_id=case,
                measured_primary_s=costs,
                best_observed_fixed_control_s=best,
                predictive_minus_best_control_s=costs["predictive"] - best,
            )
        )
    return dict(
        source="separate_matched_policy_executions",
        online_oracle_used=False,
        true_same_execution_counterfactual_observed=False,
        records=results,
    )


def analyze(root):
    manifest, manifest_sha = read_json(root, "manifest.json")
    if manifest is None or not isinstance(manifest.get("cells"), list) or not isinstance(manifest.get("runs"), list):
        raise ValueError("A readable manifest with preregistered cells and runs is required.")
    plans = [plan_cell(value, ordinal) for ordinal, value in enumerate(manifest["cells"], start=1)]
    runs = [run_record(root, value, ordinal) for ordinal, value in enumerate(manifest["runs"], start=1)]
    policies = policy_audit(root, manifest)
    source_files = {
        name: dict(
            recorded_sha256=digest(manifest.get("source_hashes", {}).get(name)),
            frozen_sha256=sha(base.registered_bytes(root, Path("source-snapshot") / name)),
        )
        for name in CORE_SOURCES
    }
    for value in source_files.values():
        value["passed"] = bool(value["recorded_sha256"] and value["recorded_sha256"] == value["frozen_sha256"])
    source_files_verified = all(value["passed"] for value in source_files.values())
    samples, sampler = gpu_samples(root)
    cells = attach_cells(plans, runs)
    attach_gpu(cells, runs, samples)
    indexed_runs = {run["run_ordinal"]: run for run in runs}
    for cell in cells:
        checks = cell_checks(cell, indexed_runs, manifest, policies, source_files_verified)
        cell["audit"] = dict(passed=all(checks.values()), checks=checks)
        cell["_run"] = indexed_runs.get(cell.get("run_ordinal"), {})
    paired = comparisons(cells)
    mixed = mixed_comparisons(paired)
    complete_pairs = [value for group in paired.values() for value in group["pairs"] if value["complete"]]
    profiles = profile_audit(plans, runs, manifest.get("protocol", {}))
    for cell in cells:
        cell.pop("_run")
    planned_ids = [cell["cell_id"] for cell in plans]
    unplanned = [block for run in runs for block in run["blocks"] if block["cell_id"] not in planned_ids]
    loaded = [run for run in runs if "raw_status_sha256" in run]
    expected_blocks = sum(cell.get("target_steps", 0) + cell.get("blocker_steps", 0) for cell in plans)
    profile_rounds = {(cell["backend"], cell.get("round")) for cell in plans if cell["phase"] == "core"}
    profile_steps = manifest.get("protocol", {}).get("profile_steps", 128)
    expected_profiles = len(profile_rounds) * 3 * profile_steps
    measured_profiles = sum(
        (profile["command"] or {}).get("optimizer_updates", 0) for run in runs for profile in run["profiles"]
    )
    measured_blocks = sum(
        (block.get(role) or {}).get("optimizer_updates", 0)
        for run in runs
        for block in run["blocks"]
        for role in ("target", "blocker")
    )
    requested_updates = sum(run.get("request_counts", {}).get("optimizer_updates", 0) for run in runs)
    recorded_patch = digest(manifest.get("current_worktree_patch_sha256"))
    actual_patch = sha(base.registered_bytes(root, "current-worktree.patch"))
    checks = dict(
        valid_unique_plan=bool(plans)
        and all(planned_ids)
        and len(set(planned_ids)) == len(planned_ids)
        and all(
            cell["backend"] in {"native", "baseline"}
            and cell["case_id"] in CORE_CASES | FAULT_CASES
            and cell["strategy"] in STRATEGIES
            and cell["phase"] in {"core", "fault"}
            and all(
                isinstance(cell.get(key), int) and cell[key] > 0 for key in ("round", "target_steps", "blocker_steps")
            )
            for cell in plans
        ),
        no_unplanned_work=not unplanned,
        all_planned_cells_audited=bool(cells) and all(cell["audit"]["passed"] for cell in cells),
        all_planned_runs_finished=len(loaded) == len(runs) and all(run["finished"] for run in runs),
        all_raw_runs_passed=bool(runs) and all(run["raw_status"] == "passed" for run in runs),
        all_started_resources_released=bool(loaded) and all(run["cleanup"]["passed"] for run in loaded),
        policy_files_verified=policies["passed"],
        frozen_core_script_files_verified=source_files_verified,
        current_worktree_patch_verified=bool(recorded_patch and recorded_patch == actual_patch),
        all_registered_profiles_audited=profiles["passed"],
        all_paired_output_and_runtime_checks_passed=bool(paired)
        and all(value["passed"] for group in paired.values() for value in group["pairs"]),
        all_complete_pairs_output_and_runtime_checks_passed=bool(complete_pairs)
        and all(value["passed"] for value in complete_pairs),
        no_failed_requests=all(run.get("request_counts", {}).get("failed", 0) == 0 for run in loaded),
        complete_optimizer_update_count=requested_updates == expected_blocks + expected_profiles,
        observed_optimizer_count_consistent=requested_updates == measured_blocks + measured_profiles,
    )
    terminal = len(loaded) == len(runs) and all(run["finished"] for run in runs)
    status = (
        "passed"
        if all(checks.values())
        else ("failed" if terminal and all(cell["work_complete"] for cell in cells) else "incomplete")
    )
    result = dict(
        schema_version=1,
        scope="Controlled fixed-step PostTrain placement matrix through two real PSRL execution backends.",
        manifest_sha256=manifest_sha,
        analysis_source_sha256=sha(Path(__file__).read_bytes()),
        analysis_helper_sha256=sha(Path(base.__file__).read_bytes()),
        required_fixed_baseline=BASELINE,
        planned_cells=len(plans),
        observed_complete_cells=sum(cell["work_complete"] for cell in cells),
        audited_cells=sum(cell["audit"]["passed"] for cell in cells),
        suite_execution=dict(
            status=manifest.get("status")
            if manifest.get("status") in {"running", "passed", "incomplete", "failed", "preparing"}
            else "unknown",
            **numeric_fields(manifest, ("budget_s", "elapsed_s")),
        ),
        policy_audit=policies,
        source_audit=dict(
            core_scripts=source_files,
            current_critical_psrl_hashes=safe_hashes(manifest.get("critical_psrl_hashes")),
            current_worktree_patch_recorded_sha256=recorded_patch,
            current_worktree_patch_frozen_sha256=actual_patch,
        ),
        profile_audit=profiles,
        auxiliary_cpu_observation=auxiliary_cpu_observation(root, runs, cells),
        gpu_sampler=sampler,
        runs=runs,
        cells=cells,
        unplanned_observations=unplanned,
        comparisons=paired,
        mixed_comparisons=mixed,
        prediction_audit=forecast_errors(cells),
        offline_control_comparison=offline_controls(cells),
        work_counts=dict(
            expected_block_updates=expected_blocks,
            expected_profile_updates=expected_profiles,
            expected_total_updates=expected_blocks + expected_profiles,
            observed_block_updates=measured_blocks,
            observed_profile_updates=measured_profiles,
            request_derived_updates=requested_updates,
        ),
        audit=dict(
            status=status,
            checks=checks,
            all_planned_cells_exported=True,
            failed_or_incomplete_observations_retained=True,
        ),
        limitations=[
            "Official benchmark score and complete LLM rollout performance are outside this fixed-step extension.",
            "Repeated time blocks share persistent services and do not establish statistical significance.",
            "Reported mixtures use predetermined equal case counts, not a measured benchmark task distribution.",
            "The capacity intervention changes the enforced service budget, not physical GPU memory pressure.",
            "Forecast scale and report age are injected faults. They do not describe naturally occurring error.",
            "Offline controls use separate matched runs. They are neither true counterfactuals nor an online oracle.",
        ],
    )

    def public(value):
        if isinstance(value, dict):
            return {key: public(item) for key, item in value.items() if not key.startswith("_")}
        return [public(item) for item in value] if isinstance(value, list) else value

    return public(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                audit=result["audit"]["status"],
                planned=result["planned_cells"],
                completed=result["observed_complete_cells"],
                audited=result["audited_cells"],
            )
        )
    )
    if result["audit"]["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
