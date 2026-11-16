"""Audit fixed-step work equivalence and export explicitly selected public evidence."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

BASELINE = "0eeead861ffafed6bf31c962d259fd847775dcf9"
SCENARIOS = ("short", "long")
OUTPUT_FIELDS = ("model", "steps", "data_sha256", "norm_sha256", "losses_sha256")
MODEL_NAMES = {"Qwen3-4B-Base", "Qwen3-1.7B-Base", "SmolLM3-3B-Base"}
SOURCE_NAMES = {
    *(f"warm_router_study/{name}.py" for name in ("main", "server", "client", "experiment", "baseline", "analyze")),
    "joint_gpu_study/worker.py",
    "psrl/workers/env_service/warm_placement.py",
}


def digest(value, length=64):
    return value if isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) else None


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def take_numbers(value, keys):
    return {key: number(value.get(key)) for key in keys if number(value.get(key)) is not None}


def interval(value):
    if not isinstance(value, dict):
        return None
    lower, upper = number(value.get("lower_s")), number(value.get("upper_s"))
    if lower is None or upper is None or not 0 <= lower <= upper:
        return None
    return dict(lower_s=lower, upper_s=upper)


def command(value):
    if not isinstance(value, dict):
        return None
    result = take_numbers(
        value,
        (
            "steps",
            "optimizer_updates",
            "load_s",
            "compute_s",
            "total_s",
            "first_loss",
            "last_loss",
            "model_bytes",
            "peak_allocated_mb",
            "cache_bytes",
        ),
    )
    result["model"] = value.get("model") if value.get("model") in MODEL_NAMES else None
    result["status"] = value.get("status") if value.get("status") in {"passed", "failed", "error"} else "unknown"
    result["cache_hit"] = value.get("cache_hit") if isinstance(value.get("cache_hit"), bool) else None
    for key in ("data_sha256", "norm_sha256", "losses_sha256"):
        result[key] = digest(value.get(key))
    samples = [sample for sample in value.get("step_s", []) if number(sample) is not None]
    if samples:
        ordered = sorted(samples)
        result["step_duration_s"] = dict(
            count=len(samples),
            minimum=ordered[0],
            maximum=ordered[-1],
            mean=sum(samples) / len(samples),
            median=ordered[len(ordered) // 2],
        )
    return result


def safe_decision(value):
    value = value if isinstance(value, dict) else {}
    result = dict(
        action=value.get("action") if value.get("action") in {"reuse", "wait", "rebuild", "defer", "reject"} else None,
        device=value.get("device_id") if value.get("device_id") in {"hot", "cold"} else None,
        estimates=[],
    )
    for estimate in value.get("estimates", []):
        if estimate.get("device_id") not in {"hot", "cold"}:
            continue
        result["estimates"].append(
            dict(
                device=estimate["device_id"],
                resident=estimate.get("resident") is True,
                memory_fits=estimate.get("memory_fits") is True,
                **{key: interval(estimate.get(key)) for key in ("waiting", "rebuild", "execution", "completion")},
            )
        )
    return result


def block(value):
    result = dict(
        scenario=value.get("scenario") if value.get("scenario") in SCENARIOS else None,
        enabled=value.get("enabled") if isinstance(value.get("enabled"), bool) else None,
        selected_device="cold" if value.get("selected") == 1 else "hot",
        complete=isinstance(value.get("target"), dict) and isinstance(value.get("blocker"), dict),
        decision=safe_decision(value.get("decision")),
        target=command(value.get("target")),
        blocker=command(value.get("blocker")),
    )
    result.update(
        take_numbers(
            value,
            (
                "decision_overhead_s",
                "target_latency_s",
                "target_wait_s",
                "tail_makespan_s",
                "blocker_remaining_actual_s",
            ),
        )
    )
    result["driver_join_both_s"] = number(value.get("response_latency_s"))
    progress = value.get("observed_progress", {})
    prediction = value.get("progress_estimate", {})
    remaining = interval(prediction.get("remaining"))
    actual = number(value.get("blocker_remaining_actual_s"))
    result["remaining_forecast"] = dict(
        completed_steps=number(progress.get("completed_steps")),
        total_steps=number(progress.get("total_steps")),
        status=prediction.get("status")
        if prediction.get("status") in {"complete", "estimated", "insufficient_history", "overrun"}
        else "unknown",
        remaining=remaining,
        per_step=interval(prediction.get("per_step")),
        actual_s=actual,
        covered=(remaining["lower_s"] <= actual <= remaining["upper_s"])
        if remaining is not None and actual is not None
        else None,
        statistical_guarantee=False,
    )
    if result["target"]:
        result["timing_s"] = dict(
            wait=result.get("target_wait_s"),
            load=result["target"].get("load_s"),
            compute=result["target"].get("compute_s"),
            target_latency=result.get("target_latency_s"),
            tail_makespan=result.get("tail_makespan_s"),
        )
    return result


def cleanup(value, router):
    gpu_after = value.get("gpu_after", {})
    devices = []
    for ordinal, snapshot in enumerate(gpu_after.values()):
        used, pids = number(snapshot.get("used_mb")), snapshot.get("pids")
        devices.append(
            dict(
                device_ordinal=ordinal,
                used_mb=used,
                compute_process_count=len(pids) if isinstance(pids, list) else None,
                released=isinstance(pids, list) and not pids and used is not None and used < 128,
            )
        )
    ledger = value.get("ledger_after_cleanup")
    ledger_counts = None
    if isinstance(ledger, list):
        ledger_counts = [
            take_numbers(
                unit,
                (
                    "leases_active",
                    "declared_gpu_memory_mb",
                    "gpu_commands_active",
                    "gpu_commands_waiting",
                    "containers_running",
                    "containers_total",
                ),
            )
            for unit in ledger
        ]
    ledger_zero = (
        (bool(ledger_counts) and all(unit.get("leases_active") == 0 for unit in ledger_counts)) if router else None
    )
    errors_clear = value.get("cleanup_errors") == []
    containers_clear = value.get("containers_remaining") in ("", [])
    physical_clear = bool(devices) and all(device["released"] for device in devices)
    return dict(
        passed=errors_clear and containers_clear and physical_clear and (ledger_zero if router else True),
        cleanup_errors_empty=errors_clear,
        cleanup_error_count=len(value["cleanup_errors"]) if isinstance(value.get("cleanup_errors"), list) else None,
        owned_containers_empty=containers_clear,
        gpu_after=devices,
        router_ledger=ledger_counts,
        all_router_leases_zero=ledger_zero,
    )


def load_run(path, router):
    raw_bytes = Path(path).read_bytes()
    raw = json.loads(raw_bytes)
    state = raw.get("status") if raw.get("status") in {"starting", "running", "passed", "failed"} else "unknown"
    record = dict(
        raw_status=state,
        raw_status_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_error_present=bool(raw.get("error")),
        environment_backend="env_worker" if router else "fixed_psrl_harbor_docker",
        git_commit=digest(raw.get("git_commit"), 40),
        source_hashes={
            key: value for key, value in raw.get("source_hashes", {}).items() if key in SOURCE_NAMES and digest(value)
        },
        budgets=take_numbers(
            raw, ("budget_s", "cleanup_budget_s", "per_gpu_reserved_mb", "declared_service_gpu_budget_mb")
        ),
        timing_s=take_numbers(raw, ("setup_s", "elapsed_s")),
        cleanup=cleanup(raw, router),
        blocks=[block(item) for item in raw.get("blocks", [])],
        profiles=[command(item) for item in raw.get("profiles", [])],
        predictor={key: interval(raw.get("predictor", {}).get(key)) for key in ("rebuild", "execution")},
    )
    policy_hash = digest(raw.get("placement_policy_sha256")) or digest(raw.get("policy_sha256"))
    if policy_hash:
        record["source_hashes"]["psrl/workers/env_service/warm_placement.py"] = policy_hash
    policy_file = Path(path).parent / "placement-policy.py"
    recorded_file_hash = hashlib.sha256(policy_file.read_bytes()).hexdigest() if policy_file.is_file() else None
    record["placement_policy"] = dict(
        recorded_sha256=policy_hash,
        frozen_file_sha256=recorded_file_hash,
        frozen_file_present=policy_file.is_file(),
        frozen_file_matches=recorded_file_hash == policy_hash if recorded_file_hash is not None else None,
    )
    image_id = raw.get("image_id", "")
    record["container_runtime"] = dict(
        image_sha256=digest(image_id.removeprefix("sha256:")) if isinstance(image_id, str) else None,
        python_source=raw.get("python_source") if raw.get("python_source") in {"image", "host_prefix"} else None,
        versions=[],
    )
    for ready in raw.get("service_ready", []):
        record["container_runtime"]["versions"].append(
            {
                name: value
                for name in ("torch", "transformers")
                if isinstance(value := ready.get(name), str)
                and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[+a-zA-Z][0-9a-zA-Z.]*)?", value)
            }
        )
    protocol = raw.get("protocol", {})
    record["protocol"] = dict(
        kind="controlled PostTrain fixed-step extension",
        official_benchmark_score=False,
        target_model=protocol.get("target_model") if protocol.get("target_model") in MODEL_NAMES else None,
        blocker_model=protocol.get("blocker_model") if protocol.get("blocker_model") in MODEL_NAMES else None,
        **take_numbers(protocol, ("target_steps", "public_examples", "arrival_after_completed_steps")),
    )
    record["timing_boundaries"] = dict(
        target_latency_s="Driver arrival to server completion, excluding client return.",
        target_rpc_wall_s="Approximate RPC submission to client return, with its own submission boundary.",
        driver_join_both_s="Driver arrival to joining both target and blocker requests.",
        excluded_from_block_latency=["service setup", "profiles", "subsequent cold-replica clearing", "final cleanup"],
    )
    requests = raw.get("requests", [])
    for original, exported in zip(raw.get("blocks", []), record["blocks"], strict=True):
        target_id = original.get("target", {}).get("id")
        matches = [
            (index, item)
            for index, item in enumerate(requests)
            if target_id is not None and item.get("payload", {}).get("id") == target_id
        ]
        if len(matches) != 1:
            continue
        ordinal, target_request = matches[0]
        exported["target_rpc_wall_s"] = number(target_request.get("wall_s"))
        if exported["selected_device"] == "cold":
            clear = next(
                (
                    item
                    for item in requests[ordinal + 1 :]
                    if item.get("index") == 1 and item.get("payload", {}).get("op") == "clear"
                ),
                None,
            )
            if clear is not None:
                exported["subsequent_cold_replica_clear"] = dict(
                    clear_s=number(clear.get("result", {}).get("clear_s")),
                    rpc_wall_s=number(clear.get("wall_s")),
                    excluded_from_block_latency=True,
                )
    record["counts"] = dict(
        blocks=len(record["blocks"]),
        complete_blocks=sum(item["complete"] for item in record["blocks"]),
        profiles=len(record["profiles"]),
        requests=len(raw.get("requests", [])),
        training_updates=sum(item.get("optimizer_updates", 0) for item in record["profiles"] if item)
        + sum(
            item[role].get("optimizer_updates", 0)
            for item in record["blocks"]
            for role in ("target", "blocker")
            if item[role]
        ),
    )
    if not router:
        record["required_fixed_commit"] = BASELINE
        record["fixed_commit_matches"] = record["git_commit"] == BASELINE
        record["runner_sha256"] = digest(raw.get("runner_sha256"))
        patch = Path(path).parent / "baseline.patch"
        record["runner_worktree_patch_sha256"] = (
            hashlib.sha256(patch.read_bytes()).hexdigest() if patch.is_file() else None
        )
    return record


def compare_commands(left, right):
    checks = {}
    for key in OUTPUT_FIELDS:
        a, b = (left or {}).get(key), (right or {}).get(key)
        checks[key] = a is not None and b is not None and a == b
    return dict(passed=all(checks.values()), fields=checks)


def find_block(run, scenario, enabled):
    found = [item for item in run["blocks"] if item["scenario"] == scenario and item["enabled"] is enabled]
    return found[0] if len(found) == 1 else None


def audit(router, baseline=None):
    checks, summaries = [], {}
    for scenario in SCENARIOS:
        on, off = find_block(router, scenario, True), find_block(router, scenario, False)
        pair = dict(scenario=scenario, complete=bool(on and off and on["complete"] and off["complete"]), outputs={})
        for role in ("target", "blocker"):
            pair["outputs"][role] = compare_commands(on.get(role) if on else None, off.get(role) if off else None)
        pair["passed"] = pair["complete"] and all(item["passed"] for item in pair["outputs"].values())
        checks.append(pair)
        if pair["complete"]:
            summary = {}
            for metric in ("target_latency_s", "tail_makespan_s"):
                a, b = on.get(metric), off.get(metric)
                if number(a) is not None and number(b) is not None and b > 0:
                    summary[metric] = dict(on=a, off=b, improvement_pct=100 * (1 - a / b))
            summaries[scenario] = summary
        if baseline is not None:
            old = find_block(baseline, scenario, False)
            same = dict(
                scenario=scenario,
                comparison="fixed baseline OFF versus Router OFF",
                complete=bool(old and off and old["complete"] and off["complete"]),
                outputs={},
            )
            for role in ("target", "blocker"):
                same["outputs"][role] = compare_commands(
                    old.get(role) if old else None, off.get(role) if off else None
                )
            same["passed"] = same["complete"] and all(item["passed"] for item in same["outputs"].values())
            checks.append(same)
    runs = [router] + ([baseline] if baseline is not None else [])
    backend_identity = None
    if baseline is not None:
        images = [item["container_runtime"]["image_sha256"] for item in runs]
        policies = [item["placement_policy"]["recorded_sha256"] for item in runs]
        versions = [item["container_runtime"]["versions"] for item in runs]
        complete_versions = all(
            len(group) == 2 and all(set(item) == {"torch", "transformers"} for item in group) for group in versions
        )
        backend_identity = dict(
            image_identical=all(images) and images[0] == images[1],
            frozen_policy_identical=all(policies) and policies[0] == policies[1],
            library_versions_identical=complete_versions
            and all(item == versions[0][0] for group in versions for item in group),
        )
    completed = all(item["raw_status"] in {"passed", "failed"} for item in runs)
    passed = (
        all(
            item["raw_status"] == "passed"
            and item["cleanup"]["passed"]
            and item["placement_policy"]["frozen_file_matches"] is not False
            for item in runs
        )
        and all(item["passed"] for item in checks)
        and (baseline is None or baseline["fixed_commit_matches"])
        and (backend_identity is None or all(backend_identity.values()))
    )
    return dict(
        status="passed" if passed else ("failed" if completed else "incomplete"),
        raw_statuses_preserved=True,
        work_equivalence=checks,
        summary=summaries,
        baseline_supplied=baseline is not None,
        backend_identity=backend_identity,
        limitation="Empirical padded intervals have no statistical coverage or global optimality guarantee.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    router = load_run(args.router, router=True)
    baseline = load_run(args.baseline, router=False) if args.baseline else None
    evidence = dict(
        schema_version=1,
        scope="Real PSRL service placement with a controlled local PostTrain variant.",
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        router=router,
        baseline=baseline,
        audit=audit(router, baseline),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                audit=evidence["audit"]["status"],
                router_status=router["raw_status"],
                baseline_status=baseline["raw_status"] if baseline else None,
            )
        )
    )


if __name__ == "__main__":
    main()
