"""Audit resident inference evidence without starting containers or GPU work."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def analyze(root, mode):
    directory = root / mode
    record = read(directory / "result.json")
    samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    start = record["work_started_at"]
    paths = list((directory / "jobs").rglob("resident.json"))
    agents = sorted((read(p) for p in paths if read(p)["status"] == "ok"), key=lambda x: x["started_at"])
    requests = [read(r) for p in paths if read(p)["status"] == "ok" for r in p.parent.parent.rglob("request-*.json")]
    count = record.get("episode_count", 2)
    assert len(agents) == count and len(requests) == 2 * count, "Completed work differs from the declared task count."
    accepted = [e for e in record["episodes"] if e["result"]["reward"] == 1 and not e["result"]["exception"]]
    failed = [e for e in record["episodes"] if e not in accepted]
    assert len({e["index"] for e in accepted}) == count, "The runner did not accept every logical episode."
    assert all(e.get("cuda_oom") for e in failed), "Unexpected non-OOM failures need separate investigation."
    assert not record.get("error") and not record.get("cleanup_error"), "The driver or cleanup failed."
    assert not record["containers_remaining"] and not record["gpu_after"]["pids"], "Owned resources remain."
    assert record["gpu_after"]["used_mb"] < 128, "GPU memory was not reclaimed."
    clean_samples = []
    for item in samples:
        clean = {"elapsed_s": item["time"] - start}
        if "gpu" in item:
            clean.update(gpu_used_mb=item["gpu"]["used_mb"], gpu_process_count=len(item["gpu"]["pids"]))
        if "telemetry_error" in item:
            clean["telemetry_error"] = item["telemetry_error"]
        for unit in item.get("ledger", []):
            clean["ledger"] = {
                k: unit[k]
                for k in (
                    "leases_active",
                    "sessions_waiting",
                    "declared_gpu_memory_mb",
                    "gpu_memory_mb",
                    "containers_running",
                    "resource_sample_age_s",
                    "resource_error",
                    "measured_gpu_memory_mb",
                    "accounted_gpu_memory_mb",
                    "owned_gpu_memory_mb",
                )
                if k in unit
            }
            clean["ledger"]["gpu_attribution_error_count"] = len(unit.get("gpu_attribution_errors", {}))
        clean_samples.append(clean)
    intervals = []
    for agent in agents:
        idle = [
            s["gpu"]["used_mb"]
            for s in samples
            if "gpu" in s and agent["idle_started_at"] <= s["time"] <= agent["idle_finished_at"]
        ]
        assert idle and min(idle) > 1024, "GPU residency was not observed during the CPU interval."
        intervals.append(
            {
                "started_s": agent["started_at"] - start,
                "ready_s": agent["ready_at"] - start,
                "idle_start_s": agent["idle_started_at"] - start,
                "idle_end_s": agent["idle_finished_at"] - start,
                "finished_s": agent["finished_at"] - start,
                "idle_gpu_min_mb": min(idle),
                "idle_gpu_max_mb": max(idle),
            }
        )
    responses = sorted(
        [
            {
                "sample_id": r["sample_id"],
                "text": r["response"]["choices"][0]["message"]["content"],
                "completion_tokens": r["response"]["usage"]["completion_tokens"],
                "latency_s": r["finished_at"] - r["started_at"],
            }
            for r in requests
        ],
        key=lambda r: (r["sample_id"], r["latency_s"]),
    )
    summary = {
        "head": record["head"],
        "logical_episodes": count,
        "attempts": len(record["episodes"]),
        "oom_attempts": len(failed),
        "completed_requests": len(requests),
        "resident_idle_s": record.get("resident_idle_s", 8),
        "burst_submitted_s": record.get("burst_submitted_at", start) - start,
        "max_active_leases": max((s.get("ledger", {}).get("leases_active", 0) for s in clean_samples), default=0),
        "ownership_credit_peak_mb": max(
            (s.get("ledger", {}).get("owned_gpu_memory_mb", 0) for s in clean_samples), default=0
        ),
        "attribution_error_samples": sum(
            s.get("ledger", {}).get("gpu_attribution_error_count", 0) > 0 for s in clean_samples
        ),
        "pool_gpu_mb": record["pool_gpu_mb"],
        "request_gpu_mb": record["request_gpu_mb"],
        "runner_sha256": record["runner_sha256"],
        "patch_sha256": hashlib.sha256((directory / "source.patch").read_bytes()).hexdigest(),
        "work_s": record["work_finished_at"] - start,
        "initialization_s": start - record["started_at"],
        "final_cleanup_s": record["finished_at"] - record["work_finished_at"],
        "total_s": record["finished_at"] - record["started_at"],
        "requests": responses,
        "episodes": intervals,
        "gpu_peak_mb": max(s["gpu_used_mb"] for s in clean_samples if "gpu_used_mb" in s),
        "max_gpu_processes": max(s["gpu_process_count"] for s in clean_samples if "gpu_process_count" in s),
        "waiting_samples": sum(s.get("ledger", {}).get("sessions_waiting", 0) > 0 for s in clean_samples),
        "telemetry_error_count": sum("telemetry_error" in s for s in clean_samples),
        "container_resource_error_count": sum(bool(s.get("ledger", {}).get("resource_error")) for s in clean_samples),
        "containers_remaining": 0,
        "gpu_processes_remaining": 0,
        "gpu_after_mb": record["gpu_after"]["used_mb"],
        "leases_after": sum(u["leases_active"] for u in record.get("ledger_after", [])) if mode == "on" else None,
    }
    return summary, clean_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="resident")
    parser.add_argument(
        "--expect-pressure", action="store_true", help="Audit real OOM pressure and trusted ON credits."
    )
    parser.add_argument("--expect-serialized", action="store_true", help="Audit the explicit low-quota pressure case.")
    args = parser.parse_args()
    result, samples = {}, {}
    for mode in ("off", "on"):
        result[mode], samples[mode] = analyze(args.root, mode)
    assert result["off"]["head"] == "0eeead861ffafed6bf31c962d259fd847775dcf9", "OFF has the wrong baseline."
    assert result["on"]["leases_after"] == 0, "Admission cleanup audit failed."
    for field in ("logical_episodes", "pool_gpu_mb", "request_gpu_mb", "resident_idle_s"):
        assert result["on"][field] == result["off"][field], f"The comparison differs in {field}."
    if args.expect_pressure:
        assert result["off"]["oom_attempts"] > 0 and result["on"]["oom_attempts"] == 0, (
            "OOM avoidance was not observed."
        )
        assert result["on"]["ownership_credit_peak_mb"] > 0, "Trusted episode ownership was not exercised."
        assert result["on"]["waiting_samples"] > 0, "Admission did not queue under pressure."
    if args.expect_serialized:
        assert result["on"]["waiting_samples"] > 0, "The pressure case did not produce a waiter."
        assert result["on"]["max_gpu_processes"] == 1 and result["off"]["max_gpu_processes"] == 2, (
            "The intended GPU concurrency contrast was not observed."
        )
    output_key = lambda x: (x["sample_id"], x["text"], x["completion_tokens"])
    assert sorted(map(output_key, result["on"]["requests"])) == sorted(map(output_key, result["off"]["requests"])), (
        "The paired requests produced different outputs."
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.prefix}-comparison.json").write_text(json.dumps(result, indent=2))
    (args.output / f"{args.prefix}-samples.json").write_text(json.dumps(samples, indent=2))
    compact = {
        mode: {k: v for k, v in row.items() if k not in ("requests", "episodes")} for mode, row in result.items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
