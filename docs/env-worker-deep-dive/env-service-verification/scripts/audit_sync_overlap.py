"""Audit completed synchronization experiments without starting GPU work."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def audit(root):
    """Match control ownership to command lifetimes and compare equivalent artifacts."""
    arms = {}
    hashes = {}
    scores = {}
    for name in ("wait", "overlap"):
        directory = root / name
        summary = json.loads((directory / "analysis.json").read_text())
        supervisor = json.loads((directory / "supervisor.json").read_text())
        if supervisor["status"] != "complete" or not supervisor["physical_release_verified"]:
            raise ValueError("Both experiment arms must complete and physically release resources.")
        manifest = json.loads((directory / "source-manifest.json").read_text())
        changed = [
            key
            for key, expected in manifest.items()
            if hashlib.sha256((directory / "source" / key).read_bytes()).hexdigest() != expected
        ]
        if changed:
            raise ValueError(f"Frozen experiment files changed: {changed!r}.")
        episodes = [json.loads(p.read_text()) for p in (directory / "resident/episodes").glob("*/episode.json")]
        commands = [command for episode in episodes for command in episode["commands"]]
        # Command boundaries bound the clock offset without claiming a historical calibration.
        low = max(
            c["admission"]["admitted_at"] - (c["result"]["prepared_monotonic_s"] - c["result"]["preparation_s"])
            for c in commands
        )
        high = min(c["admission"]["command_finished_at"] - c["result"]["finished_monotonic_s"] for c in commands)
        overlaps = []
        for event in summary["model_gate_events"]:
            if event["method"] != "nixl_pull_model" or not event.get("shared_at_entry"):
                continue
            for request in event["shared_at_entry"]:
                command = next(c for c in commands if c["admission"]["request_id"] == request)
                admission = command["admission"]
                start, end = event["started_at"], event["finished_at"]
                if not admission["admitted_at"] < start < end < admission["released_at"]:
                    raise ValueError("A logged overlap is outside the physical command lifetime.")
                delay = [command["result"]["prepared_monotonic_s"] + offset - end for offset in (low, high)]
                overlaps.append(
                    {
                        "unit": event["unit"],
                        "request": request,
                        "command_phase": command["mode"],
                        "sync_start": start,
                        "sync_end": end,
                        "transfer_s": end - start,
                        "admitted_at": admission["admitted_at"],
                        "released_at": admission["released_at"],
                        "remaining_lease_after_transfer_s": admission["released_at"] - end,
                        "estimated_compute_start_after_transfer_s_bounds": delay,
                        "inferred_stage": "preparation" if low <= high and delay[0] > 0 else "unresolved",
                    }
                )
        hashes[name] = collections.defaultdict(set)
        scores[name] = collections.defaultdict(set)
        for episode in episodes:
            train = next(c for c in episode["commands"] if c["mode"] == "train")
            key = f"updates={train['result']['updates']},lr={episode['learning_rate']}"
            hashes[name][key].add(train["result"]["tensor_sha256"])
            scores[name][key].add(episode["reward"])
        large_processes = collections.defaultdict(list)
        peak_sample = None
        for line in (directory / "resident/telemetry.jsonl").read_text().splitlines():
            sample = json.loads(line)
            gpu = sample.get("gpus", {}).get("2")
            if gpu is None:
                continue
            if peak_sample is None or gpu["used_mb"] > peak_sample["gpus"]["2"]["used_mb"]:
                peak_sample = sample
            for pid, memory in gpu.get("memory_mb_by_pid", {}).items():
                if 20000 < memory < 30000:
                    large_processes[pid].append((sample["time"], memory))
        process_windows = [
            {
                "pid": pid,
                "first_sample": min(t for t, m in samples),
                "last_sample": max(t for t, m in samples),
                "samples": len(samples),
                "peak_mib": max(m for t, m in samples),
            }
            for pid, samples in large_processes.items()
        ]
        arms[name] = {
            "step_s": summary["timing_totals"]["timing_s/step"],
            "gpu2_large_process_windows": process_windows,
            "gpu2_peak_sample": peak_sample,
            "process_scope": "Diagnostic 20 to 30 GiB process signature. Ownership is not established.",
            "initialization_s": summary["status"]["initialization_s"],
            "launch_to_cleanup_s": summary["launch_to_cleanup_s"],
            "query_p95_s": summary["query_latency_s"]["p95"],
            "model_gate_max_s": summary["model_gate_wait_s"]["max"],
            "environment_wait_mean_s": summary["environment_wait_s"]["mean"],
            "overlapping_transfers": overlaps,
            "inferred_clock_offset_bounds_s": [low, high],
            "clock_assumption": "Common host monotonic epoch and approximately constant wall offset during one arm.",
            "model_versions": summary["smg_observed_model_versions"],
            "gradient_norms": {k: v.get("actor/grad_norm") for k, v in summary["steps"].items()},
            "cuda_readback_all_pass": all(c["result"]["cuda_readback"] == 528 for c in commands),
            "all_command_capacity_released": all(c["admission"]["capacity_released"] for c in commands),
            "physical_release_verified": supervisor["physical_release_verified"],
            "source_files_checked": len(manifest),
            "environment_artifact_hashes": {k: sorted(v) for k, v in hashes[name].items()},
            "environment_scores": {k: sorted(v) for k, v in scores[name].items()},
        }
    common = hashes["wait"].keys() & hashes["overlap"].keys()
    return {
        "arms": arms,
        "step_relative_change": arms["overlap"]["step_s"] / arms["wait"]["step_s"] - 1,
        "wall_relative_change": arms["overlap"]["launch_to_cleanup_s"] / arms["wait"]["launch_to_cleanup_s"] - 1,
        "common_artifact_conditions": sorted(common),
        "common_artifacts_identical": all(hashes["wait"][k] == hashes["overlap"][k] for k in common),
        "common_scores_identical": all(scores["wait"][k] == scores["overlap"][k] for k in common),
        "limitations": [
            "Artifact hashes audit environment adaptation tensors, not policy weights transferred by NIXL.",
            "All overlapping transfers precede environment compute under the stated clock assumption.",
            "GPU memory samples are whole device peaks, not isolated synchronization allocations.",
            "Version observations do not prove request weight contents or bitwise transfer correctness.",
            "Sequential single runs do not identify the cause of total runtime differences.",
            "The overlap arm has extra GPU process samples absent in wait. This is not a clean timing comparison.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    (args.root / "mechanism-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))
