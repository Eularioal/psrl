"""Produce a bounded offline matrix audit without launching additional training."""

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def evaluate(root):
    protocol = json.loads((root / "protocol.json").read_text())
    state = json.loads((root / "status.json").read_text())
    rows = []
    for specification in protocol["arms"]:
        name = specification["name"]
        output = root / name
        path = output / "analysis.json"
        if not path.exists():
            rows.append({"arm": name, "status": state["arms"].get(name, {}).get("status", "not_started")})
            continue
        result = json.loads(path.read_text())
        supervisor = json.loads((output / "supervisor.json").read_text())
        episodes = [
            json.loads(p.read_text()) for p in (output / specification["mode"] / "episodes").glob("*/episode.json")
        ]
        commands = [c for episode in episodes for c in episode["commands"]]
        manifest = json.loads((output / "source-manifest.json").read_text())
        source_ok = all(
            hashlib.sha256((output / "source" / n).read_bytes()).hexdigest() == h for n, h in manifest.items()
        )
        work_ok = (
            len(result["steps"]) == 4
            and result["episodes_completed"] == 48
            and result["commands_completed"] == 96
            and result["queries_completed"] == 144
            and result["environment_updates_completed"] == 25088
            and len(commands) == 96
            and all(c["admission"]["capacity_released"] for c in commands)
            and all(c["result"]["cuda_readback"] == 528 for c in commands)
        )
        first_episode = min(e["started_at"] for e in episodes)
        samples = [
            json.loads(line) for line in (output / specification["mode"] / "telemetry.jsonl").read_text().splitlines()
        ]
        signatures = []
        for gpu in range(2, 2 + specification["replicas"]):
            by_pid = collections.defaultdict(list)
            for sample in samples:
                if sample["time"] < first_episode:
                    continue
                for pid, memory in sample.get("gpus", {}).get(str(gpu), {}).get("memory_mb_by_pid", {}).items():
                    by_pid[pid].append((sample["time"], memory))
            if not by_pid:
                continue
            primary = max(by_pid, key=lambda pid: len(by_pid[pid]))
            for pid, values in by_pid.items():
                large = [(t, m) for t, m in values if m > 16384]
                if pid != primary and large:
                    signatures.append(
                        {
                            "gpu": gpu,
                            "pid": pid,
                            "first": large[0][0],
                            "last": large[-1][0],
                            "samples": len(large),
                            "peak_mib": max(m for t, m in large),
                        }
                    )
        rows.append(
            {
                "arm": name,
                "status": supervisor["status"],
                "expected_work_verified": work_ok,
                "source_unchanged": source_ok,
                "physical_release_verified": supervisor.get("physical_release_verified", False),
                "step_s": result["timing_totals"]["timing_s/step"],
                "initialization_s": result["status"]["initialization_s"],
                "launch_to_cleanup_s": result["launch_to_cleanup_s"],
                "query_p95_s": result["query_latency_s"]["p95"],
                "environment_wait_mean_s": result["environment_wait_s"]["mean"],
                "actions": result["actions"],
                "generated_tokens": result["generated_tokens"],
                "command_placements": result["command_placements"],
                "model_versions": result["smg_observed_model_versions"],
                "overlapping_nixl_pulls": sum(
                    e["method"] == "nixl_pull_model" and bool(e.get("shared_at_entry"))
                    for e in result["model_gate_events"]
                ),
                "unattributed_large_secondary_processes": signatures,
                "memory_probe": "Secondary PID above 16 GiB on a rollout GPU. Ownership is unverified.",
            }
        )
    by_name = {row["arm"]: row for row in rows if "step_s" in row}
    comparisons = {}
    dynamic = by_name.get("three-env-dynamic")
    if dynamic:
        for name in ("three-env-static", "three-env-static-cap5"):
            if name in by_name:
                comparisons[name] = {
                    "dynamic_relative_time_change": dynamic["step_s"] / by_name[name]["step_s"] - 1,
                    "interpretation": "Sequential observation. Audit workload and interference before causal claims.",
                }
    return {
        "updated_at": time.time(),
        "status": state["status"],
        "rows": rows,
        "comparisons": comparisons,
        "limitations": [
            "Generated actions may differ. No repeat statistics or policy quality claim.",
            "No large secondary PID flag does not prove absence of shared host interference.",
            "NIXL event overlap with a lease does not prove overlap with its compute kernels.",
            "This audit performs no retries, tuning, or additional GPU work.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch-s", type=float, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.watch_s
    previous_state = None
    while True:
        current_state = (args.root / "status.json").read_bytes()
        if current_state == previous_state:
            if time.monotonic() >= deadline:
                break
            time.sleep(min(30, max(0, deadline - time.monotonic())))
            continue
        previous_state = current_state
        result = evaluate(args.root)
        save(args.root / "evaluation.json", result)
        lines = [f"Matrix status: {result['status']}"]
        for row in result["rows"]:
            lines.append(
                f"{row['arm']}: {row['status']}"
                + (f", four-step time {row['step_s']:.2f} s" if "step_s" in row else "")
            )
        lines += ["", *result["limitations"]]
        text_path = args.root / "evaluation.txt"
        temporary = text_path.with_suffix(".tmp")
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(text_path)
        if result["status"] != "running" or time.monotonic() >= deadline:
            print("\n".join(lines))
            break
        time.sleep(min(30, max(0, deadline - time.monotonic())))
