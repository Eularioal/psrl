"""Align archived engine observations and measure model retirement against command ownership."""

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


def read_jsonl(path):
    """Read an archived JSON stream, or tolerate its final incomplete write."""
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise


def episode_records(arm):
    """Load completed episode files, skipping an unfinished write during monitoring."""
    for path in sorted((arm / "episodes").glob("*/episode.json")):
        try:
            episode = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        yield episode


def historical_windows(reference):
    """Align actual gateway observations with the engine snapshot age during old gaps."""
    arm = reference / "elastic/resident"
    first = min(episode["started_at"] for episode in episode_records(arm))
    audit = json.loads((reference / "decision-window-audit.json").read_text())
    events = json.loads((arm.parent / "partition-audit.json").read_text())["events"]
    logs = arm / "psrl_logs"
    (gateway_log,) = logs.rglob("smg_routing_status.jsonl")
    engine_logs = sorted(logs.rglob("stats_r*_dp0.jsonl"))
    gateway = [(datetime.fromisoformat(row["ts"]).timestamp(), row) for row in read_jsonl(gateway_log)]
    engines = {
        path.stem: [(datetime.fromisoformat(row["ts"]).timestamp(), row) for row in read_jsonl(path)]
        for path in engine_logs
    }
    samples = [event for event in events if "load" in event]
    start = None
    previous = None
    best = {"duration_s": 0}
    for sample in samples:
        at = sample["at"]
        low = (
            sample["active"] > 1
            and sample["env_pressure"]
            and sample["backlog"] == 0
            and sample["load"] <= (sample["active"] - 1) * 16 * 0.65
        )
        if not low:
            start = None
        elif start is None or at - previous > audit["gap_reset_threshold_s"]:
            start = at
        if start is not None and at - start > best["duration_s"]:
            best = {"start_s": start - first, "end_s": at - first, "duration_s": at - start}
        previous = at
    gaps = []
    for gap in audit["gaps"]:
        begin, end = gap["start"], gap["end"]
        observed = [row for ts, row in gateway if begin <= ts <= end]
        unchanged = {}
        for engine, rows in engines.items():
            zero_snapshot = None
            since = None
            duration = 0
            for ts, row in rows:
                if not begin <= ts <= end:
                    continue
                if row["num_waiting_reqs"] or row["num_running_reqs"]:
                    zero_snapshot = None
                    continue
                if row["total_elapsed_time"] != zero_snapshot:
                    zero_snapshot, since = row["total_elapsed_time"], ts
                duration = max(duration, ts - since)
            unchanged[engine] = duration
        gaps.append(
            {
                "start_s": begin - first,
                "end_s": end - first,
                "duration_s": end - begin,
                "smg_samples": len(observed),
                "smg_paused_samples": sum(bool(row["paused"]) for row in observed),
                "smg_backlog_samples": sum(bool(row["pending_request_num"]) for row in observed),
                "smg_busy_samples": sum(bool(row["running_request_num"]) for row in observed),
                "unchanged_zero_snapshot_s": unchanged,
            }
        )
    return {
        "valid_decision_samples": len(samples),
        "longest_low_window": best,
        "decision_gaps": gaps,
        "decision_times_s": [row["at"] - first for row in samples],
        "gaps_with_unchanged_zero_over_10_s": sum(max(g["unchanged_zero_snapshot_s"].values()) > 10 for g in gaps),
        "engine_freshness_limit_s": 10,
        "policy_window_s": audit["required_window_s"],
        "max_gap_s": audit["gap_reset_threshold_s"],
        "origin": "First episode start in the completed startup repeat.",
    }


def handovers(root):
    """Read each arm's observed roles and map model controls to live command ownership."""
    arm = root / "resident"
    partition_file = arm / "telemetry.jsonl"
    if not partition_file.exists():
        return {}
    partition, controls = {}, {}
    ownership_samples = []
    for row in read_jsonl(partition_file):
        for event in row.get("partition", {}).get("events", []):
            partition[json.dumps(event, sort_keys=True)] = event
        for unit, model in row.get("models", {}).items():
            modes = Counter(model["shared_commands"].values())
            if modes["resident"] and modes["sleeping"]:
                ownership_samples.append(
                    {
                        "at": row["time"],
                        "unit": unit,
                        "old_resident_commands": modes["resident"],
                        "new_sleeping_commands": modes["sleeping"],
                    }
                )
            for event in model["events"]:
                controls[unit, event["method"], event["started_at"]] = dict(event, unit=unit)
    events = sorted(partition.values(), key=lambda event: event["at"])
    commands = {
        command["admission"]["request_id"]: command
        for episode in episode_records(arm)
        for command in episode["commands"]
        if command.get("admission", {}).get("request_id")
    }
    first_episode = min((row["started_at"] for row in episode_records(arm)), default=None)
    transitions = []
    for action in (event for event in events if event.get("action") == "borrow"):
        phases = []
        for event in events:
            if event["at"] < action["at"] or "replica" not in event:
                continue
            phases.append(event)
            if event.get("state") == "borrowed" or "error" in event:
                break
        if not phases:
            continue
        begin, end = action["at"], phases[-1]["at"]
        relevant = [
            event
            for event in controls.values()
            if begin <= event["started_at"] <= event["finished_at"] <= end
            and event["method"] in ("pause_generation", "nixl_sleep")
        ]
        sleep = next((event for event in relevant if event["method"] == "nixl_sleep"), None)
        owners = []
        if sleep:
            for owner in sleep["shared_at_entry"]:
                if owner not in commands:
                    continue
                command = commands[owner]
                admission = command["admission"]
                owners.append(
                    {
                        "request_id": owner,
                        "unit": admission["unit_id"],
                        "mode": command["mode"],
                        "admitted_relative_to_decision_s": admission["admitted_at"] - begin,
                        "returned_relative_to_decision_s": admission["released_at"] - begin,
                        "released_after_physical_sleep_s": admission["released_at"] - sleep["finished_at"],
                    }
                )
        transitions.append(
            {
                "decision_at": begin,
                "decision_after_first_episode_s": None if first_episode is None else begin - first_episode,
                "decision": action,
                "phases": phases,
                "borrowed": phases[-1].get("state") == "borrowed",
                "last_transition_event_after_decision_s": end - begin,
                "controls": relevant,
                "sleep": sleep,
                "tracked_sleep_owners": owners,
                "old_and_new_ownership_sample": max(
                    (
                        sample
                        for sample in ownership_samples
                        if sleep and sample["unit"] == sleep["unit"] and sample["at"] >= end
                    ),
                    key=lambda sample: sample["new_sleeping_commands"],
                    default=None,
                ),
                "commands_admitted_during_drain": [
                    owner for owner in owners if owner["admitted_relative_to_decision_s"] >= 0
                ],
                "commands_admitted_after_release": sum(
                    command["admission"]["unit_id"] == sleep["unit"] and command["admission"]["admitted_at"] > end
                    for command in commands.values()
                )
                if sleep and phases[-1].get("state") == "borrowed"
                else None,
            }
        )
    samples = [event for event in events if "load" in event]
    completed = [event["at"] for event in events if event.get("state") in ("borrowed", "resident", "quarantined")]
    gaps = [
        {
            "duration_s": sample["sample_gap_s"],
            "includes_partition_transition": any(previous["at"] <= at <= sample["at"] for at in completed),
        }
        for previous, sample in zip(samples, samples[1:])
        if (sample.get("sample_gap_s") or 0) > 6
    ]
    poll_durations = [event["poll_s"] for event in samples if "poll_s" in event]
    return {
        "transitions": transitions,
        "valid_samples": len(samples),
        "median_poll_s": statistics.median(poll_durations) if poll_durations else None,
        "skip_counts": dict(Counter(event["skipped"] for event in events if "skipped" in event)),
        "sample_gaps_over_6_s": gaps,
        "restorations": [event for event in events if "restored_version" in event],
        "errors": [
            event for event in events if "controller_error" in event or ("error" in event and "replica" in event)
        ],
    }


def paired_comparison(output, handover):
    """Report completion separately and compare speed only on equal work."""
    from analyze_integrated_partition import analyze

    labels = ("gated", "overlap")
    results = {}
    for label in labels:
        root = output / label
        analysis = root / "analysis.json"
        if not (root / "supervisor.json").exists() or not (root / "resident/telemetry.jsonl").exists():
            return None
        supervisor = json.loads((root / "supervisor.json").read_text())
        if supervisor["status"] not in ("complete", "failed"):
            return None
        raw = json.loads(analysis.read_text()) if analysis.exists() else analyze(root / "resident")
        arm = root / "resident"
        commands = [command for record in episode_records(arm) for command in record["commands"]]
        results[label] = {
            "status": supervisor["status"],
            "exit_code": raw["status"]["exit_code"],
            "steps": len(raw["steps"]),
            "episodes": raw["episodes_completed"],
            "episode_status_counts": raw["episode_status_counts"],
            "commands": raw["commands_completed"],
            "queries": raw["queries_completed"],
            "generated_tokens": raw["generated_tokens"],
            "tool_updates": raw["environment_updates_completed"],
            "initialization_s": raw["status"]["initialization_s"],
            "training_steps_s": raw["timing_totals"].get("timing_s/step"),
            "wait_for_generation_and_rollouts_s": raw["timing_totals"].get("timing_s/wait_for_gen"),
            "observed_start_to_physical_cleanup_s": raw["launch_to_cleanup_s"],
            "query_p95_s": (raw["query_latency_s"] or {}).get("p95"),
            "command_queue_p95_s": (raw["environment_wait_s"] or {}).get("p95"),
            "max_environment_preparation_s": max(
                (command["result"]["preparation_s"] for command in commands), default=None
            ),
            "max_environment_command_s": max(
                (
                    command["admission"]["command_finished_at"] - command["admission"]["admitted_at"]
                    for command in commands
                ),
                default=None,
            ),
            "command_placements": raw["command_placements"],
            "policy_gradient_norms": [step.get("actor/grad_norm") for step in raw["steps"].values()],
            "physical_release_verified": supervisor["physical_release_verified"],
            "transitions": handover[label]["transitions"],
        }
    comparison = {"arms": results}
    workload = ("steps", "episodes", "commands", "queries", "tool_updates")
    comparison["same_completed_workload"] = all(
        result["status"] == "complete" and result["physical_release_verified"] for result in results.values()
    ) and all(results["gated"][key] == results["overlap"][key] for key in workload)
    if comparison["same_completed_workload"]:
        comparison["deltas"] = {
            key: {
                "overlap_minus_gated_s": results["overlap"][key] - results["gated"][key],
                "overlap_over_gated_minus_one": results["overlap"][key] / results["gated"][key] - 1,
            }
            for key in ("initialization_s", "training_steps_s", "observed_start_to_physical_cleanup_s")
        }
    return comparison


def main():
    """Write evidence files from the two runs without starting a cluster."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    history = historical_windows(args.reference)
    (args.output / "historical-window-reconstruction.json").write_text(json.dumps(history, indent=2) + "\n")
    handover = {name: handovers(args.output / name) for name in ("gated", "overlap")}
    (args.output / "handover-timing.json").write_text(json.dumps(handover, indent=2) + "\n")
    comparison = paired_comparison(args.output, handover)
    if comparison:
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
