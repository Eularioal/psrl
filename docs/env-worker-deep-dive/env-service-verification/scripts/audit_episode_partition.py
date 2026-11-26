"""Audit a terminated episode baseline without treating partial work as a full run."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_integrated_partition import analyze


def audit(root):
    """Preserve unfinished records and report reservation and cleanup evidence."""
    output = root / "episode"
    supervisor = json.loads((output / "supervisor.json").read_text())
    if supervisor["status"] not in ("complete", "failed"):
        raise ValueError("The supervisor must terminate before final analysis.")
    result = analyze(output / "fixed")
    result["supervisor"] = supervisor
    result["complete_workload"] = (
        result["status"]["exit_code"] == 0
        and len(result["steps"]) == 2
        and result["episodes_completed"] == 24
        and result["commands_completed"] == 48
        and result["queries_completed"] == 72
        and supervisor["physical_release_verified"]
    )
    manifest = json.loads((output / "source-manifest.json").read_text())
    result["source_changes_after_freeze"] = [
        name
        for name, digest in manifest.items()
        if hashlib.sha256((output / "source" / name).read_bytes()).hexdigest() != digest
    ]
    episodes = [json.loads(path.read_text()) for path in (output / "fixed").glob("episodes/*/episode.json")]
    result["episode_reservation_audit"] = [
        {
            "uid": row["uid"],
            "status": row["status"],
            "unit": row.get("host_unit"),
            "reservation": row.get("gpu_reservation"),
            "lease_granted_s": row.get("lease_granted_at", row["started_at"]) - row["started_at"],
            "wait_after_first_reply_s": row.get("lease_granted_at", row["queries"][0]["finished_at"])
            - row["queries"][0]["finished_at"],
            "finished_s": row["finished_at"] - row["started_at"] if "finished_at" in row else None,
            "released_between_commands": any(command["admission"]["capacity_released"] for command in row["commands"]),
        }
        for row in sorted(episodes, key=lambda row: row.get("lease_granted_at", float("inf")))
    ]
    peaks = {}
    for line in (output / "fixed/telemetry.jsonl").read_text().splitlines():
        for unit in json.loads(line).get("units", []):
            peaks[unit["unit_id"]] = max(peaks.get(unit["unit_id"], 0), unit["leases_active"])
    result["peak_episode_leases"] = peaks
    result["comparison_note"] = "Only completed equal workloads support a full training speed ratio."
    (root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    audit(args.result)
