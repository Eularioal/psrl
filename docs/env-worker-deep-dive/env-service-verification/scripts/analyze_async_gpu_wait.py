"""
Decompose recorded admission intervals without running GPU work.
"""

import argparse
import json
import sys
from pathlib import Path


def analyze(trial):
    """
    Separate the unavailable budget window from the uninstrumented remainder.
    """
    events = {event["event"]: event for event in trial["events"]}
    first = events["A_return_joined"]["timing"]
    other = events["B_completed"]["timing"]
    following = events["A_next_command_completed"]["timing"]
    requested = following["requested_at"]
    admitted = following["admitted_at"]
    released = other["released_at"]
    if not requested <= admitted or admitted < released:
        raise ValueError("This analysis requires A admission after B release.")
    # Diagram [1] to [5] includes RPC execution, not only queue waiting.
    blocked = max(0.0, released - requested)
    remainder = admitted - max(requested, released)
    origin = trial["started_at"]
    return {
        "mode": trial["mode"],
        "relative_seconds": {
            "A_first_released": first["released_at"] - origin,
            "A_planning_finished": events["model_end"]["time"] - origin,
            "B_command_finished": other["command_finished_at"] - origin,
            "A_next_requested": requested - origin,
            "B_released": released - origin,
            "A_next_admitted": admitted - origin,
            "A_next_released": following["released_at"] - origin,
        },
        "planning_end_to_request_s": requested - events["model_end"]["time"],
        "own_return_join_s": events["A_return_joined"]["time"] - events["model_end"]["time"],
        "request_to_admission_s": admitted - requested,
        "B_budget_unavailable_window_s": blocked,
        "remaining_admission_interval_s": remainder,
        "B_release_to_A_admission_s": admitted - released,
        "boundary_note": "Remaining admission time has no per-RPC, lock, telemetry or device-grant spans.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    if not data.get("passed") or data.get("error") or data.get("cleanup_errors"):
        raise ValueError("Expected a completed probe without run or cleanup errors.")
    json.dump([analyze(trial) for trial in data["trials"]], sys.stdout, indent=2)
    sys.stdout.write("\n")
