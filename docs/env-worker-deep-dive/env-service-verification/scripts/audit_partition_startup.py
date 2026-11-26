"""Archive startup milestones without launching or changing an experiment."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def timestamp(line):
    """Interpret recorded node log timestamps in the declared experiment timezone."""
    match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", line)
    return datetime.strptime(match[0], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()


def analyze_startup(root):
    """Retain source lines and distinguish milestone intervals from isolated costs."""
    result = {}
    for name, mode in (("static", "fixed"), ("resident", "resident"), ("elastic", "resident")):
        arm = root / name / mode
        status = json.loads((arm / "status.json").read_text())
        launcher = (arm / "launcher.log").read_text()
        archive = arm / "startup-raylet.log"
        raylet = Path(status["ray_tmp"]) / "session_latest/logs/raylet.out"
        if not archive.exists():
            archive.write_text(
                "\n".join(
                    line
                    for line in raylet.read_text().splitlines()
                    if "Started worker process with pid" in line or "have not registered within" in line
                )
                + "\n"
            )
        lines = archive.read_text().splitlines()
        pid = re.search(r"TaskRunner pid=(\d+)", launcher)[1]
        process_line = next(line for line in lines if f"Started worker process with pid {pid}," in line)
        markers = {"start": status["started_at"], "task_process_started": timestamp(process_line)}
        evidence = {"task_process_started": process_line}
        for key, phrase in (
            ("ps_manager", "Initialized RequestStatusTracker."),
            ("coordinator", "Initialized RolloutCoordinator"),
            ("weights_ready", "Broadcast initialization complete."),
        ):
            line = next(line for line in launcher.splitlines() if "INFO:" in line and phrase in line)
            markers[key] = timestamp(line)
            evidence[key] = line
        markers["first_episode"] = min(
            json.loads(path.read_text())["started_at"] for path in arm.glob("episodes/*/episode.json")
        )
        errors = [line for line in lines if "have not registered within" in line]
        attempts = []
        for line in errors:
            failed_pid = re.search(r"worker process\((\d+)\)", line)[1]
            start = next(row for row in lines if f"Started worker process with pid {failed_pid}," in row)
            attempts.append({"pid": int(failed_pid), "start": timestamp(start), "timeout": timestamp(line)})
        result[name] = {
            "markers_epoch": markers,
            "elapsed_from_start_s": {key: round(value - markers["start"], 3) for key, value in markers.items()},
            "registration_errors": errors,
            "failed_registration_attempts": attempts,
            "source_lines": evidence,
        }
        if name == "elastic":
            partition = json.loads((root / name / "partition-audit.json").read_text())
            borrow = next(event["at"] for event in partition["events"] if event.get("action") == "borrow")
            active = []
            for path in arm.glob("episodes/*/episode.json"):
                episode = json.loads(path.read_text())
                for command in episode["commands"]:
                    admission = command["admission"]
                    if (
                        admission["unit_id"] in ("gpu-2", "gpu-3")
                        and admission["admitted_at"] <= borrow <= command["finished_at"]
                    ):
                        active.append(
                            {
                                "episode_uid": episode["uid"],
                                "mode": command["mode"],
                                "unit_id": admission["unit_id"],
                                "admitted_before_borrow_s": borrow - admission["admitted_at"],
                                "released_after_borrow_s": admission["released_at"] - borrow,
                            }
                        )
            result[name]["borrow_drain_audit"] = {
                "borrow_started_at": borrow,
                "after_first_episode_s": borrow - markers["first_episode"],
                "active_shared_commands": active,
            }
    (root / "startup-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    analyze_startup(args.result)
