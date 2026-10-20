"""Collect bounded Docker process evidence without changing service state."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess


def command(argv: list[str]) -> dict:
    """Return command output with a fixed timeout."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=8, check=False)
        return {"code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"timeout_seconds": 8}
    except OSError as error:
        return {"error": str(error)}


def read_text(path: Path) -> str:
    """Read a small proc file and preserve access errors."""
    try:
        return path.read_text()
    except OSError as error:
        return f"Unavailable: {error}"


def task_evidence(tid: str) -> dict:
    """Inspect thread state without sending a signal."""
    task = Path("/proc") / tid
    raw = read_text(task / "stat")
    result = {"tid": int(tid), "stat": raw, "stack": read_text(task / "stack"),
              "wchan": read_text(task / "wchan")}
    if ")" in raw:
        fields = raw.rsplit(")", 1)[1].split()
        result["state"] = fields[0]
        if fields[0] == "Z" and len(fields) >= 50:
            status = int(fields[49])
            result["wait_status"] = status
            result["termination_signal"] = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
    if read_text(task / "comm").strip() == "fuse-overlayfs":
        try:
            argv = (task / "cmdline").read_bytes().decode().replace("\0", " ")
            result["uses_docker_storage"] = "/var/lib/docker/fuse-overlayfs/" in argv
        except OSError as error:
            result["command_error"] = str(error)
    return result


def main() -> None:
    """Capture the current evidence and save the requested JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"observed_at": datetime.datetime.now().astimezone().isoformat()}
    listing = command(["ps", "-eLo", "pid,tid,ppid,stat,comm"])
    rows = [line.split() for line in listing.get("stdout", "").splitlines()[1:]]
    selected = [row for row in rows if len(row) >= 5 and row[4] in {"dockerd", "containerd", "fuse-overlayfs", "sync"}]
    result["threads"] = [task_evidence(row[1]) for row in selected[:64]]
    result["matching_thread_count"] = len(selected)
    result["pid1_name"] = read_text(Path("/proc/1/comm")).strip()
    result["disk"] = command(["df", "-h", "/var/lib/docker"])
    result["inodes"] = command(["df", "-i", "/var/lib/docker"])
    result["docker_ping"] = command([
        "curl", "--silent", "--show-error", "--max-time", "3", "--unix-socket",
        "/var/run/docker.sock", "http://localhost/_ping",
    ])
    log = Path("/var/log/dockerd_r2.log")
    if log.exists():
        lines = read_text(log).splitlines()
        terms = ("Starting up", "started new containerd process", "using prior storage driver",
                 "Loading containers", "Processing signal", "API listen", "Daemon has completed")
        result["daemon_log_events"] = [line for line in lines if any(term in line for term in terms)]
    kernel = command(["dmesg", "--ctime"])
    lines = kernel.get("stdout", "").splitlines()
    result["kernel_log"] = {"first": lines[:1], "last": lines[-1:],
                            "daemon_mentions": [line for line in lines if "dockerd" in line or "containerd" in line],
                            "stderr": kernel.get("stderr", ""), "timeout": kernel.get("timeout_seconds")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(result['threads'])} thread snapshots to {args.output}.")


if __name__ == "__main__":
    main()
