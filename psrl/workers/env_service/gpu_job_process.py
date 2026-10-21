"""Run or cancel one container process group with a verified process birth identity."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def identity(pid):
    """Use kernel birth ticks to reject PID reuse during cancellation."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def cancel(directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    (directory / "cancel").touch()
    try:
        record = json.loads((directory / "process.json").read_text())
    except FileNotFoundError:
        return
    pid = record["pid"]
    if identity(pid) == record["birth"]:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(directory, command, *, cwd=None, env=None):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (directory / "cancel").exists():
        return 125
    # Keep the group leader alive until its child has exited and cancellation is no longer needed.
    process = subprocess.Popen(
        ["bash", "-c", command], cwd=cwd, env={**os.environ, **(env or {})}, start_new_session=True
    )
    temporary = directory / "process.tmp"
    temporary.write_text(json.dumps({"pid": process.pid, "birth": identity(process.pid)}))
    temporary.replace(directory / "process.json")
    try:
        while process.poll() is None:
            if (directory / "cancel").exists():
                cancel(directory)
            time.sleep(0.05)
        return process.returncode
    finally:
        (directory / "process.json").unlink(missing_ok=True)


if __name__ == "__main__":
    action, job_id = sys.argv[1:3]
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("Invalid GPU job identity.")
    directory = Path("/tmp/psrl-gpu-jobs") / job_id
    if action == "cancel":
        cancel(directory)
    elif action == "run":
        options = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        sys.exit(run(directory, sys.argv[3], **options))
    else:
        raise ValueError("Unknown GPU job operation.")
