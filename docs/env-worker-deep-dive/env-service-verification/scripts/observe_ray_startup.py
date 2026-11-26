"""Sample owned Python workers before registration without pausing their execution."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def observe(output, py_spy, budget_s):
    """Keep a bounded stack audit scoped to one experiment supervisor's child."""
    deadline = time.monotonic() + budget_s
    sampled = set()
    with (output / "startup-stacks.jsonl").open("x") as log:
        while time.monotonic() < deadline:
            state_file = output / "episode/supervisor.json"
            if not state_file.exists():
                time.sleep(2)
                continue
            state = json.loads(state_file.read_text())
            if state["status"] != "initializing":
                break
            try:
                root = psutil.Process(state["child_pid"])
                processes = root.children(recursive=True)
            except (psutil.NoSuchProcess, KeyError):
                time.sleep(2)
                continue
            for process in processes:
                if time.monotonic() >= deadline:
                    return
                try:
                    if process.status() == psutil.STATUS_ZOMBIE:
                        continue
                    command = process.cmdline()
                    stage = next(
                        (
                            name
                            for name in ("setup_worker.py", "default_worker.py")
                            if any(name in arg for arg in command[:3])
                        ),
                        None,
                    )
                    if stage is None:
                        continue
                    created = process.create_time()
                    age = time.time() - created
                    interval = 20 if age >= 20 else 8 if age >= 8 else None
                    key = (process.pid, created, interval)
                    if interval is None or key in sampled:
                        continue
                    sampled.add(key)
                    row = {
                        "at": time.time(),
                        "pid": process.pid,
                        "created": created,
                        "age_s": age,
                        "stage": stage,
                        "cpu_times": process.cpu_times()._asdict(),
                        "process_status": process.status(),
                    }
                    result = subprocess.run(
                        [str(py_spy), "dump", "--pid", str(process.pid), "--nonblocking", "--full-filenames"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    start = result.stdout.find("Thread ")
                    stack = result.stdout[start:] if start >= 0 else ""
                    row.update(exit_code=result.returncode, stack=stack, error=result.stderr)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                except psutil.NoSuchProcess:
                    continue
                except (psutil.AccessDenied, subprocess.TimeoutExpired) as exc:
                    log.write(json.dumps({"at": time.time(), "pid": process.pid, "error": str(exc)}) + "\n")
                    log.flush()
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--py-spy", type=Path, required=True)
    parser.add_argument("--budget-s", type=int, default=1200)
    args = parser.parse_args()
    observe(args.output, args.py_spy, args.budget_s)
