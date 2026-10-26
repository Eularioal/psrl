"""Record the complete Compose collection belonging to one experiment directory."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def observe(root):
    ids = subprocess.check_output(["docker", "ps", "-aq"], text=True, timeout=5).split()
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True, timeout=10)) if ids else []
    projects = {
        c["Config"]["Labels"].get("com.docker.compose.project")
        for c in containers
        if any(Path(m.get("Source", "/")).is_relative_to(root) for m in c.get("Mounts", []))
    } - {None}
    records = []
    for container in containers:
        labels = container["Config"].get("Labels") or {}
        if labels.get("com.docker.compose.project") not in projects:
            continue
        host = container["HostConfig"]
        records.append(
            {
                "id": container["Id"],
                "name": container["Name"],
                "state": container["State"]["Status"],
                "labels": labels,
                "limits": {
                    k: host.get(k)
                    for k in ("Memory", "NanoCpus", "CpuQuota", "CpuPeriod", "PidsLimit", "Devices", "DeviceRequests")
                },
                "mount_sources": [m.get("Source") for m in container.get("Mounts", [])],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--seconds", type=int, default=2400)
    args = parser.parse_args()
    process = Path(f"/proc/{args.pid}/stat")
    identity = process.read_text().rsplit(")", 1)[1].split()[19]
    deadline = time.monotonic() + args.seconds
    with (args.root / "containers.jsonl").open("w") as stream:
        while time.monotonic() < deadline:
            try:
                if process.read_text().rsplit(")", 1)[1].split()[19] != identity:
                    break
            except FileNotFoundError:
                break
            sample = {"time": time.time()}
            try:
                sample["containers"] = observe(args.root.resolve())
            except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
                sample["error"] = repr(exc)
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            time.sleep(5)
    print(json.dumps({"observer_pid": os.getpid(), "finished_at": time.time()}))


if __name__ == "__main__":
    main()
