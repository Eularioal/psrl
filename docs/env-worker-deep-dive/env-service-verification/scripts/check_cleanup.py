"""Record Docker, GPU and Ray state without stopping or deleting anything."""

import argparse
import datetime
import json
import socket
import subprocess
from pathlib import Path

import psutil


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)


def collect():
    processes = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "status"]):
        info = process.info
        args = info["cmdline"] or []
        if info["status"] != psutil.STATUS_ZOMBIE and (
            info["name"] in ("raylet", "gcs_server")
            or info["name"].startswith("ray::")
            or any("ray/dashboard" in arg for arg in args)
        ):
            info["cmdline"] = [arg for arg in args if arg]
            processes.append(info)
    return dict(
        time=datetime.datetime.now().isoformat(),
        hostname=socket.gethostname(),
        docker=command(["docker", "ps", "-a", "--format", "{{.ID}} {{.Names}} {{.Status}}"]),
        gpu=command(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"]),
        ray_processes=processes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = collect()
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
