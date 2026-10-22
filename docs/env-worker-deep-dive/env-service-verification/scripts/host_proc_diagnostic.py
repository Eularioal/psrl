"""Inspect process views without allocating GPUs or changing container configuration."""

import argparse
import http.client
import json
import os
import re
import socket
from pathlib import Path
from urllib.parse import urlsplit

INITIAL_PID_NAMESPACE = 0xEFFFFFFC


def namespace_view(root):
    """Check the same namespace prerequisite as task GPU attribution."""
    result = {"path": str(root), "initial_pid_namespace": False}
    try:
        target = root / "1/ns/pid"
        result["pid1_namespace"] = os.readlink(target)
        result["initial_pid_namespace"] = target.stat().st_ino == INITIAL_PID_NAMESPACE
        result["readable_process_identity"] = all(
            os.access(root / f"1/{name}", os.R_OK) for name in ("stat", "cgroup")
        )
    except OSError as exc:
        result["error"] = type(exc).__name__
    return result


def proc_mounts():
    """List procfs mount roots, including bind mounts with escaped paths."""
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        if after.split()[0] != "proc":
            continue
        fields = before.split()
        path = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])
        mounts.append({"mount_root": fields[3], "options": fields[5], **namespace_view(Path(path))})
    return mounts


def docker_process_views():
    """Inspect only daemon identity, without reading process arguments or environment."""
    daemons = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "comm").read_text().strip() != "dockerd":
                continue
            daemons.append(
                {
                    "pid": int(proc.name),
                    "pid_namespace": os.readlink(proc / "ns/pid"),
                    "proc_view": namespace_view(proc / "root/proc"),
                }
            )
        except OSError:
            continue
    return daemons


def docker_ping():
    """Read the local Unix endpoint directly, without HTTP proxy environment lookup."""
    address = urlsplit(os.environ.get("DOCKER_HOST") or "unix:///var/run/docker.sock")
    if address.scheme != "unix" or address.netloc or not address.path or os.environ.get("DOCKER_CONTEXT"):
        return {"ok": False, "error": "An explicit local Unix socket is required."}
    connection = http.client.HTTPConnection("localhost", timeout=3)
    try:
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.settimeout(3)
        connection.sock.connect(address.path)
        connection.request("GET", "/_ping")
        response = connection.getresponse()
        body = response.read(16).decode(errors="replace").strip()
        return {"ok": response.status == 200 and body == "OK", "status": response.status, "body": body}
    except (OSError, http.client.HTTPException) as exc:
        return {"ok": False, "error": type(exc).__name__}
    finally:
        connection.close()


def pid_join_demo():
    """Use teaching data to distinguish a missing identity from zero GPU usage."""
    host_pid, node_pid, task_pid = 2000, 300, 17
    nvml = {host_pid: 512}
    node_proc = {node_pid: "task-a"}
    host_proc = {host_pid: "task-a"}
    return {
        "kind": "teaching_simulation_not_gpu_measurement",
        "same_process_ids": {"physical_host": host_pid, "experiment_node": node_pid, "task_container": task_pid},
        "wrong_join_matching_pids": [pid for pid in nvml if pid in node_proc],
        "wrong_join_meaning": "Unknown ownership, not zero usage.",
        "correct_join_task_a_memory_mib": sum(value for pid, value in nvml.items() if host_proc.get(pid) == "task-a"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-root", type=Path, default=Path(os.environ.get("PSRL_GPU_HOST_PROC", "/proc")))
    parser.add_argument(
        "--docker", action="store_true", help="Probe the local Docker Unix socket with a 3 second timeout."
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = namespace_view(args.proc_root)
    report = {
        "selected_proc": selected,
        "proc_mounts": proc_mounts(),
        "docker_daemons": docker_process_views(),
        "proxy_variables_present": sorted(
            name
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
            if os.environ.get(name)
        ),
        "pid_join_demo": pid_join_demo(),
        "scope": "Namespace prerequisite only. No NVML support, cgroup equivalence or real GPU attribution test.",
    }
    if args.docker:
        report["docker_ping"] = docker_ping()
    report["namespace_prerequisite_met"] = bool(
        selected["initial_pid_namespace"] and selected.get("readable_process_identity")
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["namespace_prerequisite_met"] and report.get("docker_ping", {"ok": True})["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
