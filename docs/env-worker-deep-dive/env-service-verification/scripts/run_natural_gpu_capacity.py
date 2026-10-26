"""Run bounded OFF and ON training pairs with owned GPU capacity pressure."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from science_router_control_ab import gpu_snapshot


def released(gpu, expected_pids, expected_used):
    deadline = time.monotonic() + 30
    while True:
        snapshot = gpu_snapshot(gpu)
        if set(snapshot["pids"]) == set(expected_pids) and abs(snapshot["used_mib"] - expected_used) < 64:
            return snapshot
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU resources did not return to the expected owner: {snapshot!r}.")
        time.sleep(1)


def run(args):
    scripts = Path(__file__).parent
    report = {"started_at": time.time(), "arms": []}
    holder = None
    holder_log = None
    initial = released(args.gpu, [], 0)
    pressure = None
    try:
        for level, mode in (("low", "off"), ("low", "on"), ("high", "on"), ("high", "off")):
            if level == "high" and holder is None:
                holder_log = (args.root / "holder.log").open("w")
                holder = subprocess.Popen(
                    [
                        sys.executable,
                        str(scripts / "hold_gpu_capacity.py"),
                        "--gpu",
                        str(args.gpu),
                        "--leave-mib",
                        "1536",
                        "--parent-pid",
                        str(os.getpid()),
                        "--status",
                        str(args.root / "holder.json"),
                    ],
                    stdout=holder_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 30
                while not (args.root / "holder.json").exists():
                    if holder.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("GPU capacity holder did not become ready.")
                    time.sleep(0.25)
                pressure = json.loads((args.root / "holder.json").read_text())
                if pressure["status"] != "ready":
                    raise RuntimeError("GPU capacity holder failed before training.")
            arm = f"{level}-{mode}"
            command = [
                sys.executable,
                str(scripts / "science_natural_gpu_ab.py"),
                "--mode",
                mode,
                "--output",
                str(args.root / arm),
                "--workload",
                str(args.root / "workload"),
                "--overlay",
                str(args.root / "gpu-overlay.yaml"),
                "--python",
                sys.executable,
                "--image",
                args.image,
                "--node-ip",
                args.node_ip,
                "--gpu",
                str(args.gpu),
                "--count",
                "4",
                "--dependencies",
                str(args.dependencies),
                "--gpu-budget-mib",
                str(int(initial["total_mib"])),
                "--command-memory-mib",
                "1024",
            ]
            if mode == "on":
                command.append("--prepared")
            if level == "high":
                command.extend(["--holder-status", str(args.root / "holder.json")])
            environment = os.environ.copy()
            source = args.on_source if mode == "on" else args.off_source
            environment.update(
                PSRL_WORKSPACE=str(source),
                PYTHONPATH=os.pathsep.join([str(scripts), str(source), str(args.verl_parent)]),
                TMPDIR="/tmp",
                RAY_TMPDIR="/tmp",
            )
            environment.pop("RAY_ADDRESS", None)
            arm_started_at = time.time()
            print(json.dumps({"arm": arm, "event": "start", "time": arm_started_at}), flush=True)
            with (args.root / f"{arm}-driver.log").open("w") as output:
                process = subprocess.Popen(
                    command,
                    cwd=source,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=900)
                except BaseException:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait(timeout=60)
                    raise
            if code:
                raise RuntimeError(f"Training driver {arm!r} failed with code {code}.")
            result = json.loads((args.root / arm / "result.json").read_text())
            if result.get("error") or result.get("result", {}).get("exception"):
                raise RuntimeError(f"Training infrastructure failed in {arm!r}.")
            expected_pids = [pressure["gpu_pid"]] if level == "high" else []
            expected_used = pressure["used_mib"] if level == "high" else 0
            snapshot = released(args.gpu, expected_pids, expected_used)
            containers = subprocess.check_output(
                ["docker", "ps", "-aq", "--filter", f"label=psrl.actor_id={result['actor_id']}"],
                text=True,
                timeout=10,
            ).split()
            if containers:
                raise RuntimeError(f"Owned containers remain after {arm!r}: {containers!r}.")
            projects = {
                path.parent.name.lower() + "__env"
                for path in (args.root / arm / "jobs").rglob("config.json")
                if path.parent.name.startswith("task__")
            }
            for project in projects:
                for listing in (["docker", "ps", "-aq"], ["docker", "network", "ls", "-q"]):
                    remaining = subprocess.check_output(
                        [*listing, "--filter", f"label=com.docker.compose.project={project}"],
                        text=True,
                        timeout=10,
                    ).split()
                    if remaining:
                        raise RuntimeError(f"Compose resources remain for {project!r}: {remaining!r}.")
            report["arms"].append(
                {
                    "arm": arm,
                    "command": command,
                    "source": str(source),
                    "gpu_after": snapshot,
                    "released_compose_projects": sorted(projects),
                    "started_at": arm_started_at,
                    "finished_at": time.time(),
                }
            )
            print(json.dumps({"arm": arm, "event": "finished", "time": time.time()}), flush=True)
    finally:
        if holder is not None:
            holder.terminate()
            holder.wait(timeout=15)
        if holder_log is not None:
            holder_log.close()
        report["finished_at"] = time.time()
        report["gpu_after"] = released(args.gpu, [], 0)
        (args.root / "matrix.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for option in ("root", "on-source", "off-source", "dependencies", "verl-parent"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--image", default="pxtest:latest")
    parser.add_argument("--gpu", type=int, default=6)
    run(parser.parse_args())
