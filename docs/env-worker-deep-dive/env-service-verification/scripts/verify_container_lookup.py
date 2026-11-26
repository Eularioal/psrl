"""Verify live Docker identity and capture a bounded Compose trace without GPU work."""

import argparse
import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from psrl.workers.env_service.docker_api import find_main_container, ping_docker


async def verify(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = f"psrl-query-{uuid.uuid4().hex[:10]}"
    lease, actor = f"{project}-lease", f"{project}-actor"
    report = {
        "project": project,
        "started_at": time.time(),
        "scope": "One CPU container. Three sequential query pairs and one traced Compose query. No training or GPU.",
        "query_pairs": [],
    }
    compose = output / "compose.json"
    compose.write_text(
        json.dumps(
            {
                "services": {
                    "main": {
                        "image": args.image,
                        "pull_policy": "never",
                        "entrypoint": ["sleep", "120"],
                        "network_mode": "none",
                        "cpus": 0.1,
                        "mem_limit": "128m",
                        "pids_limit": 32,
                        "labels": {"psrl.env_lease": lease, "psrl.actor_id": actor},
                    }
                }
            },
            indent=2,
        )
    )
    command = ["docker", "compose", "--project-name", project, "-f", str(compose)]

    def run(arguments, timeout=15):
        return subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=timeout)

    async def lookup():
        trace = []
        started = time.monotonic()
        identifier = await find_main_container(project, lease, actor, trace=trace)
        return {"id": identifier, "elapsed_s": time.monotonic() - started, "trace": trace}

    try:
        await ping_docker()
        report["docker_version"] = run(["docker", "version", "--format", "{{json .Server.Version}}"]).stdout.strip()
        run([*command, "up", "-d", "--wait"])
        for _ in range(3):
            started = time.monotonic()
            cli = run([*command, "ps", "-q", "main"])
            elapsed = time.monotonic() - started
            api = await lookup()
            if cli.stdout.strip() != api["id"]:
                raise RuntimeError("Compose and the API resolved different container identities.")
            report["query_pairs"].append({"compose_s": elapsed, "api": api})
        trace_command = [
            "strace",
            "-f",
            "-tt",
            "-T",
            "-s",
            "128",
            "-o",
            str(output / "compose.strace"),
            "-e",
            "trace=process,connect,openat,poll,ppoll,epoll_wait",
            *command,
            "ps",
            "-q",
            "main",
        ]
        traced = run(trace_command)
        report["traced_id"] = traced.stdout.strip()
        old = report["query_pairs"][-1]["api"]["id"]
        run([*command, "up", "-d", "--wait", "--force-recreate"])
        report["after_recreation"] = await lookup()
        if report["after_recreation"]["id"] == old:
            raise RuntimeError("Container recreation did not change the identity.")
        run([*command, "stop", "-t", "1"])
        try:
            await lookup()
        except RuntimeError as error:
            report["stopped_rejected"] = str(error)
        else:
            raise RuntimeError("The stopped container was accepted.")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            run([*command, "down", "--timeout", "1"], timeout=20)
            remaining = run(["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"])
            report["physical_release_verified"] = not remaining.stdout.strip()
            if remaining.stdout.strip():
                raise RuntimeError("Owned containers remain after cleanup.")
        finally:
            report["finished_at"] = time.time()
            (output / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="An existing local image containing sleep.")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if os.environ.get("DOCKER_CONTEXT"):
        parser.error("Use an explicit local Unix Docker socket without a context override.")
    asyncio.run(verify(arguments))
