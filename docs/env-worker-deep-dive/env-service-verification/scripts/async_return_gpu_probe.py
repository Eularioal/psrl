"""Compare real CUDA and vLLM planning with synchronous and service owned GPU return."""

import argparse
import asyncio
import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import aiohttp
import ray
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand


async def plan(session, endpoint, tokens, observation, controlled_delay=None):
    """Generate a fixed token budget from the same stable CUDA observation in both arms."""
    if controlled_delay is not None:
        await asyncio.sleep(controlled_delay)
        return {"kind": "controlled_planning_delay_not_vllm", "usage": {"completion_tokens": 0}}
    payload = {
        "model": "async-return-probe",
        "messages": [
            {"role": "system", "content": "Plan the next checks for a GPU programming task."},
            {
                "role": "user",
                "content": f"The isolated CUDA memory check returned {observation}. Explain the next checks.",
            },
        ],
        "max_tokens": tokens,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
    }
    async with session.post(endpoint + "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        return await response.json()


async def experiment(args, endpoint, evidence):
    run_id = "async-return-" + uuid.uuid4().hex[:10]
    evidence["actor_id"] = run_id
    ray.init(
        address="local",
        namespace=run_id,
        _node_ip_address=args.node_ip,
        _temp_dir=tempfile.mkdtemp(prefix="/tmp/ar-"),
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
    )
    router_class = EnvRouter
    if args.trace_control:
        from gpu_command_trace import TracedRouter

        router_class = ray.remote(num_cpus=0)(TracedRouter)
    router = router_class.options(name=run_id).remote(
        {
            "node_policy": "any",
            "slot_ttl_s": 600,
            "units": [
                {
                    "unit_id": "probe",
                    "node_ip": args.node_ip,
                    "cpu_quota": 4,
                    "memory_mb": 4096,
                    "gpu_indices": [args.env_gpu],
                    "gpu_memory_mb": 6144,
                    "max_containers": 2,
                    "gpu_reservation": "command",
                }
            ],
        }
    )
    envs, leases, paths = [], [], []
    try:
        libs = resolve_driver_libs(read_kernel_driver_version())
        for name in ("A", "B"):
            admission = await router.try_admit.remote(name, EnvDemand(2, 2048, True, 4096), args.node_ip)
            lease = admission.lease
            assert lease is not None, "Both persistent host leases must fit."
            leases.append(lease)
            overlay = build_lease_overlay(lease, libs)
            path = write_overlay(overlay)
            paths.append(path)
            directory = args.output / name
            (directory / "environment").mkdir(parents=True)
            env = ManagedDockerEnvironment(
                environment_dir=directory / "environment",
                environment_name="async-return-probe",
                session_id=run_id + name.lower(),
                trial_paths=TrialPaths(directory / "trial"),
                task_env_config=EnvironmentConfig(docker_image=args.image, cpus=2, memory_mb=2048),
                extra_docker_compose=[path],
                lease_overlay=overlay,
                actor_id=run_id,
                gpu_commands={"router_name": run_id, "lease_id": lease.lease_id},
                network_policy=NetworkPolicy(network_mode="no-network"),
            )
            envs.append(env)
            await env.start(force_build=False)
            await env.upload_file(Path(__file__).with_name("cuda_pressure.py"), "/tmp/probe.py")
        a, b = envs
        cuda = "python3 /tmp/probe.py --gib 1 --hold 0.2 --output /tmp/result.json"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90), trust_env=False) as session:
            if args.controlled_planning_s is None:
                await plan(session, endpoint, args.tokens, 42)
            for env in envs:
                result = await env.exec_gpu(cuda, timeout_sec=30)
                assert result.return_code == 0, result
            evidence["trials"] = []
            for index, mode in enumerate(args.modes):
                overlap = mode == "async"
                events = []
                started = time.time()

                def event(kind, _events=events, **extra):
                    _events.append({"event": kind, "time": time.time(), **extra})

                a.last_gpu_command = None

                async def backfill():
                    while not (a.last_gpu_command and "command_finished_at" in a.last_gpu_command):
                        await asyncio.sleep(0.01)
                    event("B_requested")
                    result = await b.exec_gpu(cuda, timeout_sec=30, wait_timeout_sec=60)
                    assert result.return_code == 0, result
                    event("B_completed", timing=dict(b.last_gpu_command))

                async with asyncio.TaskGroup() as group:
                    group.create_task(backfill())
                    result = await a.exec_gpu(cuda, timeout_sec=30, overlap_return=overlap)
                    assert result.return_code == 0, result
                    first_record = a.last_gpu_command
                    event("A_output_available", timing=dict(first_record))
                    snapshot = await router.snapshot.remote()
                    event("ledger_at_output", units=[dataclasses.asdict(unit) for unit in snapshot])
                    event("model_start")
                    reply = await plan(
                        session,
                        endpoint,
                        args.tokens,
                        json.loads(result.stdout)["kernel_output"],
                        args.controlled_planning_s,
                    )
                    event("model_end", usage=reply["usage"])
                    await a.wait_for_gpu_return()
                    event("A_return_joined", timing=dict(first_record))
                    result = await a.exec_gpu(cuda, timeout_sec=30, wait_timeout_sec=60)
                    assert result.return_code == 0, result
                    event("A_next_command_completed", timing=dict(a.last_gpu_command))
                trial = {
                    "index": index,
                    "mode": "async" if overlap else "sync",
                    "started_at": started,
                    "finished_at": time.time(),
                    "events": events,
                    "reply": reply,
                }
                trial["duration_s"] = trial["finished_at"] - started
                model_start = next(e["time"] for e in events if e["event"] == "model_start")
                returned = first_record["released_at"]
                trial["model_overlaps_return"] = model_start < returned
                backfill_record = next(e["timing"] for e in events if e["event"] == "B_completed")
                assert backfill_record["admitted_at"] >= returned, "B must not receive capacity before A returns."
                expected_tokens = args.tokens if args.controlled_planning_s is None else 0
                assert reply["usage"]["completion_tokens"] == expected_tokens, (
                    "Both arms must generate equal token counts."
                )
                evidence["trials"].append(trial)
                if args.trace_control:
                    evidence["control_trace"] = await router.export_trace.remote()
                (args.output / "result.json").write_text(json.dumps(evidence, indent=2))
                print(
                    json.dumps({k: trial[k] for k in ("index", "mode", "duration_s", "model_overlaps_return")}),
                    flush=True,
                )
        evidence["unauthorized_gpu_return_codes"] = []
        for env in envs:
            denied = await env.exec(cuda, timeout_sec=30)
            evidence["unauthorized_gpu_return_codes"].append(denied.return_code)
            assert denied.return_code != 0, "Returned containers must not retain CUDA access."
        evidence["passed"] = True
    finally:
        evidence["cleanup_errors"] = []

        async def cleanup():
            try:
                async with asyncio.timeout(60):
                    for env in envs:
                        try:
                            await env.stop(delete=True)
                        except Exception as exc:
                            evidence["cleanup_errors"].append(repr(exc))
                    for lease in leases:
                        try:
                            await router.release.remote(lease.lease_id)
                        except Exception as exc:
                            evidence["cleanup_errors"].append(repr(exc))
                    for path in paths:
                        path.unlink(missing_ok=True)
            finally:
                ray.shutdown()

        task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    node = EnvNodeService(None, args.node_ip)
    selected = (args.env_gpu,) if args.controlled_planning_s is not None else (args.model_gpu, args.env_gpu)
    before = await node.gpu_lending_snapshot(selected)
    assert all(value["used_mb"] < 64 for value in before.values()), "Selected GPUs must be idle."
    await node.check_gpu_command_support()
    if args.controlled_planning_s is not None:
        evidence = {
            "scope": "Real Ray Router, Harbor Docker CUDA, simulated fixed planning delay, no vLLM or training",
            "planning_mode": "controlled_delay",
            "controlled_planning_s": args.controlled_planning_s,
            "gpu_before": before,
        }
        try:
            async with asyncio.timeout(args.work_budget_s):
                await experiment(args, None, evidence)
        except BaseException as exc:
            evidence["error"] = repr(exc)
            raise
        finally:
            evidence["gpu_final"] = await node.gpu_lending_snapshot(selected)
            (args.output / "result.json").write_text(json.dumps(evidence, indent=2))
        return
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model),
        "--served-model-name",
        "async-return-probe",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enforce-eager",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.35",
        "--distributed-executor-backend",
        "uni",
        "--language-model-only",
        "--gdn-prefill-backend",
        "triton",
    ]
    evidence = {
        "scope": "Real Ray Router, Harbor Docker CUDA and independent vLLM GPU, no training",
        "gpu_before": before,
    }
    evidence["server_command"] = command
    server_env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=str(args.model_gpu), VLLM_WORKER_MULTIPROC_METHOD="spawn", OMP_NUM_THREADS="1"
    )
    server_env.pop("RAY_ADDRESS", None)
    with (args.output / "vllm.log").open("w") as log:
        server = subprocess.Popen(
            command, env=server_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        evidence["server_pid"] = server.pid
        try:
            async with asyncio.timeout(300):
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2), trust_env=False) as session:
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError("vLLM exited during startup. Inspect vllm.log.")
                        try:
                            async with session.get(endpoint + "/health") as response:
                                if response.status == 200:
                                    break
                        except (aiohttp.ClientError, TimeoutError):
                            pass
                        await asyncio.sleep(1)
            async with asyncio.timeout(300):
                await experiment(args, endpoint, evidence)
        except BaseException as exc:
            evidence["error"] = repr(exc)
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    await asyncio.to_thread(server.wait, timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    await asyncio.to_thread(server.wait, timeout=10)
            evidence["server_exit_code"] = server.returncode
            evidence["gpu_final"] = await node.gpu_lending_snapshot((args.model_gpu, args.env_gpu))
            (args.output / "result.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--model-gpu", type=int, default=0)
    parser.add_argument("--env-gpu", type=int, default=7)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--trace-control", action="store_true")
    parser.add_argument("--work-budget-s", type=float, default=180)
    parser.add_argument("--controlled-planning-s", type=float)
    parser.add_argument("--modes", nargs="+", choices=("sync", "async"), default=["sync", "async", "async", "sync"])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.work_budget_s <= 0:
        parser.error("--work-budget-s must be positive")
    if arguments.controlled_planning_s is None and arguments.model is None:
        parser.error("--model is required for a real vLLM run")
    if arguments.controlled_planning_s is not None and not 0 <= arguments.controlled_planning_s <= 15:
        parser.error("--controlled-planning-s must be between 0 and 15")
    asyncio.run(main(arguments))
