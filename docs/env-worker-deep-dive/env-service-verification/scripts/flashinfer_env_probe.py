"""Validate official FlashInfer tasks through native EnvWorker command leases."""

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

from examples.gpu_benchmarks.run import gpu_snapshot


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=15).strip()


async def run(args):
    import ray
    from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths
    from psrl.workers.env_service.client import EnvServiceClient
    from psrl.workers.env_service.overlay import build_lease_overlay, read_kernel_driver_version, resolve_driver_libs
    from psrl.workers.env_service.router import EnvRouter
    from psrl.workers.env_service.types import EnvDemand
    from psrl.workers.env_worker.harbor import EnvWorkerEnvironment

    root = args.root.resolve()
    tag = "fi-probe-" + uuid.uuid4().hex[:10]
    record = {
        "status": "starting",
        "tag": tag,
        "started": time.time(),
        "budget_s": 600,
        "environment_backend": "env_worker",
        "gpu_reservation": "command",
        "scope": "Real Ray EnvRouter and native EnvWorker; no agent rollout or PSRL training step.",
        "sources": {
            name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "harbor")
        },
    }

    def save():
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(record, indent=2, default=str))
        temporary.replace(root / "status.json")

    router = handle = environment = None
    save()
    try:
        record["docker"] = docker("info", "--format", "{{.ServerVersion}}")
        image_id = docker("image", "inspect", args.image, "--format", "{{.Id}}")
        record["image_id"] = image_id
        before = gpu_snapshot(args.gpu)
        record["gpu_before"] = before
        if before["pids"] or before["used_mb"] > 128:
            raise RuntimeError("The selected independent GPU must be idle.")
        ray.init(
            address="local",
            namespace=tag,
            _node_ip_address=args.node_ip,
            _temp_dir=f"/tmp/{tag}",
            num_cpus=4,
            num_gpus=0,
            include_dashboard=False,
            log_to_driver=False,
        )
        record["ray_nodes"] = ray.nodes()
        alive = [node for node in record["ray_nodes"] if node["Alive"]]
        if len(alive) != 1 or alive[0]["NodeManagerAddress"] != args.node_ip:
            raise RuntimeError("Unexpected private Ray node identity.")
        async with asyncio.timeout(600):
            router = EnvRouter.options(name=tag).remote(
                {
                    "node_policy": "any",
                    "units": [
                        {
                            "unit_id": tag,
                            "node_ip": args.node_ip,
                            "cpu_quota": 4,
                            "memory_mb": 16384,
                            "gpu_indices": [args.gpu],
                            "gpu_memory_mb": int(before["total_mb"]),
                            "max_containers": 1,
                            "gpu_reservation": "command",
                        }
                    ],
                }
            )
            await asyncio.wait_for(router.ping.remote(), 45)
            client = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=5)
            handle = await client.acquire(tag, EnvDemand(4, 16384, True, 8192), max_wait_s=30)
            record["lease"] = dataclasses.asdict(handle.lease)
            overlay = build_lease_overlay(handle.lease, resolve_driver_libs(read_kernel_driver_version()))
            prefix = Path(sys.prefix).resolve()
            paths = [(root, "/work", False), (prefix, str(prefix), True), (args.upstream, "/opt/fi", True)]
            paths.append((Path("/usr/local/cuda-12.8"), "/usr/local/cuda", True))
            mounts = [
                {"type": "bind", "source": str(src), "target": dst, "read_only": readonly}
                for src, dst, readonly in paths
            ]
            (root / "environment").mkdir(exist_ok=True)
            environment = EnvWorkerEnvironment(
                environment_dir=root / "environment",
                environment_name="flashinfer-official-probe",
                session_id=tag,
                actor_id=tag,
                trial_paths=TrialPaths(root / "trial"),
                task_env_config=EnvironmentConfig(docker_image=image_id, cpus=4, memory_mb=16384),
                network_policy=NetworkPolicy(network_mode="no-network"),
                mounts=mounts,
                persistent_env={
                    "PATH": f"{prefix}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
                    "PYTHONPATH": "/opt/fi",
                    "CUDA_HOME": "/usr/local/cuda",
                    "OMP_NUM_THREADS": "1",
                    "MAX_JOBS": "2",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HF_HUB_OFFLINE": "1",
                },
                lease_overlay=overlay,
                gpu_execution={"router_name": tag, "lease_id": handle.lease.lease_id, "mode": "command"},
            )
            await environment.start()
            record["status"] = "running"
            save()
            python = str(prefix / "bin/python")
            preflight = shlex.join(
                [python, "-c", "import torch; x=torch.ones(8,device='cuda'); assert x.sum().item()==8"]
            )
            result = await environment.exec_gpu(preflight, timeout_sec=60, wait_timeout_sec=30, overlap_return=False)
            record["cuda_preflight"] = {"return_code": result.return_code, "stderr": result.stderr}
            if result.return_code:
                raise RuntimeError("Container CUDA execution failed.")
            record["gpu_between_commands"] = gpu_snapshot(args.gpu)
            record["ledger_between_commands"] = [dataclasses.asdict(x) for x in await router.snapshot.remote()]
            command = shlex.join([python, "/work/flashinfer_official_eval.py"])
            result = await environment.exec_gpu(command, timeout_sec=420, wait_timeout_sec=30, overlap_return=False)
            (root / "benchmark.stdout").write_text(result.stdout or "")
            (root / "benchmark.stderr").write_text(result.stderr or "")
            record["return_code"] = result.return_code
            record["command_timing"] = environment.last_gpu_command
            if result.return_code:
                raise RuntimeError("Official FlashInfer command failed; inspect benchmark.stderr.")
            official = json.loads((root / "official-results.json").read_text())
            if not official["passed"]:
                raise RuntimeError("The official evaluator did not pass.")
            record["ledger_after_command"] = [dataclasses.asdict(x) for x in await router.snapshot.remote()]
            if any(x["declared_gpu_memory_mb"] for x in record["ledger_after_command"]):
                raise RuntimeError("Command GPU quota was not returned.")
            record["status"] = "passed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        errors = []
        if environment is not None:
            try:
                await asyncio.wait_for(environment.stop(delete=True), 45)
            except BaseException as exc:
                errors.append(repr(exc))
        if handle is not None:
            try:
                await asyncio.wait_for(handle.release(), 15)
            except BaseException as exc:
                errors.append(repr(exc))
        if router is not None:
            try:
                await asyncio.wait_for(router.shutdown_nodes.remote(), 30)
            except BaseException as exc:
                errors.append(repr(exc))
            finally:
                ray.kill(router)
        if ray.is_initialized():
            ray.shutdown()
        try:
            ids = docker("ps", "-aq", "--filter", f"label=psrl.actor_id={tag}").split()
            if ids:
                docker("rm", "-f", *ids)
            record["containers_remaining"] = docker("ps", "-aq", "--filter", f"label=psrl.actor_id={tag}")
            for _ in range(10):
                record["gpu_after"] = gpu_snapshot(args.gpu)
                if not record["gpu_after"]["pids"] and record["gpu_after"]["used_mb"] < 128:
                    break
                await asyncio.sleep(1)
            if record["containers_remaining"] or record["gpu_after"]["pids"] or record["gpu_after"]["used_mb"] >= 128:
                errors.append("Physical resources have not been confirmed released.")
        except BaseException as exc:
            errors.append(repr(exc))
        record["cleanup_errors"] = errors
        if errors:
            record["status"] = "failed"
        record["finished"] = time.time()
        save()
    if record["status"] != "passed":
        raise RuntimeError("The bounded probe did not pass.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    asyncio.run(run(parser.parse_args()))
