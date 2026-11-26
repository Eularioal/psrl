"""Compare a fixed first-fit environment partition with drained vLLM replica lending."""

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import ray
import yaml
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.elastic_partition import ElasticPartition, PartitionDevice
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.probe import NvmlProbe
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand

SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT / "resident_sharing_study"))
from inference import StudyServer  # noqa: E402


class Study:
    """Retain real admission, device fencing, command return, and model control."""

    server_type = StudyServer

    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.tag = "partition-" + uuid.uuid4().hex[:8]
        self.probe = NvmlProbe()
        self.servers = {}
        self.routers = {}
        self.clients = {}
        self.environments = []
        self.handles = []
        self.overlays = []
        self.evidence = {"status": "running", "tag": self.tag, "trials": [], "cleanup_errors": []}
        self.started = time.monotonic()

    def save(self):
        self.evidence["elapsed_s"] = time.monotonic() - self.started
        tmp = self.out / "evidence.tmp"
        tmp.write_text(json.dumps(self.evidence, indent=2, default=str) + "\n")
        tmp.replace(self.out / "evidence.json")

    async def prepare(self):
        args = self.args
        self.before = self.probe.read_gpu_snapshot(args.gpus)
        if any(d["pids"] or d["used_mb"] > 64 for d in self.before.values()):
            raise RuntimeError("All study GPUs must be idle before initialization.")
        self.evidence.update(
            before=self.before,
            devices={k: dataclasses.asdict(v) for k, v in self.probe.read_gpu_devices().items() if k in args.gpus},
            args=vars(args),
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            comparison="Current source fixed first-fit partition versus dynamic partition, not historical PSRL OFF.",
            boundary="Real Ray, EnvRouter, Docker, vLLM and PostTrain commands. Explicit entry, no SMG or PPO.",
            sources={n: importlib.util.find_spec(n).origin for n in ("psrl", "verl", "transfer_queue", "vllm")},
        )
        snapshot = self.out / "source"
        snapshot.mkdir()
        for src in (
            Path(__file__),
            SCRIPT / "partition_worker.py",
            SCRIPT / "real_gpu_lending.py",
            SCRIPT / "resident_sharing_study/inference.py",
            Path(args.repo) / "examples/posttrain_bench/local_model.py",
            Path(args.repo) / "psrl/workers/env_service/elastic_partition.py",
        ):
            shutil.copy2(src, snapshot / src.name)
        shutil.copy2(snapshot / "local_model.py", snapshot / "task.py")
        self.save()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus[2:]))
        ray.init(
            address="local",
            num_cpus=16,
            num_gpus=2,
            include_dashboard=False,
            log_to_driver=False,
            namespace=self.tag,
            _node_ip_address=args.node_ip,
            _temp_dir=f"/tmp/r-{self.tag[-8:]}",
            object_store_memory=256 * 1024**2,
            _memory=4 * 1024**3,
        )
        nodes = [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]]
        if nodes != [args.node_ip]:
            raise RuntimeError(f"Unexpected Ray nodes: {nodes!r}.")
        self.evidence["ray_nodes"] = nodes
        runtime = {
            "env_vars": {
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(snapshot),
                        str(SCRIPT),
                        str(SCRIPT / "resident_sharing_study"),
                        args.repo,
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
                "RAY_ADDRESS": ray.get_runtime_context().gcs_address,
            }
        }

        async def prepare_server(index):
            server = (
                ray.remote(num_cpus=2, num_gpus=1, max_concurrency=32)(self.server_type)
                .options(
                    runtime_env=runtime,
                )
                .remote(
                    str(args.model),
                    sharing=True,
                    output_tokens=args.tokens,
                    max_model_len=2048,
                    gpu_memory_utilization=0.25,
                    max_num_seqs=16,
                )
            )
            self.servers[f"starting-{index}"] = server
            ready = await server.ready.remote()
            gpu = int(ready["cuda_visible_devices"])
            if gpu not in args.gpus[2:]:
                raise RuntimeError(f"Wrong inference device: {ready!r}.")
            del self.servers[f"starting-{index}"]
            self.servers[str(gpu)] = server
            self.evidence.setdefault("workers", {})[str(gpu)] = ready
            await server.query.remote("warmup")
            self.save()

        async with asyncio.TaskGroup() as group:
            for index in range(2):
                group.create_task(prepare_server(index))
        self.resident = self.probe.read_gpu_snapshot(args.gpus)
        self.evidence["resident"] = self.resident
        for gpu in args.gpus:
            name = str(gpu)
            shared = name in self.servers
            unit = dict(
                unit_id=f"{self.tag}-{name}",
                node_ip=args.node_ip,
                cpu_quota=8,
                memory_mb=65536,
                gpu_indices=[gpu],
                gpu_memory_mb=90000,
                gpu_reservation="command",
                max_containers=args.slots,
            )
            if shared:
                profiles = await self.servers[name].collective_rpc.remote("env_sharing_profile")
                unit.update(
                    shared_rollout_replica_rank=gpu,
                    model_headroom_mb=math.ceil(max(p["transient_mb"] for p in profiles) + 2048),
                )
            config = dict(
                node_policy="prefer_other",
                slot_ttl_s=600,
                units=[unit],
                gpu_sharing=dict(max_command_s=300, reuse_sleeping=True, safety_margin_mb=1024),
            )
            router = EnvRouter.options(name=unit["unit_id"]).remote(config, [args.node_ip])
            self.routers[name] = router
            await router.set_gpu_placement.remote({args.node_ip: args.gpus[2:]})
            if shared:
                await router.bind_shared_rollout_gpu.remote(unit["unit_id"], self.servers[name])
            self.clients[name] = EnvServiceClient(
                router, args.node_ip, router_name=unit["unit_id"], heartbeat_interval_s=2
            )
        self.evidence["initialization_s"] = time.monotonic() - self.started
        self.save()

    async def command(self, device, trial, job, mode, *, steps=None):
        args = self.args
        folder = self.out / trial / f"job-{job}"
        artifact = folder / "artifact"
        artifact.mkdir(parents=True, exist_ok=True)
        task_dir = folder / mode
        task_dir.mkdir()
        router_name = f"{self.tag}-{device}"
        handle = await self.clients[device].acquire(
            f"{trial}-{job}-{mode}",
            EnvDemand(2, 16384, True, 16384),
            max_wait_s=30,
        )
        self.handles.append(handle)
        mounts = [
            f"{self.out / 'source'}:/study:ro",
            f"{args.env_model}:/model:ro",
            f"{args.data / ('public' if mode == 'train' else 'private')}:/data:ro",
            f"{artifact}:/artifact:{'rw' if mode == 'train' else 'ro'}",
        ]
        for prefix in dict.fromkeys(
            (Path(sys.executable).parent.parent, Path(sys.executable).resolve().parent.parent)
        ):
            mounts.append(f"{prefix}:{prefix}:ro")
        settings = {
            "services": {
                "main": {
                    "volumes": mounts,
                    "environment": {
                        "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
                        "OMP_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "2",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "HF_HUB_OFFLINE": "1",
                    },
                }
            }
        }
        (task_dir / "docker-compose.yaml").write_text(yaml.safe_dump(settings))
        overlay = build_lease_overlay(handle.lease, resolve_driver_libs(read_kernel_driver_version()))
        overlay_path = write_overlay(overlay)
        self.overlays.append(overlay_path)
        env = ManagedDockerEnvironment(
            environment_dir=task_dir,
            environment_name=f"{router_name}-{job}-{mode}",
            session_id=uuid.uuid4().hex[:10],
            trial_paths=TrialPaths(task_dir / "trial"),
            task_env_config=EnvironmentConfig(docker_image=args.image, cpus=2, memory_mb=16384),
            extra_docker_compose=[overlay_path],
            lease_overlay=overlay,
            gpu_execution={"router_name": router_name, "lease_id": handle.lease.lease_id, "mode": "command"},
            network_policy=NetworkPolicy(network_mode="no-network"),
        )
        self.environments.append(env)
        try:
            await env.start(force_build=False)
            result = await env.exec_gpu(
                shlex.join(
                    [
                        sys.executable,
                        "/study/partition_worker.py",
                        "--mode",
                        mode,
                        "--steps",
                        str(args.steps if steps is None else steps),
                    ]
                ),
                timeout_sec=240,
                wait_timeout_sec=30,
                overlap_return=False,
            )
            (task_dir / "stdout.txt").write_text(result.stdout or "")
            (task_dir / "stderr.txt").write_text(result.stderr or "")
            if result.return_code:
                raise RuntimeError(f"Environment command failed: {(result.stderr or '')[-1500:]!r}.")
            value = json.loads(
                next(
                    line.split("=", 1)[1]
                    for line in result.stdout.splitlines()
                    if line.startswith("PARTITION_RESULT=")
                )
            )
            if value["cuda_readback"] != 528:
                raise RuntimeError("Container CUDA probe failed.")
            return value
        finally:
            await env.stop(delete=True)
            self.environments.remove(env)
            await handle.release()
            self.handles.remove(handle)

    async def trial(self, policy, repetition):
        args = self.args
        trial = f"{repetition}-{policy}"
        row = {"id": trial, "policy": policy, "commands": [], "queries": [], "transitions": []}
        self.evidence["trials"].append(row)
        baseline = self.probe.read_gpu_snapshot(args.gpus)
        started = time.monotonic()

        async def sleep(device):
            begin = time.monotonic()
            await self.servers[device].pause_generation.remote(mode="wait", clear_cache=False)
            await self.servers[device].nixl_sleep.remote(level=1)
            snapshot = self.probe.read_gpu_snapshot([int(device)])[int(device)]
            if snapshot["used_mb"] > baseline[int(device)]["used_mb"] - 1024:
                raise RuntimeError("Sleep failed to release model memory.")
            row["transitions"].append(
                dict(
                    kind="sleep",
                    device=device,
                    duration_s=time.monotonic() - begin,
                    at_s=begin - started,
                    gpu=snapshot,
                )
            )
            self.save()

        async def wake(device):
            begin = time.monotonic()
            snapshot = self.probe.read_gpu_snapshot([int(device)])[int(device)]
            if set(snapshot["pids"]) - set(baseline[int(device)]["pids"]):
                raise RuntimeError("Environment GPU processes remain before model restoration.")
            await self.servers[device].nixl_wake_up.remote()
            await self.servers[device].resume_generation.remote()
            row["transitions"].append(
                dict(
                    kind="wake",
                    device=device,
                    duration_s=time.monotonic() - begin,
                    at_s=begin - started,
                    gpu=self.probe.read_gpu_snapshot([int(device)]),
                )
            )
            self.save()

        controller = ElasticPartition(
            [PartitionDevice(str(i), args.slots, 65536, i in args.gpus[2:]) for i in args.gpus],
            sleep,
            wake,
            dynamic=policy == "dynamic",
            inference_limit=4,
            transition_timeout_s=270,
        )

        async def job(index):
            for mode in ("train", "verify"):
                submitted = time.monotonic()
                async with controller.environment(16384) as device:
                    admitted = time.monotonic()
                    result = await self.command(device, trial, index, mode)
                    row["commands"].append(
                        dict(
                            job=index,
                            mode=mode,
                            device=device,
                            result=result,
                            wait_s=admitted - submitted,
                            service_s=time.monotonic() - admitted,
                        )
                    )
                    self.save()

        async def query(index):
            await asyncio.sleep((index // 4) * 10)
            submitted = time.monotonic()
            async with controller.inference() as device:
                admitted = time.monotonic()
                result = await self.servers[device].query.remote(index)
                row["queries"].append(
                    dict(
                        index=index,
                        device=device,
                        result=result,
                        wait_s=admitted - submitted,
                        latency_s=time.monotonic() - submitted,
                    )
                )
                self.save()

        async with asyncio.TaskGroup() as group:
            monitor = group.create_task(controller.run())
            work = [group.create_task(job(i)) for i in range(args.jobs)]
            work += [group.create_task(query(i)) for i in range(args.requests)]
            await asyncio.gather(*work)
            row["work_done_s"] = time.monotonic() - started
            while any(d.inference and d.state != "inference" for d in controller.devices):
                if any(d.state == "quarantined" for d in controller.devices):
                    raise RuntimeError("Trial ended with a quarantined GPU.")
                await asyncio.sleep(0.1)
            monitor.cancel()
        row.update(
            total_with_restore_s=time.monotonic() - started,
            events=controller.events,
            gpu_after=self.probe.read_gpu_snapshot(args.gpus),
        )
        for device, server in self.servers.items():
            row.setdefault("restored_queries", {})[device] = await server.query.remote("restore-check")
        self.save()

    async def cleanup(self):
        for name, action in (
            *[("container", lambda env=env: env.stop(delete=True)) for env in self.environments],
            *[("lease", handle.release) for handle in self.handles],
            *[("engine", lambda server=s: server.stop.remote()) for s in self.servers.values()],
            *[("router", lambda router=r: router.shutdown_nodes.remote()) for r in self.routers.values()],
        ):
            try:
                await asyncio.wait_for(action(), 30)
            except Exception as exc:
                self.evidence["cleanup_errors"].append({name: repr(exc)})
        for actor in (*self.servers.values(), *self.routers.values()):
            ray.kill(actor, no_restart=True)
        ray.shutdown()
        for overlay in self.overlays:
            overlay.unlink(missing_ok=True)
        for _ in range(40):
            after = self.probe.read_gpu_snapshot(self.args.gpus)
            if all(not v["pids"] and v["used_mb"] < 64 for v in after.values()):
                break
            await asyncio.sleep(0.5)
        self.evidence["after"] = after
        containers = []
        for gpu in self.args.gpus:
            containers += subprocess.check_output(
                ["docker", "ps", "-aq", "--filter", f"label=psrl.env_unit={self.tag}-{gpu}"],
                text=True,
                timeout=10,
            ).split()
        self.evidence["containers_after"] = containers
        self.evidence["physical_release_confirmed"] = not containers and all(
            not v["pids"] and v["used_mb"] < 64 for v in after.values()
        )
        self.save()


async def main(args):
    study = Study(args)
    try:
        async with asyncio.timeout(args.total_timeout):
            await study.prepare()
            for repetition, policy in enumerate(args.modes):
                async with asyncio.timeout(args.trial_timeout):
                    await study.trial(policy, repetition)
            commands = [c for trial in study.evidence["trials"] for c in trial["commands"]]
            hashes = {c["result"]["tensor_sha256"] for c in commands if c["mode"] == "train"}
            scores = {c["result"]["reward"] for c in commands if c["mode"] == "verify"}
            if len(hashes) != 1 or len(scores) != 1:
                raise RuntimeError("Matched work produced different weights or scores.")
            study.evidence.update(status="passed", tensor_hashes=list(hashes), scores=list(scores))
    except BaseException:
        study.evidence.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        await study.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--env-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=448)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--modes", nargs="+", choices=("fixed", "dynamic"), default=["fixed", "dynamic"])
    parser.add_argument("--total-timeout", type=float, default=1100)
    parser.add_argument("--trial-timeout", type=float, default=400)
    asyncio.run(main(parser.parse_args()))
