"""Compare serial compute, existing resident sharing, and cooperative compute lending."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

import ray
import yaml
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from inference import StudyServer
from protocol import read, save
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.probe import NvmlProbe
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand

STUDY = Path(__file__).resolve().parent


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=15).strip()


async def wait_json(path, predicate, *, timeout=60):
    """Bound protocol waits independently of inference and worker execution."""
    async with asyncio.timeout(timeout):
        while True:
            value = read(path)
            if value is not None and predicate(value):
                return value
            await asyncio.sleep(0.001)


class Experiment:
    """Keep shared initialization out of paired steady execution measurements."""

    command_timeout_s = 280
    command_deadline_s = 300
    ready_timeout_s = 90
    environment_memory_mb = 8192
    environment_gpu_mb = 2048
    worker_script = "worker.py"
    task_snapshot = "executed-vision.py"
    boundary = "Real EnvRouter command leases, Harbor Docker, vLLM, and local CIFAR task. No PPO or MPS."

    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.trial_root = self.out
        self.gpu_index = args.gpu_index
        self.before = None
        self.tag = f"resident-study-{uuid.uuid4().hex[:8]}"
        self.local = Path(tempfile.mkdtemp(prefix=self.tag + "-", dir="/tmp"))
        self.service_root = self.local
        self.environments = []
        self.handles = []
        self.commands = []
        self.overlays = []
        self.server = self.router = None
        self.probe = NvmlProbe()
        self.evidence = {"passed": False, "tag": self.tag, "trials": [], "cleanup_errors": []}

    def record(self):
        save(self.out / "evidence.json", self.evidence)

    def task_source(self):
        return Path(self.args.repo) / "examples/mlgym_bench/vision.py"

    def server_options(self):
        return {}

    def extra_mounts(self, role):
        return []

    def study_plan(self):
        return {
            "epochs": [2, 64],
            "updates": [8, 256],
            "profiles": ["spaced", "burst"],
            "rounds": self.args.rounds,
            "requests_per_trial": 3,
            "tokens_per_request": 64,
            "trial_timeout_s": 20,
            "mps": False,
            "model_offload": False,
            "candidate_pool": "Only the explicitly shared model GPU is configured.",
        }

    async def prepare(self):
        self.before = self.probe.read_gpu_snapshot([self.gpu_index])
        if self.before[self.gpu_index]["pids"] or self.before[self.gpu_index]["used_mb"] > 64:
            raise RuntimeError("The selected GPU must be idle at startup.")
        self.evidence.update(
            before=self.before,
            devices={self.gpu_index: asdict(self.probe.read_gpu_devices()[self.gpu_index])},
            local_control=str(self.local),
            boundary=self.boundary,
            sources={
                name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "vllm")
            },
            baseline_commit="0eeead861ffafed6bf31c962d259fd847775dcf9",
            comparison_kind="Same current source mechanism comparison. A is compute serial, not historical PSRL OFF.",
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            image=docker("image", "inspect", self.args.image, "--format", "{{.Id}}"),
            plan=self.study_plan(),
        )
        source = self.task_source()
        shutil.copy2(source, self.out / self.task_snapshot)
        shutil.copytree(STUDY, self.out / "executed-study", ignore=shutil.ignore_patterns("__pycache__"))
        self.evidence["source_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source, STUDY.parent / "real_gpu_lending.py", *STUDY.glob("*.py"))
        }
        self.record()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        ray.init(
            address="local",
            num_cpus=8,
            num_gpus=1,
            include_dashboard=False,
            log_to_driver=False,
            namespace=self.tag,
            _node_ip_address=self.args.node_ip,
            _temp_dir=f"/tmp/r-{self.tag[-8:]}",
            object_store_memory=256 * 1024**2,
            _memory=2 * 1024**3,
        )
        nodes = [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]]
        if nodes != [self.args.node_ip]:
            raise RuntimeError(f"Unexpected Ray node identities: {nodes!r}.")
        self.evidence["ray_nodes"] = nodes
        worker_path = os.pathsep.join(
            (str(STUDY), str(STUDY.parent), self.args.repo, os.environ.get("PYTHONPATH", ""))
        )
        runtime = {"env_vars": {"PYTHONPATH": worker_path, "RAY_ADDRESS": ray.get_runtime_context().gcs_address}}
        self.server = (
            ray.remote(num_cpus=2, num_gpus=1, max_concurrency=16)(StudyServer)
            .options(runtime_env=runtime)
            .remote(str(self.args.model), sharing=True, **self.server_options())
        )
        self.evidence["worker"] = await self.server.ready.remote()
        self.record()
        self.evidence["warmup_requests"] = [await self.server.query.remote(i) for i in range(2)]
        profile = await self.server.collective_rpc.remote("env_sharing_profile")
        headroom = math.ceil(max(p["transient_mb"] for p in profile) + 2048)
        self.evidence["model_profile"] = profile
        self.evidence["model_resident"] = self.probe.read_gpu_snapshot([self.gpu_index])
        config = {
            "node_policy": "prefer_other",
            "slot_ttl_s": 120,
            "gpu_sharing": {
                "safety_margin_mb": 1024,
                "max_command_s": self.command_deadline_s,
                "reuse_sleeping": False,
            },
            "units": [
                {
                    "unit_id": self.tag,
                    "node_ip": self.args.node_ip,
                    "cpu_quota": 6,
                    "memory_mb": self.environment_memory_mb * 3,
                    "gpu_indices": [self.gpu_index],
                    "gpu_memory_mb": self.before[self.gpu_index]["total_mb"],
                    "gpu_reservation": "command",
                    "max_containers": 2,
                    "shared_rollout_replica_rank": 0,
                    "model_headroom_mb": headroom,
                }
            ],
        }
        self.router = EnvRouter.options(name=self.tag).remote(config, [self.args.node_ip])
        await self.router.set_gpu_placement.remote({self.args.node_ip: [self.gpu_index]})
        self.evidence["binding"] = await self.router.bind_shared_rollout_gpu.remote(self.tag, self.server)
        self.evidence["router_config"] = config
        self.client = EnvServiceClient(self.router, self.args.node_ip, router_name=self.tag, heartbeat_interval_s=2)
        self.train_control, self.verify_control = await self.start_environments()
        self.record()

    async def start_environments(self):
        """Prepare the two separately leased task roles."""
        return await self.start_environment("train"), await self.start_environment("verify")

    async def start_environment(self, role):
        handle = await self.client.acquire(
            role,
            EnvDemand(2, self.environment_memory_mb, True, self.environment_gpu_mb),
            max_wait_s=20,
        )
        self.handles.append(handle)
        local = self.service_root / role
        local.mkdir(parents=True)
        mounts = [
            f"{self.out / 'executed-study'}:/study:ro",
            f"{self.out / self.task_snapshot}:/opt/task.py:ro",
            f"{self.args.data / 'public'}:/data/public:ro",
            f"{local}:/control:rw",
        ]
        mounts += self.extra_mounts(role)
        prefixes = dict.fromkeys((Path(sys.executable).parent.parent, Path(sys.executable).resolve().parent.parent))
        mounts += [f"{prefix}:{prefix}:ro" for prefix in prefixes]
        if role == "verify":
            mounts += [
                f"{self.args.data / 'private'}:/data/private:ro",
                f"{self.service_root / 'train'}:/submissions:ro",
            ]
        environment_dir = self.trial_root / role / "environment"
        environment_dir.mkdir(parents=True)
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
        (environment_dir / "docker-compose.yaml").write_text(yaml.safe_dump(settings))
        overlay = build_lease_overlay(handle.lease, resolve_driver_libs(read_kernel_driver_version()))
        overlay_path = write_overlay(overlay)
        self.overlays.append(overlay_path)
        environment = ManagedDockerEnvironment(
            environment_dir=environment_dir,
            environment_name=f"{self.tag}-{role}",
            session_id=uuid.uuid4().hex[:10],
            trial_paths=TrialPaths(self.trial_root / role / "trial"),
            task_env_config=EnvironmentConfig(
                docker_image=self.args.image, cpus=2, memory_mb=self.environment_memory_mb
            ),
            extra_docker_compose=[overlay_path],
            lease_overlay=overlay,
            gpu_execution={"router_name": self.tag, "lease_id": handle.lease.lease_id, "mode": "command"},
            network_policy=NetworkPolicy(network_mode="no-network"),
        )
        self.environments.append(environment)
        await environment.start(force_build=False)
        command = [
            sys.executable,
            f"/study/{self.worker_script}",
            "--control",
            "/control",
            "--source",
            "/opt/task.py",
            "--data",
            "/data/public",
            "--role",
            role,
        ]
        if role == "verify":
            command += ["--private", "/data/private"]
        task = asyncio.create_task(
            environment.exec_gpu(shlex.join(command), timeout_sec=self.command_timeout_s, wait_timeout_sec=20)
        )
        self.commands.append(task)
        ready = await wait_json(local / "ready.json", lambda v: True, timeout=self.ready_timeout_s)
        if ready["cuda_readback"] != 528:
            raise RuntimeError("The container CUDA readback failed.")
        self.evidence[f"{role}_ready"] = ready
        return local

    async def trial(self, policy, epochs, profile, repetition, *, warmup=False):
        trial_id = f"{epochs}-{profile}-{repetition}-{policy}"
        request = {
            "trial_id": trial_id,
            "epochs": epochs,
            "checkpoint": f"{trial_id}.pt",
            "go": False,
            "pause": False,
            "epoch": 0,
        }
        save(self.train_control / "request.json", request)
        await wait_json(self.train_control / "state.json", lambda v: v.get("trial_id") == trial_id)
        start_ns = time.monotonic_ns()
        queries = []
        if policy != "serial":
            request["go"] = True
            save(self.train_control / "request.json", request)
        for index in range(0 if warmup else 3):
            offset = 0.01 + (index * self.interval_s if profile == "spaced" else 0)
            await asyncio.sleep(max(0, offset - (time.monotonic_ns() - start_ns) / 1e9))
            submitted_ns = time.monotonic_ns()
            receipt = None
            if policy == "cooperative":
                request.update(pause=True, epoch=index + 1)
                save(self.train_control / "request.json", request)

                def stopped(value, epoch=index + 1):
                    return value.get("trial_id") == trial_id and (
                        value.get("state") == "completed"
                        or value.get("state") == "paused"
                        and value.get("epoch") == epoch
                    )

                async with asyncio.timeout(2):
                    while True:
                        state = read(self.train_control / "state.json")
                        result = read(self.train_control / "result.json")
                        if state and stopped(state) or result and stopped(result):
                            receipt = result if result and stopped(result) else state
                            break
                        await asyncio.sleep(0.001)
            admitted_ns = time.monotonic_ns()
            result = await self.server.query.remote(index)
            ended_ns = time.monotonic_ns()
            if policy == "cooperative":
                request["pause"] = False
                save(self.train_control / "request.json", request)
                if receipt["state"] == "paused":
                    await wait_json(
                        self.train_control / "state.json",
                        lambda value, epoch=index + 1: (
                            value.get("trial_id") == trial_id
                            and value.get("state") == "resumed"
                            and value.get("epoch") == epoch
                        ),
                        timeout=2,
                    )
            queries.append(
                {
                    "scheduled_ns": start_ns + int(offset * 1e9),
                    "submitted_ns": submitted_ns,
                    "admitted_ns": admitted_ns,
                    "ended_ns": ended_ns,
                    "receipt": receipt,
                    **result,
                }
            )
        if policy == "serial":
            request["go"] = True
            save(self.train_control / "request.json", request)
        training = await wait_json(
            self.train_control / "result.json", lambda v: v.get("trial_id") == trial_id, timeout=20
        )
        complete_ns = max(training["ended_ns"], *(q["ended_ns"] for q in queries)) if queries else training["ended_ns"]
        save(self.verify_control / "request.json", {"trial_id": trial_id, "checkpoint": f"/submissions/{trial_id}.pt"})
        verified = await wait_json(self.verify_control / "result.json", lambda v: v.get("trial_id") == trial_id)
        if training["optimizer_updates"] != epochs * 4 or any(len(q["token_ids"]) != 64 for q in queries):
            raise RuntimeError("A trial completed different amounts of useful work.")
        row = {
            "id": trial_id,
            "policy": policy,
            "epochs": epochs,
            "profile": profile,
            "repetition": repetition,
            "warmup": warmup,
            "makespan_s": (complete_ns - start_ns) / 1e9,
            "training": training,
            "verified": verified,
            "queries": queries,
            "gpu": self.probe.read_gpu_snapshot([self.gpu_index]),
        }
        self.evidence["trials"].append(row)
        self.record()
        print(
            json.dumps(
                {
                    "trial": trial_id,
                    "seconds": row["makespan_s"],
                    "score": verified["reward"],
                    "updates": training["optimizer_updates"],
                }
            ),
            flush=True,
        )

    async def run_trials(self):
        self.interval_s = 1.5 * sum(r["engine_duration_s"] for r in self.evidence["warmup_requests"]) / 2
        self.evidence["spaced_request_interval_s"] = self.interval_s
        await self.trial("serial", 2, "warmup", 0, warmup=True)
        policies = ("serial", "existing", "cooperative")
        for repetition in range(self.args.rounds):
            order = policies[repetition:] + policies[:repetition]
            for epochs in (2, 64):
                for profile in ("spaced", "burst"):
                    for policy in order:
                        async with asyncio.timeout(20):
                            await self.trial(policy, epochs, profile, repetition)
        rows = [r for r in self.evidence["trials"] if not r["warmup"]]
        for epochs in (2, 64):
            selected = [r for r in rows if r["epochs"] == epochs]
            if len({r["training"]["tensor_sha256"] for r in selected}) != 1:
                raise RuntimeError("Pausing or concurrency changed the final model tensors.")
            if len({r["verified"]["reward"] for r in selected}) != 1:
                raise RuntimeError("Independent verification changed across policies.")
        tokens = {tuple(q["token_ids"]) for r in rows for q in r["queries"]}
        if len(tokens) != 1:
            raise RuntimeError("Generation outputs changed across policies.")
        self.evidence["workload_and_outputs_matched"] = True
        self.evidence["reports"] = await self.router.gpu_job_reports.remote(self.handles[0].lease.lease_id)
        self.evidence["passed"] = True

    async def cleanup(self):
        for role in ("train", "verify"):
            directory = self.service_root / role
            if directory.exists():
                save(directory / "request.json", {"stop": True})
        for command in self.commands:
            try:
                output = await asyncio.wait_for(asyncio.shield(command), 15)
                if output.return_code:
                    self.evidence["cleanup_errors"].append(
                        {"worker_exit": output.return_code, "stderr": output.stderr}
                    )
            except Exception as exc:
                self.evidence["cleanup_errors"].append({"command": repr(exc)})
        actions = [("container", lambda env=env: env.stop(delete=True)) for env in self.environments]
        actions += [("lease", handle.release) for handle in self.handles]
        if self.server is not None:
            actions.append(("engine", lambda: self.server.stop.remote()))
        if self.router is not None:
            actions.append(("router", lambda: self.router.shutdown_nodes.remote()))
        for name, action in actions:
            try:
                await asyncio.wait_for(action(), 30)
            except Exception as exc:
                self.evidence["cleanup_errors"].append({name: repr(exc)})
        for actor in (self.server, self.router):
            if actor is not None:
                ray.kill(actor, no_restart=True)
        ray.shutdown()
        for overlay in self.overlays:
            overlay.unlink(missing_ok=True)
        after = None
        if self.before is not None:
            for _ in range(40):
                after = self.probe.read_gpu_snapshot([self.gpu_index])
                if (
                    not after[self.gpu_index]["pids"]
                    and after[self.gpu_index]["used_mb"] <= self.before[self.gpu_index]["used_mb"] + 32
                ):
                    break
                await asyncio.sleep(0.5)
        self.evidence["after"] = after
        self.evidence["containers_after"] = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={self.tag}")
        self.evidence["physical_release_confirmed"] = (
            after is not None
            and not after[self.gpu_index]["pids"]
            and after[self.gpu_index]["used_mb"] <= self.before[self.gpu_index]["used_mb"] + 32
            and not self.evidence["containers_after"]
        )
        self.evidence["passed"] &= self.evidence["physical_release_confirmed"] and not self.evidence["cleanup_errors"]
        shutil.copytree(self.local, self.out / "control-and-checkpoints", dirs_exist_ok=True)
        shutil.rmtree(self.local)
        self.record()
        self.probe.close()


async def main(args):
    experiment = Experiment(args)
    started = time.monotonic()
    try:
        async with asyncio.timeout(720):
            await experiment.prepare()
            await experiment.run_trials()
    except BaseException:
        experiment.evidence["error"] = traceback.format_exc()
        raise
    finally:
        await experiment.cleanup()
        experiment.evidence["elapsed_s"] = time.monotonic() - started
        experiment.record()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--rounds", type=int, default=3, choices=(1, 2, 3))
    asyncio.run(main(parser.parse_args()))
