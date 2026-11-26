"""Compare automatic and explicit reservations using bounded PostTrain commands."""

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import os
import shlex
import shutil
import signal
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
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand
from roofline_sweep import gpu_snapshot, save

SCRIPT = Path(__file__).resolve().parent


class Study:
    ray_gpu_count = 0

    def __init__(self, args):
        self.args = args
        self.out = args.output
        self.tag = "autores-" + uuid.uuid4().hex[:8]
        self.routers, self.clients, self.envs, self.handles, self.overlays = {}, {}, [], [], []
        self.data = dict(status="initializing", tag=self.tag, args=vars(args), trials=[], cleanup_errors=[])
        self.started = time.monotonic()

    def save(self):
        self.data["elapsed_s"] = time.monotonic() - self.started
        save(self.out / "evidence.json", self.data)

    async def prepare(self):
        self.out.mkdir(parents=True, exist_ok=False)
        self.data["before"] = gpu_snapshot(self.args.gpu)
        if self.data["before"]["pids"] or self.data["before"]["used_mib"] > 64:
            raise RuntimeError("Selected GPU is occupied.")
        self.gpu = self.data["before"]["index"]
        subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, check=True, timeout=8)
        self.data["sources"] = {n: importlib.util.find_spec(n).origin for n in ("psrl", "verl", "transfer_queue")}
        self.data["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        self.data["boundary"] = (
            "One H20, real EnvRouter and managed Docker PostTrain train/verify commands. "
            "Controlled inter-command gaps, no rollout model or PPO. Current source in every arm."
        )
        source = self.out / "source"
        source.mkdir()
        for src in (
            Path(__file__),
            SCRIPT / "partition_worker.py",
            self.args.repo / "examples/posttrain_bench/local_model.py",
            self.args.repo / "patch/transformers/fast_package_metadata.py",
        ):
            shutil.copy2(src, source / src.name)
        shutil.copy2(source / "local_model.py", source / "task.py")
        shutil.copytree(
            self.args.repo / "psrl/workers/env_service",
            source / "env_service",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (source / "entry.py").write_text(
            "import importlib.metadata\nimport runpy\nfrom fast_package_metadata import packages_distributions\n"
            "original = importlib.metadata.packages_distributions\n"
            "importlib.metadata.packages_distributions = packages_distributions\n"
            "try:\n    import transformers\nfinally:\n    importlib.metadata.packages_distributions = original\n"
            'runpy.run_path("/study/partition_worker.py", run_name="__main__")\n'
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu) if self.ray_gpu_count else ""
        ray.init(
            address="local",
            num_cpus=12,
            num_gpus=self.ray_gpu_count,
            include_dashboard=False,
            log_to_driver=False,
            namespace=self.tag,
            _node_ip_address=self.args.node_ip,
            _temp_dir=f"/tmp/r-{self.tag[-8:]}",
            object_store_memory=256 * 1024**2,
            _memory=4 * 1024**3,
        )
        nodes = [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]]
        if nodes != [self.args.node_ip]:
            raise RuntimeError(f"Unexpected Ray nodes {nodes!r}.")
        self.data["ray_nodes"] = nodes
        os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        for mode in ("episode", "command", "auto"):
            name = self.tag + "-" + mode
            unit = dict(
                unit_id=name,
                node_ip=self.args.node_ip,
                cpu_quota=8,
                memory_mb=65536,
                gpu_indices=[self.gpu],
                gpu_memory_mb=20000,
                max_containers=2,
                gpu_reservation=mode,
            )
            if mode != "episode":
                unit["max_gpu_commands"] = 1
            router = EnvRouter.options(name=name).remote(dict(node_policy="any", slot_ttl_s=300, units=[unit]), [])
            self.routers[mode] = router
            self.clients[mode] = EnvServiceClient(
                router,
                self.args.node_ip,
                router_name=name,
                heartbeat_interval_s=2,
                poll_interval_s=0.1,
                max_poll_interval_s=0.1,
            )
        self.save()

    async def create(self, mode, trial, role, contract="independent_commands"):
        folder = self.out / trial / role
        folder.mkdir(parents=True)
        artifact = folder / "artifact"
        artifact.mkdir()
        submitted = time.monotonic()
        handle = await self.clients[mode].acquire(
            f"{trial}-{role}", EnvDemand(2, 16384, True, 14000, gpu_execution_contract=contract), max_wait_s=240
        )
        self.handles.append(handle)
        admitted = time.monotonic()
        mounts = [
            f"{self.out / 'source'}:/study:ro",
            f"{self.args.model}:/model:ro",
            f"{self.args.data}:/data:ro",
            f"{artifact}:/artifact:rw",
        ]
        for prefix in dict.fromkeys(
            (Path(sys.executable).parent.parent, Path(sys.executable).resolve().parent.parent)
        ):
            mounts.append(f"{prefix}:{prefix}:ro")
        compose = dict(
            services=dict(
                main=dict(
                    volumes=mounts,
                    environment=dict(
                        PATH=f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
                        OMP_NUM_THREADS="2",
                        OPENBLAS_NUM_THREADS="1",
                        MKL_NUM_THREADS="2",
                        CUBLAS_WORKSPACE_CONFIG=":4096:8",
                        HF_HUB_OFFLINE="1",
                    ),
                )
            )
        )
        (folder / "docker-compose.yaml").write_text(yaml.safe_dump(compose))
        overlay = build_lease_overlay(handle.lease, resolve_driver_libs(read_kernel_driver_version()))
        path = write_overlay(overlay)
        self.overlays.append(path)
        env = ManagedDockerEnvironment(
            environment_dir=folder,
            environment_name=f"{self.tag}-{trial}-{role}",
            session_id=uuid.uuid4().hex[:10],
            trial_paths=TrialPaths(folder / "trial"),
            task_env_config=EnvironmentConfig(docker_image=self.args.image, cpus=2, memory_mb=16384),
            extra_docker_compose=[path],
            lease_overlay=overlay,
            gpu_execution=dict(
                router_name=self.tag + "-" + mode, lease_id=handle.lease.lease_id, mode=handle.lease.gpu_reservation
            ),
            network_policy=NetworkPolicy(network_mode="no-network"),
        )
        self.envs.append(env)
        await env.start(force_build=False)
        return (
            env,
            handle,
            dict(
                admission_wait_s=admitted - submitted,
                setup_s=time.monotonic() - admitted,
                lease=dataclasses.asdict(handle.lease),
                submitted_s=submitted,
            ),
        )

    async def close(self, env, handle):
        await env.stop(delete=True)
        self.envs.remove(env)
        await handle.release()
        self.handles.remove(handle)

    async def command(self, env, phase, folder, steps=None):
        started = time.monotonic()
        result = await env.exec_gpu(
            shlex.join(
                [
                    sys.executable,
                    "/study/entry.py",
                    "--mode",
                    phase,
                    "--data-root",
                    "/data/" + ("public" if phase == "train" else "private"),
                    "--steps",
                    str(self.args.steps if steps is None else steps),
                ]
            ),
            timeout_sec=180,
            wait_timeout_sec=240,
        )
        finished = time.monotonic()
        (folder / (phase + "-stdout.txt")).write_text(result.stdout or "")
        (folder / (phase + "-stderr.txt")).write_text(result.stderr or "")
        if result.return_code:
            raise RuntimeError(f"Command failed in {folder}: {(result.stderr or '')[-1000:]}.")
        value = json.loads(
            next(s.split("=", 1)[1] for s in result.stdout.splitlines() if s.startswith("PARTITION_RESULT="))
        )
        if value["cuda_readback"] != 528:
            raise RuntimeError("Container CUDA failed.")
        return dict(
            phase=phase,
            started_s=started,
            finished_s=finished,
            wall_s=finished - started,
            result=value,
            reservation=dict(env.last_gpu_command),
        )

    async def trial(self, mode, gap, repetition):
        name = f"{repetition}-{mode}-{gap:g}"
        row = dict(id=name, mode=mode, gap_s=gap, sessions={})
        self.data["trials"].append(row)
        start = time.monotonic()
        ready = asyncio.Event()

        async def session(role):
            if role == "b":
                await ready.wait()
            env, handle, record = await self.create(mode, name, role)
            row["sessions"][role] = record
            record["commands"] = []
            try:
                record["commands"].append(await self.command(env, "train", self.out / name / role))
                if role == "a":
                    record["gap_started_s"] = time.monotonic()
                    ready.set()
                    await asyncio.sleep(gap)
                    record["gap_ended_s"] = time.monotonic()
                record["commands"].append(await self.command(env, "verify", self.out / name / role))
                record["completed_s"] = time.monotonic()
            finally:
                await self.close(env, handle)
                record["closed_s"] = time.monotonic()
                self.save()

        async with asyncio.TaskGroup() as group:
            group.create_task(session("a"))
            group.create_task(session("b"))
        row["total_s"] = time.monotonic() - start
        row["after"] = gpu_snapshot(self.args.gpu)
        if row["after"]["pids"] or row["after"]["used_mib"] > 64:
            raise RuntimeError("A trial left GPU processes or memory behind.")
        self.save()

    async def conservative_case(self):
        env, handle, record = await self.create("auto", "unknown-contract", "a", contract="unknown")
        try:
            record["command"] = await self.command(env, "train", self.out / "unknown-contract/a")
            record["ledger_after_command"] = [
                dataclasses.asdict(s) for s in await self.routers["auto"].snapshot.remote()
            ]
            if handle.lease.gpu_reservation != "episode":
                raise RuntimeError("Unknown protocol lost its session reservation.")
            self.data["conservative_case"] = record
        finally:
            await self.close(env, handle)

    async def cleanup(self):
        for kind, action in [
            *(("container", lambda e=e: e.stop(delete=True)) for e in self.envs),
            *(("lease", h.release) for h in self.handles),
            *(("node", lambda r=r: r.shutdown_nodes.remote()) for r in self.routers.values()),
        ]:
            try:
                await asyncio.wait_for(action(), 20)
            except Exception as exc:
                self.data["cleanup_errors"].append(dict(kind=kind, error=repr(exc)))
        for router in self.routers.values():
            ray.kill(router, no_restart=True)
        ray.shutdown()
        for overlay in self.overlays:
            overlay.unlink(missing_ok=True)
        for _ in range(30):
            after = gpu_snapshot(self.args.gpu)
            if not after["pids"] and after["used_mib"] < 64:
                break
            await asyncio.sleep(0.5)
        containers = []
        for mode in ("auto", "command", "episode"):
            containers += subprocess.check_output(
                ["docker", "ps", "-aq", "--filter", f"label=psrl.env_unit={self.tag}-{mode}"], text=True, timeout=10
            ).split()
        self.data.update(
            after=after,
            containers_after=containers,
            physical_release_confirmed=not containers and not after["pids"] and after["used_mib"] < 64,
        )
        self.save()


async def main(args):
    study = Study(args)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        async with asyncio.timeout(args.timeout):
            await study.prepare()
            # Ray initialization can replace process signal handlers.
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, task.cancel)
            study.data["status"] = "running"
            study.save()
            await study.conservative_case()
            for repetition in range(args.repeats):
                for gap in args.gaps:
                    modes = ("episode", "auto", "command") if repetition % 2 == 0 else ("command", "auto", "episode")
                    for mode in modes:
                        await study.trial(mode, gap, repetition)
            commands = [c for t in study.data["trials"] for s in t["sessions"].values() for c in s["commands"]]
            hashes = {c["result"]["tensor_sha256"] for c in commands if c["phase"] == "train"}
            scores = {c["result"]["reward"] for c in commands if c["phase"] == "verify"}
            if len(hashes) != 1 or len(scores) != 1:
                raise RuntimeError("Equivalent work produced unequal weights or scores.")
            study.data.update(status="passed", hashes=list(hashes), scores=list(scores))
    except BaseException:
        study.data.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        await study.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gaps", nargs="+", type=float, default=[6, 30])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=2400)
    asyncio.run(main(parser.parse_args()))
