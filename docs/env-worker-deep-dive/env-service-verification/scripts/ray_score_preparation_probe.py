"""Compare preparation timing with unchanged service tests through real Router actors."""

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import shlex
import time
import uuid
from pathlib import Path

import ray
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.overlay import build_lease_overlay, read_kernel_driver_version, resolve_driver_libs
from psrl.workers.env_service.router import create_env_router
from psrl.workers.env_service.types import EnvDemand
from psrl.workers.env_worker.harbor import EnvWorkerEnvironment

psrl_logger = logging.getLogger(__name__)
GROUP = (
    "test/registered/prefill_only/test_multi_item_scoring_adapter.py::"
    "TestMultiItemScoringClassificationAdvancedAdapter"
)


def origins():
    """Report the dependency roots used by the driver and a Ray worker."""
    return {name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue")}


async def scenario(args):
    """Keep host leases fixed while moving only CPU import work before GPU admission."""
    token = f"score-prepare-{uuid.uuid4().hex[:8]}"
    args.output.mkdir(parents=True, exist_ok=False)
    unit = dict(
        unit_id=token,
        node_ip=args.node_ip,
        cpu_quota=16,
        memory_mb=131072,
        max_containers=2,
        gpu_indices=[args.gpu],
        gpu_memory_mb=65536,
        gpu_reservation="command",
        command_gpu_pool=token,
    )
    core = create_env_router(dict(router_name=token, units=[unit], node_policy="any", slot_ttl_s=900))
    probe = EnvNodeService(None, args.node_ip)
    evidence = dict(token=token, status="running", commands={}, batches=[], unit=unit, group=GROUP)
    envs, leases, tasks = {}, [], []
    started = time.time()

    def save():
        path = args.output / "evidence.tmp"
        path.write_text(json.dumps(evidence, indent=2) + "\n")
        path.replace(args.output / "evidence.json")

    async def execute(name, label, prepared):
        submitted = time.time()
        env = envs[name]
        if prepared:
            result = await env.exec_gpu_prepared(
                "/assets/prepared_score_probe.py",
                dict(label=label, group=GROUP),
                prepare_timeout_sec=60,
                timeout_sec=180,
                wait_timeout_sec=240,
            )
        else:
            command = shlex.join(["python3", "/assets/prepared_score_probe.py", "--label", label, "--group", GROUP])
            result = await env.exec_gpu(command, timeout_sec=180, wait_timeout_sec=240)
        finished = time.time()
        (args.output / f"{label}.log").write_text((result.stdout or "") + (result.stderr or ""))
        if result.return_code:
            raise RuntimeError(f"Original service group failed for {label!r}: {result.return_code!r}.")
        group = json.loads((args.output / f"{label}.group.json").read_text())
        nodes = group["nodes"]
        if len(nodes) != 4 or not all(node["passed"] for node in nodes.values()):
            raise RuntimeError(f"Expected four passing original service tests for {label!r}.")
        evidence["commands"][label] = dict(
            submitted=submitted,
            finished=finished,
            wall_s=finished - submitted,
            reservation=dict(env.last_gpu_command),
            preparation=json.loads((args.output / f"{label}.preparation.json").read_text()),
            passed=len(nodes),
        )
        save()

    try:
        before = await probe.prepare((args.gpu,))
        evidence["before"] = before
        if not all(s["used_mb"] < 64 and not s["pids"] for s in before.values()):
            raise RuntimeError("The experiment requires an idle dedicated GPU.")
        libs = resolve_driver_libs(read_kernel_driver_version())
        for name in ("a", "b"):
            admission = await core.try_admit.remote(f"{token}-{name}", EnvDemand(8, 65536, True, 49152), args.node_ip)
            if not admission.lease:
                raise RuntimeError(f"Session admission failed: {admission!r}.")
            lease = admission.lease
            leases.append(lease)
            overlay = build_lease_overlay(lease, libs)
            main = overlay["services"]["main"]
            main["ipc"] = "host"
            main["volumes"] += [
                f"{args.assets / 'code'}:/code:ro",
                f"{args.assets / 'hf-cache'}:/hf-cache:ro",
                f"{args.assets / 'upstream/tasks/ss-sgl22544-score-multi-item-delimiters/tests'}:/tests:ro",
                f"{Path(__file__).resolve().parent}:/assets:ro",
                f"{args.output}:/run:rw",
            ]
            main["environment"] = [
                "LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/lib64:/usr/local/cuda/lib64",
                "PYTHONDONTWRITEBYTECODE=1",
                "OMP_NUM_THREADS=4",
                "PYTHONPATH=/code/python",
            ]
            empty = args.output / f"environment-{name}"
            empty.mkdir()
            env = EnvWorkerEnvironment(
                environment_dir=empty,
                environment_name=f"{token}-{name}",
                session_id=f"{token}-{name}",
                trial_paths=TrialPaths(args.output / f"trial-{name}"),
                actor_id=token,
                task_env_config=EnvironmentConfig(docker_image=args.image, cpus=8, memory_mb=65536),
                network_policy=NetworkPolicy(network_mode="no-network"),
                lease_overlay=overlay,
                gpu_execution={"mode": "command", "router_name": token, "lease_id": lease.lease_id},
            )
            envs[name] = env
            await env.start()
            prep = await env.exec(
                "bash -c 'python3 -I /tests/validate_vendor.py && source /tests/prep.sh'", timeout_sec=60
            )
            if prep.return_code:
                raise RuntimeError(f"Offline container preparation failed: {prep.stderr!r}.")
            await execute(name, f"{name}-warm", False)
        evidence["setup_s"] = time.time() - started
        for batch, prepared in enumerate((False, True, True, False)):
            batch_started = time.time()
            tasks = [asyncio.create_task(execute(name, f"{batch}-{name}", prepared)) for name in envs]
            await asyncio.gather(*tasks)
            evidence["batches"].append(dict(batch=batch, prepared=prepared, wall_s=time.time() - batch_started))
            save()
        evidence["status"] = "passed"
    except BaseException as error:
        evidence.update(status="failed", error=repr(error))
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        errors = []
        for env in envs.values():
            try:
                await asyncio.wait_for(env.stop(), 40)
            except Exception as error:
                errors.append(repr(error))
        for lease in leases:
            try:
                await asyncio.wait_for(core.release.remote(lease.lease_id), 15)
            except Exception as error:
                errors.append(repr(error))
        try:
            await asyncio.wait_for(core.shutdown_nodes.remote(), 15)
        except Exception as error:
            errors.append(repr(error))
        ray.kill(core, no_restart=True)
        evidence["cleanup_errors"] = errors
        evidence["final_gpu"] = await probe.gpu_lending_snapshot((args.gpu,))
        evidence["elapsed_s"] = time.time() - started
        await probe.close()
        save()
        if errors or any(s["used_mb"] >= 64 or s["pids"] for s in evidence["final_gpu"].values()):
            raise RuntimeError("Experiment resources were not physically released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--budget-seconds", type=int, default=540, choices=range(60, 541))
    args = parser.parse_args()
    args.assets, args.output = args.assets.resolve(), args.output.resolve()
    logging.basicConfig(level=logging.INFO)
    namespace = f"score-{uuid.uuid4().hex[:8]}"
    os.environ.pop("RAY_ADDRESS", None)
    ray.init(
        address="local",
        namespace=namespace,
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        _node_ip_address=args.node_ip,
        _temp_dir=f"/tmp/{namespace}",
        object_store_memory=128 * 1024 * 1024,
    )
    try:
        worker = ray.get(ray.remote(origins).remote())
        if worker != origins() or [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]] != [args.node_ip]:
            raise RuntimeError("Ray node identity or dependency roots differ.")
        asyncio.run(asyncio.wait_for(scenario(args), args.budget_seconds))
    finally:
        ray.shutdown()
