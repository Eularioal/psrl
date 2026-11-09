"""Compare fixed work through real Ray actors and persistent GPU task containers."""

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import logging
import os
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


def origins():
    return {name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue")}


async def wait_file(path):
    async with asyncio.timeout(60):
        while not path.exists():
            await asyncio.sleep(0.05)


async def scenario(args, mode, repeat):
    token = f"rpc-place-{uuid.uuid4().hex[:8]}"
    output = args.output / f"{repeat}-{mode}"
    output.mkdir(parents=True, exist_ok=False)
    units = [
        dict(
            unit_id=f"{token}-{i}",
            node_ip=args.node_ip,
            cpu_quota=8 if i == 0 else 2,
            memory_mb=16384 if i == 0 else 4096,
            max_containers=2 if i == 0 else 1,
            gpu_indices=[index],
            gpu_memory_mb=4096,
            gpu_reservation="command",
            command_gpu_pool=token if mode == "pooled" else "",
        )
        for i, index in enumerate(args.gpus)
    ]
    core = create_env_router(dict(router_name=token, units=units, node_policy="any", slot_ttl_s=600))
    probe = EnvNodeService(None, args.node_ip)
    evidence = dict(mode=mode, prepared_a=args.prepared_a, token=token, status="running", commands={}, units=units)
    envs, leases, tasks = {}, [], []
    started = time.time()
    preparation_slots = asyncio.Semaphore(args.window or 3)
    evidence["preparation_window"] = args.window

    def save():
        temporary = output / "evidence.tmp"
        temporary.write_text(json.dumps(evidence, indent=2))
        temporary.replace(output / "evidence.json")

    async def execute(name, phase, repeats=1):
        env = envs[name]
        label = f"{name}_{phase}"
        commands = [f"touch /run/{label}.ready"]
        for i in range(repeats):
            commands.append(
                "python /assets/command_context_probe.py --source /assets/versions/base "
                "--tests /assets/assets/task/tests/postmerge_tests/test/registered/kernels "
                f"--output /run/{label}_{i}.json"
            )
        requested = time.time()
        if (args.window is not None and phase != "warm") or (args.prepared_a and name == "a" and phase == "measured"):
            module = Path(__file__).with_name("prepared_correctness_probe.py")
            await env.upload_file(module, "/tmp/prepared_correctness_probe.py")
            async with preparation_slots:
                operation = asyncio.create_task(
                    env.exec_gpu_prepared(
                        "/tmp/prepared_correctness_probe.py",
                        {"label": label},
                        python="python",
                        prepare_timeout_sec=60,
                        timeout_sec=100,
                        wait_timeout_sec=100,
                    )
                )
                try:
                    async with asyncio.timeout(70):
                        while not (output / f"{label}.preparation.json").exists():
                            if operation.done():
                                await operation
                                raise RuntimeError("Preparation exited without readiness evidence.")
                            await asyncio.sleep(0.05)
                except BaseException:
                    operation.cancel()
                    await asyncio.gather(operation, return_exceptions=True)
                    raise
            try:
                result = await operation
            except BaseException:
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                raise
        else:
            result = await env.exec_gpu(
                " && ".join(commands),
                env={
                    "TRITON_CACHE_DIR": "/tmp/triton-cache",
                    "CUDA_CACHE_PATH": "/tmp/cuda-cache",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "OMP_NUM_THREADS": "1",
                },
                timeout_sec=100,
                wait_timeout_sec=100,
            )
        finished = time.time()
        (output / f"{label}.log").write_text((result.stdout or "") + (result.stderr or ""))
        assert result.return_code == 0, f"Correctness command failed for {label!r}: {result.stderr!r}."
        evidence["commands"][label] = dict(
            requested=requested,
            finished=finished,
            wall_s=finished - requested,
            reservation=dict(env.last_gpu_command),
            probes=[json.loads((output / f"{label}_{i}.json").read_text()) for i in range(repeats)],
        )
        save()

    try:
        before = await probe.prepare(tuple(args.gpus))
        evidence["before"] = before
        assert all(s["used_mb"] < 64 and not s["pids"] for s in before.values()), "Select idle GPUs."
        libs = resolve_driver_libs(read_kernel_driver_version())
        for name in ("a", "b", "c"):
            cpu, memory = (4, 6144) if name != "c" else (2, 4096)
            admission = await core.try_admit.remote(
                f"{token}-{name}", EnvDemand(cpu, memory, True, 3072), args.node_ip
            )
            lease = admission.lease
            assert lease, f"Session admission failed: {admission!r}."
            leases.append(lease)
            assert lease.unit_id == units[int(name == "c")]["unit_id"], "Unexpected home placement."
            overlay = build_lease_overlay(lease, libs)
            main = overlay["services"]["main"]
            main["volumes"] += [
                f"{args.python_env}:{args.python_env}:ro",
                f"{args.assets}:/assets:ro",
                f"{output}:/run:rw",
            ]
            main["environment"] += [
                f"PATH={args.python_env}/bin:/usr/local/bin:/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE=1",
            ]
            empty = output / f"environment-{name}"
            empty.mkdir()
            env = EnvWorkerEnvironment(
                environment_dir=empty,
                environment_name=f"{token}-{name}",
                session_id=f"{token}-{name}",
                trial_paths=TrialPaths(output / f"trial-{name}"),
                actor_id=token,
                task_env_config=EnvironmentConfig(docker_image=args.image, cpus=cpu, memory_mb=memory),
                network_policy=NetworkPolicy(network_mode="no-network"),
                lease_overlay=overlay,
                gpu_execution={"mode": "command", "router_name": token, "lease_id": lease.lease_id},
            )
            envs[name] = env
            await env.start()
            await execute(name, "warm")
        evidence["host_before"] = [dataclasses.asdict(s) for s in await core.snapshot.remote()]
        batch_start = time.time()
        if args.window is not None:

            async def trajectory(name):
                for turn in range(2):
                    await execute(name, f"turn{turn}")

            tasks.extend(asyncio.create_task(trajectory(name)) for name in envs)
            await asyncio.gather(*tasks)
            evidence["batch_s"] = time.time() - batch_start
        else:
            b = asyncio.create_task(execute("b", "measured", 3))
            tasks.append(b)
            await wait_file(output / "b_measured.ready")
            c = asyncio.create_task(execute("c", "measured"))
            tasks.append(c)
            await wait_file(output / "c_measured.ready")
            request = f"{token}-cancel"
            cid = await envs["a"]._gpu_container_id()
            pending = await core.acquire_gpu_command.remote(leases[0].lease_id, cid, request)
            assert pending["status"] == "wait", "Both devices should be reserved during cancellation check."
            await core.cancel_gpu_command.remote(leases[0].lease_id, request)
            evidence["queued_cancellation"] = pending
            a = asyncio.create_task(execute("a", "measured"))
            tasks.append(a)
            await asyncio.gather(*tasks)
            evidence["batch_s"] = time.time() - batch_start
        evidence["host_after"] = [dataclasses.asdict(s) for s in await core.snapshot.remote()]
        fields = ("cpu_used", "memory_used_mb", "leases_active")
        assert [[s[k] for k in fields] for s in evidence["host_before"]] == [
            [s[k] for k in fields] for s in evidence["host_after"]
        ], "Host charges changed during GPU commands."
        if args.window is None:
            second = evidence["commands"]["a_measured"]
            expected = args.gpus[mode == "pooled"]
            assert second["reservation"]["gpu_indices"] == [expected] or tuple(
                second["reservation"]["gpu_indices"]
            ) == (expected,), "A did not use the expected device."
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
                await asyncio.wait_for(core.release.remote(lease.lease_id), 40)
            except Exception as error:
                errors.append(repr(error))
        try:
            await asyncio.wait_for(core.shutdown_nodes.remote(), 30)
        except Exception as error:
            errors.append(repr(error))
        ray.kill(core, no_restart=True)
        evidence["final_gpu"] = await probe.gpu_lending_snapshot(tuple(args.gpus))
        evidence["cleanup_errors"] = errors
        evidence["elapsed_s"] = time.time() - started
        await probe.close()
        save()
        assert not errors and all(s["used_mb"] < 64 and not s["pids"] for s in evidence["final_gpu"].values()), (
            "Owned GPU resources must be physically released."
        )


async def main(args):
    for repeat in range(args.repeats):
        if args.preparation_windows:
            for window in args.preparation_windows if repeat % 2 == 0 else list(reversed(args.preparation_windows)):
                args.window = window
                await asyncio.wait_for(scenario(args, "pooled", f"{repeat}-window{window}"), 240)
        else:
            args.window = None
            for mode in args.modes if repeat % 2 == 0 else list(reversed(args.modes)):
                psrl_logger.info(f"Starting repeat={repeat!r}, mode={mode!r}.")
                await asyncio.wait_for(scenario(args, mode, repeat), 240)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--python-env", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--prepared-a", action="store_true")
    parser.add_argument("--repeats", type=int, default=2, choices=(1, 2))
    parser.add_argument("--modes", nargs="+", choices=("fixed", "pooled"), default=["fixed", "pooled"])
    parser.add_argument("--preparation-windows", nargs="+", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    namespace = f"placement-{uuid.uuid4().hex[:8]}"
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
        nodes = [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]]
        assert nodes == [args.node_ip], f"Unexpected Ray node identity: {nodes!r}."
        worker = ray.get(ray.remote(origins).remote())
        (args.output / "runtime.json").write_text(
            json.dumps(dict(driver=origins(), worker=worker, nodes=nodes), indent=2)
        )
        assert origins() == worker, "Driver and worker dependency sources differ."
        asyncio.run(main(args))
    finally:
        ray.shutdown()
