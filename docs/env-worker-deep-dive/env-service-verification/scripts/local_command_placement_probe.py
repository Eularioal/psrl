"""Verify local command placement with real containers and SWE-Serve correctness tests."""

import argparse
import asyncio
import dataclasses
import json
import logging
import shlex
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.gpu_execution import CommandGpuReservation, GpuExecutionBackend
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.overlay import build_lease_overlay, read_kernel_driver_version, resolve_driver_libs
from psrl.workers.env_service.router import EnvRouterCore
from psrl.workers.env_service.types import EnvDemand
from psrl.workers.env_worker.harbor import EnvWorkerEnvironment

psrl_logger = logging.getLogger(__name__)


class Remote:
    """Replace only Ray transport with direct calls to production services."""

    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        return SimpleNamespace(remote=getattr(self.target, name))


class LocalRouter(EnvRouterCore):
    async def _gpu_command_node(self, node_ip):
        return Remote(self.node)


class LocalPolicy(CommandGpuReservation):
    @property
    def router(self):
        return Remote(self.core)


async def wait_file(path, timeout=45):
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(0.1)


async def scenario(args, pooled, evidence):
    token = f"placement-{uuid.uuid4().hex[:10]}"
    mode = "pooled" if pooled else "fixed"
    output = args.output / mode
    output.mkdir(parents=True, exist_ok=False)
    node = EnvNodeService(None, "local")
    units = [
        {
            "unit_id": f"{token}-{i}",
            "node_ip": "local",
            "cpu_quota": 8 if i == 0 else 2,
            "memory_mb": 16384 if i == 0 else 4096,
            "max_containers": 2,
            "gpu_indices": [index],
            "gpu_memory_mb": 4096,
            "gpu_reservation": "command",
            "command_gpu_pool": token if pooled else "",
        }
        for i, index in enumerate(args.gpus)
    ]
    core = LocalRouter({"node_policy": "any", "units": units, "slot_ttl_s": 600})
    core.node = node
    envs, leases = [], []
    blocker = None
    evidence.update(mode=mode, token=token, status="running", units=units)
    started = time.monotonic()

    def save():
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2))

    async def execute(env, name):
        result = await env.exec_gpu(
            f"python /assets/command_context_probe.py --source /assets/versions/base "
            f"--tests /assets/assets/task/tests/postmerge_tests/test/registered/kernels --output /run/{name}.json",
            env={
                "TRITON_CACHE_DIR": "/tmp/triton-cache",
                "CUDA_CACHE_PATH": "/tmp/cuda-cache",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "OMP_NUM_THREADS": "1",
            },
            timeout_sec=100,
            wait_timeout_sec=15,
        )
        (output / f"{name}.log").write_text((result.stdout or "") + (result.stderr or ""))
        assert result.return_code == 0, f"Correctness command failed: {result.stderr!r}."
        evidence[name] = {
            "command": dict(env.last_gpu_command),
            "probe": json.loads((output / f"{name}.json").read_text()),
        }
        save()

    try:
        await node.prepare(tuple(args.gpus))
        await node.check_gpu_command_support()
        evidence["before"] = await node.gpu_lending_snapshot(tuple(args.gpus))
        assert all(s["used_mb"] < 64 and not s["pids"] for s in evidence["before"].values()), "Select idle GPUs."
        libs = resolve_driver_libs(read_kernel_driver_version())
        for name in ("a", "b"):
            lease = (await core.try_admit(f"{token}-{name}", EnvDemand(4, 6144, True, 3072), "local")).lease
            assert lease and lease.unit_id == units[0]["unit_id"], "Host demand must place both sessions on unit0."
            leases.append(lease)
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
                task_env_config=EnvironmentConfig(docker_image=args.image, cpus=4, memory_mb=6144),
                network_policy=NetworkPolicy(network_mode="no-network"),
                lease_overlay=overlay,
                gpu_execution={"mode": "command", "router_name": "local", "lease_id": lease.lease_id},
            )
            policy = LocalPolicy("local", lease.lease_id)
            policy.core = core
            env._execution_backend = GpuExecutionBackend(policy)
            envs.append(env)
            await env.start()
        a, b = envs
        evidence["host_before"] = [dataclasses.asdict(s) for s in core._ledger.snapshot()]
        await a.exec("echo preserved > /tmp/session-state")
        await execute(a, "a_first")
        holder = (
            "import os,time,json,torch\nfrom pathlib import Path\n"
            "x=torch.ones(64*1024*1024,device='cuda',dtype=torch.uint8)\n"
            "torch.cuda.synchronize()\n"
            "Path('/run/holder-ready.json').write_text(json.dumps({'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
            "deadline=time.monotonic()+90\n"
            "while not Path('/run/holder-stop').exists() and time.monotonic()<deadline: time.sleep(.1)\n"
            "assert Path('/run/holder-stop').exists(), 'Holder deadline expired.'\n"
        )
        blocker = b.start_gpu_job(f"python -c {shlex.quote(holder)}", timeout_sec=120, wait_timeout_sec=15)["job_id"]
        await wait_file(output / "holder-ready.json")
        assert b._gpu_jobs.jobs[blocker].reservation["unit_id"] == units[0]["unit_id"], (
            "Holder must occupy original GPU."
        )
        if pooled:
            await execute(a, "a_second")
            assert evidence["a_second"]["command"]["unit_id"] == units[1]["unit_id"], "A must move to GPU1."
            assert not b._gpu_jobs.jobs[blocker].task.done(), "A must complete while the original GPU is reserved."
        else:
            cid = await a._gpu_container_id()
            waiting = await core.acquire_gpu_command(leases[0].lease_id, cid, "fixed-wait")
            assert waiting["status"] == "wait", "Fixed binding must queue while its original GPU is reserved."
            evidence["fixed_wait"] = waiting
            await core.cancel_gpu_command(leases[0].lease_id, "fixed-wait")
        evidence["while_holder_active"] = [dataclasses.asdict(s) for s in core._ledger.snapshot()]
        evidence["holder_active_after_a"] = not b._gpu_jobs.jobs[blocker].task.done()
        (output / "holder-stop").touch()
        await asyncio.wait_for(b._gpu_jobs.jobs[blocker].task, 30)
        assert b._gpu_jobs.jobs[blocker].state == "completed", "Holder must physically return its GPU."
        if not pooled:
            await execute(a, "a_second")
        assert (await a.exec("cat /tmp/session-state")).stdout.strip() == "preserved", "Workspace must survive."
        evidence["reports"] = await core.gpu_job_reports(leases[0].lease_id)
        evidence["host_after"] = [dataclasses.asdict(s) for s in core._ledger.snapshot()]
        host_fields = ("cpu_used", "memory_used_mb", "leases_active")
        assert [[s[k] for k in host_fields] for s in evidence["host_before"]] == [
            [s[k] for k in host_fields] for s in evidence["host_after"]
        ], "GPU relocation must not migrate or duplicate host charges."
        evidence["status"] = "passed"
    except BaseException as error:
        evidence.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "holder-stop").touch()
        errors = []
        for env in envs:
            try:
                await env.stop()
            except Exception as error:
                errors.append(repr(error))
        for lease in leases:
            try:
                await core.release(lease.lease_id)
            except Exception as error:
                errors.append(repr(error))
        if core._reaper:
            core._reaper.cancel()
            await asyncio.gather(core._reaper, return_exceptions=True)
        evidence["final_gpu"] = await node.gpu_lending_snapshot(tuple(args.gpus))
        evidence["cleanup_errors"] = errors
        evidence["elapsed_s"] = time.monotonic() - started
        await node.close()
        save()
        assert not errors and all(s["used_mb"] < 64 and not s["pids"] for s in evidence["final_gpu"].values()), (
            "Owned GPU resources must be physically released."
        )


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    for pooled in (False, True):
        evidence = {}
        psrl_logger.info(f"Starting local placement scenario with pooled={pooled!r}.")
        await asyncio.wait_for(scenario(args, pooled, evidence), timeout=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--python-env", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(parser.parse_args()))
