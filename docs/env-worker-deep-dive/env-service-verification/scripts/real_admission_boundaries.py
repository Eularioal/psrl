"""Exercise independent GPU memory, CPU and host-memory admission with real Docker."""

import argparse
import asyncio
import dataclasses
import json
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import ray
from examples.sciaccel_rl.runner import _apply_env_lease
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.client import EnvAdmissionTimeout, EnvServiceClient
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=30).strip()


async def check_case(args, name, token):
    out = args.output / name
    out.mkdir(parents=True)
    ip = ray.util.get_node_ip_address()
    gpu = name == "gpu-memory"
    unit = dict(unit_id=f"{name}-{token}", node_ip=ip, cpu_quota=8, memory_mb=8192, max_containers=8)
    if gpu:
        unit.update(gpu_indices=[0], gpu_memory_mb=1200)
    elif name == "cpu":
        unit["cpu_quota"] = 1.5
    else:
        unit["memory_mb"] = 768
    demand = EnvDemand(cpu=1, memory_mb=512, needs_gpu=gpu, gpu_memory_mb=512 if gpu else 0)
    router = EnvRouter.remote(dict(node_policy="any", slot_ttl_s=60, units=[unit]))
    client = EnvServiceClient(router, ip, heartbeat_interval_s=5, poll_interval_s=0.2, max_poll_interval_s=0.3)
    evidence = dict(unit=unit, demand=dataclasses.asdict(demand), started_at=time.time())
    envs, handles, paths, commands = [], [], [], []

    async def start(handle, suffix):
        handles.append(handle)
        composes, kwargs = [], {}
        paths.append(await _apply_env_lease(handle.lease, composes, kwargs, "", ""))
        trial_dir = out / suffix
        trial_dir.mkdir()
        env = DockerEnvironment(
            environment_dir=args.environment.resolve(),
            environment_name=f"boundary-{token}",
            session_id=f"eb-{token}-{name}-{suffix}",
            trial_paths=TrialPaths(trial_dir=trial_dir),
            task_env_config=EnvironmentConfig(cpus=1, memory_mb=512, gpus=int(gpu)),
            override_gpus=0,
            extra_docker_compose=composes,
        )
        envs.append(env)
        await env.start(force_build=False)
        return env

    async def snapshot():
        return [dataclasses.asdict(s) for s in await router.snapshot.remote()]

    def containers():
        return docker("ps", "-q", "--filter", f"label=psrl.env_unit={unit['unit_id']}").split()

    try:
        for i in range(2 if gpu else 1):
            h = await client.acquire(f"{name}-{token}-{i}", demand, max_wait_s=10)
            env = await start(h, f"initial-{i}")
            if gpu:
                await env.upload_file(Path(__file__).with_name("cuda_driver_probe.py"), "/tmp/probe.py")
                commands.append(
                    asyncio.create_task(env.exec("PROBE_HOLD_SECONDS=15 python /tmp/probe.py", timeout_sec=40))
                )
            else:
                result = await env.exec('python -c "print(6 * 7)"', timeout_sec=10)
                evidence["initial_exec"] = result.model_dump()
                assert result.return_code == 0 and result.stdout.strip() == "42"
        if gpu:
            # Check readiness by artifact creation, not by assuming startup duration.
            deadline = time.monotonic() + 20
            for env in envs:
                while time.monotonic() < deadline:
                    result = await env.exec("test -f /logs/artifacts/cuda-probe.json", timeout_sec=5)
                    if result.return_code == 0:
                        break
                    await asyncio.sleep(0.2)
                assert result.return_code == 0, "CUDA probe did not become ready."
        evidence["before_wait"] = dict(snapshot=await snapshot(), containers=containers())
        started = time.monotonic()
        try:
            unexpected = await client.acquire(f"{name}-{token}-blocked", demand, max_wait_s=2)
        except EnvAdmissionTimeout:
            evidence["admission_at_boundary"] = "timeout"
        else:
            handles.append(unexpected)
            raise AssertionError("Budget excess was admitted.")
        evidence["wait_s"] = time.monotonic() - started
        evidence["while_blocked"] = dict(
            snapshot=await snapshot(),
            containers=containers(),
            gpu_telemetry=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True
            ),
        )
        assert evidence["before_wait"]["containers"] == evidence["while_blocked"]["containers"]
        assert evidence["while_blocked"]["snapshot"][0]["leases_active"] < unit["max_containers"]
        if gpu:
            evidence["cuda_results"] = [r.model_dump() for r in await asyncio.gather(*commands)]
            assert all(r["return_code"] == 0 for r in evidence["cuda_results"])
        await envs[0].stop(delete=True)
        await handles[0].release()
        evidence["after_one_release"] = dict(snapshot=await snapshot(), containers=containers())
        h = await client.acquire(f"{name}-{token}-resumed", demand, max_wait_s=10)
        env = await start(h, "resumed")
        if gpu:
            await env.upload_file(Path(__file__).with_name("cuda_driver_probe.py"), "/tmp/probe.py")
            result = await env.exec("PROBE_HOLD_SECONDS=0 python /tmp/probe.py", timeout_sec=30)
        else:
            result = await env.exec('python -c "print(6 * 7)"', timeout_sec=10)
        evidence["resumed"] = dict(
            lease=dataclasses.asdict(h.lease),
            execution=result.model_dump(),
            containers=containers(),
            snapshot=await snapshot(),
        )
        assert result.return_code == 0
        evidence["passed"] = True
    except BaseException as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        for env in envs:
            await env.stop(delete=True)
        for h in handles:
            await h.release()
        evidence["final_snapshot"] = await snapshot()
        evidence["remaining_containers"] = containers()
        for path in paths:
            path.unlink(missing_ok=True)
        ray.kill(router)
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(dict(case=name, passed=evidence["passed"], wait_s=evidence["wait_s"])), flush=True)
    return evidence


async def main(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex[:8]
    xml = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True))
    first = xml.findall("gpu")[0]
    assert int(first.findtext("minor_number")) == 0, "Use a verified index/minor match to isolate the budget test."
    used = float(
        subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        ).strip()
    )
    assert used < 100, "GPU 0 is not idle."
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        _temp_dir=f"/tmp/evbounds-{token}",
        namespace=f"evbounds-{token}",
        log_to_driver=False,
    )
    try:
        summary = dict(node_ip=ray.util.get_node_ip_address(), gpu_uuid=first.findtext("uuid"), cases={})
        for name in ("gpu-memory", "cpu", "host-memory"):
            result = await check_case(args, name, token)
            summary["cases"][name] = dict(passed=result["passed"], remaining=result["remaining_containers"])
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--environment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
