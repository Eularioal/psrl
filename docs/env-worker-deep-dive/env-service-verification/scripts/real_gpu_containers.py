"""Exercise shared GPU containers, driver mismatch and device identity with real Harbor."""

import argparse
import asyncio
import dataclasses
import json
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import ray
from examples.sciaccel_rl.runner import _apply_env_lease
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.overlay import write_overlay
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand


async def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        _temp_dir=f"/tmp/evg-{token}",
        namespace=f"evg-{token}",
        log_to_driver=False,
    )
    ip = ray.util.get_node_ip_address()
    router = EnvRouter.remote(
        dict(
            node_policy="any",
            slot_ttl_s=60,
            units=[
                dict(
                    unit_id="shared",
                    node_ip=ip,
                    cpu_quota=4,
                    memory_mb=4096,
                    gpu_indices=[1],
                    gpu_memory_mb=2048,
                    max_containers=2,
                )
            ],
        )
    )
    client = EnvServiceClient(router, ip, heartbeat_interval_s=5)
    demand = EnvDemand(cpu=1, memory_mb=512, needs_gpu=True, gpu_memory_mb=512)
    environments, handles, paths = [], [], []
    evidence = dict(
        kernel=Path("/proc/driver/nvidia/version").read_text(),
        libcuda_symlink=str(Path("/usr/lib64/libcuda.so.1").resolve()),
    )
    xml = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True))
    evidence["gpu_identity"] = [
        dict(nvml_index=i, uuid=g.findtext("uuid"), minor=int(g.findtext("minor_number")))
        for i, g in enumerate(xml.findall("gpu"))
    ]
    try:
        for i in range(2):
            handle = await client.acquire(f"shared-{token}-{i}", demand, max_wait_s=10)
            handles.append(handle)
            composes, kwargs = [], {}
            path = await _apply_env_lease(handle.lease, composes, kwargs, "", "")
            paths.append(path)
            td = out / f"trial-{i}"
            td.mkdir(exist_ok=True)
            env = DockerEnvironment(
                environment_dir=args.environment.resolve(),
                environment_name=f"shared-{token}",
                session_id=f"evg-{token}-{i}",
                trial_paths=TrialPaths(trial_dir=td),
                task_env_config=EnvironmentConfig(cpus=1, memory_mb=512, gpus=1),
                override_gpus=0,
                extra_docker_compose=composes,
            )
            environments.append(env)
        await asyncio.gather(*(env.start(force_build=False) for env in environments))
        await asyncio.gather(
            *(
                env.upload_file(Path(__file__).with_name("cuda_driver_probe.py"), "/tmp/probe.py")
                for env in environments
            )
        )
        third = await router.try_admit.remote("third", demand, ip)
        evidence["third_admission"] = dataclasses.asdict(third)
        commands = [
            asyncio.create_task(env.exec("PROBE_HOLD_SECONDS=20 python /tmp/probe.py", timeout_sec=60))
            for env in environments
        ]
        await asyncio.sleep(4)
        evidence["during"] = dict(
            leases=[dataclasses.asdict(h.lease) for h in handles],
            snapshot=[dataclasses.asdict(s) for s in await router.snapshot.remote()],
            telemetry=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True
            ),
            containers=subprocess.check_output(
                ["docker", "ps", "--filter", "label=psrl.env_unit=shared", "--format", "{{.ID}} {{.Names}}"], text=True
            ),
        )
        results = await asyncio.gather(*commands)
        evidence["shared_results"] = [r.model_dump() for r in results]
        assert all(r.return_code == 0 for r in results), "Both shared containers must execute CUDA."
        evidence["shared_probes"] = [json.loads(r.stdout.strip()) for r in results]
        assert evidence["shared_probes"][0]["uuid_hex"] == evidence["shared_probes"][1]["uuid_hex"]
        assert third.status == "wait"
        for env in environments:
            await env.stop(delete=True)
        for handle in handles:
            await handle.release()
        evidence["after_shared"] = [dataclasses.asdict(s) for s in await router.snapshot.remote()]

        # Deliberately follow the mismatched symlink as a negative control.
        wrong_library = out / "wrong-libcuda.so"
        shutil.copyfile(Path("/usr/lib64/libcuda.so.1").resolve(), wrong_library)
        wrong = write_overlay({"services": {"main": {"volumes": [f"{wrong_library}:/usr/lib64/libcuda.so.1:ro"]}}})
        paths.append(wrong)
        handle = await client.acquire(f"wrong-{token}", demand, max_wait_s=10)
        handles.append(handle)
        composes, kwargs = [], {}
        paths.append(await _apply_env_lease(handle.lease, composes, kwargs, "", ""))
        td = out / "wrong-driver"
        td.mkdir(exist_ok=True)
        env = DockerEnvironment(
            environment_dir=args.environment.resolve(),
            environment_name=f"wrong-{token}",
            session_id=f"evg-{token}-wrong",
            trial_paths=TrialPaths(trial_dir=td),
            task_env_config=EnvironmentConfig(cpus=1, memory_mb=512, gpus=1),
            override_gpus=0,
            extra_docker_compose=[*composes, wrong],
        )
        environments.append(env)
        await env.start(force_build=False)
        result = await env.exec(
            "python -c \"import ctypes; print(ctypes.CDLL('/usr/lib64/libcuda.so.1').cuInit(0))\"", timeout_sec=30
        )
        evidence["wrong_driver_cuinit"] = result.model_dump()
        evidence["wrong_driver_nvidia_smi"] = (
            await env.exec("nvidia-smi --query-gpu=uuid --format=csv,noheader", timeout_sec=30)
        ).model_dump()
        evidence["driver_mismatch_reproduced"] = result.return_code == 0 and result.stdout.strip() == "803"
        evidence["wrong_driver_loaded_maps"] = (
            await env.exec(
                "python -c \"import ctypes; c=ctypes.CDLL('/usr/lib64/libcuda.so.1'); v=ctypes.c_int(); c.cuDriverGetVersion(ctypes.byref(v)); print(v.value); print(''.join(l for l in open('/proc/self/maps') if 'libcuda' in l))\"",
                timeout_sec=30,
            )
        ).model_dump()
    except Exception as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        for env in environments:
            await env.stop(delete=True)
        for handle in handles:
            await handle.release()
        evidence["final_snapshot"] = [dataclasses.asdict(s) for s in await router.snapshot.remote()]
        for path in paths:
            path.unlink(missing_ok=True)
        (out / "wrong-libcuda.so").unlink(missing_ok=True)
        ray.kill(router)
        ray.shutdown()
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str))
    print(json.dumps(evidence, indent=2, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--environment", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
