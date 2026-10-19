"""Exercise GPU identity and PTX through the complete original EnvWorker path."""

import argparse
import asyncio
import dataclasses
import json
import subprocess
import uuid
from pathlib import Path

import ray
from omegaconf import OmegaConf
from psrl.workers.env_service.probe import read_gpu_devices
from psrl.workers.env_worker.manager import EnvWorkerManager
from psrl.workers.env_worker.sandbox import SandboxSpec


async def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex[:8]
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=8,
        include_dashboard=False,
        _temp_dir=f"/tmp/evworkergpu-{token}",
        namespace=f"evworkergpu-{token}",
        log_to_driver=False,
    )
    ip = ray.util.get_node_ip_address()
    config = OmegaConf.create(
        {
            "psrl": {
                "env_worker": dict(
                    placement="dedicated",
                    dedicated_node_ips=[ip],
                    workers_per_node=2,
                    cpu_slots_per_worker=1,
                    gpu_slots_per_worker=1,
                    worker_num_cpus=1,
                    exec_default_timeout_s=60,
                    max_observation_chars=10000,
                    idle_sandbox_timeout_s=180,
                    routing={"method": "least_loaded"},
                )
            }
        }
    )
    manager = EnvWorkerManager(config)
    handles = []
    evidence = {"node_ip": ip, "probes": []}
    try:
        inventory = read_gpu_devices()
        for _ in range(2):
            handle = await manager.coordinator_handle().create_sandbox.remote(
                SandboxSpec(
                    image="python:3.13-slim",
                    gpus=1,
                    cpus=1,
                    memory="512m",
                    network="none",
                    labels={"verification.run": token},
                ),
                timeout_s=10,
            )
            handles.append(handle)
            await handle.write_file("/tmp/probe.py", Path(__file__).with_name("cuda_driver_probe.py").read_bytes())
            result = await handle.exec(
                "mkdir -p /logs/artifacts && PROBE_HOLD_SECONDS=0 python /tmp/probe.py", timeout_s=60
            )
            assert result.exit_code == 0, result.stdout
            probe = json.loads((await handle.read_file("/logs/artifacts/cuda-probe.json")).decode())
            expected = inventory[handle.worker_id]
            assert probe["uuid_hex"] == expected.uuid.removeprefix("GPU-").replace("-", ""), probe
            container = json.loads(
                subprocess.check_output(["docker", "inspect", f"psrl-env-{handle.sandbox_id}"], text=True)
            )[0]
            mounts = [item["PathOnHost"] for item in container["HostConfig"]["Devices"]]
            assert expected.path in mounts, mounts
            assert not any(device.path in mounts for index, device in inventory.items() if index != expected.index), (
                mounts
            )
            assert probe["device_count"] == 1 and probe["kernel_output"] == 42
            evidence["probes"].append(
                dict(worker_id=handle.worker_id, expected=dataclasses.asdict(expected), devices=mounts, actual=probe)
            )
        assert {h.worker_id for h in handles} == {0, 1}
        evidence["passed"] = True
    finally:
        for handle in handles:
            await handle.destroy()
        evidence["final_stats"] = await manager.coordinator_handle().stats.remote()
        manager.shutdown()
        ray.shutdown()
        evidence["remaining"] = subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"label=verification.run={token}"], text=True
        ).split()
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    assert not evidence["remaining"]
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args().output))
