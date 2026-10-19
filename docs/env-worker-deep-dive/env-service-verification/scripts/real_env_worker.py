"""Run the existing EnvWorkerManager/Coordinator/Worker path on actual Docker."""

import argparse
import asyncio
import dataclasses
import json
import subprocess
import uuid
from pathlib import Path

import ray
from omegaconf import OmegaConf
from psrl.workers.env_worker.manager import EnvWorkerManager
from psrl.workers.env_worker.sandbox import SandboxSpec


async def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex[:8]
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        _temp_dir=f"/tmp/evworker-{token}",
        namespace=f"evworker-{token}",
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
                    gpu_slots_per_worker=0,
                    worker_num_cpus=1,
                    exec_default_timeout_s=10,
                    max_observation_chars=10000,
                    idle_sandbox_timeout_s=60,
                    routing={"method": "least_loaded"},
                )
            }
        }
    )
    manager = EnvWorkerManager(config)
    coordinator = manager.coordinator_handle()
    handles = []
    evidence = dict(node_ip=ip, scope="Existing Path A EnvWorker; not Harbor or EnvRouter", token=token)
    spec = SandboxSpec(
        image="python:3.13-slim",
        cpus=1,
        memory="256m",
        network="none",
        labels={"verification.run": token},
        startup_timeout_s=20,
    )
    try:
        first = await coordinator.create_sandbox.remote(spec, timeout_s=5)
        handles.append(first)
        second = await coordinator.create_sandbox.remote(spec, timeout_s=5)
        handles.append(second)
        evidence["workers"] = [h.worker_id for h in handles]
        assert first.worker_id != second.worker_id
        evidence["full"] = await coordinator.stats.remote()
        try:
            extra = await coordinator.create_sandbox.remote(spec, timeout_s=1)
        except ray.exceptions.RayTaskError as exc:
            evidence["third_rejected"] = str(exc)
            assert "No env worker capacity" in str(exc)
        else:
            handles.append(extra)
            raise AssertionError("Third sandbox exceeded the pool capacity.")
        init = await first.exec("export VERIFY_STATE=42; cd /tmp; printf original > state.txt", timeout_s=5)
        later = await first.exec('printf "%s\\n" "$VERIFY_STATE"; pwd; cat state.txt', timeout_s=5)
        evidence["persistent_shell"] = [dataclasses.asdict(init), dataclasses.asdict(later)]
        assert later.exit_code == 0 and "42\n/tmp\noriginal" in later.stdout
        await first.write_file("/tmp/binary.bin", b"\x00\xffenv-worker\n")
        data = await first.read_file("/tmp/binary.bin")
        evidence["file_roundtrip_hex"] = data.hex()
        assert data == b"\x00\xffenv-worker\n"
        failed_command = await first.exec("false", timeout_s=5)
        next_command = await first.exec("echo still-alive", timeout_s=5)
        evidence["command_failure"] = dataclasses.asdict(failed_command)
        evidence["after_command_failure"] = dataclasses.asdict(next_command)
        assert failed_command.exit_code == 1 and next_command.exit_code == 0
        ids = subprocess.check_output(
            ["docker", "ps", "-q", "--filter", f"label=verification.run={token}"], text=True
        ).split()
        evidence["docker_containers"] = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True))
        assert len(ids) == 2
        await first.destroy()
        await first.destroy()
        evidence["after_idempotent_destroy"] = await coordinator.stats.remote()
        replacement = await coordinator.create_sandbox.remote(spec, timeout_s=5)
        handles.append(replacement)
        result = await replacement.exec("echo replacement-ready", timeout_s=5)
        evidence["replacement"] = dict(worker_id=replacement.worker_id, result=dataclasses.asdict(result))
        assert result.exit_code == 0
        evidence["passed"] = True
    except BaseException as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        for handle in handles:
            await handle.destroy()
        evidence["final_capacity"] = await coordinator.stats.remote()
        evidence["remaining_containers"] = subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"label=verification.run={token}"], text=True
        ).split()
        manager.shutdown()
        ray.shutdown()
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(
        json.dumps({k: evidence[k] for k in ["passed", "workers", "final_capacity", "remaining_containers"]}, indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
