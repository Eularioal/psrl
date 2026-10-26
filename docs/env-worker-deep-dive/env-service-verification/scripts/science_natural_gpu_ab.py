"""Compare one task's concurrent native training commands on one physical GPU."""

import argparse
import asyncio
import dataclasses
import inspect
import json
import time
import uuid
from pathlib import Path

from science_router_control_ab import gpu_snapshot, save, shell


def sample_gpu(index):
    import pynvml as nv

    result = gpu_snapshot(index)
    nv.nvmlInit()
    try:
        utilization = nv.nvmlDeviceGetUtilizationRates(nv.nvmlDeviceGetHandleByIndex(index))
        result.update(utilization_gpu=utilization.gpu, utilization_memory=utilization.memory)
    finally:
        nv.nvmlShutdown()
    return result


async def run(args):
    import ray
    import yaml
    from examples.sciaccel_rl.config import HarborConfig, SciAccelRuntimeConfig
    from examples.sciaccel_rl.runner import run_harbor_episode

    args.output.mkdir(parents=True, exist_ok=False)
    token = "natural-" + uuid.uuid4().hex[:10]
    record = {"mode": args.mode, "actor_id": token, "started_at": time.time(), "config": vars(args)}
    record["runner_source"] = inspect.getfile(run_harbor_episode)
    before = gpu_snapshot(args.gpu)
    if args.holder_status:
        holder = json.loads(args.holder_status.read_text())
        birth = Path(f"/proc/{holder['pid']}/stat").read_text().rsplit(")", 1)[1].split()[19]
        if (
            holder["status"] != "ready"
            or holder["birth"] != birth
            or holder["gpu_uuid"] != before["uuid"]
            or before["pids"] != [holder["gpu_pid"]]
            or abs(before["used_mib"] - holder["used_mib"]) > 64
        ):
            raise RuntimeError("GPU pressure owner or device state does not match the recorded allocation.")
        record["pressure_holder"] = holder
    elif before["pids"] or before["used_mib"] > 64:
        raise RuntimeError(f"GPU is occupied: {before!r}.")
    record["gpu_before"] = before
    await shell("curl", "--max-time", "3", "--unix-socket", "/var/run/docker.sock", "http://localhost/_ping")
    task = args.output / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'schema_version = "1.4"\n[task]\nname = "verification/mlgym-cifar-native-concurrency"\n'
        f'[environment]\ndocker_image = "{args.image}"\ncpus = 16\nmemory_mb = 65536\ngpus = 1\n'
        'network_mode = "no-network"\n[agent]\ntimeout_sec = 660\n[verifier]\ntimeout_sec = 30\n'
    )
    save(
        task / "instruction.md",
        {
            "mode": args.mode,
            "count": args.count,
            "python": args.python,
            "prepared": args.prepared,
        },
    )
    (task / "environment/docker-compose.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "main": {
                        "volumes": [
                            f"{Path(args.python).parent.parent}:{Path(args.python).parent.parent}:ro",
                            f"{args.workload}:/opt/workload:ro",
                        ],
                        "environment": {
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "HF_HUB_OFFLINE": "1",
                            "HF_DATASETS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                        },
                    }
                }
            }
        )
    )
    if args.dependencies:
        compose = yaml.safe_load((task / "environment/docker-compose.yaml").read_text())
        compose["services"]["main"]["volumes"].append(f"{args.dependencies}:/opt/deps:ro")
        compose["services"]["main"]["environment"]["PYTHONPATH"] = "/opt/deps"
        (task / "environment/docker-compose.yaml").write_text(yaml.safe_dump(compose))
    (task / "tests/test.sh").write_text(
        f"#!/bin/sh\nset -eu\n{args.python} - <<'PY'\n"
        "import json\nfrom pathlib import Path\n"
        f"items = [json.loads(Path(f'/logs/artifacts/cifar-{{i}}.json').read_text()) for i in range({args.count})]\n"
        "ok = all(x['status'] == 'ok' and x['steps'] == 10 for x in items)\n"
        "Path('/logs/verifier/reward.json').write_text(json.dumps({'reward': float(ok)}))\nPY\n"
    )
    stop = asyncio.Event()
    router = None
    monitor = None
    env_args = {}
    try:
        if args.mode == "on":
            from psrl.workers.env_service.client import EnvServiceClient
            from psrl.workers.env_service.router import EnvRouter
            from psrl.workers.env_service.types import EnvDemand

            ray.init(
                address="local",
                namespace=token,
                _node_ip_address=args.node_ip,
                _temp_dir=f"/tmp/ng-{token[-10:]}",
                num_cpus=4,
                num_gpus=0,
                include_dashboard=False,
                log_to_driver=False,
            )
            record["ray_nodes"] = ray.nodes()
            router = EnvRouter.options(name=token).remote(
                {
                    "node_policy": "any",
                    "slot_ttl_s": 900,
                    "units": [
                        {
                            "unit_id": "one-gpu",
                            "node_ip": args.node_ip,
                            "cpu_quota": 16,
                            "memory_mb": 65536,
                            "gpu_indices": [args.gpu],
                            "gpu_memory_mb": args.gpu_budget_mib,
                            "max_containers": 1,
                            "gpu_reservation": "command",
                        }
                    ],
                }
            )
            await asyncio.wait_for(router.ping.remote(), 60)
            client = EnvServiceClient(router, args.node_ip, heartbeat_interval_s=5, router_name=token)
            env_args = {"env_client": client, "env_demand": EnvDemand(16, 65536, True, args.command_memory_mib)}

        async def sample():
            with (args.output / "telemetry.jsonl").open("w") as stream:
                while not stop.is_set():
                    item = {"time": time.time()}
                    try:
                        item["gpu"] = await asyncio.to_thread(sample_gpu, args.gpu)
                        if router:
                            item["units"] = [
                                dataclasses.asdict(x) for x in await asyncio.wait_for(router.snapshot.remote(), 3)
                            ]
                    except Exception as exc:  # noqa: BLE001
                        item["error"] = repr(exc)
                    stream.write(json.dumps(item) + "\n")
                    stream.flush()
                    await asyncio.sleep(0.25)

        monitor = asyncio.create_task(sample())
        config = SciAccelRuntimeConfig(
            harbor=HarborConfig(
                jobs_dir=str(args.output / "jobs"),
                agent_name="science_natural_gpu_agent:NaturalGpuAgent",
                override_gpus=0,
                gpu_compose_override=str(args.overlay),
                max_concurrent_episodes=1,
                memory_mb_override=65536,
            ),
            task_timeout_sec=660,
            verifier_timeout_sec=30,
        )
        record["work_started_at"] = time.time()
        async with asyncio.timeout(780):
            result = await run_harbor_episode(
                task_path=str(task),
                model_base_url="http://127.0.0.1:1/v1",
                model_name="unused",
                config=config,
                needs_gpu=True,
                session_id=token,
                actor_id=token,
                regrade_unverified=False,
                **env_args,
            )
        record["result"] = dataclasses.asdict(result)
        record["work_finished_at"] = time.time()
    except BaseException as exc:
        record["error"] = repr(exc)
    finally:
        stop.set()
        if monitor:
            await monitor
        if router:
            record["final_ledger"] = [dataclasses.asdict(x) for x in await router.snapshot.remote()]
            try:
                await asyncio.wait_for(router.shutdown_nodes.remote(), 20)
            finally:
                ray.kill(router)
        if ray.is_initialized():
            ray.shutdown()
        record["finished_at"] = time.time()
        record["gpu_after"] = gpu_snapshot(args.gpu)
        save(args.output / "result.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--holder-status", type=Path)
    parser.add_argument("--gpu-budget-mib", type=int, default=81920)
    parser.add_argument("--command-memory-mib", type=int, default=4096)
    asyncio.run(run(parser.parse_args()))
