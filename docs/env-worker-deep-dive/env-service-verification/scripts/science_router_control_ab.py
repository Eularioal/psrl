"""Compare the fixed historical runner with command reservations on one physical GPU."""

import argparse
import asyncio
import dataclasses
import inspect
import json
import time
import uuid
from pathlib import Path


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str))


async def shell(*args, timeout=15):
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"Command {args[:2]!r} failed: {stderr.decode()!r}.")
    return stdout.decode()


def gpu_snapshot(gpu):
    import pynvml as nv

    nv.nvmlInit()
    try:
        device = nv.nvmlDeviceGetHandleByIndex(gpu)
        memory = nv.nvmlDeviceGetMemoryInfo(device, version=nv.nvmlMemory_v2)
        return {
            "uuid": nv.nvmlDeviceGetUUID(device),
            "minor": nv.nvmlDeviceGetMinorNumber(device),
            "used_mib": memory.used / 1024**2,
            "total_mib": memory.total / 1024**2,
            "pids": [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(device)],
        }
    finally:
        nv.nvmlShutdown()


def prepare(args):
    import yaml
    from psrl.workers.env_service.overlay import build_lease_overlay, read_kernel_driver_version, resolve_driver_libs
    from psrl.workers.env_service.types import EnvDemand, EnvLease

    args.output.mkdir(parents=True, exist_ok=True)
    demand = EnvDemand(1, 1024, True, (args.gib + 1) * 1024)
    lease = EnvLease("overlay", "overlay", "probe", args.node_ip, (args.gpu,), demand, 0, 0)
    overlay = build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version()))
    overlay["services"]["main"].pop("labels", None)
    (args.output / "gpu-overlay.yaml").write_text(yaml.safe_dump(overlay))
    save(args.output / "design.json", vars(args))
    save(args.output / "gpu-before.json", gpu_snapshot(args.gpu))


def prepare_tasks(args):
    tasks = []
    for index in range(4):
        task = args.output / "tasks" / str(index)
        (task / "environment").mkdir(parents=True, exist_ok=True)
        (task / "tests").mkdir(exist_ok=True)
        (task / "task.toml").write_text(
            'schema_version = "1.4"\n[task]\n'
            f'name = "verification/router-control-{index}"\n'
            "[environment]\ncpus = 1\nmemory_mb = 1024\ngpus = 1\n"
            f'docker_image = "{args.image}"\nnetwork_mode = "no-network"\n'
            "[agent]\ntimeout_sec = 240\n[verifier]\ntimeout_sec = 30\n"
        )
        save(task / "instruction.md", {"mode": args.mode, "gib": args.gib, "hold_s": 2, "cpu_s": 8})
        (task / "tests/test.sh").write_text(
            "#!/bin/sh\nset -eu\npython3 - <<'PY'\n"
            "import json\nfrom pathlib import Path\n"
            "for i in range(2):\n"
            "    item = json.loads(Path(f'/logs/artifacts/pressure-{i}.json').read_text())\n"
            "    assert item['status'] == 'ok' and item['kernel_output'] == 42\n"
            "Path('/logs/verifier/reward.json').write_text(json.dumps({'reward': 1.0}))\nPY\n"
        )
        tasks.append(task)
    return tasks


async def run(args):
    import ray
    from examples.sciaccel_rl.config import HarborConfig, SciAccelRuntimeConfig
    from examples.sciaccel_rl.runner import run_harbor_episode

    args.output.mkdir(parents=True, exist_ok=True)
    token = "control-ab-" + uuid.uuid4().hex[:10]
    record = {
        "mode": args.mode,
        "actor_id": token,
        "started_at": time.time(),
        "runner_source": inspect.getfile(run_harbor_episode),
        "scope": "Science runner and Harbor with synthetic CUDA tasks, no LLM or optimizer",
        "attempts": [],
    }
    save(args.output / "result.json", record)
    before = gpu_snapshot(args.gpu)
    record["gpu_before"] = before
    if before["pids"] or before["used_mib"] > 64:
        raise RuntimeError(f"Experiment GPU is occupied: {before!r}.")
    await shell("curl", "--max-time", "3", "--unix-socket", "/var/run/docker.sock", "http://localhost/_ping")
    router, client = None, None
    tasks = prepare_tasks(args)
    sessions = []
    monitor = None
    stop = asyncio.Event()
    env_args = {}
    try:
        if args.mode == "on" or args.ray_for_off:
            ray.init(
                address="local",
                namespace=token,
                _node_ip_address=args.node_ip,
                _temp_dir=f"/tmp/rc-{token[-10:]}",
                num_cpus=4,
                num_gpus=0,
                include_dashboard=False,
                log_to_driver=False,
            )
        if args.mode == "on":
            from gpu_command_trace import TracedRouter
            from psrl.workers.env_service.client import EnvServiceClient
            from psrl.workers.env_service.types import EnvDemand

            router = (
                ray.remote(num_cpus=0)(TracedRouter)
                .options(name=token)
                .remote(
                    {
                        "node_policy": "any",
                        "slot_ttl_s": 300,
                        "units": [
                            {
                                "unit_id": "one-gpu",
                                "node_ip": args.node_ip,
                                "cpu_quota": 4,
                                "memory_mb": 4096,
                                "gpu_indices": [args.gpu],
                                "gpu_memory_mb": 81920,
                                "max_containers": 4,
                                "gpu_reservation": "command",
                            }
                        ],
                    }
                )
            )
            await asyncio.wait_for(router.ping.remote(), timeout=60)
            client = EnvServiceClient(router, args.node_ip, heartbeat_interval_s=5, router_name=token)
            env_args = {"env_client": client, "env_demand": EnvDemand(1, 1024, True, (args.gib + 1) * 1024)}
        record["ray_initialized"] = ray.is_initialized()
        record["prewarm_mode"] = args.prewarm
        record["preparation_started_at"] = time.time()

        async def warm():
            record["prewarm_started_at"] = time.time()
            result = await router.wait_prewarm.remote()
            record["prewarm_finished_at"] = time.time()
            record["prewarm_nodes"] = result

        warming = None
        if router and args.prewarm != "lazy":
            await router.begin_prewarm.remote()
            if args.prewarm == "serial":
                await warm()
            else:
                warming = asyncio.create_task(warm())
        record["independent_preparation_started_at"] = time.time()
        await asyncio.sleep(args.preparation_seconds)
        record["independent_preparation_finished_at"] = time.time()
        if warming:
            await warming
        record["preparation_finished_at"] = time.time()
        config = SciAccelRuntimeConfig(
            harbor=HarborConfig(
                jobs_dir=str(args.output / "jobs"),
                agent_name="science_router_control_agent:RouterControlAgent",
                override_gpus=0,
                gpu_compose_override=str(args.overlay),
                max_concurrent_episodes=4,
                memory_mb_override=1024,
            ),
            task_timeout_sec=240,
            verifier_timeout_sec=30,
        )

        async def sample():
            with (args.output / "telemetry.jsonl").open("w") as stream:
                while not stop.is_set():
                    item = {"time": time.time()}
                    try:
                        item["gpu"] = await asyncio.to_thread(gpu_snapshot, args.gpu)
                        if router:
                            item["units"] = [
                                dataclasses.asdict(x)
                                for x in await asyncio.wait_for(router.snapshot.remote(), timeout=3)
                            ]
                    except Exception as exc:  # noqa: BLE001
                        item["error"] = repr(exc)
                    stream.write(json.dumps(item) + "\n")
                    stream.flush()
                    await asyncio.sleep(0.5)

        monitor = asyncio.create_task(sample())
        slots = asyncio.Semaphore(4)

        async def episode(index):
            for attempt in range(1, 9):
                async with slots:
                    session = f"{token}-{index}-{attempt}"
                    sessions.append(session)
                    started = time.time()
                    result = await run_harbor_episode(
                        task_path=str(tasks[index % 4]),
                        model_base_url="http://127.0.0.1:1/v1",
                        model_name="unused",
                        config=config,
                        needs_gpu=True,
                        session_id=session,
                        actor_id=token,
                        regrade_unverified=False,
                        **env_args,
                    )
                    paths = list((args.output / "jobs" / session).rglob("router-control.json"))
                    agent = json.loads(paths[0].read_text()) if len(paths) == 1 else {"status": "missing"}
                    accepted = agent["status"] == "ok" and result.reward == 1 and not result.exception
                    item = {
                        "index": index,
                        "task": index % 4,
                        "attempt": attempt,
                        "session": session,
                        "started_at": started,
                        "finished_at": time.time(),
                        "accepted": accepted,
                        "agent": agent,
                        "result": dataclasses.asdict(result),
                    }
                    record["attempts"].append(item)
                    save(args.output / "result.json", record)
                    print(
                        json.dumps(
                            {
                                "mode": args.mode,
                                "index": index,
                                "attempt": attempt,
                                "accepted": accepted,
                                "status": agent["status"],
                            }
                        ),
                        flush=True,
                    )
                    if accepted:
                        return
                    if agent["status"] != "cuda_oom":
                        raise RuntimeError(f"Non-OOM task failure: {item!r}.")
                await asyncio.sleep(2 + index % 3)
            raise RuntimeError(f"Retry budget exhausted for task {index!r}.")

        record["work_started_at"] = time.time()
        record["waves"] = []
        async with asyncio.timeout(args.budget):
            wave_size = args.wave_size or args.count
            for first in range(0, args.count, wave_size):
                wave = {"first": first, "started_at": time.time()}
                async with asyncio.TaskGroup() as group:
                    for index in range(first, min(first + wave_size, args.count)):
                        group.create_task(episode(index))
                wave["finished_at"] = time.time()
                wave["duration_s"] = wave["finished_at"] - wave["started_at"]
                record["waves"].append(wave)
                save(args.output / "result.json", record)
        record["work_finished_at"] = time.time()
        record["work_duration_s"] = record["work_finished_at"] - record["work_started_at"]
        record["passed"] = True
    except BaseException as exc:
        record["error"] = repr(exc)
        record["passed"] = False
    finally:
        stop.set()
        if monitor:
            await monitor
        if router:
            try:
                record["final_ledger"] = [dataclasses.asdict(x) for x in await router.snapshot.remote()]
                save(args.output / "control-trace.json", await router.export_trace.remote())
            finally:
                try:
                    await asyncio.wait_for(router.shutdown_nodes.remote(), timeout=15)
                finally:
                    ray.kill(router)
                    ray.shutdown()
        if ray.is_initialized():
            ray.shutdown()
        record["cleanup_errors"] = []
        for session in sessions:
            try:
                ids = (
                    await shell("docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={session}")
                ).split()
                if ids:
                    await shell("docker", "rm", "-f", *ids, timeout=30)
            except Exception as exc:  # noqa: BLE001
                record["cleanup_errors"].append(repr(exc))
        record["remaining_owned_containers"] = (
            await shell("docker", "ps", "-aq", "--filter", f"label=psrl.actor_id={token}")
        ).split()
        for _ in range(10):
            record["gpu_after"] = gpu_snapshot(args.gpu)
            if not record["gpu_after"]["pids"] and record["gpu_after"]["used_mib"] < 64:
                break
            await asyncio.sleep(1)
        record["finished_at"] = time.time()
        save(args.output / "result.json", record)
    if not record.get("passed") or record["cleanup_errors"] or record["remaining_owned_containers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    parser.add_argument("--mode", choices=["off", "on"], default="off")
    parser.add_argument("--gib", type=int, default=50)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--wave-size", type=int, default=0)
    parser.add_argument("--budget", type=int, default=480)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prewarm", choices=["lazy", "serial", "overlap"], default="lazy")
    parser.add_argument("--preparation-seconds", type=float, default=0)
    parser.add_argument("--ray-for-off", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare:
        prepare(arguments)
    else:
        asyncio.run(run(arguments))
