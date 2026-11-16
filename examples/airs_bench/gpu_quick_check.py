"""Run bounded real CUDA submissions and isolated grading through PSRL workers."""

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from examples.airs_bench.config import AirsBenchRuntimeConfig
from examples.airs_bench.grading import PYTHON, build_verifier_spec, grade_submission
from examples.gpu_benchmarks.run import gpu_snapshot


async def episode(task, runtime, coordinator):
    from psrl.environments.mlgym_env import build_sandbox_spec
    from psrl.workers.env_worker.coordinator import acquire_sandbox

    source = Path(runtime.data_root) / "airs_prepared" / task["task"]
    verifier, command = build_verifier_spec(runtime, str(source), f"tasks/{task['task']}.yaml", {})
    spec = build_sandbox_spec(runtime, str(source), {"psrl.airs_task_id": task["task"]}, verifier)
    handle = await acquire_sandbox(coordinator, spec, 60)
    try:
        await handle.write_file(
            "/home/agent/workspace/gpu_baseline.py", Path(__file__).with_name("gpu_baseline.py").read_bytes()
        )
        result = await handle.exec(f"{runtime.sandbox_python or PYTHON} /home/agent/workspace/gpu_baseline.py", 120)
        if result.exit_code != 0 or result.timed_out:
            raise RuntimeError(f"GPU baseline failed: {dataclasses.asdict(result)!r}.")
        telemetry = await handle.exec("cat /home/agent/workspace/gpu-baseline.json", 10)
        cuda = json.loads(telemetry.stdout.strip())
        if cuda["peak_cuda_bytes"] <= 0:
            raise RuntimeError("The baseline did not allocate CUDA tensors.")
        metrics = await grade_submission(handle, verifier, command, task["metric"], 120)
        return {"gpu": cuda, "metrics": metrics, "baseline": dataclasses.asdict(result), "passed": True}
    finally:
        await handle.destroy()


async def run(args):
    import ray
    from omegaconf import OmegaConf
    from psrl.workers.env_worker.manager import EnvWorkerManager

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = gpu_snapshot(args.gpu)
    if before["pids"] or before["used_mb"] > 128:
        raise RuntimeError("Select an idle GPU for bounded verification.")
    subprocess.run(["docker", "info"], check=True, timeout=10, stdout=subprocess.DEVNULL)
    tag = "airs-" + uuid.uuid4().hex[:10]
    record = {
        "tag": tag,
        "started": time.time(),
        "gpu_before": before,
        "tasks": [],
        "sources": {name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue")},
    }
    manager = None
    runtime = AirsBenchRuntimeConfig(
        data_root=str(args.data.resolve()),
        image=args.image,
        sandbox_gpus=1,
        sandbox_cpus=2,
        sandbox_memory="8g",
        verifier_cpus=2,
        verifier_memory="8g",
        episode_timeout_s=240,
        per_action_timeout_s=120,
        apps_case_timeout_s=2,
        apps_candidate_timeout_s=6,
        sandbox_python=args.python,
    )
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        ray.init(
            address="local",
            namespace=tag,
            num_cpus=8,
            num_gpus=1,
            include_dashboard=False,
            _node_ip_address=args.node_ip,
            _temp_dir="/tmp/" + tag,
            log_to_driver=False,
        )
        record["ray_nodes"] = ray.nodes()
        config = OmegaConf.create(
            {
                "psrl": {
                    "env_worker": {
                        "placement": "dedicated",
                        "dedicated_node_ips": [args.node_ip],
                        "workers_per_node": 1,
                        "cpu_slots_per_worker": 1,
                        "gpu_slots_per_worker": 1,
                        "worker_num_cpus": 1,
                        "cpu_budget_per_worker": 6,
                        "memory_budget_bytes_per_worker": 20 * 1024**3,
                        "routing": {"method": "least_loaded"},
                        "exec_default_timeout_s": 120,
                        "max_observation_chars": 16000,
                        "coordinator_max_concurrency": 16,
                    }
                }
            }
        )
        manager = EnvWorkerManager(config)
        tasks = json.loads((args.data / "manifest.json").read_text())["tasks"]
        if args.tasks:
            tasks = [task for task in tasks if task["task"] in args.tasks]
            if {task["task"] for task in tasks} != set(args.tasks):
                raise ValueError("Requested task is absent from the manifest.")
        for task in tasks:
            item = {"task": task["task"], "started": time.time()}
            try:
                item.update(await asyncio.wait_for(episode(task, runtime, manager.coordinator_handle()), 240))
            except Exception as error:
                item.update(passed=False, error=repr(error))
            item["finished"] = time.time()
            record["tasks"].append(item)
            (out / "evidence.json").write_text(json.dumps(record, indent=2))
            print(json.dumps(item), flush=True)
    finally:
        if manager is not None:
            manager.shutdown()
        ray.shutdown()
        record["finished"] = time.time()
        record["gpu_after"] = gpu_snapshot(args.gpu)
        record["containers_after"] = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{json .}}"], text=True, timeout=10
        ).splitlines()
        record["passed"] = bool(record["tasks"]) and all(t["passed"] for t in record["tasks"])
        (out / "evidence.json").write_text(json.dumps(record, indent=2))
    if not record["passed"] or record["gpu_after"]["pids"]:
        raise RuntimeError("One or more AIRS checks failed or GPU cleanup remains incomplete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image", default="psrl/airs-bench-agent:latest")
    parser.add_argument("--python", default="")
    parser.add_argument("--tasks", nargs="+")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
