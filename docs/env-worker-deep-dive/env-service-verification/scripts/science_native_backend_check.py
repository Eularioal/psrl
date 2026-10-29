"""Run one cached LAPS science oracle through a bounded PSRL episode."""

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=15).strip()


async def main(args):
    from examples.sciaccel_rl.config import HarborConfig, SciAccelRuntimeConfig
    from examples.sciaccel_rl.runner import run_harbor_episode

    args.output.mkdir(parents=True, exist_ok=False)
    task = args.output / "task"
    shutil.copytree(args.task, task)
    hashes = {
        str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest() for p in task.rglob("*") if p.is_file()
    }
    for path in task.rglob("Dockerfile"):
        path.unlink()
    config = (task / "task.toml").read_text()
    config = config.replace("[environment]\n", f'[environment]\ndocker_image = "{args.image}"\n')
    config = config.replace(
        "[verifier.environment]\n", f'[verifier.environment]\ndocker_image = "{args.verifier_image}"\n'
    )
    config = config.replace('network_mode = "allowlist"', 'network_mode = "no-network"')
    config = re.sub(r"^allowed_hosts = .*$", "allowed_hosts = []", config, flags=re.M)
    (task / "task.toml").write_text(config)
    tag = "native-science-" + uuid.uuid4().hex[:10]
    evidence = {
        "mode": args.mode,
        "tag": tag,
        "source": inspect.getfile(run_harbor_episode),
        "task_hashes": hashes,
        "deployment_changes": (
            "Cached images, no-network oracle, no build. Scientific sources, solve and grading unchanged."
        ),
    }
    evidence["sources"] = {
        name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "harbor")
    }
    evidence["runner_sha256"] = hashlib.sha256(Path(inspect.getfile(run_harbor_episode)).read_bytes()).hexdigest()
    router = None
    options = {}
    started = time.monotonic()
    try:
        if args.mode == "native":
            import ray
            from psrl.workers.env_service.client import EnvServiceClient
            from psrl.workers.env_service.router import EnvRouter

            ray.init(
                address="local",
                namespace=tag,
                _node_ip_address=args.node_ip,
                _temp_dir=f"/tmp/{tag}",
                num_cpus=2,
                num_gpus=0,
                include_dashboard=False,
            )
            evidence["ray_nodes"] = ray.nodes()
            router = EnvRouter.options(name=tag).remote(
                {
                    "node_policy": "cpu_only",
                    "units": [
                        {
                            "unit_id": tag,
                            "node_ip": args.node_ip,
                            "cpu_quota": 9,
                            "memory_mb": 17000,
                            "gpu_indices": [],
                            "gpu_memory_mb": 0,
                            "max_containers": 1,
                        }
                    ],
                }
            )
            await asyncio.wait_for(router.ping.remote(), 45)
            options["env_client"] = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=5)
        harbor = {"jobs_dir": str(args.output / "jobs"), "agent_name": "oracle"}
        if args.mode == "native":
            harbor["environment_backend"] = "env_worker"
        cfg = SciAccelRuntimeConfig(harbor=HarborConfig(**harbor), task_timeout_sec=180, verifier_timeout_sec=240)
        before = time.monotonic()
        kwargs = dict(
            task_path=str(task),
            model_base_url="http://127.0.0.1:1/v1",
            model_name="unused-oracle",
            config=cfg,
            session_id=tag,
            actor_id=tag,
            **options,
        )
        result = await asyncio.wait_for(run_harbor_episode(**kwargs), 480)
        evidence["episode_s"] = time.monotonic() - before
        evidence["result"] = dataclasses.asdict(result)
        evidence["passed"] = result.exception is None and result.rewards.get("reward_repair") == 1.0
        evidence["trials"] = [json.loads(p.read_text()) for p in (args.output / "jobs").rglob("result.json")]
        if router:
            evidence["snapshot"] = [dataclasses.asdict(x) for x in await router.snapshot.remote()]
    finally:
        ids = docker("ps", "-aq", "--filter", f"label=psrl.actor_id={tag}").split()
        if ids:
            docker("rm", "-f", *ids)
        evidence["remaining_containers"] = docker("ps", "-aq", "--filter", f"label=psrl.actor_id={tag}")
        if router:
            await router.shutdown_nodes.remote()
            ray.kill(router)
            ray.shutdown()
        evidence["total_s"] = time.monotonic() - started
        (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str))
    assert evidence.get("passed") and not evidence["remaining_containers"], "Science grading and cleanup must pass."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["native", "baseline"], required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--verifier-image", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
