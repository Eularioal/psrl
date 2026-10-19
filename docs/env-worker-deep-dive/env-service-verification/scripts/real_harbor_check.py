"""Run a real Harbor oracle with a real EnvRouter lease and record container evidence."""

import argparse
import asyncio
import dataclasses
import json
import subprocess
import time
import uuid
from pathlib import Path

import ray
import yaml
from examples.sciaccel_rl.env_demand import demand_from_task
from examples.sciaccel_rl.runner import _apply_env_lease
from harbor.job import Job
from harbor.models.job.config import AgentConfig, JobConfig
from harbor.models.trial.config import TaskConfig
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.router import EnvRouter


def docker(*args):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=20)
    result.check_returncode()
    return result.stdout.strip()


async def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:10]
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        _temp_dir=f"/tmp/evh-{stamp}",
        namespace=f"evh-{stamp}",
        log_to_driver=False,
    )
    ip = ray.util.get_node_ip_address()
    demand = demand_from_task(str(args.task), gpu_memory_mb_override=1024)
    unit = dict(unit_id="real-harbor", node_ip=ip, cpu_quota=16, memory_mb=32768, max_containers=2)
    if demand.needs_gpu:
        unit.update(gpu_indices=[args.gpu], gpu_memory_mb=8000)
    router = EnvRouter.remote(dict(units=[unit], node_policy="any", slot_ttl_s=60))
    client = EnvServiceClient(router, ip, heartbeat_interval_s=5)
    handle = await client.acquire(f"verify-{stamp}", demand, max_wait_s=15)
    evidence = dict(
        task=str(args.task),
        agent=args.agent,
        node_ip=ip,
        lease=dataclasses.asdict(handle.lease),
        containers={},
        related_containers={},
        samples=[],
        started_at=time.time(),
    )
    paths, env = [], {}
    overlay = await _apply_env_lease(handle.lease, paths, env, "", "")
    actor_overlay = out / "actor-label.yaml"
    actor_overlay.write_text(yaml.safe_dump({"services": {"main": {"labels": [f"psrl.actor_id=verify-{stamp}"]}}}))
    env["extra_docker_compose"] = [*paths, actor_overlay]
    env.setdefault("kwargs", {})["actor_id"] = f"verify-{stamp}"
    evidence["overlay"] = yaml.safe_load(overlay.read_text())
    (out / "lease.json").write_text(json.dumps(evidence, indent=2))
    stop = asyncio.Event()

    async def monitor():
        while not stop.is_set():
            ids = await asyncio.to_thread(
                docker, "ps", "-aq", "--filter", f"label=psrl.env_lease={handle.lease.lease_id}"
            )
            for cid in ids.split():
                if cid not in evidence["containers"]:
                    data = json.loads(await asyncio.to_thread(docker, "inspect", cid))[0]
                    evidence["containers"][cid] = dict(
                        name=data["Name"],
                        labels=data["Config"]["Labels"],
                        devices=data["HostConfig"]["Devices"],
                        mounts=data["Mounts"],
                    )
            related = await asyncio.to_thread(docker, "ps", "-aq", "--filter", f"name={args.task.name[:24]}")
            for cid in related.split():
                if cid not in evidence["related_containers"]:
                    data = json.loads(await asyncio.to_thread(docker, "inspect", cid))[0]
                    evidence["related_containers"][cid] = dict(
                        name=data["Name"], labels=data["Config"]["Labels"], devices=data["HostConfig"]["Devices"]
                    )
            snapshots = await router.snapshot.remote()
            evidence["samples"].append(
                dict(at=time.time(), containers=ids.split(), snapshots=[dataclasses.asdict(s) for s in snapshots])
            )
            (out / "live.json").write_text(json.dumps(evidence, indent=2, default=str))
            try:
                await asyncio.wait_for(stop.wait(), 2)
            except asyncio.TimeoutError:
                pass

    monitor_task = asyncio.create_task(monitor())
    try:
        job = await Job.create(
            JobConfig(
                jobs_dir=out / "jobs",
                job_name=f"verify-{stamp}",
                tasks=[TaskConfig(path=args.task.resolve())],
                agents=[AgentConfig(name=args.agent)],
                environment=env,
                n_attempts=1,
                n_concurrent_trials=1,
                quiet=True,
            )
        )
        result = await asyncio.wait_for(job.run(), args.timeout)
        (out / "harbor-result.json").write_text(result.model_dump_json(indent=2))
        evidence["trials"] = [
            dict(
                task=t.task_name,
                reward=t.verifier_result.rewards if t.verifier_result else None,
                exception=t.exception_info.model_dump(mode="json") if t.exception_info else None,
            )
            for t in result.trial_results or []
        ]
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        try:
            await monitor_task
        except Exception as exc:
            evidence["monitor_error"] = repr(exc)
        remaining = await asyncio.to_thread(
            docker, "ps", "-aq", "--filter", f"label=psrl.env_lease={handle.lease.lease_id}"
        )
        evidence["remaining_before_release"] = remaining.split()
        if remaining:
            await asyncio.to_thread(docker, "rm", "-f", *remaining.split())
        await handle.release()
        evidence["final_snapshot"] = [dataclasses.asdict(s) for s in await router.snapshot.remote()]
        evidence["elapsed_s"] = time.time() - evidence["started_at"]
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str))
        overlay.unlink(missing_ok=True)
        ray.kill(router)
        ray.shutdown()
    print(
        json.dumps(
            {
                k: evidence.get(k)
                for k in ("trials", "error", "elapsed_s", "remaining_before_release", "final_snapshot")
            },
            indent=2,
        )
    )
    assert evidence["trials"] and all(
        t["exception"] is None and t["reward"] is not None for t in evidence["trials"]
    ), "Harbor must return a graded trial."
    assert evidence["containers"], "Must observe real containers carrying the lease label."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", default="oracle")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=2400)
    asyncio.run(main(parser.parse_args()))
