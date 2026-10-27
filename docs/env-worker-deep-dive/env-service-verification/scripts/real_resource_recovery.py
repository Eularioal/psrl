"""Bounded real Harbor/Docker resource and Ray node-handoff fault experiment.

Uses a cached Python image, four CPU-only containers, one isolated Ray cluster,
and unique ownership labels. No model training or image builds are performed.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import ray
import yaml
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.container_budget import ContainerBudget
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.overlay import build_lease_overlay
from psrl.workers.env_service.router import RayEnvRouter
from psrl.workers.env_service.types import EnvDemand


def command(*args, check=True):
    return subprocess.run(args, text=True, capture_output=True, timeout=15, check=check)


def docker(*args):
    return command("docker", *args).stdout.strip()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class HandoffNode(EnvNodeService):
    """Pause only at the fault-injection point and retain production cleanup and ack."""

    def __init__(self, router, ip, marker):
        super().__init__(router, ip)
        self.marker = Path(marker)

    async def cleanup(self):
        pending = await self.router.pending_reaped.remote(self.node_ip)
        if pending and self.marker.exists():
            save(self.marker.with_suffix(".received.json"), [lease.lease_id for lease in pending])
            return len(pending)
        return await super().cleanup()


class HandoffRouter(RayEnvRouter):
    def __init__(self, config, marker):
        super().__init__(config)
        self.marker = marker

    def _create_node(self, ip):
        # This experiment owns a single-node local Ray cluster.
        return ray.remote(num_cpus=0)(HandoffNode).remote(ray.get_runtime_context().current_actor, ip, self.marker)

    async def node_handle(self, ip):
        return await self._gpu_command_node(ip)

    def diagnostics(self):
        return {
            "health": {ip: asdict(state) for ip, state in self._node_health.items()},
            "sources": {name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue")},
        }


@ray.remote(num_cpus=0)
class Owner:
    def sources(self):
        return {name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue")}

    async def acquire(self, router, ip, demand, session):
        self.client = EnvServiceClient(router, ip, heartbeat_interval_s=0.3)
        self.handle = await self.client.acquire(session, demand, max_wait_s=15)
        return self.handle.lease


async def until(read, accept, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = await read()
        if accept(value):
            return value
        await asyncio.sleep(0.3)
    raise TimeoutError(f"Observation deadline exceeded; last observation: {value!r}")


async def experiment(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    unit = f"resource-{token}"
    marker = out / "pause-cleanup"
    marker.touch()
    evidence = {"started_at": time.time(), "token": token, "unit": unit, "passed": False}
    save(out / "ownership.json", {"unit": unit, "ray_temp": f"/tmp/er-{token}"})
    evidence["dependencies"] = {
        name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "harbor", "ray")
    }
    repo = Path(__file__).resolve().parents[4]
    evidence["source_sha256"] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (repo / "psrl/workers/env_service").glob("*.py")
    }
    evidence["docker"] = json.loads(docker("info", "--format", "{{json .}}"))
    evidence["image"] = json.loads(docker("image", "inspect", args.image))[0]["Id"]
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        _memory=512 * 1024**2,
        _node_ip_address=args.node_ip,
        _temp_dir=f"/tmp/er-{token}",
        namespace=unit,
        log_to_driver=False,
    )
    evidence["ray_nodes"] = ray.nodes()
    assert [row["NodeManagerAddress"] for row in ray.nodes() if row["Alive"]] == [args.node_ip]
    budget = ContainerBudget(0.5, 128, 1, environments=2, sidecar_cpu=0.25, sidecar_memory_mb=64)
    demand = EnvDemand(cpu=budget.cpu, memory_mb=budget.memory_mb)
    config = {
        "node_policy": "any",
        "slot_ttl_s": 3,
        "health": {"interval_s": 0.2, "timeout_s": 30, "max_failures": 3},
        "telemetry": {"interval_s": 0.5, "timeout_s": 6, "max_age_s": 10},
        "units": [{"unit_id": unit, "node_ip": args.node_ip, "cpu_quota": 2, "memory_mb": 512, "max_containers": 1}],
    }
    router = ray.remote(num_cpus=0)(HandoffRouter).remote(config, str(marker))
    owner = Owner.remote()
    envs, projects = [], set()
    try:
        evidence["worker_sources"] = await owner.sources.remote()
        evidence["router_before"] = await router.diagnostics.remote()
        lease = await owner.acquire.remote(router, args.node_ip, demand, unit)
        evidence["lease"] = asdict(lease)
        environment = out / "environment"
        environment.mkdir(exist_ok=True)
        (environment / "docker-compose.yaml").write_text(
            yaml.safe_dump(
                {
                    "services": {
                        "sidecar": {
                            "image": args.image,
                            "command": ["python3", "-c", "import time; x=bytearray(16*1024**2); time.sleep(180)"],
                        }
                    }
                }
            )
        )
        # Two environments model the main and separate-verifier container sets.
        for role in ("agent", "verifier"):
            trial = out / role
            trial.mkdir(exist_ok=True)
            env = ManagedDockerEnvironment(
                environment_dir=environment,
                environment_name=unit,
                session_id=f"{unit}-{role}",
                trial_paths=TrialPaths(trial_dir=trial),
                task_env_config=EnvironmentConfig(docker_image=args.image, cpus=1, memory_mb=128),
                lease_overlay=build_lease_overlay(lease, None),
                container_budget=asdict(budget),
                actor_id=unit,
            )
            envs.append(env)
            await env.start(force_build=False)
        ids = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={unit}").split()
        rows = json.loads(docker("inspect", *ids))
        projects.update(row["Config"]["Labels"]["com.docker.compose.project"] for row in rows)
        evidence["containers"] = [
            {
                "id": row["Id"],
                "labels": row["Config"]["Labels"],
                "memory": row["HostConfig"]["Memory"],
                "nano_cpus": row["HostConfig"]["NanoCpus"],
            }
            for row in rows
        ]
        save(out / "owned-projects.json", sorted(projects))
        assert len(rows) == 4
        for row in rows:
            main = row["Config"]["Labels"]["com.docker.compose.service"] == "main"
            assert row["HostConfig"]["Memory"] == (128 if main else 64) * 1024**2
            assert row["HostConfig"]["NanoCpus"] == (500_000_000 if main else 250_000_000)

        async def snapshot():
            return asdict((await router.snapshot.remote())[0])

        evidence["physical_sample"] = await until(
            snapshot, lambda s: s["containers_running"] == 4 and (s["measured_memory_mb"] or 0) > 0
        )
        assert evidence["physical_sample"]["leases_active"] == 1
        sidecar = next(row["Id"] for row in rows if row["Config"]["Labels"]["com.docker.compose.service"] == "sidecar")
        if not args.skip_memory_probe:
            started = str(int(time.time()))
            oom = command("docker", "exec", sidecar, "python3", "-c", "x=bytearray(256*1024**2)", check=False)
            evidence["memory_limit_probe"] = {"returncode": oom.returncode, "stdout": oom.stdout, "stderr": oom.stderr}
            events = docker(
                "events",
                "--since",
                started,
                "--until",
                str(int(time.time()) + 1),
                "--filter",
                f"container={sidecar}",
                "--filter",
                "event=oom",
                "--format",
                "{{json .}}",
            )
            evidence["oom_events"] = [json.loads(line) for line in events.splitlines()]
            assert oom.returncode == 137 and evidence["oom_events"]
        else:
            evidence["memory_probe_skipped"] = True
        save(out / "live.json", evidence)

        old_node = await router.node_handle.remote(args.node_ip)
        ray.kill(owner, no_restart=True)

        async def received():
            return marker.with_suffix(".received.json").exists()

        await until(received, bool, timeout=10)
        evidence["pending_after_delivery"] = [
            item.lease_id for item in await router.pending_reaped.remote(args.node_ip)
        ]
        assert evidence["pending_after_delivery"] == [lease.lease_id]
        blocked = await router.try_admit.remote(f"next-{unit}", demand, args.node_ip)
        evidence["admission_before_cleanup"] = asdict(blocked)
        assert blocked.status == "wait"
        evidence["containers_before_node_kill"] = docker(
            "ps", "-aq", "--filter", f"label=psrl.env_unit={unit}"
        ).split()
        assert evidence["containers_before_node_kill"]
        ray.kill(old_node, no_restart=True)
        marker.unlink()
        recovered = None
        attempts = []
        # Recovery retains the production three-failure budget and probe deadline.
        for _ in range(3):
            await asyncio.sleep(0.3)
            recovered = await router.try_admit.remote(f"next-{unit}", demand, args.node_ip)
            attempts.append({"admission": asdict(recovered), "diagnostics": await router.diagnostics.remote()})
            if recovered.status == "granted":
                break
        evidence["recovery_admissions"] = attempts
        assert recovered.status == "granted", attempts
        assert not docker("ps", "-aq", "--filter", f"label=psrl.env_lease={lease.lease_id}")
        assert not await router.pending_reaped.remote(args.node_ip)
        new_node = await router.node_handle.remote(args.node_ip)
        assert new_node._actor_id != old_node._actor_id
        evidence["node_replaced"] = True
        await router.release.remote(recovered.lease.lease_id)
        await new_node.cleanup.remote()
        evidence["final_sample"] = await until(
            snapshot, lambda s: s["containers_running"] == 0 and s["leases_active"] == 0
        )
        evidence["passed"] = True
    except BaseException as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        marker.unlink(missing_ok=True)
        ray.kill(owner, no_restart=True)
        for env in envs:
            try:
                await asyncio.wait_for(env.stop(delete=True), timeout=15)
            except Exception as exc:
                evidence.setdefault("cleanup_errors", []).append(repr(exc))
        ids = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={unit}").split()
        if ids:
            docker("rm", "-f", *ids)
        for project in projects:
            networks = docker("network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}").split()
            if networks:
                docker("network", "rm", *networks)
        evidence["remaining_containers"] = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={unit}").split()
        try:
            await asyncio.wait_for(router.shutdown_nodes.remote(), timeout=10)
        except Exception as exc:
            evidence.setdefault("cleanup_errors", []).append(repr(exc))
        ray.kill(router, no_restart=True)
        ray.shutdown()
        evidence["finished_at"] = time.time()
        save(out / "evidence.json", evidence)
        print(
            json.dumps({"passed": evidence["passed"], "remaining_containers": evidence["remaining_containers"]}),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--skip-memory-probe", action="store_true", help="Run only remaining recovery checks.")
    asyncio.run(experiment(parser.parse_args()))
