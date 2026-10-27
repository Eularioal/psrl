"""Exercise production worker RPC supervision with real Ray actors and Harbor containers.

The episode body is a tiny CPU fixture. The model, data loader and PS storage are
not started. A receipt-discarding control reproduces the unobserved failure mechanism.
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
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from omegaconf import OmegaConf
from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager
from psrl.workers.agent_loop.worker import PSRL_AgentLoopWorker
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.container_budget import ContainerBudget
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.overlay import build_lease_overlay
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

WorkerBase = PSRL_AgentLoopWorker.__ray_metadata__.modified_class


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=10).strip()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sources():
    return {name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "harbor")}


class FixtureWorker(WorkerBase):
    def __init__(self, router, ip, out, name, image):
        self.router, self.ip, self.out, self.name, self.image = router, ip, Path(out), name, image
        self.agent_programs = set()
        self._programs_closing = False
        self.config = OmegaConf.create(
            {"gen_actor_rollout_ref": {"rollout": {"agent": {"default_agent_loop": "fixture"}}}}
        )

    async def _run_agent_loop(self, agent_name, batch):
        client = EnvServiceClient(self.router, self.ip, heartbeat_interval_s=0.3)
        handle = await client.acquire(self.name, EnvDemand(cpu=0.5, memory_mb=128), max_wait_s=30)
        trial = self.out / self.name
        trial.mkdir()
        environment = trial / "environment"
        environment.mkdir()
        env = ManagedDockerEnvironment(
            environment_dir=environment,
            environment_name=self.name,
            session_id=self.name,
            trial_paths=TrialPaths(trial_dir=trial),
            task_env_config=EnvironmentConfig(docker_image=self.image, cpus=1, memory_mb=128),
            lease_overlay=build_lease_overlay(handle.lease, None),
            container_budget=asdict(ContainerBudget(0.5, 128, 0)),
            actor_id=self.name,
        )
        try:
            await env.start(force_build=False)
            result = await env.exec("python3 -c 'print(6 * 7)'", timeout_sec=10)
            assert result.return_code == 0 and result.stdout.strip() == "42", "Real container execution failed."
            save(trial / "ready.json", {"lease": asdict(handle.lease), "stdout": result.stdout, "sources": sources()})
            if not self.name.endswith("success"):
                await asyncio.Event().wait()
        finally:
            await env.stop(delete=True)
            await handle.release()


@ray.remote(num_cpus=0)
class BufferRegistry:
    def ensure_train_buffer_exists(self, buffer_id):
        return True


class FixtureManager(PSRL_AgentLoopManager):
    def __init__(self, worker, registry):
        self.agent_loop_workers = [worker]
        self.ps_manager_handle = registry
        self._refill_breaker_diagnosis = None
        self._train_buffer_waiters = {}
        self._train_chunk_waiters = {}
        self._val_buffer_waiters = {}
        self._resolved_train_chunks = {}
        self.train_data_buffers = {}
        self.train_accumulated_buffers = {}
        self.task = None

    def launch(self, observe):
        batch = TensorDict({}, batch_size=[1])
        tu.assign_non_tensor_stack(batch, "uid", [0])
        if observe:
            self.task = asyncio.create_task(self._supervise_worker_batch(0, batch))
        else:
            self.unobserved = self.agent_loop_workers[0].add_agent_program.remote(batch)

    def status(self):
        return {
            "diagnosis": self._refill_breaker_diagnosis,
            "done": bool(self.task and self.task.done()),
            "waiters": sum(map(len, self._train_buffer_waiters.values())),
            "sources": sources(),
        }


async def until(read, accept, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = await read()
        if accept(value):
            return value
        await asyncio.sleep(0.2)
    raise TimeoutError(f"Observation deadline exceeded. Last value: {value!r}")


async def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    unit = f"worker-check-{token}"
    evidence = {"passed": False, "started_at": time.time(), "cases": {}, "sources": sources()}
    repo = Path(__file__).resolve().parents[4]
    evidence["source_sha256"] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (repo / "psrl/workers/agent_loop").glob("*.py")
    }
    save(out / "ownership.json", {"unit": unit, "ray_temp": f"/tmp/ws-{token}"})
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
        _node_ip_address=args.node_ip,
        _temp_dir=f"/tmp/ws-{token}",
        namespace=unit,
        object_store_memory=128 * 1024**2,
        _memory=512 * 1024**2,
    )
    assert [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]] == [args.node_ip], "Wrong Ray node identity."
    router = EnvRouter.remote(
        {
            "node_policy": "any",
            "slot_ttl_s": 3,
            "telemetry": {"interval_s": 0.5, "timeout_s": 6},
            "units": [
                {"unit_id": unit, "node_ip": args.node_ip, "cpu_quota": 1, "memory_mb": 256, "max_containers": 1}
            ],
        }
    )
    registry = BufferRegistry.remote()
    actors, projects = [], set()
    try:
        for case in ("success", "unobserved-death", "supervised-death"):
            name = f"{unit}-{case}"
            worker = ray.remote(num_cpus=0)(FixtureWorker).remote(router, args.node_ip, str(out), name, args.image)
            manager = ray.remote(num_cpus=0)(FixtureManager).remote(worker, registry)
            actors.extend((worker, manager))
            await manager.launch.remote(case != "unobserved-death")

            async def ready(name=name):
                return (out / name / "ready.json").exists()

            await until(ready, bool)
            record = {"worker": json.loads((out / name / "ready.json").read_text())}
            if case == "success":
                record["manager"] = await until(lambda manager=manager: manager.status.remote(), lambda s: s["done"])
                assert record["manager"]["diagnosis"] is None, "Successful execution tripped the breaker."
            else:
                ids = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={unit}").split()
                assert len(ids) == 1, "Death injection needs one real live container."
                rows = json.loads(docker("inspect", *ids))
                projects.update(row["Config"]["Labels"]["com.docker.compose.project"] for row in rows)
                save(out / "owned-projects.json", sorted(projects))
                waiter = manager.wait_for_training_batch.remote(0)
                waiter_task = asyncio.ensure_future(waiter)
                await until(lambda manager=manager: manager.status.remote(), lambda s: s["waiters"] == 1)
                killed_at = time.monotonic()
                ray.kill(worker, no_restart=True)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(waiter_task), timeout=2 if case == "unobserved-death" else 10
                    )
                    raise AssertionError("No training batch exists in this fixture.")
                except TimeoutError:
                    assert case == "unobserved-death", "Supervised death left the trainer waiting."
                    record["wait_result"] = "timeout"
                    ray.cancel(waiter)
                    waiter_task.cancel()
                    await asyncio.gather(waiter_task, return_exceptions=True)
                except ray.exceptions.RayTaskError as exc:
                    assert case == "supervised-death" and "ActorDiedError" in str(exc), str(exc)
                    record["wait_result"] = "worker_failure"
                    record["failure"] = str(exc)
                record["notification_elapsed_s"] = time.monotonic() - killed_at
                record["manager"] = await manager.status.remote()
                if case == "supervised-death":
                    try:
                        await manager.wait_for_training_batch.remote(1)
                        raise AssertionError("A late waiter missed the failure latch.")
                    except ray.exceptions.RayTaskError as exc:
                        assert "ActorDiedError" in str(exc), str(exc)
                        record["late_waiter_failed"] = True

            async def resources():
                return asdict((await router.snapshot.remote())[0])

            record["released"] = await until(resources, lambda s: s["leases_active"] == s["containers_running"] == 0)
            assert not docker("ps", "-aq", "--filter", f"label=psrl.env_unit={unit}"), "Owned container leaked."
            ray.kill(worker, no_restart=True)
            ray.kill(manager, no_restart=True)
            evidence["cases"][case] = record
            save(out / "live.json", evidence)
        evidence["passed"] = True
    except BaseException as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        ids = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={unit}").split()
        if ids:
            docker("rm", "-f", *ids)
        for project in projects:
            ids = docker("network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}").split()
            if ids:
                docker("network", "rm", *ids)
        try:
            await asyncio.wait_for(router.shutdown_nodes.remote(), timeout=15)
        except Exception as exc:
            evidence["shutdown_error"] = repr(exc)
        ray.kill(router, no_restart=True)
        ray.kill(registry, no_restart=True)
        ray.shutdown()
        evidence["finished_at"] = time.time()
        evidence["remaining_containers"] = docker("ps", "-aq", "--filter", f"label=psrl.env_unit={unit}").split()
        save(out / "evidence.json", evidence)
        print(
            json.dumps({"passed": evidence["passed"], "remaining_containers": evidence["remaining_containers"]}),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--image", required=True)
    asyncio.run(main(parser.parse_args()))
