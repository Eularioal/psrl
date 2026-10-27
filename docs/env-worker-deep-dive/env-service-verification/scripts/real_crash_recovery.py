"""Kill lease owners and a real agent worker while Harbor containers are running."""

import argparse
import asyncio
import dataclasses
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import ray
import yaml
from examples.sciaccel_rl.runner import _apply_env_lease
from harbor.environments.docker.docker import DockerEnvironment
from harbor.job import Job
from harbor.models.job.config import AgentConfig, JobConfig
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import TaskConfig
from harbor.models.trial.paths import TrialPaths
from omegaconf import OmegaConf
from psrl.workers.agent_loop.worker import PSRL_AgentLoopWorker
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=30).strip()


def inspect_ids(ids):
    if not ids:
        return []
    rows = json.loads(docker("inspect", *ids))
    return [dict(id=r["Id"], name=r["Name"], state=r["State"]["Status"], labels=r["Config"]["Labels"]) for r in rows]


async def snapshot(router):
    return [dataclasses.asdict(s) for s in await router.snapshot.remote()]


@ray.remote
class LeaseOwner:
    """Minimal owner isolates TTL cleanup from the separate actor reaper."""

    async def start(self, router, environment, out, token):
        self.client = EnvServiceClient(router, ray.util.get_node_ip_address(), heartbeat_interval_s=1)
        self.handle = await self.client.acquire(f"ttl-{token}", EnvDemand(cpu=1, memory_mb=256))
        composes, kwargs = [], {}
        self.overlay = await _apply_env_lease(self.handle.lease, composes, kwargs, "", "")
        self.env = DockerEnvironment(
            environment_dir=Path(environment),
            environment_name=f"crash-{token}",
            session_id=f"ec-{token}-ttl",
            trial_paths=TrialPaths(trial_dir=Path(out)),
            task_env_config=EnvironmentConfig(cpus=1, memory_mb=256),
            extra_docker_compose=composes,
        )
        await self.env.start(force_build=False)
        return dict(pid=os.getpid(), lease=dataclasses.asdict(self.handle.lease), overlay=str(self.overlay))


class CrashWorker(PSRL_AgentLoopWorker.__ray_metadata__.modified_class):
    """Keep the real worker constructor/reaper and add only a fixture launch method."""

    async def start_crash_fixture(self, router, task, out, token):
        self.fixture_client = EnvServiceClient(router, ray.util.get_node_ip_address(), heartbeat_interval_s=1)
        self.fixture_handle = await self.fixture_client.acquire(f"crash-{token}", EnvDemand(cpu=1, memory_mb=256))
        composes, kwargs = [], {}
        overlay = await _apply_env_lease(self.fixture_handle.lease, composes, kwargs, "", "")
        actor_overlay = Path(out) / "actor-label.yaml"
        actor_overlay.write_text(
            yaml.safe_dump({"services": {"main": {"labels": [f"psrl.actor_id={self._actor_id}"]}}})
        )
        kwargs["extra_docker_compose"] = [*composes, actor_overlay]
        kwargs.setdefault("kwargs", {})["actor_id"] = self._actor_id
        self.fixture_job = await Job.create(
            JobConfig(
                jobs_dir=Path(out) / "jobs",
                job_name=f"crash-{token}",
                tasks=[TaskConfig(path=Path(task))],
                agents=[AgentConfig(name="oracle")],
                environment=kwargs,
                n_attempts=1,
                n_concurrent_trials=1,
                quiet=True,
            )
        )
        self.fixture_task = asyncio.create_task(self.fixture_job.run())
        return dict(
            pid=os.getpid(),
            actor_label=self._actor_id,
            reaper_pid=self._reaper_proc.pid,
            lease=dataclasses.asdict(self.fixture_handle.lease),
            overlay=str(overlay),
        )

    def fixture_status(self):
        if not self.fixture_task.done():
            return "running"
        try:
            result = self.fixture_task.result()
        except BaseException as exc:
            return repr(exc)
        return result.model_dump_json()


def create_task(out, token):
    task = out / f"evcrash-{token}"
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    dockerfile = "FROM python:3.13-slim\nRUN mkdir -p /logs/artifacts /logs/verifier\nWORKDIR /app\n"
    (task / "environment/Dockerfile").write_text(dockerfile)
    (task / "tests/Dockerfile").write_text(dockerfile + "COPY test.sh /tests/test.sh\nRUN chmod +x /tests/test.sh\n")
    (task / "instruction.md").write_text("Write 42 into /logs/artifacts/answer.txt.\n")
    (task / "solution/solve.sh").write_text("#!/bin/bash\nset -eu\necho 42 > /logs/artifacts/answer.txt\n")
    (task / "tests/test.sh").write_text(
        "#!/bin/bash\nset -eu\ntouch /tmp/verifier-ready\nsleep 300\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "task.toml").write_text("""schema_version = "1.4"
[task]
name = "verification/crash-recovery"
[environment]
cpus = 1
memory_mb = 256
build_timeout_sec = 180
network_mode = "allowlist"
allowed_hosts = ["example.com"]
[agent]
timeout_sec = 30
[verifier]
timeout_sec = 360
environment_mode = "separate"
[verifier.environment]
cpus = 1
memory_mb = 256
build_timeout_sec = 180
network_mode = "no-network"
""")
    return task


async def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex[:8]
    evidence = dict(started_at=time.time(), token=token, timeline=[])
    ray.init(
        address="local",
        num_cpus=8,
        num_gpus=0,
        include_dashboard=False,
        _temp_dir=f"/tmp/evcrash-{token}",
        namespace=f"evcrash-{token}",
        log_to_driver=False,
    )
    ip = ray.util.get_node_ip_address()
    evidence["node_ip"] = ip
    router = EnvRouter.remote(
        dict(
            node_policy="any",
            slot_ttl_s=4,
            units=[dict(unit_id=f"crash-{token}", node_ip=ip, cpu_quota=8, memory_mb=4096, max_containers=4)],
        )
    )
    client = EnvServiceClient(router, ip, heartbeat_interval_s=1)
    actors, projects, overlays = [], set(), []
    tq_started = False

    def record_projects(rows):
        projects.update(r["labels"]["com.docker.compose.project"] for r in rows)

    try:
        ttl_dir = out / "ttl-only"
        ttl_dir.mkdir()
        owner = LeaseOwner.remote()
        actors.append(owner)
        owner_info = await owner.start.remote(router, str(args.environment.resolve()), str(ttl_dir), token)
        overlays.append(owner_info["overlay"])
        lease_id = owner_info["lease"]["lease_id"]
        before = inspect_ids(docker("ps", "-aq", "--filter", f"label=psrl.env_lease={lease_id}").split())
        record_projects(before)
        assert before, "Owner must have started a real container."
        ray.kill(owner, no_restart=True)
        await asyncio.sleep(7)
        after_ttl = dict(
            snapshot=await snapshot(router),
            containers=inspect_ids(docker("ps", "-aq", "--filter", f"label=psrl.env_lease={lease_id}").split()),
        )
        assert after_ttl["snapshot"][0]["leases_active"] == 0
        if args.expect_fixed:
            assert not after_ttl["containers"], "Expired containers must be reaped without a new acquire."
        else:
            assert after_ttl["containers"], "This check expects to observe deferred local cleanup."
        # A new admission must also wait for any outstanding node cleanup.
        replacement = await client.acquire(f"replacement-{token}", EnvDemand(cpu=1, memory_mb=256), max_wait_s=3)
        after_next_acquire = inspect_ids(docker("ps", "-aq", "--filter", f"label=psrl.env_lease={lease_id}").split())
        await replacement.release()
        assert not after_next_acquire
        evidence["ttl_only"] = dict(
            owner=owner_info, before=before, after_ttl=after_ttl, after_next_acquire=after_next_acquire
        )
        (out / "live.json").write_text(json.dumps(evidence, indent=2))
        print(f"TTL experiment: containers after TTL={len(after_ttl['containers'])}.", flush=True)

        import transfer_queue as tq

        tq.init(
            OmegaConf.create({"backend": {"SimpleStorage": {"total_storage_size": 16, "num_data_storage_units": 1}}})
        )
        tq_started = True
        config = OmegaConf.create(
            dict(
                gen_actor_rollout_ref=dict(
                    model={"_target_": "verl.workers.config.model.HFModelConfig", "path": args.model},
                    rollout=dict(
                        agent=dict(agent_loop_config_path="examples/sciaccel_rl/config/sciaccel_agent_config.yaml")
                    ),
                ),
                psrl=dict(
                    logging_path=str(out / "worker-logs"),
                    deployment=dict(n_rollout_instances=1, n_validate_instances=0),
                    colocate_validate_and_train=False,
                    rollout_gateway=dict(server_max_concurrency=4),
                ),
                data={},
                trainer=dict(project_name="env-verification", experiment_name=token),
            )
        )
        worker = (
            ray.remote(CrashWorker).options(max_concurrency=4).remote(config, None, "", "", worker_id=0, worker_num=1)
        )
        actors.append(worker)
        task = create_task(out, token)
        info = await worker.start_crash_fixture.remote(router, str(task), str(out), token)
        overlays.append(info["overlay"])
        evidence["real_worker"] = info
        deadline = time.monotonic() + 240
        verifier_id = None
        while time.monotonic() < deadline:
            rows = inspect_ids(docker("ps", "-aq", "--filter", f"name=evcrash-{token}").split())
            record_projects(rows)
            for row in rows:
                labels = row["labels"]
                if "__verifier" in row["name"] and labels.get("com.docker.compose.service") == "main":
                    check = subprocess.run(
                        ["docker", "exec", row["id"], "test", "-f", "/tmp/verifier-ready"],
                        capture_output=True,
                        timeout=10,
                    )
                    if check.returncode == 0:
                        verifier_id = row["id"]
            if verifier_id:
                break
            status = await worker.fixture_status.remote()
            assert status == "running", status
            await asyncio.sleep(1)
        assert verifier_id, "Separate verifier must be running before the crash."
        evidence["before_worker_kill"] = rows
        print(f"Real worker is alive with {len(rows)} Harbor containers. Killing actor now.", flush=True)
        killed_at = time.monotonic()
        ray.kill(worker, no_restart=True)
        for _ in range(15):
            rows = inspect_ids(docker("ps", "-aq", "--filter", f"name=evcrash-{token}").split())
            evidence["timeline"].append(
                dict(elapsed_s=time.monotonic() - killed_at, containers=rows, snapshot=await snapshot(router))
            )
            await asyncio.sleep(1)
        evidence["after_worker_reaper"] = rows
        evidence["after_lease_cleanup"] = inspect_ids(docker("ps", "-aq", "--filter", f"name=evcrash-{token}").split())
        evidence["findings"] = dict(
            tagged_containers_present_at_kill=sum(
                r["labels"].get("psrl.actor_id") == info["actor_label"] for r in evidence["before_worker_kill"]
            ),
            tagged_containers_remaining=any(r["labels"].get("psrl.actor_id") == info["actor_label"] for r in rows),
            unlabelled_containers_survive=bool(evidence["after_lease_cleanup"]),
            lease_released=(await snapshot(router))[0]["leases_active"] == 0,
        )
        assert not evidence["findings"]["tagged_containers_remaining"]
        assert evidence["findings"]["lease_released"]
        if args.expect_fixed:
            assert not evidence["after_lease_cleanup"], "Worker crash left Harbor containers behind."
            assert all(
                "psrl.env_lease" in row["labels"] and "psrl.actor_id" in row["labels"]
                for row in evidence["before_worker_kill"]
            )
        print(json.dumps(evidence["findings"]), flush=True)
    except BaseException as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        # Evidence above is captured before manual cleanup. Restrict cleanup to the
        # Compose projects created by this unique experiment token.
        for actor in actors:
            ray.kill(actor, no_restart=True)
        for project in projects:
            ids = docker("ps", "-aq", "--filter", f"label=com.docker.compose.project={project}").split()
            if ids:
                docker("rm", "-f", *ids)
            networks = docker("network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}").split()
            if networks:
                docker("network", "rm", *networks)
        evidence["after_manual_cleanup"] = docker("ps", "-aq", "--filter", f"name=evcrash-{token}").split()
        evidence["final_snapshot"] = await snapshot(router)
        for path in overlays:
            Path(path).unlink(missing_ok=True)
        if tq_started:
            tq.close()
        ray.kill(router)
        ray.shutdown()
        evidence["finished_at"] = time.time()
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--environment", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expect-fixed", action="store_true")
    asyncio.run(main(p.parse_args()))
