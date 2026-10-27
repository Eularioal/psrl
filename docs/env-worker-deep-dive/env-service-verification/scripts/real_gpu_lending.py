"""Verify two bounded Harbor CUDA and vLLM handoffs on one real GPU.

The inference-only server uses production handoff and engine control methods.
It loads checkpoint weights directly and omits the training NIXL transport.
"""

import argparse
import asyncio
import dataclasses
import importlib
import json
import os
import subprocess
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

import ray
from examples.sciaccel_rl.runner import _apply_env_lease
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from omegaconf import OmegaConf
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.lending import GpuPhaseClient
from psrl.workers.env_service.lending_node import GpuLendingNodeMixin
from psrl.workers.env_service.router import EnvRouter
from psrl.workers.env_service.types import EnvDemand
from psrl.workers.gen.gpu_handoff import GpuControlGate, gpu_control
from psrl.workers.gen.vllm_async_server import PSRL_vLLMHttpServer


class InferenceOnlyServer(PSRL_vLLMHttpServer):
    """Use the production GPU protocol without starting training or a parameter server."""

    def __init__(self, model):
        from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.config = OmegaConf.create({"enable_sleep_mode": True})
        self.psrl_config = OmegaConf.create({"tms": {"range": "none", "enable_cuda_graph": False}})
        self.gen_interface = SimpleNamespace(rollout_replica_idx=0)
        self._gpu_control = GpuControlGate()
        self._gpu_sync_active = False
        self._gpu_weights_valid = True
        self._last_env_owner = None
        self._env_handoff_ready = False
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model,
                tensor_parallel_size=1,
                dtype="bfloat16",
                enable_sleep_mode=True,
                enforce_eager=True,
                max_model_len=512,
                max_num_seqs=1,
                gpu_memory_utilization=0.15,
                kv_cache_memory_bytes=512 * 1024 * 1024,
                enable_prefix_caching=False,
                disable_log_stats=True,
            )
        )
        self.grpc_servicer = VllmEngineServicer(self.engine, time.time())

    @gpu_control
    async def nixl_sleep(self, level):
        # No parameter transfer is started in this inference-only experiment.
        await self.sleep(level)

    @gpu_control
    async def nixl_wake_up(self):
        await self.wake_up()

    async def ready(self):
        return {
            "gpu_ids": ray.get_gpu_ids(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "node_ip": ray.util.get_node_ip_address(),
            "sources": {
                name: importlib.import_module(name).__file__ for name in ("psrl", "verl", "transfer_queue", "vllm")
            },
        }

    async def query(self):
        from vllm import SamplingParams

        assert self._gpu_control.borrower is None, "Cannot query while Docker owns the GPU."
        result = None
        async for result in self.engine.generate(
            "Question: What is 6 times 7? Answer:",
            SamplingParams(temperature=0, max_tokens=16),
            request_id=uuid.uuid4().hex,
        ):
            pass
        assert result is not None and result.finished, "Generation did not finish."
        return {"text": result.outputs[0].text, "token_ids": list(result.outputs[0].token_ids)}

    async def stop(self):
        self.engine.shutdown(timeout=10)


async def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    unit_id = f"lending-{token}"
    router_name = f"router-{token}"
    evidence = {
        "status": "running",
        "unit_id": unit_id,
        "budget": {"gpus": 1, "docker_rounds": 2, "model_max_tokens": 16, "work_timeout_s": 480},
        "boundary": "Real Ray, EnvRouter, leases, node checks, Harbor Docker and vLLM. No training, NIXL or SMG.",
        "events": [],
    }
    observer = GpuLendingNodeMixin()
    server = router = handle = env = overlay = None
    started = time.monotonic()

    def save():
        evidence["elapsed_s"] = round(time.monotonic() - started, 3)
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str) + "\n")

    async def record(stage):
        item = {"stage": stage, "elapsed_s": round(time.monotonic() - started, 3)}
        item["gpu"] = await observer.gpu_lending_snapshot([0])
        item["ray_available_gpu"] = ray.available_resources().get("GPU", 0)
        evidence["events"].append(item)
        save()
        print(json.dumps(item), flush=True)

    async def workflow():
        nonlocal server, router, handle, env, overlay
        server = ray.remote(num_cpus=2, num_gpus=1, max_concurrency=16)(InferenceOnlyServer).remote(str(args.model))
        evidence["worker"] = await server.ready.remote()
        assert evidence["worker"]["node_ip"] == args.node_ip, "Worker reached the wrong node."
        evidence["baseline_generation"] = await server.query.remote()
        await record("model_initial")
        cfg = {
            "node_policy": "cpu_only",
            "default_gpu_memory_mb": 512,
            "slot_ttl_s": 120,
            "heartbeat_interval_s": 5,
            "units": [
                {
                    "unit_id": unit_id,
                    "node_ip": args.node_ip,
                    "gpu_indices": [0],
                    "cpu_quota": 2,
                    "memory_mb": 1024,
                    "gpu_memory_mb": 1024,
                    "max_containers": 1,
                    "rollout_replica_rank": 0,
                }
            ],
            "gpu_lending": {
                "transition_timeout_s": 45,
                "max_env_phase_s": 90,
                "return_tolerance_mb": 16,
                "safety_margin_mb": 1024,
            },
        }
        evidence["config"] = cfg
        router = EnvRouter.options(name=router_name).remote(cfg, [args.node_ip])
        await router.bind_rollout_gpu.remote(unit_id, server, None)
        client = EnvServiceClient(
            router, args.node_ip, default_gpu_memory_mb=512, heartbeat_interval_s=5, router_name=router_name
        )
        demand = EnvDemand(cpu=1, memory_mb=512, needs_gpu=True, gpu_memory_mb=512)
        handle = await client.acquire(f"episode-{token}", demand, max_wait_s=10)
        evidence["lease"] = dataclasses.asdict(handle.lease)
        await router.set_gpu_phase.remote(handle.lease.lease_id, "env", 0)
        phases = GpuPhaseClient(router, handle.lease.lease_id)
        await record("env_initial")

        composes, kwargs = [], {}
        overlay = await _apply_env_lease(handle.lease, composes, kwargs, "", "")
        trial_dir = out / "trial"
        trial_dir.mkdir(exist_ok=True)
        env = ManagedDockerEnvironment(
            environment_dir=args.environment.resolve(),
            environment_name=unit_id,
            session_id=f"lending-{token}",
            trial_paths=TrialPaths(trial_dir=trial_dir),
            task_env_config=EnvironmentConfig(cpus=1, memory_mb=512, gpus=1),
            override_gpus=0,
            extra_docker_compose=composes,
            **kwargs["kwargs"],
            gpu_lending={"router_name": router_name, "lease_id": handle.lease.lease_id},
        )
        await env.start(force_build=False)
        await env.upload_file(Path(__file__).with_name("cuda_driver_probe.py"), "/tmp/probe.py")
        evidence["second_admission"] = dataclasses.asdict(
            await router.try_admit.remote("second", demand, args.node_ip)
        )
        assert evidence["second_admission"]["status"] == "wait", "A second borrower must wait."
        evidence["rounds"] = []
        for index in range(2):
            command = asyncio.create_task(
                env.exec(
                    "PROBE_HOLD_SECONDS=4 python /tmp/probe.py",
                    timeout_sec=20,
                )
            )
            await asyncio.sleep(2)
            await record(f"docker_cuda_{index}")
            result = await command
            assert result.return_code == 0, result.stderr
            probe = json.loads(result.stdout.strip())
            expected_uuid = evidence["events"][0]["gpu"][0]["uuid"].removeprefix("GPU-").replace("-", "")
            assert probe["uuid_hex"] == expected_uuid, "Container and vLLM must use the same physical GPU."
            await phases.set_phase("model")
            await record(f"model_{index}")
            containers = await observer._lease_containers(handle.lease.lease_id)
            assert containers and all(c["State"]["Paused"] for c in containers), "Model phase must freeze containers."
            generation = await server.query.remote()
            assert generation["token_ids"] == evidence["baseline_generation"]["token_ids"], "Weights changed on wake."
            evidence["rounds"].append({"cuda": probe, "generation": generation, "paused": True})
            await phases.set_phase("env")
            await record(f"env_{index}")
            check = await env.exec("test -f /logs/artifacts/cuda-probe.json && echo file-preserved", timeout_sec=10)
            assert check.return_code == 0 and "file-preserved" in check.stdout, "Container state was lost."
        await handle.release()
        evidence["snapshot_after_release"] = [dataclasses.asdict(x) for x in await router.snapshot.remote()]
        assert all(x["leases_active"] == 0 for x in evidence["snapshot_after_release"]), "Lease leaked."
        await record("released_model_awake")
        evidence["status"] = "passed"

    try:
        initial = await observer.gpu_lending_snapshot([0])
        assert not initial[0]["pids"] and initial[0]["used_mb"] < 100, "Verification requires an idle GPU."
        evidence["initial_gpu"] = initial
        evidence["identity_xml"] = await asyncio.to_thread(
            subprocess.check_output,
            ["nvidia-smi", "-q", "-x"],
            text=True,
            timeout=10,
        )
        ray.init(
            address="local",
            num_cpus=6,
            num_gpus=1,
            include_dashboard=False,
            _node_ip_address=args.node_ip,
            _temp_dir=f"/tmp/gl-{token}",
            namespace=f"gl-{token}",
            log_to_driver=True,
        )
        evidence["ray_nodes"] = [{"id": n["NodeID"], "ip": n["NodeManagerAddress"]} for n in ray.nodes() if n["Alive"]]
        save()
        await asyncio.wait_for(workflow(), timeout=480)
    except BaseException:
        evidence["status"] = "failed"
        evidence["error"] = traceback.format_exc()
        raise
    finally:
        cleanup = []
        for name, action in (
            ("environment", (lambda: env.stop(delete=True)) if env else None),
            ("lease", handle.release if handle else None),
            ("engine", (lambda: server.stop.remote()) if server else None),
        ):
            if action:
                try:
                    await asyncio.wait_for(action(), 30)
                    cleanup.append({"target": name, "status": "ok"})
                except BaseException as exc:  # noqa: BLE001
                    cleanup.append({"target": name, "error": repr(exc)})
        for actor in (server, router):
            if actor:
                ray.kill(actor)
        if ray.is_initialized():
            ray.shutdown()
        if overlay:
            overlay.unlink(missing_ok=True)
        await asyncio.sleep(2)
        evidence["cleanup"] = cleanup
        evidence["final_gpu"] = await observer.gpu_lending_snapshot([0])
        evidence["final_containers"] = (
            await asyncio.to_thread(
                subprocess.check_output,
                ["docker", "ps", "-aq", "--filter", f"label=psrl.env_unit={unit_id}"],
                text=True,
                timeout=10,
            )
        ).split()
        save()
        print(json.dumps({"status": evidence["status"], "cleanup": cleanup, "output": str(out)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    asyncio.run(run(parser.parse_args()))
