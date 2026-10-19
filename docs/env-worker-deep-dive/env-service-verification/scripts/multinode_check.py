"""Check real node placement, remote telemetry and Gate B on an isolated two-node Ray cluster."""

import argparse
import ctypes
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote
class Probe:
    def where(self):
        return dict(
            ip=ray.util.get_node_ip_address(), node_id=ray.get_runtime_context().get_node_id(), pid=os.getpid()
        )

    def hold(self):
        from psrl.workers.env_service.overlay import read_kernel_driver_version

        version = read_kernel_driver_version()
        self.cuda = ctypes.CDLL(f"/usr/lib64/libcuda.so.{version}")
        assert self.cuda.cuInit(0) == 0
        self.ctx = ctypes.c_void_p()
        assert self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, 0) == 0
        self.ptr = ctypes.c_uint64()
        assert self.cuda.cuMemAlloc_v2(ctypes.byref(self.ptr), ctypes.c_size_t(512 * 1024 * 1024)) == 0
        return self.where()

    def free(self):
        self.cuda.cuMemFree_v2(self.ptr)
        self.cuda.cuCtxDestroy_v2(self.ctx)

    async def telemetry(self):
        from psrl.workers.env_service.probe import read_gpu_telemetry

        return await read_gpu_telemetry()

    async def admission(self, router):
        from psrl.workers.env_service.client import EnvAdmissionTimeout, EnvServiceClient
        from psrl.workers.env_service.types import EnvDemand

        client = EnvServiceClient(router, ray.util.get_node_ip_address(), poll_interval_s=0.2, max_poll_interval_s=0.3)
        try:
            handle = await client.acquire(uuid.uuid4().hex, EnvDemand(needs_gpu=True, gpu_memory_mb=512), max_wait_s=2)
        except EnvAdmissionTimeout:
            return dict(status="timeout", telemetry=await self.telemetry())
        result = dict(status="granted", lease=dataclasses.asdict(handle.lease), telemetry=await self.telemetry())
        await handle.release()
        return result

    async def node_url(self, router):
        from psrl.workers.env_service.client import EnvServiceClient
        from psrl.workers.env_service.types import EnvDemand

        client = EnvServiceClient(router, ray.util.get_node_ip_address(), heartbeat_interval_s=5)
        handle = await client.acquire(uuid.uuid4().hex, EnvDemand(cpu=1), max_wait_s=10)
        url = handle.lease.node_service_url
        await handle.release()
        return url

    async def gate(self, node_service_url=""):
        import psrl.workers.gen.session_router as sr

        sr._GATE_B_MAX_WAIT_SEC = 2
        sr._GATE_B_POLL_INTERVAL_SEC = 0.2
        start = time.monotonic()
        await sr._wait_for_gpu_idle(
            "/dev/nvidia0", "remote-gpu-session", shared=False, node_service_url=node_service_url, gpu_indices=(0,)
        )
        return dict(where=self.where(), elapsed_s=time.monotonic() - start, telemetry=await self.telemetry())


def join(args):
    from ray._private.node import Node
    from ray._private.parameter import RayParams

    node = Node(
        RayParams(
            gcs_address=args.join,
            node_ip_address=args.peer,
            num_cpus=8,
            num_gpus=0,
            object_store_memory=256 * 1024 * 1024,
            temp_dir=args.temp,
        ),
        head=False,
        shutdown_at_exit=True,
    )
    try:
        deadline = time.monotonic() + 900
        while not (args.output / "stop").exists() and time.monotonic() < deadline:
            time.sleep(1)
    finally:
        node.kill_all_processes(check_alive=False, allow_graceful=True)


def head(args):
    from omegaconf import OmegaConf
    from psrl.workers.env_service.node_policy import NodeRules, restrict_to_unit_nodes, select_worker_nodes
    from psrl.workers.env_service.router import EnvRouter

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    temp = f"/tmp/evm-{token}"
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    info = ray.init(
        address="local",
        num_cpus=8,
        num_gpus=0,
        object_store_memory=256 * 1024 * 1024,
        include_dashboard=False,
        _temp_dir=temp,
        namespace=f"evm-{token}",
        log_to_driver=False,
    )
    head_ip = ray.util.get_node_ip_address()
    command = f"source {shlex.quote(args.activation)} >/tmp/evm-activation.log 2>&1\n"
    command += "export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1\n"
    command += f"cd {shlex.quote(os.getcwd())}\nPYTHONPATH=. {shlex.join([sys.executable, str(Path(__file__).resolve()), '--join', info.address_info['gcs_address'], '--peer', args.peer, '--temp', temp, '--output', str(out)])}"
    peer_log = (out / "peer.log").open("w")
    peer = subprocess.Popen(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={out / 'known-hosts'}",
            args.peer,
            command,
        ],
        stdout=peer_log,
        stderr=subprocess.STDOUT,
    )
    evidence = dict(head_ip=head_ip, peer_ip=args.peer)
    actors = []
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            nodes = {n["NodeManagerAddress"]: n["NodeID"] for n in ray.nodes() if n["Alive"]}
            if args.peer in nodes:
                break
            if peer.poll() is not None:
                raise RuntimeError(f"Peer exited {peer.returncode}")
            time.sleep(1)
        assert args.peer in nodes, "Peer did not join the isolated cluster."
        evidence["nodes"] = nodes
        for ip in (head_ip, args.peer):
            actors.append(
                Probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip], soft=False)).remote()
            )
        evidence["placement"] = ray.get([p.where.remote() for p in actors], timeout=120)
        evidence["before"] = ray.get([p.telemetry.remote() for p in actors])
        assert all(t[0][1] < 100 for t in evidence["before"]), "GPU 0 must be idle before this experiment."
        ray.get(actors[1].hold.remote())
        units = [
            dict(
                unit_id=f"node{i}",
                node_ip=ip,
                cpu_quota=4,
                memory_mb=4096,
                gpu_indices=[0],
                gpu_memory_mb=768,
                max_containers=2,
            )
            for i, ip in enumerate((head_ip, args.peer))
        ]
        router = EnvRouter.remote(dict(units=units, node_policy="any", slot_ttl_s=60))
        actors.append(router)
        evidence["admission"] = ray.get([p.admission.remote(router) for p in actors[:2]], timeout=30)
        urls = ray.get([p.node_url.remote(router) for p in actors[:2]], timeout=60)
        evidence["node_service_urls"] = urls
        evidence["gate_b"] = ray.get(
            [p.gate.remote(urls[1] if args.expect_fixed else "") for p in actors[:2]], timeout=120
        )
        if args.expect_fixed:
            assert all(row["elapsed_s"] >= 2 for row in evidence["gate_b"])
            evidence["remote_gate_b_passed"] = True
        assert evidence["admission"][0]["status"] == "granted" and evidence["admission"][1]["status"] == "timeout"
        evidence["gate_b_cross_node_gap_reproduced"] = (
            evidence["gate_b"][0]["elapsed_s"] < 1 and evidence["gate_b"][1]["elapsed_s"] >= 2
        )
        ray.get(actors[1].free.remote())
        evidence["after"] = ray.get([p.telemetry.remote() for p in actors[:2]])
        rules = NodeRules.build("avoid", train_rollout_node_ips=[head_ip])
        kept, _ = select_worker_nodes([(nid, ip) for ip, nid in nodes.items()], rules)
        kept, _ = restrict_to_unit_nodes(kept, [args.peer])
        evidence["selected_worker_nodes"] = kept
        (out / "telemetry.json").write_text(json.dumps(evidence, indent=2))

        if args.skip_workers:
            return

        # Initialize real workers and their real tokenizer, HTTP client and TransferQueue.
        import transfer_queue as tq
        from psrl.workers.agent_loop.worker import PSRL_AgentLoopWorker

        tq.init(
            OmegaConf.create({"backend": {"SimpleStorage": {"total_storage_size": 16, "num_data_storage_units": 1}}})
        )
        cfg = OmegaConf.create(
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
        evidence["real_workers"] = []
        for i in range(2):
            nid, ip = kept[i % len(kept)]
            worker = PSRL_AgentLoopWorker.options(
                name=f"verify-worker-{i}",
                max_concurrency=2,
                scheduling_strategy=NodeAffinitySchedulingStrategy(nid, soft=False),
            ).remote(cfg, None, "", "", worker_id=i, worker_num=2)
            actors.append(worker)
            ray.get(worker.set_agent_loop_manager.remote(None), timeout=180)
            actor_id = worker._actor_id.hex()
            state = ray._private.state.actors(actor_id)
            evidence["real_workers"].append(
                dict(
                    actor_id=actor_id,
                    expected_node=nid,
                    actual_node=state["Address"]["NodeID"],
                    ip=ip,
                    state=state["State"],
                )
            )
        assert all(w["actual_node"] == w["expected_node"] and w["state"] == "ALIVE" for w in evidence["real_workers"])
        tq.close()
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for actor in actors:
            ray.kill(actor)
        (out / "stop").touch()
        try:
            peer.wait(timeout=30)
        except subprocess.TimeoutExpired:
            peer.terminate()
        ray.shutdown()
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--peer", required=True)
    p.add_argument("--activation")
    p.add_argument("--model")
    p.add_argument("--join")
    p.add_argument("--temp")
    p.add_argument("--expect-fixed", action="store_true")
    p.add_argument("--skip-workers", action="store_true")
    a = p.parse_args()
    join(a) if a.join else head(a)
