"""Check real Ray router reuse and isolated node probe faults without GPUs."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import ray
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.router import RayEnvRouter, create_env_router


class ProbeNode(EnvNodeService):
    """Inject probe failures only into the node actor owned by this check."""

    fault = ""

    async def set_fault(self, fault):
        self.fault = fault

    async def prepare(self, indices):
        if self.fault == "unavailable":
            raise RuntimeError("Injected node probe failure.")
        if self.fault == "timeout":
            await asyncio.sleep(5)
        return await super().prepare(indices)

    async def cleanup(self):
        pending = await super().cleanup()
        return max(pending, int(self.fault == "pending"))


class ProbeRouter(RayEnvRouter):
    """Keep production recovery logic and replace only the isolated probe endpoint."""

    def _create_node(self, node_ip):
        return ray.remote(num_cpus=0)(ProbeNode).remote(ray.get_runtime_context().current_actor, node_ip)

    async def set_fault(self, node_ip, fault):
        service, _ = await self._node_task(node_ip)
        await service.set_fault.remote(fault)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tag = "router-check-" + uuid.uuid4().hex[:8]
    handles = []
    result = {"passed": False, "scope": "Real Ray and Docker probes with isolated synthetic node faults"}
    try:
        ray.init(
            address="local",
            namespace=tag,
            _node_ip_address=args.node_ip,
            _temp_dir=f"/tmp/{tag}",
            num_cpus=2,
            num_gpus=0,
            object_store_memory=80 * 1024 * 1024,
            include_dashboard=False,
            log_to_driver=False,
        )
        assert [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]] == [args.node_ip]
        config = {"router_name": tag, "prewarm": {"enable": False}}
        first = create_env_router(config)
        handles.append(first)
        second = create_env_router(config)
        assert first._actor_id == second._actor_id, "Same configuration must reuse the actor."
        try:
            create_env_router({**config, "default_gpu_memory_mb": 1})
        except ValueError:
            result["R1_changed_configuration_rejected"] = True
        else:
            raise AssertionError("Changed configuration was accepted.")
        result["R1_same_actor_reused"] = True
        config = {
            "node_policy": "any",
            "prewarm": {"enable": False},
            "health": {"timeout_s": 1, "max_failures": 1},
            "units": [
                {"unit_id": tag, "node_ip": args.node_ip, "cpu_quota": 1, "memory_mb": 256, "max_containers": 1}
            ],
        }
        router = ray.remote(num_cpus=0)(ProbeRouter).remote(config)
        handles.append(router)
        ray.get(router.wait_prewarm.remote(), timeout=30)
        initial = ray.get(router.recover_node.remote(args.node_ip), timeout=15)
        assert initial["healthy"], "Real Docker probe must initially pass."
        states = {}
        for fault in ("unavailable", "", "timeout", "", "pending", ""):
            ray.get(router.set_fault.remote(args.node_ip, fault), timeout=10)
            state = ray.get(router.recover_node.remote(args.node_ip), timeout=15)
            assert state["healthy"] == (not fault), "Recovery must preserve the probe outcome."
            if fault in ("unavailable", "timeout"):
                assert state["retry_exhausted"], "Fault must exhaust the configured retry budget."
            if fault == "pending":
                assert state["pending_cleanup"], "Unconfirmed cleanup must remain visible."
            states[fault or "recovered"] = state
        result["R2_states"] = states
        result["passed"] = True
    finally:
        for handle in reversed(handles):
            try:
                ray.get(handle.shutdown_nodes.remote(), timeout=20)
            finally:
                ray.kill(handle, no_restart=True)
        ray.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
