"""Verify the same managed tool protocol using real CIFAR training in both modes."""

import asyncio
import dataclasses
import json
import shlex
import time

from examples.sciaccel_rl.gpu_tools import dispatch_gpu_tool
from harbor.agents.nop import NopAgent


class UnifiedGpuAgent(NopAgent):
    async def run(self, instruction, environment, context):
        import ray

        spec = json.loads(instruction)
        router = ray.get_actor(spec["router_name"])
        record = {"mode": spec["mode"], "jobs": [], "started_at": time.time()}
        destination = self.logs_dir / "unified-gpu.json"

        async def tool(**request):
            return await dispatch_gpu_tool(environment, "psrl-gpu " + json.dumps(request))

        try:
            for index in range(spec["count"]):
                command = shlex.join([spec["python"], "/opt/workload/train.py", "--index", str(index)])
                start = await tool(op="start", command=command, timeout_sec=180, wait_timeout_sec=120)
                job_id = start["job_id"]
                if index == 0:
                    # Cancel a queued real training invocation without disturbing its owner.
                    await asyncio.sleep(0.05)
                    queued = await tool(op="start", command=command, timeout_sec=180, wait_timeout_sec=120)
                    cancelled = await tool(op="cancel", job_id=queued["job_id"])
                    assert cancelled["state"] == "cancelled"
                    record["queued_cancel"] = cancelled
                    observed = await tool(op="wait", job_id=job_id, timeout_sec=0.01)
                    record["short_observation"] = observed
                result = await tool(op="wait", job_id=job_id, timeout_sec=240, released=True)
                record["jobs"].append(result)
                assert result["state"] == "completed" and result["exit_code"] == 0, result
                reservation = result["reservation"]
                assert reservation["reservation_mode"] == spec["mode"]
                assert reservation["capacity_released"] == (spec["mode"] == "command")
                units = [dataclasses.asdict(unit) for unit in await router.snapshot.remote()]
                record.setdefault("after_job_units", []).append(units)
                assert sum(unit["leases_active"] for unit in units) == 1
                declared = sum(unit["declared_gpu_memory_mb"] for unit in units)
                assert (declared > 0) == (spec["mode"] == "episode")
                if index == 0:
                    record["telemetry"] = await tool(op="telemetry", job_id=job_id)
            await environment.finish_gpu_jobs()
            context.metadata = {"scope": "Real CIFAR updates via unified GPU jobs, no model rollout"}
        finally:
            record["finished_at"] = time.time()
            destination.write_text(json.dumps(record, indent=2))
