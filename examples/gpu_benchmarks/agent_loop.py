"""Collect benchmark trajectories through the existing Harbor session loop."""

import asyncio
import os

from examples.sciaccel_rl.agent_loop import SciAccelAgentLoop, _harbor_loop
from examples.sciaccel_rl.runner import run_harbor_episode
from psrl.workers.agent_loop.loops.utils import register
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.harbor_resources import demand_from_task


@register("gpu_benchmark")
class GpuBenchmarkAgentLoop(SciAccelAgentLoop):
    """Require EnvRouter admission for every benchmark episode."""

    def __init__(self, context, **kwargs):
        if not context.config.psrl.get("env_service", {}).get("enable", False):
            raise ValueError("GPU benchmark recipes require psrl.env_service.enable=True.")
        super().__init__(context=context, **kwargs)
        if self.runtime_config.harbor.environment_backend not in ("docker", "env_worker"):
            raise ValueError("GPU benchmarks require a managed container backend.")
        units = self.env_service_cfg.get("units", [])
        if not units or any(unit.get("gpu_reservation", "episode") != "episode" for unit in units):
            raise ValueError("These benchmark verifiers require explicit episode GPU units.")
        self._benchmark_client = None

    async def _run_harbor_in_thread(self, task_path, model_base_url, model_name, **kwargs):
        async def execute():
            if self._benchmark_client is None:
                self._benchmark_client = EnvServiceClient.from_config(self.env_service_cfg)
            if self._benchmark_client is None:
                raise RuntimeError("The benchmark cannot run without EnvRouter admission.")
            gpu_memory_mb = kwargs.pop("gpu_memory_mb", 0)
            demand = demand_from_task(task_path, needs_gpu_hint=True, gpu_memory_mb_override=gpu_memory_mb)
            return await run_harbor_episode(
                task_path=task_path,
                model_base_url=model_base_url,
                model_name=model_name,
                config=self.runtime_config,
                env_client=self._benchmark_client,
                env_demand=demand,
                session_router_url=self.session_router_url,
                max_model_len=int(self.rollout_config.get("max_model_len", 40960)),
                max_turns=self.max_turns,
                actor_id=os.getenv("PSRL_ACTOR_ID", ""),
                thinking_template=self.thinking_template,
                regrade_unverified=False,
                **kwargs,
            )

        return await asyncio.wrap_future(_harbor_loop.run(execute()))
