"""Original science agent plus an identical, declared CUDA pressure phase in both arms."""

import json
import time

from examples.sciaccel_rl.agent import TruncatingTerminus2
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.nop import NopAgent


class PressureTerminus2(TruncatingTerminus2):
    async def run(self, instruction, environment, context):
        self._pressure_environment = environment
        self._pressure_done = False
        return await super().run(instruction, environment, context)

    async def _exec_pressure(self, command, **kwargs):
        return await self._pressure_environment.exec(command, **kwargs)

    async def _query_llm(self, *args, **kwargs):
        reply = await super()._query_llm(*args, **kwargs)
        if not self._pressure_done:
            self._pressure_done = True
            started = time.time()
            command = await self._exec_pressure(
                "python3 /opt/verification/cuda_pressure.py --gib 50 --hold 45", timeout_sec=75
            )
            record = {
                "started_at": started,
                "finished_at": time.time(),
                "scope": "CUDA injection after first genuine model response, before executing its commands",
                "command": command.model_dump(),
            }
            (self.logs_dir / "gpu-pressure-command.json").write_text(json.dumps(record, indent=2))
            if command.return_code != 0:
                raise NonZeroAgentExitCodeError(
                    "Controlled CUDA phase failed: " + (command.stdout or "") + (command.stderr or "")
                )
        return reply


class PressureProbeAgent(NopAgent):
    async def run(self, instruction, environment, context):
        command = await environment.exec(
            "python3 /opt/verification/cuda_pressure.py --gib 50 --hold 15", timeout_sec=60
        )
        (self.logs_dir / "gpu-pressure-command.json").write_text(json.dumps(command.model_dump(), indent=2))
        if command.return_code not in (0, 42):
            raise RuntimeError("Unexpected probe failure: " + (command.stdout or "") + (command.stderr or ""))
        context.metadata = {"pressure": json.loads(command.stdout.strip())}
