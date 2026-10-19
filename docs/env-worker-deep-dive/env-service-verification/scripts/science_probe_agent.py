"""Fixed CUDA-only Harbor agent, explicitly without model generation or training."""

import json
import time

from harbor.agents.nop import NopAgent


class ProbeAgent(NopAgent):
    @staticmethod
    def name():
        return "science-cuda-only-verification"

    async def run(self, instruction, environment, context):
        started = time.time()
        result = await environment.exec("python /app/probe.py", timeout_sec=45)
        record = {
            "started": started,
            "finished": time.time(),
            "command": result.model_dump(),
            "scope": "CUDA-only Harbor verification, no model query or training",
        }
        (self.logs_dir / "budget-agent.json").write_text(json.dumps(record, indent=2))
        assert result.return_code == 0, result.stderr
        context.metadata = {"cuda": json.loads(result.stdout.strip())}
