"""Fixed CUDA workload with one real model query for a bounded science training comparison."""

import json
import time

from harbor.agents.nop import NopAgent
from openai import AsyncOpenAI


class BudgetAgent(NopAgent):
    """Keep resource demand identical across arms while collecting a real training trajectory."""

    def __init__(self, *args, api_base=None, extra_body=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_base = api_base
        self.extra_body = extra_body or {}

    @staticmethod
    def name():
        return "science-budget-verification"

    async def run(self, instruction, environment, context):
        started = time.time()
        async with AsyncOpenAI(base_url=self.api_base, api_key="EMPTY", max_retries=0, timeout=90) as client:
            reply = await client.chat.completions.create(
                model=self.model_name.removeprefix("openai/"),
                messages=[{"role": "user", "content": "In one sentence, describe why a GPU memory budget is useful."}],
                max_tokens=32,
                temperature=0.7,
                extra_body=self.extra_body,
            )
        result = await environment.exec("python /app/probe.py", timeout_sec=45)
        record = {
            "started": started,
            "finished": time.time(),
            "model_reply": reply.model_dump(),
            "command": result.model_dump(),
        }
        (self.logs_dir / "budget-agent.json").write_text(json.dumps(record, indent=2))
        assert result.return_code == 0, result.stderr
        context.n_input_tokens = reply.usage.prompt_tokens
        context.n_output_tokens = reply.usage.completion_tokens
        context.metadata = {"fixed_workload": True, "cuda": json.loads(result.stdout.strip())}
