"""Run one upstream preference-training iteration using real local judge feedback."""

import asyncio
import json
import time
from pathlib import Path

import rl_finetune as training
import torch


def main():
    training.n_steps = 1
    training.n_prompts_to_sample = 2
    training.batch_size = 1
    training.max_new_tokens = 16
    training.n_examples_to_print = 2
    original_evaluate = training.evaluate
    original_step = torch.optim.Adam.step
    record = {"training_iterations": 1, "optimizer_steps": 0, "judge_waits": []}

    async def evaluate(*args, **kwargs):
        start = time.monotonic()
        before = torch.cuda.memory_allocated()
        result = await original_evaluate(*args, **kwargs)
        record["judge_waits"].append(
            {
                "elapsed_s": time.monotonic() - start,
                "allocated_before": before,
                "allocated_after": torch.cuda.memory_allocated(),
                "preferences": result,
            }
        )
        return result

    def step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        record["optimizer_steps"] += 1
        return result

    training.evaluate = evaluate
    torch.optim.Adam.step = step
    started = time.monotonic()
    try:
        asyncio.run(training.main())
    finally:
        torch.optim.Adam.step = original_step
        training.evaluate = original_evaluate
        record["elapsed_s"] = time.monotonic() - started
        Path("/logs/artifacts/feedback-smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    if not record["optimizer_steps"]:
        raise RuntimeError("Real preference feedback did not produce an optimizer update.")


if __name__ == "__main__":
    main()
