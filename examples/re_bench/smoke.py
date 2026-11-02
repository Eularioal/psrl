"""Exercise real RE-Bench model operations without manufacturing preference labels."""

import json
import math
import os
import sys
import time
from pathlib import Path

import tiktoken
import torch

sys.path.insert(0, "/home/agent/solution")
from model import GPT


def main():
    started = time.monotonic()
    torch.manual_seed(0)
    prompts = [json.loads(line)["prompt"] for line in Path("/home/agent/prompts.jsonl").read_text().splitlines()[:2]]
    model = GPT.from_pretrained("gpt2-xl").to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    records = []

    def observe(phase):
        torch.cuda.synchronize()
        records.append(
            {
                "phase": phase,
                "pid": os.getpid(),
                "elapsed_s": time.monotonic() - started,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
        )

    observe("model_loaded")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        responses = model.generate_strings_autobatch(prompts, max_new_tokens=8, batch_size=2)
    observe("generated")
    # Public prompt tokens exercise CUDA training without claiming judge feedback.
    tokens = tiktoken.get_encoding("gpt2").encode_ordinary(prompts[0])[:65]
    batch = torch.tensor([tokens], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
    if not torch.isfinite(loss):
        raise RuntimeError("The real model produced a nonfinite smoke loss.")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    observe("optimizer_step")
    result = {
        "mode": "real_assets_cuda_smoke_without_judge",
        "model": "gpt2-xl",
        "torch": torch.__version__,
        "samples": [{"prompt": p, "response": r} for p, r in zip(prompts, responses, strict=True)],
        "loss": loss.item(),
        "optimizer_steps": 1,
        "observations": records,
        "finite_loss": math.isfinite(loss.item()),
    }
    Path("/logs/artifacts/smoke.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
