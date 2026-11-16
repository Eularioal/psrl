"""Run a GPU-backed scaffold and grade Rust programs on hidden CodeContests tests."""

import asyncio
import importlib.util
import json
from pathlib import Path

import tiktoken
import torch
from rust_codecontests_utils import evaluate_rust_code
from transformers import AutoModelForCausalLM


async def evaluate():
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained("/data/public/model", local_files_only=True).cuda().eval()
    encoding = tiktoken.get_encoding("gpt2")
    calls = []

    def score_code(description, code):
        prompt = encoding.encode(description)[:128]
        completion = encoding.encode(code)[:384]
        ids = torch.tensor([prompt + completion], device="cuda")
        labels = ids.clone()
        labels[:, : len(prompt)] = -100
        with torch.no_grad():
            loss = model(ids, labels=labels).loss.item()
        calls.append({"tokens": ids.numel(), "loss": loss})
        return loss

    path = Path("/logs/artifacts/solver.py")
    spec = importlib.util.spec_from_file_location("candidate", path)
    candidate = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), candidate.__dict__)
    problems = json.loads(Path("/protected/problems.json").read_text())
    results = []
    for problem in problems:
        public = {key: problem[key] for key in ["name", "description", "public_tests", "time_limit"]}
        code = await candidate.solve(public, score_code)
        report = await evaluate_rust_code(asyncio.Semaphore(1), problem, code, "/tmp")
        results.append(report)
    valid = len(calls) >= len(problems)
    return (
        {
            "reward": sum(r["succeeded"] for r in results) / len(results) if valid else 0,
            "local_evaluation": 1,
            "eval_valid": 1,
            "gpu_inference_calls": len(calls),
        },
        results,
        calls,
    )


def main():
    metrics = {"reward": 0, "local_evaluation": 1, "eval_valid": 0}
    output = Path("/logs/verifier")
    try:
        metrics, results, calls = asyncio.run(evaluate())
        (output / "code-results.json").write_text(json.dumps(results, indent=2))
        (output / "gpu-inference.json").write_text(json.dumps(calls, indent=2))
    except Exception as exc:
        (output / "error.json").write_text(json.dumps({"error": repr(exc)}))
    (output / "reward.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
