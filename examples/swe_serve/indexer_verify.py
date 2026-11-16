"""Check the real NSA projection on bounded CUDA inputs against float32 matrix multiplication."""

import importlib.util
import json
import os
from pathlib import Path

import torch


def main():
    os.chdir("/code")
    spec = importlib.util.spec_from_file_location("trusted_indexer_workload", "/tests/speed_check/bench_indexer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = module._build_indexer()
    cases = []
    with torch.no_grad():
        for tokens in (1, 8, 32):
            inputs = torch.randn(tokens, module.HIDDEN, device="cuda", dtype=torch.bfloat16)
            expected = inputs.float() @ model.weights_proj.weight.float().t()
            actual = model._weights_proj_bf16_in_fp32_out(inputs)
            torch.cuda.synchronize()
            valid = actual.is_cuda and actual.dtype == torch.float32 and actual.shape == expected.shape
            valid = valid and torch.isfinite(actual).all().item()
            valid = valid and torch.allclose(actual, expected, rtol=1e-3, atol=0.01)
            cases.append({"tokens": tokens, "valid": bool(valid), "max_error": (actual - expected).abs().max().item()})
    valid = all(case["valid"] for case in cases)
    metrics = {
        "reward": int(valid),
        "eval_valid": int(valid),
        "cuda": 1,
        "local_evaluation": 1,
        "tests": len(cases),
        "passed_tests": sum(case["valid"] for case in cases),
    }
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    (output / "projection-cases.json").write_text(json.dumps(cases, indent=2))
    (output / "reward.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
