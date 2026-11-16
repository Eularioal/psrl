"""Verify conditional prefix sums against a protected upstream reference."""

import importlib.util
import json
import secrets
from pathlib import Path


def main():
    import torch
    from reference import prefix_sum_torch

    output = Path("/logs/verifier")
    reward = {"reward": 0, "local_evaluation": 1, "eval_valid": 1, "correctness": 0}
    try:
        path = Path("/logs/artifacts/solution.py")
        spec = importlib.util.spec_from_file_location("candidate", path)
        candidate = importlib.util.module_from_spec(spec)
        exec(compile(path.read_bytes(), str(path), "exec"), candidate.__dict__)
        for length in (1, 1024, 1048576):
            torch.manual_seed(secrets.randbits(32))
            x = torch.randint(-10, 10, (length,), dtype=torch.int32, device="cuda")
            expected = prefix_sum_torch(x)
            actual = candidate.prefix_sum(x.clone())
            assert actual.device == expected.device, "Preserve the CUDA device."
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        reward.update(reward=1, correctness=1)
    except Exception as exc:
        (output / "error.json").write_text(json.dumps({"error": repr(exc)}))
    (output / "reward.json").write_text(json.dumps(reward))


if __name__ == "__main__":
    main()
