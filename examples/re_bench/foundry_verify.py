"""Replay a submitted fine-tuning program and compare weights with a private run."""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    import reference_train
    from score import compare_models

    metrics = {"reward": 0, "local_evaluation": 1, "eval_valid": 0}
    output = Path("/logs/verifier")
    try:
        reference_train.train("/tmp/reference-model")
        started = time.monotonic()
        subprocess.run(
            [sys.executable, "/logs/artifacts/finetune.py", "/tmp/candidate-model"],
            check=True,
            timeout=90,
            env=dict(os.environ),
        )
        elapsed = time.monotonic() - started
        difference = compare_models("/tmp/candidate-model", "/tmp/reference-model")
        valid = all(math.isfinite(v) for v in difference.values()) and difference["norm_diff"] < 1e-3
        metrics.update(
            reward=1 / (1 + elapsed) if valid else 0,
            eval_valid=1,
            weights_match=int(valid),
            seconds=elapsed,
            **difference,
        )
    except Exception as exc:
        (output / "error.json").write_text(json.dumps({"error": repr(exc)}))
    (output / "reward.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
