"""Replay upstream language loss for a locally sized embedding repair task."""

import json
import math
import sys
from pathlib import Path


def main():
    sys.path[:0] = ["/data/public", "/opt/embedding"]
    import score

    output = Path("/logs/verifier")
    metrics = {"reward": 0, "local_evaluation": 1, "eval_valid": 0}
    try:
        score.n_layer, score.n_head, score.n_embd = 12, 12, 768
        score.block_size, score.batch_size, score.eval_iters = 128, 1, 1
        report = score.evaluate_model("/logs/artifacts/fixed_model.pth", False)
        loss = report["message"]["loss_validation"]
        if not math.isfinite(loss) or loss <= 0:
            raise ValueError("Expected finite positive validation loss.")
        metrics.update(reward=1 / (1 + loss), loss=loss, eval_valid=1)
        (output / "upstream-report.json").write_text(json.dumps(report))
    except Exception as exc:
        (output / "error.json").write_text(json.dumps({"error": repr(exc)}))
    (output / "reward.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
