"""Use the protected upstream FLOP bound and empirical scaling interpolation."""

import json
import math
from pathlib import Path

from scaling_score import compute_flops, ground_truth_loss_equation


def main():
    metrics = {"reward": 0, "local_evaluation": 1, "eval_valid": 0}
    try:
        answer = json.loads(Path("/logs/artifacts/answer.json").read_text())
        width, steps, predicted = answer["n_embd"], answer["max_iters"], answer["loss_prediction"]
        if not isinstance(width, int) or not isinstance(steps, int) or min(width, steps) <= 0:
            raise ValueError("Positive integer width and steps required.")
        flops = compute_flops(width, steps)
        if flops > 5.5e17 or not math.isfinite(predicted):
            raise ValueError("Invalid compute budget or loss prediction.")
        loss = float(ground_truth_loss_equation(n_embd=width, max_iters=steps))
        optimum = float(ground_truth_loss_equation(n_embd=554, max_iters=88914))
        score = 1 - (loss - optimum) - abs(predicted - loss)
        metrics.update(reward=score, eval_valid=1, flops=flops, interpolated_loss=loss)
    except Exception as exc:
        Path("/logs/verifier/error.json").write_text(json.dumps({"error": repr(exc)}))
    Path("/logs/verifier/reward.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
