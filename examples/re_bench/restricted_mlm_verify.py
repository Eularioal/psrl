"""Use upstream masked token loss with a private bounded text split."""

import importlib.util
import json
import math
import sys
from pathlib import Path


def load(name, path):
    """Execute the submitted source bytes without accepting an artifact bytecode cache."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def main():
    sys.path.insert(0, "/data/public")
    output = Path("/logs/verifier")
    reward = {"reward": 0, "local_evaluation": 1, "eval_valid": 0}
    try:
        scorer = load("upstream_mlm_score", "/opt/restricted/score.py")
        scorer.batch_size = 2
        scorer.sequence_length = 16
        scorer.n_eval_batches = 2
        scorer.data_dir = "/protected"
        model = load("candidate_model", "/logs/artifacts/model.py").get_trained_model()
        loss = scorer.get_loss(model, True)
        if not math.isfinite(loss) or loss <= 0:
            raise ValueError("Expected a finite positive masked token loss.")
        reward.update(reward=1 / (1 + loss), eval_valid=1, loss=loss)
    except Exception as exc:
        (output / "error.json").write_text(json.dumps({"error": repr(exc)}))
    (output / "reward.json").write_text(json.dumps(reward))


if __name__ == "__main__":
    main()
