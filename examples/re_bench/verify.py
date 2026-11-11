"""Use the official RE-Bench evaluator in an independent native Env container."""

import asyncio
import json
import math
import os
import shutil
import sys
from pathlib import Path


def main(artifacts=Path("/logs/artifacts"), output=Path("/logs/verifier")):
    mode = os.environ["REBENCH_MODE"]
    if mode not in ("smoke", "feedback-smoke", "bounded-local", "official-evaluation"):
        raise ValueError("Unknown RE-Bench verification mode.")
    if mode == "bounded-local" and os.environ["REBENCH_JUDGE_BACKEND"] != "local":
        raise ValueError("Bounded local evaluation requires a local judge.")
    if mode == "smoke":
        result = json.loads((artifacts / "smoke.json").read_text())
        if (
            result.get("model") != "gpt2-xl"
            or result.get("optimizer_steps") != 1
            or not result.get("finite_loss")
            or not math.isfinite(result["loss"])
            or len(result.get("samples", [])) != 2
            or not all(item["response"].strip() for item in result["samples"])
            or len({x["pid"] for x in result["observations"]}) != 1
            or not all(x["allocated_bytes"] > 0 for x in result["observations"])
        ):
            raise ValueError("Real-model CUDA smoke did not complete.")
        rewards = {"reward": 0.0, "smoke_valid": 1, "official_evaluation": 0, "judge_valid": 0}
    else:
        model = artifacts / "finetuned_model"
        if not model.is_file() or not (artifacts / "model.py").is_file():
            result = {"submission_error": "Submit the trained model and model.py."}
            rewards = {
                "reward": 0.0,
                "valid_submission": 0,
                "official_evaluation": 0,
                "local_evaluation": int(os.environ["REBENCH_JUDGE_BACKEND"] == "local"),
                "judge_valid": 0,
                "eval_valid": int(mode != "feedback-smoke"),
                "feedback_smoke_valid": 0,
            }
            (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
            (output / "reward.json").write_text(json.dumps(rewards, indent=2) + "\n")
            return
        shutil.copyfile(artifacts / "model.py", "/home/agent/solution/model.py")
        # Separate verifiers expose their log mount to the host, not their copied artifacts.
        os.environ["REBENCH_JUDGE_DIR"] = "/logs/verifier/rebench-judge"
        sys.path.insert(0, "/opt/rebench")
        from score import evaluate_model

        result = asyncio.run(
            asyncio.wait_for(
                evaluate_model(
                    model,
                    "/protected/prompts.jsonl",
                    "/protected/reference_outputs",
                    False,
                    max_prompts=2 if mode in ("feedback-smoke", "bounded-local") else None,
                    max_tokens_to_sample=16 if mode in ("feedback-smoke", "bounded-local") else 60,
                ),
                timeout=float(os.environ["REBENCH_EVAL_TIMEOUT"]),
            )
        )
        score = result.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Official evaluator did not return a valid preference score.")
        local = os.environ["REBENCH_JUDGE_BACKEND"] == "local"
        partial = mode == "feedback-smoke"
        result["judge"] = {"backend": os.environ["REBENCH_JUDGE_BACKEND"], "model": os.environ["REBENCH_JUDGE_MODEL"]}
        result["evaluated_prompts"] = 2 if mode in ("feedback-smoke", "bounded-local") else 512
        rewards = {
            "reward": 0.0 if partial else score,
            "valid_submission": 1,
            "official_evaluation": int(not local and not partial),
            "local_evaluation": int(local and not partial),
            "judge_valid": 1,
            "eval_valid": int(not partial),
            "feedback_smoke_valid": int(partial),
        }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "reward.json").write_text(json.dumps(rewards, indent=2) + "\n")


if __name__ == "__main__":
    main()
