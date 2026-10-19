"""Extract bounded training evidence from the original console and Harbor records."""

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


def collect(root: Path):
    text = (root / "GRPO-sciaccel-Qwen35-4B-laps_repair_easy-L1.log").read_text()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    cluster = json.loads((root / "cluster.json").read_text())
    line = next(line for line in text.splitlines() if "step:1 - " in line)
    metrics = {}
    for segment in line.split(" - ")[1:]:
        key, value = segment.split(":", 1)
        value = re.sub(r"np\.(?:float64|int64)\((.*?)\)", r"\1", value)
        metrics[key] = float(value)
    episodes = [
        {"uid": int(uid), "termination": reason, "reward": float(reward)}
        for uid, reason, reward in re.findall(r"\[uid=(\d+)\] Episode done: terminate=(\w+), reward=([0-9.]+)\.", text)
    ]
    warmup_match = re.search(r"'lr_warmup_steps': (\d+)", text)
    warmup = int(warmup_match.group(1)) if warmup_match else None
    row = pd.read_parquet(root / "one-task.parquet").iloc[0]
    reward_key = row["extra_info"].get("reward_key", "reward")
    trials = []
    for path in root.glob("harbor-jobs/**/result.json"):
        result = json.loads(path.read_text())
        if "verifier_result" in result:
            verifier_rewards = (result.get("verifier_result") or {}).get("rewards") or {}
            trials.append(
                {
                    "path": str(path.relative_to(root)),
                    "rewards": result.get("verifier_result"),
                    "training_reward": verifier_rewards.get(reward_key, 0.0),
                    "exception": result.get("exception_info"),
                }
            )
    result = {
        "exit_code": cluster["exit_code"],
        "original_ray_pointer_unchanged": cluster["original_pointer_unchanged"],
        "training_completed_log": "Training completed successfully!" in text,
        "episodes": episodes,
        "training_reward_key": reward_key,
        "verifier_trial_rewards": [trial["training_reward"] for trial in trials],
        "metrics": metrics,
        "warmup_steps": warmup,
        "nonzero_first_step_learning_rate": warmup == 0 and metrics["actor/lr"] > 0,
        "policy_gradient_nonzero": math.isfinite(metrics["actor/grad_norm"]) and metrics["actor/grad_norm"] > 0,
        "effective_policy_update_validated": (
            warmup == 0
            and metrics["actor/lr"] > 0
            and math.isfinite(metrics["actor/grad_norm"])
            and metrics["actor/grad_norm"] > 0
        ),
        "train_worker_logs": {},
        "harbor_trials": trials,
    }
    for path in root.glob("psrl_logs/*/TrainWorker_R*.log"):
        worker_text = path.read_text()
        result["train_worker_logs"][path.name] = {
            "train_actor_completed": "[End Event] TRAIN - Train actor" in worker_text,
            "push_model_completed": "[End Event] PUSH - Push model" in worker_text,
        }
    assert result["exit_code"] == 0 and result["training_completed_log"]
    assert metrics["training/global_step"] == 1 and metrics["termination/trained_tokens"] > 0
    assert math.isfinite(metrics["actor/grad_norm"]) and metrics["actor/grad_norm"] >= 0
    assert math.isfinite(metrics["actor/loss"])
    assert len(result["train_worker_logs"]) == 2
    assert all(all(worker.values()) for worker in result["train_worker_logs"].values())
    (root / "training-evidence.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({key: value for key, value in result.items() if key not in ("metrics", "harbor_trials")}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    collect(parser.parse_args().output)
