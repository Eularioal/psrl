"""Freeze disjoint bounded writing and medical examples from the pinned upstream checkout."""

import argparse
import hashlib
import json
from pathlib import Path


def json_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_rubrics(items):
    """Keep original signed criteria with both useful-answer and harmful-answer coverage."""
    positive = [item for item in items if item["points"] > 0][:2]
    negative = [item for item in items if item["points"] < 0][:2]
    if len(positive) != 2 or len(negative) != 2:
        return []
    return positive + negative


def prepare(repo, assets, judge_model):
    from transformers import AutoTokenizer

    repo, assets, judge_model = (Path(value).resolve() for value in (repo, assets, judge_model))
    if not (judge_model / "config.json").is_file() or not list(judge_model.glob("*.safetensors")):
        raise FileNotFoundError("The frozen local judge requires real model weights.")
    models = json.loads((assets / "models.json").read_text())
    tokenizers = [AutoTokenizer.from_pretrained(item["path"], local_files_only=True) for item in models.values()]

    def prompt_fits(prompt):
        return all(len(tokenizer.encode(prompt, add_special_tokens=True)) <= 384 for tokenizer in tokenizers)

    root = repo / "src/eval/tasks"
    arena = root / "arenahardwriting/evaluation_code/data/arena-hard-v2.0"
    questions = {item["uid"]: item for item in json_lines(arena / "question.jsonl")}
    reference_file = arena / "model_answer/Qwen3-1.7B.jsonl"
    writing = []
    for item in sorted(json_lines(reference_file), key=lambda row: row["uid"]):
        question = questions[item["uid"]]
        if question["category"] != "creative_writing" or not 20 <= len(question["prompt"]) <= 650:
            continue
        if not prompt_fits(question["prompt"]):
            continue
        answer = item["messages"][-1]["content"]["answer"].strip()
        if answer:
            writing.append({"id": item["uid"], "prompt": question["prompt"], "answer": answer[:512]})
    health_file = root / "healthbench/evaluation_code/data/healthbench.jsonl"
    health = []
    for item in sorted(json_lines(health_file), key=lambda row: row["prompt_id"]):
        prompt = "\n".join(f"{turn['role']}: {turn['content']}" for turn in item["prompt"])
        answer = (item.get("ideal_completions_data") or {}).get("ideal_completion", "").strip()
        rubrics = select_rubrics(item["rubrics"])
        if 20 <= len(prompt) <= 800 and answer and rubrics and prompt_fits(prompt):
            health.append({"id": item["prompt_id"], "prompt": prompt, "answer": answer[:512], "rubrics": rubrics})
    for benchmark, rows, sources in (
        ("arenahardwriting", writing, [arena / "question.jsonl", reference_file]),
        ("healthbench", health, [health_file]),
    ):
        if len(rows) < 4 or len({row["id"] for row in rows[:4]}) != 4:
            raise ValueError(f"Insufficient distinct real examples for {benchmark!r}.")
        destination = assets / "data" / benchmark
        destination.mkdir(parents=True, exist_ok=False)
        for name, split in (("public", rows[:2]), ("private", rows[2:4])):
            directory = destination / name
            directory.mkdir()
            (directory / "rows.json").write_text(json.dumps(split, indent=2))
        manifest = {
            "protocol": "bounded_real_prompts_and_frozen_local_verdict_probabilities",
            "public_ids": [row["id"] for row in rows[:2]],
            "private_ids": [row["id"] for row in rows[2:4]],
            "reference_character_limit": 512,
            "generation_token_limit": 64,
            "health_rubrics": "Two original positive and two original negative criteria per example.",
            "sources": {
                str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
            },
        }
        (destination / "sources.json").write_text(json.dumps(manifest, indent=2))
    metadata = {
        "name": "local-frozen-Qwen3.5-4B",
        "path": str(judge_model),
        "config_sha256": hashlib.sha256((judge_model / "config.json").read_bytes()).hexdigest(),
        "weight_files": {path.name: path.stat().st_size for path in sorted(judge_model.glob("*.safetensors"))},
        "read_only": True,
    }
    (assets / "judge.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "assets", "judge-model"):
        parser.add_argument("--" + key, required=True)
    prepare(**vars(parser.parse_args()))
