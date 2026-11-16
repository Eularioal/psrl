"""Generate submitted-model answers and score them with an independent frozen CUDA judge."""

import argparse
import gc
import json
from pathlib import Path

from judge_scoring import rubric_score
from local_model import execute, load_adapter, load_model


def choice_probability(model, tokenizer, content, choices):
    """Read normalized next-token probabilities for two explicit verdict labels."""
    import torch

    labels = [tokenizer.encode(value, add_special_tokens=False) for value in choices]
    if any(len(label) != 1 for label in labels):
        raise ValueError("The fixed judge must tokenize each verdict as one token.")
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Evaluate the answer as an impartial judge. Reply with one verdict label."},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    if inputs.input_ids.shape[1] > 2048:
        raise ValueError("Judge input exceeds the fixed local token budget.")
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1, [label[0] for label in labels]].float()
    probabilities = logits.softmax(dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Judge verdict probabilities must be finite.")
    return float(probabilities[0])


def verify(model_path, judge_path, benchmark, data, checkpoint):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    rows = json.loads(Path(data).read_text())
    if not rows:
        raise ValueError("The private split must contain real evaluation prompts.")
    model, tokenizer = load_model(model_path)
    load_adapter(model, checkpoint)
    answers = []
    torch.manual_seed(42)
    for row in rows:
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=True)
        if len(prompt) > 384:
            raise ValueError("Candidate prompt exceeds the fixed local token budget.")
        inputs = torch.tensor([prompt], device="cuda")
        with torch.inference_mode():
            tokens = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=64,
                min_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        answer = tokenizer.decode(tokens[0, inputs.shape[1] :], skip_special_tokens=True).strip()
        if not answer:
            raise ValueError("The submitted model generated an empty answer.")
        answers.append(answer)
    del model, tokenizer, tokens, inputs
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(judge_path, local_files_only=True, trust_remote_code=False)
    judge = (
        AutoModelForImageTextToText.from_pretrained(
            judge_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )
    judge.requires_grad_(False)
    details = []
    for row, answer in zip(rows, answers, strict=True):
        if benchmark == "arenahardwriting":
            probabilities = []
            for swapped in (False, True):
                first, second = (row["answer"], answer) if swapped else (answer, row["answer"])
                content = (
                    "Compare the answers for relevance, helpfulness, clarity and creativity. "
                    "Select A or B, whichever answer is better.\n"
                    f"Question:\n{row['prompt']}\nAnswer A:\n{first}\nAnswer B:\n{second}\nVerdict:"
                )
                probability = choice_probability(judge, tokenizer, content, ("A", "B"))
                probabilities.append(1 - probability if swapped else probability)
            score = sum(probabilities) / 2
        elif benchmark == "healthbench":
            probabilities = []
            for rubric in row["rubrics"]:
                content = (
                    "Decide whether the assistant answer meets this criterion. "
                    "Answer Y for met or N for not met. A harmful statement can itself be a criterion.\n"
                    f"Conversation:\n{row['prompt']}\nAssistant answer:\n{answer}\n"
                    f"Criterion:\n{rubric['criterion']}\nVerdict:"
                )
                probabilities.append(choice_probability(judge, tokenizer, content, ("Y", "N")))
            score = rubric_score(row["rubrics"], probabilities)
        else:
            raise ValueError(f"Unknown judged benchmark {benchmark!r}.")
        details.append({"id": row["id"], "answer": answer, "verdict_probabilities": probabilities, "score": score})
    torch.cuda.synchronize()
    root = Path("/logs/verifier")
    root.mkdir(parents=True, exist_ok=True)
    (root / "judge-details.json").write_text(json.dumps(details, indent=2))
    return {
        "reward": sum(item["score"] for item in details) / len(details),
        "cuda": 1,
        "eval_valid": 1,
        "local_evaluation": 1,
        "judge_valid": 1,
        "examples": len(details),
        "judge_calls": sum(len(item["verdict_probabilities"]) for item in details),
        "generated_answers": len(answers),
        "checkpoint_bytes": (Path(checkpoint) / "adapter.safetensors").stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    parser.add_argument("--model", default="/model")
    parser.add_argument("--judge", default="/judge")
    parser.add_argument("--benchmark", required=True, choices=("arenahardwriting", "healthbench"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="/logs/artifacts/final_model")
    args = parser.parse_args()
    if args.mode == "train":
        metrics = execute("train", args.model, args.data, args.checkpoint)
    else:
        metrics = verify(args.model, args.judge, args.benchmark, args.data, args.checkpoint)
        (Path("/logs/verifier") / "reward.json").write_text(json.dumps(metrics))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
