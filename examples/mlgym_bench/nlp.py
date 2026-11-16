"""Run bounded MLGym language tasks with independent hidden-data GPU evaluation."""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


def execute(mode, task):
    import torch
    from safetensors.torch import load_file, save_file

    if not torch.cuda.is_available():
        raise RuntimeError("Language tasks require CUDA.")
    torch.manual_seed(42)
    training = mode == "train"
    data = Path("/data/public" if training else "/data/private")
    rows = json.loads((data / "rows.json").read_text())
    checkpoint = Path("/logs/artifacts/model.safetensors")
    if task == "languageModelingFineWeb":
        spec = importlib.util.spec_from_file_location("trusted_fineweb", "/opt/task/upstream.py")
        upstream = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = upstream
        spec.loader.exec_module(upstream)
        model = upstream.Model(upstream.ModelConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64)).cuda()
        selected = dict(model.named_parameters())
    else:
        from transformers import AutoTokenizer, BertForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(
            "/model", num_labels=3, local_files_only=True, attn_implementation="eager"
        ).cuda()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        selected = {k: v for k, v in model.named_parameters() if k.startswith("classifier.")}
        for parameter in selected.values():
            parameter.requires_grad_(True)
    model.eval()
    if not training:
        state = load_file(str(checkpoint), device="cuda")
        if set(state) != set(selected):
            raise ValueError("Checkpoint keys do not match the fixed architecture.")
        for name, value in state.items():
            if value.shape != selected[name].shape or not torch.isfinite(value).all():
                raise ValueError("Checkpoint contains invalid weights.")
            selected[name].data.copy_(value)
    optimizer = torch.optim.AdamW(selected.values(), lr=0.001) if training else None
    losses, correct = [], 0
    for row in rows[:2] if training else rows:
        with torch.set_grad_enabled(training):
            if task == "languageModelingFineWeb":
                tokens = list(row["text"].encode("utf-8"))[:129]
                if len(tokens) < 3:
                    raise ValueError("Language sample is too short.")
                x = torch.tensor([tokens[:-1]], device="cuda")
                y = torch.tensor([tokens[1:]], device="cuda")
                _, loss = model(x, y)
            else:
                inputs = tokenizer(row["text1"], row["text2"], return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                result = model(**inputs, labels=torch.tensor([row["label"]], device="cuda"))
                loss = result.loss
                correct += int(result.logits.argmax(-1).item() == row["label"])
        if not torch.isfinite(loss):
            raise ValueError("Loss must be finite.")
        losses.append(float(loss.detach()))
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    if training:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in selected.items()}, str(checkpoint))
    torch.cuda.synchronize()
    mean = sum(losses) / len(losses)
    return {
        "reward": math.exp(-mean) if task == "languageModelingFineWeb" else correct / len(losses),
        "loss": mean,
        "cuda": 1,
        "local_evaluation": 1,
        "eval_valid": 1,
        "examples": len(losses),
        "optimizer_updates": len(losses) if training else 0,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    parser.add_argument("--task", required=True, choices=("languageModelingFineWeb", "naturalLanguageInferenceMNLI"))
    args = parser.parse_args()
    result = execute(args.mode, args.task)
    if args.mode == "verify":
        Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
        Path("/logs/verifier/reward.json").write_text(json.dumps(result))
    print(json.dumps(result))
