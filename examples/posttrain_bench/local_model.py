"""Tune a fixed pretrained model and independently score hidden answer likelihood."""

import argparse
import json
import math
from pathlib import Path


def load_model(model_path):
    """Load the fixed CUDA base model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Post-training requires a working CUDA device.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )
    return model, tokenizer


def load_adapter(model, checkpoint):
    """Validate and load the submitted final normalization weight."""
    import torch
    from safetensors.torch import load_file

    name = "model.norm.weight"
    parameter = dict(model.named_parameters())[name]
    state = load_file(str(Path(checkpoint) / "adapter.safetensors"), device="cuda")
    if set(state) != {name} or state[name].shape != parameter.shape:
        raise ValueError("Submission parameters do not match the trusted base model.")
    if not torch.isfinite(state[name]).all():
        raise ValueError("Submission contains nonfinite weights.")
    with torch.no_grad():
        parameter.copy_(state[name])


def encode_example(tokenizer, row):
    """Encode one supervised example with the benchmark's fixed token limits."""
    import torch

    prompt = tokenizer.encode(row["prompt"], add_special_tokens=True)[-192:]
    answer = tokenizer.encode(row["answer"], add_special_tokens=False)[:64]
    if not prompt or not answer:
        raise ValueError("Each example must contain prompt and answer tokens.")
    inputs = torch.tensor([prompt + answer], device="cuda")
    labels = inputs.clone()
    labels[:, : len(prompt)] = -100
    return inputs, labels


def execute(mode, model_path, data, checkpoint):
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(42)
    model, tokenizer = load_model(model_path)
    parameters = dict(model.named_parameters())
    name = "model.norm.weight"
    if name not in parameters:
        raise ValueError("The fixed model must expose its final normalization weight.")
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    parameter = parameters[name]
    rows = json.loads(Path(data).read_text())
    if not rows:
        raise ValueError("The dataset must contain real examples.")
    checkpoint = Path(checkpoint)
    model.eval()
    updates = 0
    if mode == "train":
        parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([parameter], lr=0.001)
    else:
        load_adapter(model, checkpoint)
    losses = []
    for row in rows[:2] if mode == "train" else rows:
        inputs, labels = encode_example(tokenizer, row)
        with torch.set_grad_enabled(mode == "train"):
            loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("Model loss must be finite.")
        losses.append(float(loss.detach()))
        if mode == "train":
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
    if mode == "train":
        checkpoint.mkdir(parents=True, exist_ok=True)
        save_file({name: parameter.detach().cpu().contiguous()}, str(checkpoint / "adapter.safetensors"))
    torch.cuda.synchronize()
    mean_loss = sum(losses) / len(losses)
    return {
        "reward": math.exp(-mean_loss),
        "answer_nll": mean_loss,
        "cuda": 1,
        "eval_valid": 1,
        "local_evaluation": 1,
        "examples": len(losses),
        "optimizer_updates": updates,
        "checkpoint_bytes": (checkpoint / "adapter.safetensors").stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    parser.add_argument("--model", default="/model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="/logs/artifacts/final_model")
    args = parser.parse_args()
    metrics = execute(args.mode, args.model, args.data, args.checkpoint)
    if args.mode == "verify":
        destination = Path("/logs/verifier/reward.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
