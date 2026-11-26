"""Run fixed PostTrain updates and independent likelihood scoring in owned containers."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

from protocol import PausePoint, read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--role", choices=("train", "verify"), required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location("trusted_posttrain", args.source)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    readback = (torch.arange(32, device="cuda") + 1).sum().item()
    model, tokenizer = task.load_model("/model")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    name = "model.norm.weight"
    parameter = dict(model.named_parameters())[name]
    initial = parameter.detach().clone()
    rows = json.loads(((args.data if args.role == "train" else args.private) / "rows.json").read_text())
    examples = [task.encode_example(tokenizer, row) for row in (rows[:2] if args.role == "train" else rows)]
    if not examples:
        raise ValueError("The worker requires actual task examples.")
    torch.cuda.synchronize()
    save(
        args.control / "ready.json",
        {
            "cuda_readback": readback,
            "torch": torch.__version__,
            "pid": os.getpid(),
            "model": str(model.config._name_or_path),
            "example_lengths": [inputs.shape[-1] for inputs, _ in examples],
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        },
    )
    previous = None
    deadline = time.monotonic() + 270
    while time.monotonic() < deadline:
        request = read(args.control / "request.json")
        if request and request.get("stop"):
            return
        if not request or request["trial_id"] == previous:
            time.sleep(0.002)
            continue
        previous = request["trial_id"]
        checkpoint = args.control / request["checkpoint"]
        torch.cuda.reset_peak_memory_stats()
        if args.role == "train":
            with torch.no_grad():
                parameter.copy_(initial)
            parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW([parameter], lr=0.001)
            hook = PausePoint(args.control, previous, torch.cuda.synchronize)
            hook("prepared", 0)
            losses = []
            for step in range(request["steps"]):
                inputs, labels = examples[step % len(examples)]
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("Training loss must be finite.")
                loss.backward()
                optimizer.step()
                hook("updated", step + 1)
                losses.append(float(loss.detach()))
            checkpoint.mkdir()
            saved = parameter.detach().cpu().contiguous()
            save_file({name: saved}, str(checkpoint / "adapter.safetensors"))
            torch.cuda.synchronize()
            ended_ns = time.monotonic_ns()
            result = {
                "optimizer_updates": request["steps"],
                "started_ns": hook.started_ns,
                "ended_ns": ended_ns,
                "events": hook.events,
                "update_times_ns": hook.update_times_ns,
                "tensor_sha256": hashlib.sha256(saved.float().numpy().tobytes()).hexdigest(),
                "last_loss": losses[-1],
            }
            del optimizer, loss
        else:
            task.load_adapter(model, checkpoint)
            losses = []
            with torch.no_grad():
                for inputs, labels in examples:
                    loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
                    losses.append(float(loss))
            mean = sum(losses) / len(losses)
            if not math.isfinite(mean):
                raise ValueError("Independent verification must produce a finite score.")
            result = {"reward": math.exp(-mean), "answer_nll": mean, "examples": len(losses), "eval_valid": 1}
        result.update(
            trial_id=previous,
            state="completed",
            allocated_mb=torch.cuda.memory_allocated() / 1024**2,
            reserved_mb=torch.cuda.memory_reserved() / 1024**2,
            peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
        )
        save(args.control / "result.json", result)
    raise TimeoutError("The bounded PostTrain worker expired.")


if __name__ == "__main__":
    main()
