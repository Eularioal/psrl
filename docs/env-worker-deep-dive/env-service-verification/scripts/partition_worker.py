"""Execute fixed PostTrain norm updates or independent private likelihood verification."""

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "verify"), required=True)
    parser.add_argument("--steps", type=int, default=448)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("task", "/study/task.py")
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    readback = (torch.arange(32, device="cuda") + 1).sum().item()
    model, tokenizer = task.load_model("/model")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rows = json.loads((args.data_root / "rows.json").read_text())
    examples = [task.encode_example(tokenizer, row) for row in (rows[:2] if args.mode == "train" else rows)]
    checkpoint = Path("/artifact")
    prepared = time.monotonic()
    result = {
        "cuda_readback": readback,
        "preparation_s": prepared - started,
        "prepared_monotonic_s": prepared,
    }
    if args.mode == "train":
        name = "model.norm.weight"
        parameter = dict(model.named_parameters())[name]
        parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([parameter], lr=args.learning_rate)
        for step in range(args.steps):
            inputs, labels = examples[step % len(examples)]
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite.")
            loss.backward()
            optimizer.step()
        saved = parameter.detach().cpu().contiguous()
        save_file({name: saved}, str(checkpoint / "adapter.safetensors"))
        result.update(updates=args.steps, tensor_sha256=hashlib.sha256(saved.float().numpy().tobytes()).hexdigest())
    else:
        task.load_adapter(model, checkpoint)
        with torch.no_grad():
            losses = [float(model(input_ids=x, labels=y, use_cache=False).loss) for x, y in examples]
        mean = sum(losses) / len(losses)
        if not math.isfinite(mean):
            raise ValueError("Verification loss is not finite.")
        result.update(reward=math.exp(-mean), answer_nll=mean, examples=len(examples))
    torch.cuda.synchronize()
    result["finished_monotonic_s"] = time.monotonic()
    result.update(compute_s=time.monotonic() - prepared, peak_mb=torch.cuda.max_memory_allocated() / 1024**2)
    print("PARTITION_RESULT=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
