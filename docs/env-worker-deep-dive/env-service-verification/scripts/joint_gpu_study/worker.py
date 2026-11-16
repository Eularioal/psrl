"""Replay bounded PostTrain commands with an identical resident model budget."""

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

MODELS = {"Qwen3-1.7B-Base", "Qwen3-4B-Base", "SmolLM3-3B-Base"}
MIB = 1024**2


def emit(value):
    print("JRESULT " + json.dumps(value, allow_nan=False), flush=True)


class Worker:
    """Keep immutable base models and reset each command's trainable parameter."""

    def __init__(self, args):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.config_class, self.model_class, self.tokenizer_class = AutoConfig, AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(2)
        torch.manual_seed(42)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one working GPU to each worker.")
        self.gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        expected = args.gpu_uuid or os.environ.get("EXPECTED_GPU_UUID")
        if not expected or self.gpu_uuid.removeprefix("GPU-").lower() != expected.removeprefix("GPU-").lower():
            raise RuntimeError(f"Unexpected GPU identity {self.gpu_uuid!r}.")
        self.assets, self.budget = args.assets.resolve(), args.budget_mb * MIB
        self.activation = 1024 * MIB
        self.cache, self.sizes = OrderedDict(), {}

    def clear(self, names=None):
        start = time.monotonic()
        removed = []
        for name in list(self.cache) if names is None else names:
            del self.cache[name]
            removed.append(name)
        gc.collect()
        self.torch.cuda.empty_cache()
        self.torch.cuda.synchronize()
        return time.monotonic() - start, removed

    def resident_bytes(self):
        return sum(value["bytes"] for value in self.cache.values())

    def model_bytes(self, name):
        if name not in self.sizes:
            config = self.config_class.from_pretrained(self.assets / "models" / name, local_files_only=True)
            with self.torch.device("meta"):
                model = self.model_class.from_config(config, torch_dtype=self.torch.bfloat16)
            self.sizes[name] = sum(v.numel() * v.element_size() for v in model.parameters())
            self.sizes[name] += sum(v.numel() * v.element_size() for v in model.buffers())
        return self.sizes[name]

    def run(self, request):
        torch = self.torch
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        name = request["model"]
        if name not in MODELS:
            raise ValueError(f"Unsupported fixed model {name!r}.")
        data = Path(request["data"]).resolve()
        if not data.is_relative_to(self.assets):
            raise ValueError("Read data only from the fixed asset mount.")
        raw = data.read_bytes()
        rows = json.loads(raw)[:2]
        if len(rows) != 2:
            raise ValueError("Each command requires two existing public examples.")
        hit, evict_s, evicted, load_s = name in self.cache, 0.0, [], 0.0
        size = self.model_bytes(name)
        if size + self.activation > self.budget:
            raise ValueError("The model and conservative activation allowance exceed the budget.")
        if not hit:
            while self.cache and self.resident_bytes() + size + self.activation > self.budget:
                elapsed, removed = self.clear([next(iter(self.cache))])
                evict_s += elapsed
                evicted.extend(removed)
            loaded = time.monotonic()
            path = self.assets / "models" / name
            tokenizer = self.tokenizer_class.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            model = (
                self.model_class.from_pretrained(
                    path,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                )
                .cuda()
                .eval()
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            norm = dict(model.named_parameters())["model.norm.weight"]
            self.cache[name] = dict(model=model, tokenizer=tokenizer, norm=norm.detach().cpu().clone(), bytes=size)
            torch.cuda.synchronize()
            load_s = time.monotonic() - loaded
        self.cache.move_to_end(name)
        entry = self.cache[name]
        model, tokenizer = entry["model"], entry["tokenizer"]
        parameter = dict(model.named_parameters())["model.norm.weight"]
        parameter.data.copy_(entry["norm"].to(device="cuda"))
        parameter.grad = None
        parameter.requires_grad_(True)
        torch.manual_seed(42)
        computed = time.monotonic()
        optimizer = torch.optim.AdamW([parameter], lr=0.001)
        losses = []
        for row in rows:
            prompt = tokenizer.encode(row["prompt"], add_special_tokens=True)[-192:]
            answer = tokenizer.encode(row["answer"], add_special_tokens=False)[:64]
            if not prompt or not answer:
                raise ValueError("Each example must contain prompt and answer tokens.")
            inputs = torch.tensor([prompt + answer], device="cuda")
            labels = inputs.clone()
            labels[:, : len(prompt)] = -100
            loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Model loss must be finite.")
            losses.append(float(loss.detach()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        compute_s = time.monotonic() - computed
        digest = hashlib.sha256(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        del inputs, labels, loss, optimizer
        parameter.grad = None
        parameter.requires_grad_(False)
        peak = torch.cuda.max_memory_allocated()
        if peak > self.budget:
            raise RuntimeError("Observed allocated memory exceeded the identical worker budget.")
        mean = sum(losses) / len(losses)
        return dict(
            model=name,
            loss=mean,
            answer_nll=mean,
            losses=losses,
            reward=math.exp(-mean),
            optimizer_updates=2,
            norm_sha256=digest,
            data_sha256=hashlib.sha256(raw).hexdigest(),
            cache_hit=hit,
            load_s=load_s,
            compute_s=compute_s,
            evict_s=evict_s,
            evicted=evicted,
            total_s=time.monotonic() - start,
            cache_keys=list(self.cache),
            cache_bytes=self.resident_bytes(),
            model_bytes=size,
            peak_allocated_mb=peak / MIB,
            allocated_mb=torch.cuda.memory_allocated() / MIB,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--budget-mb", type=int, required=True)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    worker = Worker(args)
    emit(dict(status="ready", gpu_uuid=worker.gpu_uuid, budget_mb=args.budget_mb))
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                operation = request.get("op", "run")
                if operation in {"clear", "shutdown"}:
                    elapsed, removed = worker.clear()
                    result = dict(clear_s=elapsed, removed=removed, cache_keys=[], cache_bytes=0)
                elif operation in {"run", "profile"}:
                    result = worker.run(request)
                else:
                    raise ValueError(f"Unsupported operation {operation!r}.")
                emit(dict(id=request.get("id"), status="passed", **result))
                if operation == "shutdown":
                    break
            except Exception as exc:
                emit(dict(id=request.get("id"), status="failed", error=f"{type(exc).__name__}: {exc}"))
    finally:
        worker.clear()


if __name__ == "__main__":
    main()
