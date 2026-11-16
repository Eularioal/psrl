"""Measure real model staging, private memory and GPU validation costs."""

import argparse
import ctypes
import gc
import json
import os
import time
from pathlib import Path

MODELS = {
    "language": (
        "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        "float32",
    ),
    "classification": (
        "models--tomaarsen--Qwen3-Reranker-0.6B-seq-cls/snapshots/6a5829f5079c66e78d911e06fe21931cc00232f7",
        "bfloat16",
    ),
}


def memory():
    """Distinguish process proportional memory from shared file mappings."""
    fields = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            if name in {"Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty", "Anonymous"}:
                fields[name + "_kb"] = int(value.split()[0])
    return fields


def trim():
    """Release unused Python objects and ordinary CPU allocator pages."""
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)


def tensor_key(value):
    """Preserve tied parameter aliases instead of charging or converting them twice."""
    return value.data_ptr(), value.storage_offset(), tuple(value.shape), tuple(value.stride())


class ModelRuntime:
    """Retain cheap native mappings and optionally materialize converted CPU states."""

    def __init__(self, cache):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        self.torch = torch
        self.classes = dict(language=AutoModelForCausalLM, classification=AutoModelForSequenceClassification)
        self.raw, self.configs = {}, {}
        self.prepared = {}
        self.prep_cost, self.bytes = {}, {}
        self.versions = dict(torch=torch.__version__, transformers=transformers.__version__)
        self.initial_memory = memory()
        actual = str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
        expected = os.environ["PSRL_EXPECTED_GPU_UUID"].removeprefix("GPU-")
        if actual != expected:
            raise RuntimeError("The container did not select the expected private GPU.")
        x = torch.arange(8, device="cuda")
        if int((x * x).sum()) != 140:
            raise RuntimeError("The CUDA preflight kernel returned an incorrect result.")
        torch.cuda.synchronize()
        self.gpu_uuid = actual
        self.load_times = {}
        for name, (relative, _) in MODELS.items():
            begin = time.monotonic()
            source = (
                self.classes[name]
                .from_pretrained(
                    cache / "hub" / relative, local_files_only=True, torch_dtype="auto", attn_implementation="eager"
                )
                .eval()
            )
            self.configs[name] = source.config
            self.raw[name] = source.state_dict()
            self.load_times[name] = time.monotonic() - begin
            dtype = getattr(torch, MODELS[name][1])
            unique = {tensor_key(value): value for value in self.raw[name].values()}
            self.bytes[name] = sum(v.numel() * torch.empty((), dtype=dtype).element_size() for v in unique.values())
            del source
            trim()

    def prepare(self, name):
        """Perform a real precision conversion without caching inference answers."""
        torch = self.torch
        begin = time.monotonic()
        dtype = getattr(torch, MODELS[name][1])
        converted = {}
        state = {}
        for key, value in self.raw[name].items():
            identity = tensor_key(value)
            if identity not in converted:
                converted[identity] = value.to(dtype=dtype)
            state[key] = converted[identity]
        self.prepared[name] = state
        elapsed = time.monotonic() - begin
        self.prep_cost[name] = elapsed
        return elapsed

    def evict(self, name):
        """Drop converted tensors while preserving native file mappings."""
        del self.prepared[name]
        trim()

    def execute(self, name, *, prepared, seed=0):
        """Construct a fresh GPU model and run the same deterministic validation."""
        torch = self.torch
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        dtype = getattr(torch, MODELS[name][1])
        with torch.device("meta"):
            model = (
                self.classes[name]
                .from_config(self.configs[name], torch_dtype=dtype, attn_implementation="eager")
                .eval()
                .to(dtype=dtype)
            )
        model.to_empty(device="cuda")
        state = self.prepared[name] if prepared else self.raw[name]
        model.load_state_dict(state, strict=True)
        torch.cuda.synchronize()
        loaded = time.monotonic()
        tokens = (torch.arange(8, device="cuda", dtype=torch.long) + seed + 100).unsqueeze(0)
        with torch.inference_mode():
            logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits.float()
            value = logits[:, -1, :32].sum().item() if name == "language" else logits.sum().item()
        torch.cuda.synchronize()
        computed = time.monotonic()
        peak = torch.cuda.max_memory_allocated()
        del model, tokens, logits
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return dict(
            value=value,
            load_gpu_s=loaded - started,
            compute_s=computed - loaded,
            total_s=time.monotonic() - started,
            gpu_peak_bytes=peak,
        )


def profile(args):
    """Run a bounded crossover profile before choosing a cache policy experiment."""
    result = dict(
        status="running", scope="Real checkpoint staging and GPU inference. Not an original benchmark verifier."
    )
    runtime = ModelRuntime(args.cache)
    result.update(
        versions=runtime.versions,
        gpu_uuid=runtime.gpu_uuid,
        payload_bytes=runtime.bytes,
        native_load_s=runtime.load_times,
        initial_memory=runtime.initial_memory,
        native_mappings_memory=memory(),
        models={},
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name in MODELS:
        reference = runtime.execute(name, prepared=False)
        before = memory()
        prepare_s = runtime.prepare(name)
        after = memory()
        rows = []
        for prepared in (False, True, True, False):
            row = runtime.execute(name, prepared=prepared)
            row["prepared"] = prepared
            if abs(row["value"] - reference["value"]) > max(1e-4, abs(reference["value"]) * 1e-5):
                raise RuntimeError("Staging changed the deterministic model validation result.")
            rows.append(row)
        runtime.evict(name)
        result["models"][name] = dict(
            reference=reference,
            prepare_s=prepare_s,
            before=before,
            prepared=after,
            after_eviction=memory(),
            commands=rows,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["status"] = "passed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/hf-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile(args)
