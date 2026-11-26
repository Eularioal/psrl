"""Measure a local BF16 compute/bandwidth reference and decode candidate knees."""

import argparse
import json
import math
import os
import struct
import time
from functools import partial
from pathlib import Path


def parameter_count(model):
    count = 0
    for path in model.glob("*.safetensors"):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        count += sum(math.prod(value["shape"]) for name, value in header.items() if name != "__metadata__")
    if count == 0:
        raise ValueError("No safetensors parameters were found.")
    return count


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from psrl.workers.env_service.probe import NvmlProbe

    probe = NvmlProbe()
    before = probe.read_gpu_snapshot([args.gpu])
    if any(row["pids"] or row["used_mb"] > 64 for row in before.values()):
        raise RuntimeError("Roofline calibration device is occupied.")
    torch.set_num_threads(1)
    record = {"started_at": time.time(), "before": before, "gemm": [], "bandwidth": []}

    def measure(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        values = []
        for _ in range(3):
            start.record()
            for _ in range(20):
                fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) / 20000)
        return values

    for size in (2048, 4096, 8192):
        a = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        seconds = measure(partial(torch.mm, a, b, out=c))
        record["gemm"].append(
            {"size": size, "seconds": seconds, "tflops": [2 * size**3 / value / 1e12 for value in seconds]}
        )
        del a, b, c
    for count in (64 * 1024**2, 256 * 1024**2):
        a = torch.ones(count, device="cuda", dtype=torch.bfloat16)
        b = torch.ones_like(a)
        c = torch.empty_like(a)
        seconds = measure(partial(torch.add, a, b, out=c))
        record["bandwidth"].append(
            {"elements": count, "seconds": seconds, "tb_s": [3 * count * 2 / value / 1e12 for value in seconds]}
        )
        del a, b, c
    compute = max(value for row in record["gemm"] for value in row["tflops"])
    bandwidth = max(value for row in record["bandwidth"] for value in row["tb_s"])
    ridge = compute / bandwidth
    config = json.loads((args.model / "config.json").read_text())
    params = parameter_count(args.model)
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // heads)
    rows = []
    for context in (128, 384, 2048, 2304, 8192):
        kv_bytes = 2 * layers * kv_heads * head_dim * context * 2
        flops = 2 * params + 4 * layers * heads * head_dim * context
        denominator = flops - ridge * kv_bytes
        knee = ridge * params * 2 / denominator if denominator > 0 else None
        rows.append(
            {
                "context": context,
                "kv_bytes_per_request": kv_bytes,
                "estimated_flops_per_decode_token": flops,
                "candidate_knee_requests": knee,
            }
        )
    record.update(
        measured_tflops=compute,
        measured_tb_s=bandwidth,
        ridge_flop_per_byte=ridge,
        model_parameters=params,
        model_config=config,
        decode_estimates=rows,
        finished_at=time.time(),
        scope="Local GEMM/add references. Approximate decoder traffic, not an SM saturation measurement.",
    )
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
