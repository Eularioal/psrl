"""Capture actual bounded CUDA forwards of native operators and small vision models."""

import argparse
import importlib
import inspect
import json
import signal
import sys
from pathlib import Path


def cases():
    """Provide explicit valid constructor and forward inputs for local correctness checks."""
    import torch

    def tensor(*shape):
        return torch.randn(*shape, device="cuda")

    values = []

    def add(module, name, init=(), args=(), kwargs=None, level=1):
        values.append((f"fastkernels.tasks.baseline.L{level}.{module}", name, init, args, kwargs or {}))

    for module, name in (
        ("gelu", "GELU"),
        ("relu", "ReLU"),
        ("silu", "SiLU"),
        ("sigmoid", "Sigmoid"),
        ("tanh", "Tanh"),
        ("quickgelu", "QuickGELU"),
        ("mish", "Mish"),
        ("squared_relu", "SquaredReLU"),
        ("silu_and_mul", "SiluAndMul"),
        ("squared_relu_and_mul", "SquaredReluAndMul"),
        ("dropout", "Dropout"),
        ("l2_norm", "L2Norm"),
        ("softmax", "Softmax"),
        ("softmax", "LogSoftmax"),
    ):
        add(module, name, args=(tensor(2, 8, 32),))
    for module, name in (
        ("layer_norm", "LayerNorm"),
        ("rms_norm", "RMSNorm"),
        ("rms_norm_native", "RMSNormNative"),
        ("t5_layer_norm", "T5LayerNorm"),
    ):
        add(module, name, (32,), (tensor(2, 8, 32),))
    add("rms_norm_gated", "RMSNormGated", (32,), (tensor(2, 8, 32), tensor(2, 8, 32)))
    add("group_norm", "GroupNorm", (2, 4), (tensor(2, 4, 8, 8),))
    add("batch_norm2d", "BatchNorm2d", (4,), (tensor(2, 4, 8, 8),))
    for module, name in (("avg_pool2d", "AvgPool2d"), ("max_pool2d", "MaxPool2d")):
        add(module, name, (2,), (tensor(2, 4, 8, 8),))
    add("global_avg_pool2d", "GlobalAvgPool2d", args=(tensor(2, 4, 8, 8),))
    for module, name in (("conv1d", "Conv1d"), ("conv1d_native", "Conv1dNative")):
        add(module, name, (4, 8, 3), (tensor(2, 4, 16),))
    add("conv_transpose1d", "ConvTranspose1d", (4, 8, 3), (tensor(2, 4, 8),))
    add("conv2d", "Conv2d", (4, 8, 3), (tensor(2, 4, 16, 16),))
    add("conv3d", "Conv3d", (2, 4, (2, 2, 2)), (tensor(1, 2, 8, 8, 8),))
    add("linear", "Linear", (32, 16), (tensor(2, 8, 32),))
    add("linear", "Matmul", args=(tensor(2, 8, 32), tensor(16, 32), tensor(16)))
    add("linear", "BMM", args=(tensor(2, 8, 16), tensor(2, 16, 8)))
    add("bmm", "BatchMatMul", args=(tensor(2, 8, 16), tensor(2, 16, 8)))
    add("dense_attention", "DenseAttention", ("sdpa",), (tensor(2, 8, 4, 16),) * 3)
    add("dp3_sinusoidal_pos_emb", "DP3SinusoidalPosEmb", (32,), (tensor(2),))
    add("sinusoidal_embed", "SinusoidalEmbed", (32,), (tensor(2),))
    add("sam3_position_encoding", "Sam3PositionEncoding", (32,), (tensor(2, 4, 8, 8),))
    add("sam3_rope", "Sam3RoPE2D", (16, (2, 2)), (tensor(2, 4, 4, 16), tensor(2, 4, 4, 16)))
    add("ttt_e2e_rope", "TTTE2ERoPE", (16, 32), (tensor(2, 8, 4, 16), torch.arange(8, device="cuda")))
    add("tensor_ops", "Pad", args=(tensor(2, 8, 16), (1, 1)))
    add("tensor_ops", "OneHot", args=(torch.tensor([0, 1, 2], device="cuda"), 4))
    add("tensor_ops", "Cat", args=([tensor(2, 8), tensor(2, 8)],))
    add("tensor_ops", "Exp", args=(tensor(2, 8, 16),))
    add("sparse_mm", "SparseMM", args=(tensor(8, 8), tensor(8, 16)))
    add("moe_sum", "MoeSum", args=(tensor(8, 32), 2))
    add(
        "siglip2",
        "SigLIP2Model",
        args=(tensor(1, 3, 32, 32),),
        kwargs={"patch_size": 4, "embed_dim": 32, "depth": 1, "num_heads": 4, "pos_embed_grid_size": (8, 8)},
        level=4,
    )
    add(
        "swinv2",
        "SwinV2Model",
        args=(tensor(1, 3, 32, 32),),
        kwargs={"embed_dim": 16, "depths": (1, 1), "num_heads": (2, 4), "window_size": 4},
        level=4,
    )
    add("mobilenetv4", "MobileNetV4Model", args=(tensor(1, 3, 64, 64),), level=4)
    return values


def capture(output, targets=None, case_timeout=30):
    import torch
    from fastkernels import capture as upstream
    from fastkernels.list import discover_operator_targets

    if not torch.cuda.is_available():
        raise RuntimeError("Capture requires a working CUDA device.")
    if not 1 <= case_timeout <= 90:
        raise ValueError("Capture cases require a timeout between one and ninety seconds.")
    torch.manual_seed(42)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specs = cases()
    if targets is not None:
        selected = set(targets)
        specs = [item for item in specs if f"{item[0]}:{item[1]}" in selected]
        if {f"{item[0]}:{item[1]}" for item in specs} != selected:
            raise ValueError("Capture targets must identify explicitly supported forward cases.")
    records = []
    loaded = []
    for module, name, init, args, kwargs in specs:
        try:
            cls = getattr(importlib.import_module(module), name)
            loaded.append((module, name, cls, init, args, kwargs))
        except (ImportError, AttributeError) as exc:
            records.append({"target": f"{module}:{name}", "captured": False, "error": repr(exc)})
    classes = []
    for target in discover_operator_targets():
        module = sys.modules.get(f"fastkernels.tasks.baseline.L{target.level}.{target.name}")
        cls = getattr(module, target.class_name, None)
        if inspect.isclass(cls):
            classes.append(cls)
    instrumented = upstream._instrument(classes)

    def expired(signum, frame):
        raise TimeoutError("The operator capture exceeded its configured time budget.")

    signal.signal(signal.SIGALRM, expired)
    for module, name, cls, init, args, kwargs in loaded:
        upstream._RECORDS.clear()
        upstream._HOOK_RECORDS.clear()
        hooks = []
        try:
            signal.alarm(case_timeout)
            model = cls(*init, **kwargs).cuda().eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.is_floating_point():
                        parameter.uniform_(-0.1, 0.1)
                hooks = upstream._register_verification_hooks(model, instrumented)
                result = model(*args)
                torch.cuda.synchronize()
            verification = upstream._verify_forward_capture()
            if verification.get("issues"):
                raise ValueError(f"Capture verification failed: {verification!r}.")
            report = upstream._build_report(
                f"psrl-local-{name}",
                "bounded_cuda_forward",
                torch.float32,
                2,
                len(classes),
                {"pretrained": False},
                verification,
            )
            report["engine"] = "examples.fastkernels_bench.capture_cases"
            report["device"] = {"name": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}
            path = output / f"{module.rsplit('.', 1)[-1]}-{name}.json"
            path.write_text(json.dumps(report, indent=2))
            records.append({"target": f"{module}:{name}", "captured": True, "operators": len(report["operators"])})
            del model, result
        except Exception as exc:
            records.append({"target": f"{module}:{name}", "captured": False, "error": repr(exc)})
        finally:
            signal.alarm(0)
            for hook in hooks:
                hook.remove()
        print(json.dumps(records[-1]), flush=True)
        output.with_suffix(".json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--case-timeout", type=int, default=30)
    capture(**vars(parser.parse_args()))
