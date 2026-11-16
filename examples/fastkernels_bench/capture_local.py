"""Capture actual CUDA forwards of bounded upstream models with random weights."""

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path


def capture(output, family="recommendation-vision"):
    import torch
    from fastkernels import capture as upstream
    from fastkernels.list import discover_operator_targets
    from fastkernels.tasks.baseline.L4.clip_text_model import CLIPTextModel
    from fastkernels.tasks.baseline.L4.convnextv2 import ConvNextV2ForImageClassification
    from fastkernels.tasks.baseline.L4.dlrmv2 import DLRMv2, DLRMv2Config
    from fastkernels.tasks.baseline.L4.lightgcn import LightGCN, LightGCNConfig
    from psrl_bench_config import TextConfig, VisionConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Capture requires an actual CUDA device.")
    torch.manual_seed(42)
    targets = discover_operator_targets()
    classes = []
    for target in targets:
        module = sys.modules.get(f"fastkernels.tasks.baseline.L{target.level}.{target.name}")
        cls = getattr(module, target.class_name, None)
        if inspect.isclass(cls):
            classes.append(cls)
    instrumented = upstream._instrument(classes)
    upstream._RECORDS.clear()
    upstream._HOOK_RECORDS.clear()

    def execute(model, args):
        model = model.cuda().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-0.1, 0.1)
            hooks = upstream._register_verification_hooks(model, instrumented)
            try:
                model(*args)
                torch.cuda.synchronize()
            finally:
                for hook in hooks:
                    hook.remove()

    if family == "clip-text":
        execute(CLIPTextModel(TextConfig()), (torch.randint(0, 128, (2, 8), device="cuda"),))
    else:
        execute(
            DLRMv2(DLRMv2Config(num_embeddings_per_feature=[32] * 4)),
            (torch.randn(2, 13, device="cuda"), [torch.randint(0, 32, (2, 1), device="cuda") for _ in range(4)]),
        )
        users = torch.tensor([0, 1, 2, 3], device="cuda")
        items = torch.tensor([1, 2, 3, 4], device="cuda")
        adjacency = LightGCN.build_adjacency(users, items, 16, 16, device=torch.device("cuda"))
        execute(LightGCN(LightGCNConfig(16, 16, 32, 2)), (users, items, adjacency))
        execute(
            ConvNextV2ForImageClassification(VisionConfig()),
            (torch.randn(2, 3, 32, 32, device="cuda"),),
        )
    report = upstream._build_report(
        "psrl-local-random-initialized-models",
        "bounded_forward",
        torch.float32,
        2,
        len(classes),
        {"pretrained": False},
        upstream._verify_forward_capture(),
    )
    report["engine"] = "examples.fastkernels_bench.capture_local"
    report["device"] = {"name": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}
    fp4 = importlib.import_module("fastkernels.tasks.baseline.L2.trtllm_fp4_moe")
    report["trtllm_fp4_moe_available"] = fp4.trtllm_fp4_moe_available()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"captured": len(report["operators"]), "device": report["device"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", choices=["recommendation-vision", "clip-text"], default="recommendation-vision")
    capture(**vars(parser.parse_args()))
