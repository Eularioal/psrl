"""Run a bounded MLGym image training or private scoring command."""

import argparse
import importlib.util
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "verify"), required=True)
    parser.add_argument("--steps", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    import torch

    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("vision", "/study/examples/mlgym_bench/vision.py")
    vision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision)
    readback = (torch.arange(32, device="cuda") + 1).sum().item()
    task_root = args.data_root.parent
    prepared = time.monotonic()
    result = vision.evaluate(
        args.mode,
        task_root / "public",
        Path("/artifact/model.pt"),
        task_root / "private",
        epochs=args.steps,
        learning_rate=args.learning_rate,
    )
    torch.cuda.synchronize()
    finished = time.monotonic()
    result.update(
        cuda_readback=readback,
        preparation_s=prepared - started,
        prepared_monotonic_s=prepared,
        finished_monotonic_s=finished,
        compute_s=finished - prepared,
        peak_mb=torch.cuda.max_memory_allocated() / 1024**2,
    )
    if args.mode == "train":
        result["updates"] = result["optimizer_updates"]
    else:
        result["metric_name"] = "accuracy" if "accuracy" in result else "BLEU-4"
    print("PARTITION_RESULT=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
