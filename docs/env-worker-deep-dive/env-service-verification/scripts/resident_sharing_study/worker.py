"""Run the real local vision task with optional cooperative compute pause points."""

import argparse
import hashlib
import importlib.util
import time
from pathlib import Path

from protocol import PausePoint, read, save


def load_vision(source):
    """Use the supplied task implementation without importing other benchmark packages."""
    spec = importlib.util.spec_from_file_location("study_vision", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--role", choices=("train", "verify"), required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    module = load_vision(args.source)
    value = (torch.arange(32, device="cuda") + 1).sum().item()
    save(
        args.control / "ready.json",
        {"cuda_readback": value, "torch": torch.__version__, "pid": __import__("os").getpid()},
    )
    previous = None
    deadline = time.monotonic() + 280
    while time.monotonic() < deadline:
        request = read(args.control / "request.json")
        if request and request.get("stop"):
            return
        if not request or request["trial_id"] == previous:
            time.sleep(0.002)
            continue
        previous = request["trial_id"]
        checkpoint = args.control / request["checkpoint"]
        if args.role == "verify":
            result = module.evaluate("verify", args.data, checkpoint, args.private)
        else:
            hook = PausePoint(args.control, previous, torch.cuda.synchronize)
            result = module.evaluate(
                "train",
                args.data,
                checkpoint,
                Path("/unmounted-private"),
                epochs=request["epochs"],
                training_hook=hook,
            )
            torch.cuda.synchronize()
            ended_ns = time.monotonic_ns()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            digest = hashlib.sha256()
            for key, tensor in sorted(state.items()):
                digest.update(key.encode())
                digest.update(tensor.contiguous().numpy().tobytes())
            result.update(
                started_ns=hook.started_ns,
                ended_ns=ended_ns,
                events=hook.events,
                tensor_sha256=digest.hexdigest(),
            )
        save(args.control / "result.json", {"trial_id": previous, "state": "completed", **result})
    raise TimeoutError("The bounded study worker expired.")


if __name__ == "__main__":
    main()
