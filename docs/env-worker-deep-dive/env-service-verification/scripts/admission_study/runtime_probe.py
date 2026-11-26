"""Measure immutable runtime locality with fresh containers and identical imports."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

IMPORTS = """
import json, time
times = {}
t = time.monotonic()
import torch
times['torch_s'] = time.monotonic() - t
t = time.monotonic()
import transformers
times['transformers_s'] = time.monotonic() - t
t = time.monotonic()
import sklearn
times['sklearn_s'] = time.monotonic() - t
t = time.monotonic()
value = torch.ones(8, device='cuda').sum().item()
times['cuda_s'] = time.monotonic() - t
assert value == 8, 'CUDA execution failed.'
print(json.dumps(dict(times=times, cuda_sum=value,
    versions=[torch.__version__, transformers.__version__, sklearn.__version__])))
"""


def run(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "warm_router_study"))
    from baseline import gpu_snapshot, static_gpu_overlay

    before = gpu_snapshot(args.gpu)
    if before["pids"] or before["used_mb"] > 128:
        raise RuntimeError("The probe requires a physically idle GPU.")
    evidence = dict(before=before, observations=[], immutable_files={})
    for relative in (
        "bin/python3.12",
        "lib/python3.12/site-packages/torch/__init__.py",
        "lib/python3.12/site-packages/transformers/__init__.py",
    ):
        hashes = [hashlib.sha256((root / relative).read_bytes()).hexdigest() for root in (args.prefix, args.cache)]
        if hashes[0] != hashes[1]:
            raise RuntimeError(f"Runtime file differs: {relative!r}.")
        evidence["immutable_files"][relative] = hashes[0]
    overlay = static_gpu_overlay(before)["services"]["main"]
    try:
        for policy in ("static", "local", "local", "static"):
            name = "psrl-locality-probe-" + uuid.uuid4().hex[:10]
            root = args.prefix if policy == "static" else args.cache
            command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", "4"]
            for item in overlay["devices"]:
                command.extend(["--device", item])
            for item in overlay["volumes"] + [f"{root}:{args.prefix}:ro"]:
                command.extend(["--volume", item])
            for item in overlay["environment"] + [
                "OMP_NUM_THREADS=1",
                "HF_HUB_OFFLINE=1",
                "LITELLM_LOCAL_MODEL_COST_MAP=True",
            ]:
                command.extend(["--env", item])
            command.extend(["--entrypoint", str(args.prefix / "bin/python"), args.image, "-c", IMPORTS])
            start = time.monotonic()
            try:
                result = subprocess.run(command, text=True, capture_output=True, timeout=90, check=True)
                observation = dict(
                    policy=policy,
                    wall_s=time.monotonic() - start,
                    result=json.loads(result.stdout),
                    stderr=result.stderr,
                )
                evidence["observations"].append(observation)
                print(json.dumps(observation), flush=True)
            finally:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
    finally:
        evidence["after"] = gpu_snapshot(args.gpu)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    parser.add_argument("--gpu", type=int, default=0)
    run(parser.parse_args())
