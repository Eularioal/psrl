"""Prepare trusted dependencies before running one unchanged score test group."""

import argparse
import json
import os
import resource
import runpy
import subprocess
import sys
import time
from pathlib import Path


def prepare(payload):
    """Return the original group runner after CPU dependency imports."""
    os.environ.update(
        PYTHONPATH="/code/python",
        HF_HOME="/hf-cache",
        HF_HUB_CACHE="/hf-cache/hub",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        SGLANG_DISABLE_CUDNN_CHECK="1",
        XDG_CACHE_HOME="/tmp/.cache",
        TORCH_EXTENSIONS_DIR="/tmp/.cache/torch_extensions",
        TRITON_CACHE_DIR="/tmp/.cache/triton",
        TEST_MODEL_NAME="/hf-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        TEST_CLASSIFICATION_BASE_MODEL=(
            "/hf-cache/hub/models--tomaarsen--Qwen3-Reranker-0.6B-seq-cls/"
            "snapshots/6a5829f5079c66e78d911e06fe21931cc00232f7"
        ),
    )
    started = time.time()
    import pytest
    import torch
    import transformers

    assert pytest and transformers and not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA."
    evidence = dict(
        started=started,
        ready=time.time(),
        maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_initialized=torch.cuda.is_initialized(),
    )
    output = Path("/run") / f"{payload['label']}.preparation.json"
    output.write_text(json.dumps(evidence, indent=2))

    def execute():
        evidence.update(activated=time.time(), visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
        authorized = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not authorized.startswith("GPU-") or "," in authorized:
            raise RuntimeError("Expected exactly one authorized GPU UUID.")
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=10
        ).splitlines()
        index = next(row.split(", ")[0] for row in rows if row.split(", ")[1] == authorized)
        os.environ["CUDA_VISIBLE_DEVICES"] = index
        actual = str(torch.cuda.get_device_properties(0).uuid)
        if actual.removeprefix("GPU-") != authorized.removeprefix("GPU-"):
            raise RuntimeError("Numeric device identity differs from the authorized GPU.")
        evidence.update(numeric_gpu=index, actual_gpu=actual)
        output.write_text(json.dumps(evidence, indent=2))
        sys.argv = [
            "/tests/run_upstream_e2e_group.py",
            payload["group"],
            "/tests/fail_to_pass.txt",
            "/tests/pass_to_pass.txt",
            "/tests/upstream_e2e_expected_deselected.txt",
            f"/run/{payload['label']}.group.json",
        ]
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as error:
            if error.code not in (0, None):
                raise

    return execute


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    prepare(dict(label=args.label, group=args.group))()
