"""Prepare imports without CUDA before running the unchanged correctness probe."""

import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path


def prepare(payload):
    label = payload["label"]
    started = time.time()
    import pytest
    import torch
    import triton

    assert pytest and triton and not torch.cuda.is_initialized(), "Preparation must not initialize CUDA."
    # The read-only assets remain unchanged while this experiment executes.
    probe = Path("/assets/command_context_probe.py")
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()
    evidence = dict(
        started=started,
        ready=time.time(),
        maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_initialized=torch.cuda.is_initialized(),
        probe_sha256=digest,
    )
    result_path = Path(f"/run/{label}.preparation.json")
    result_path.write_text(json.dumps(evidence, indent=2))

    def execute():
        assert hashlib.sha256(probe.read_bytes()).hexdigest() == digest, "Prepared probe version changed."
        os.environ.update(
            TRITON_CACHE_DIR="/tmp/triton-cache",
            CUDA_CACHE_PATH="/tmp/cuda-cache",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            OMP_NUM_THREADS="1",
        )
        evidence["activated"] = time.time()
        evidence["visible_gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        result_path.write_text(json.dumps(evidence, indent=2))
        spec = importlib.util.spec_from_file_location("correctness_probe", probe)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = [
            str(probe),
            "--source",
            "/assets/versions/base",
            "--tests",
            "/assets/assets/task/tests/postmerge_tests/test/registered/kernels",
            "--output",
            f"/run/{label}_0.json",
        ]
        try:
            module.main()
        except SystemExit as error:
            if error.code != 0:
                raise

    return execute
