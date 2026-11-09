"""Measure real SWE-Serve correctness tests across fresh GPU processes."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    import pytest
    import torch
    import triton

    imported = time.perf_counter()
    sys.path.insert(0, str(args.source))
    events = []
    reports = []

    def listener(**event):
        events.append(
            {
                "kernel": event["src"].name,
                "cache_hit": event["cache_hit"],
                "hash": event["metadata"]["hash"],
                "times": vars(event["times"]),
            }
        )

    class Reporter:
        def pytest_runtest_logreport(self, report):
            reports.append(
                {"test": report.nodeid, "phase": report.when, "outcome": report.outcome, "s": report.duration}
            )

    triton.knobs.compilation.listener = listener
    # The CUDA probe is outside the test timer but inside command startup.
    assert torch.cuda.is_available(), "CUDA is unavailable."
    assert torch.ones(16, device="cuda").sum().item() == 16, "CUDA probe failed."
    torch.cuda.synchronize()
    ready = time.perf_counter()
    code = pytest.main(
        [
            str(args.tests / "test_gdn_fused_qkv_split.py"),
            str(args.tests / "test_gdn_fused_proj_p2p.py"),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        plugins=[Reporter()],
    )
    torch.cuda.synchronize()
    finished = time.perf_counter()
    calls = [r for r in reports if r["phase"] == "call"]
    source = args.source / "sglang/jit_kernel/triton/gdn_fused_proj.py"
    result = {
        "pid_in_container": os.getpid(),
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_source": torch.__file__,
        "triton_source": triton.__file__,
        "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "import_s": imported - started,
        "cuda_init_s": ready - imported,
        "pytest_s": finished - ready,
        "measured_process_s": finished - started,
        "test_call_s": sum(r["s"] for r in calls),
        "passed": sum(r["outcome"] == "passed" for r in calls),
        "exit_code": int(code),
        "compiler_events": events,
        "reports": reports,
        "memory_allocated_before_exit": torch.cuda.memory_allocated(),
        "memory_reserved_before_exit": torch.cuda.memory_reserved(),
    }
    args.output.write_text(json.dumps(result, indent=2))
    # No skipped or missing test is accepted as a successful characterization.
    raise SystemExit(0 if int(code) == 0 and result["passed"] == 6 else 1)


if __name__ == "__main__":
    main()
