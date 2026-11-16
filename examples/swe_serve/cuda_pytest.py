"""Run selected production regression tests on CUDA and attest device operations per case."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class CudaOperations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        values = tree_leaves((args, kwargs, result))
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in values):
            self.count += 1
        return result


class CudaValidation:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        with CudaOperations() as operations:
            yield
        torch.cuda.synchronize()
        record = {"nodeid": item.nodeid, "cuda_operations": operations.count}
        target = Path("/logs/verifier") / ("cuda-" + hashlib.sha256(item.nodeid.encode()).hexdigest()[:16] + ".json")
        target.write_text(json.dumps(record))
        if operations.count == 0:
            pytest.fail("The selected CUDA regression performed no CUDA tensor operations.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("The CUDA regression profile requires a working CUDA device.")
    torch.set_default_device("cuda")
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[CudaValidation()]))
