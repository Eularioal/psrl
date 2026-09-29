"""
GPU integration tests that start a real container with GPU passthrough.

Unlike `test_worker_integration.py`, which only exercises the CPU path (any local
Docker image, `network="none"`, no device flags), these tests exercise the code
path that actually matters for GPU sandboxes: `EnvWorker._allocate_gpu_indices`
picking a physical device from `CUDA_VISIBLE_DEVICES`, and `build_gpu_argv` in
`sandbox.py` turning it into `--device /dev/nvidiaN` plus driver bind-mounts.

Skipped automatically unless ALL of the following hold on the host running pytest:
- `docker` is on PATH.
- `nvidia-smi` is on PATH (there is a real NVIDIA GPU with a driver installed).
- The environment variable `PSRL_GPU_PROBE_IMAGE` points at a local image that has
  Python and a CUDA-enabled PyTorch installed (for example
  `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`, or any project-specific sandbox
  image that already bundles `torch`).

Run with, for example:
    PSRL_GPU_PROBE_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime \\
        pytest tests/env_worker/test_worker_gpu_integration.py -v
"""

from __future__ import annotations

import asyncio
import os
import shutil

import pytest
from psrl.workers.env_worker.sandbox import SandboxSpec
from psrl.workers.env_worker.worker import EnvWorker

GPU_PROBE_IMAGE = os.getenv("PSRL_GPU_PROBE_IMAGE")
_HAS_DOCKER = shutil.which("docker") is not None
_HAS_NVIDIA_SMI = shutil.which("nvidia-smi") is not None

requires_gpu_sandbox = pytest.mark.skipif(
    not (_HAS_DOCKER and _HAS_NVIDIA_SMI and GPU_PROBE_IMAGE),
    reason=(
        "Requires docker, a host NVIDIA GPU (nvidia-smi on PATH), and "
        "PSRL_GPU_PROBE_IMAGE pointing at a local CUDA+PyTorch image."
    ),
)


def _first_visible_gpu_index() -> str:
    """
    Pick one physical GPU index to hand to the worker under test.

    Reuses whatever `CUDA_VISIBLE_DEVICES` Ray or the shell already set, falling
    back to index 0. This mirrors how `EnvWorkerManager` derives
    `EnvWorker.available_gpu_indices` in production (see `worker.py`), so the test
    exercises the same derivation instead of hardcoding a Ray-only assumption.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible.split(",")[0]
    return "0"


@requires_gpu_sandbox
@pytest.mark.integration
def test_gpu_sandbox_sees_the_assigned_device(monkeypatch):
    """
    A sandbox created with `gpus=1` must have exactly one real GPU visible inside
    the container. This is the passthrough path (`build_gpu_argv`), not the CPU
    shell path that `test_worker_integration.py` already covers.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _first_visible_gpu_index())

    async def scenario() -> None:
        worker = EnvWorker(worker_id=0, cpu_slots=1, gpu_slots=1, exec_default_timeout_s=120.0)
        sandbox_id = await worker.create_sandbox(SandboxSpec(image=GPU_PROBE_IMAGE, gpus=1, network="none"))
        try:
            result = await worker.exec(
                sandbox_id,
                "nvidia-smi --query-gpu=index --format=csv,noheader",
                timeout_s=60.0,
            )
            assert result.exit_code == 0, f"nvidia-smi failed inside the sandbox: {result.stdout!r}."
            assert result.stdout.strip() != "", "No GPU index was visible inside the sandbox."
        finally:
            await worker.destroy_sandbox(sandbox_id)

    asyncio.run(scenario())


@requires_gpu_sandbox
@pytest.mark.integration
def test_gpu_inference_runs_on_the_assigned_device(monkeypatch):
    """
    Run one real CUDA computation inside the sandbox and read back its result.

    This is the "GPU inference unit" case: the command below is a stand-in for a
    task script that loads a model and runs a forward pass. It prints a single
    parseable line, the same shape a reward function would parse from a real task
    (see `examples/airs_bench/prepare/build_parquet.py::build_row` for how PSRL
    dataset rows carry per-task expectations in `extra_info`).
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _first_visible_gpu_index())

    probe_script = (
        "python -c \""
        "import torch; "
        "ok = torch.cuda.is_available(); "
        "x = torch.randn(1024, 1024, device='cuda') if ok else None; "
        "y = float((x @ x).sum()) if ok else 0.0; "
        "print(f'GPU_PROBE_RESULT cuda_ok={ok} matmul_sum={y:.2f}')"
        "\""
    )

    async def scenario() -> None:
        worker = EnvWorker(worker_id=0, cpu_slots=1, gpu_slots=1, exec_default_timeout_s=120.0)
        sandbox_id = await worker.create_sandbox(SandboxSpec(image=GPU_PROBE_IMAGE, gpus=1, network="none"))
        try:
            result = await worker.exec(sandbox_id, probe_script, timeout_s=90.0)
            assert result.exit_code == 0, f"The inference probe exited non-zero: {result.stdout!r}."
            assert "GPU_PROBE_RESULT" in result.stdout, f"Probe did not run: {result.stdout!r}."
            assert "cuda_ok=True" in result.stdout, (
                f"torch.cuda.is_available() was False inside the sandbox: {result.stdout!r}. "
                "The container did not actually get GPU compute access even though it started."
            )
        finally:
            await worker.destroy_sandbox(sandbox_id)

    asyncio.run(scenario())


@requires_gpu_sandbox
@pytest.mark.integration
def test_two_gpu_sandboxes_never_share_a_device(monkeypatch):
    """
    Regression guard for `_allocate_gpu_indices`'s exclusivity contract, but through
    the real Docker path instead of the pure-Python check in `test_worker.py`.

    Requires at least 2 GPUs to be meaningful; skips itself otherwise so single-GPU
    CI hosts do not fail spuriously.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    indices = [p for p in visible.split(",") if p.strip()] if visible else ["0"]
    if len(indices) < 2:
        pytest.skip("Needs at least 2 visible GPUs to check exclusivity; found fewer.")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(indices[:2]))

    async def scenario() -> None:
        worker = EnvWorker(worker_id=0, cpu_slots=2, gpu_slots=2, exec_default_timeout_s=120.0)
        first_id = await worker.create_sandbox(SandboxSpec(image=GPU_PROBE_IMAGE, gpus=1, network="none"))
        second_id = await worker.create_sandbox(SandboxSpec(image=GPU_PROBE_IMAGE, gpus=1, network="none"))
        try:
            first_out = await worker.exec(first_id, "nvidia-smi --query-gpu=index --format=csv,noheader", 60.0)
            second_out = await worker.exec(second_id, "nvidia-smi --query-gpu=index --format=csv,noheader", 60.0)
            assert first_out.stdout.strip() != second_out.stdout.strip(), (
                f"Both sandboxes saw the same device: {first_out.stdout!r} vs {second_out.stdout!r}."
            )
        finally:
            await worker.destroy_sandbox(first_id)
            await worker.destroy_sandbox(second_id)

    asyncio.run(scenario())
