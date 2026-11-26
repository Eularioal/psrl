"""Check CUDA copy completion during an independent compute process without MPS."""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


def compute(ready, go, finished, result):
    """Run a fixed amount of matrix work in a separate CUDA context."""
    import torch

    torch.set_num_threads(2)
    a = torch.full((8192, 8192), 0.001, dtype=torch.float16, device="cuda")
    b = torch.full_like(a, 0.002)
    c = torch.empty_like(a)
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    ready.set()
    if not go.wait(30):
        raise TimeoutError("The copy process did not start the experiment.")
    started = time.time()
    for _ in range(512):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    ended = time.time()
    result.put(
        {
            "started_at": started,
            "finished_at": ended,
            "matmuls": 512,
            "element": c[0, 0].item(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "pid": os.getpid(),
        }
    )
    finished.set()


def main(output):
    import torch
    from psrl.workers.env_service.probe import NvmlProbe

    torch.set_num_threads(2)
    device = int(os.environ["CUDA_VISIBLE_DEVICES"])
    before = NvmlProbe().read_gpu_snapshot([device])
    if any(v["pids"] or v["used_mb"] > 64 for v in before.values()):
        raise RuntimeError("The requested GPU is occupied.")
    record = {"before": before, "mode": "separate processes without MPS", "scope": "CUDA copy, not PSRL NIXL sync"}
    context = mp.get_context("spawn")
    ready, go, finished = (context.Event() for _ in range(3))
    queue = context.Queue()
    child = context.Process(target=compute, args=(ready, go, finished, queue))
    child.start()
    try:
        if not ready.wait(90):
            raise TimeoutError("The compute process did not initialize.")
        host = [torch.full((32 * 1024**2,), i / 32, dtype=torch.float16, pin_memory=True) for i in range(32)]
        weights = [torch.empty_like(block, device="cuda") for block in host]
        torch.cuda.synchronize()
        go.set()
        time.sleep(0.1)
        started = time.time()
        for target, block in zip(weights, host):
            target.copy_(block, non_blocking=True)
        torch.cuda.synchronize()
        copied = time.time()
        active = not finished.is_set()
        valid = all(torch.equal(target.cpu(), block) for target, block in zip(weights, host))
        record.update(
            copy_started_at=started,
            copy_finished_at=copied,
            copy_s=copied - started,
            copied_bytes=sum(b.numel() * b.element_size() for b in host),
            compute_still_active_after_copy=active,
            weights_exact=valid,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        child.join(40)
        if child.is_alive() or child.exitcode != 0:
            raise RuntimeError("The bounded compute process did not exit successfully.")
        record["compute"] = queue.get(timeout=3)
        record["overlap_observed"] = (
            record["compute"]["started_at"] < started < copied < record["compute"]["finished_at"]
        )
        record["status"] = "complete" if valid else "failed"
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)
        if child.is_alive():
            child.kill()
            child.join(5)
        output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
