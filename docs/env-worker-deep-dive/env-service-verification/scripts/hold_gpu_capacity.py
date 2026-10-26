"""Hold a bounded CUDA allocation to simulate a smaller available GPU memory capacity."""

import argparse
import ctypes as c
import json
import os
import signal
import time
from pathlib import Path

from science_router_control_ab import gpu_snapshot


def hold(args):
    before = gpu_snapshot(args.gpu)
    if before["pids"] or before["used_mib"] > 64:
        raise RuntimeError("The selected GPU is occupied before the pressure allocation.")
    os.environ["CUDA_VISIBLE_DEVICES"] = before["uuid"]
    cuda = c.CDLL("libcuda.so.1")
    context, pointer = c.c_void_p(), c.c_uint64()
    record = {
        "pid": os.getpid(),
        "birth": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        "gpu_uuid": before["uuid"],
        "started_at": time.time(),
        "leave_mib": args.leave_mib,
        "max_hold_s": args.max_hold_s,
        "scope": "Explicitly injected memory occupancy, not native training demand",
    }
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    def call(name, *values):
        code = getattr(cuda, name)(*values)
        if code:
            raise RuntimeError(f"CUDA call {name!r} failed with code {code}.")

    def save():
        temporary = args.status.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(args.status)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        call("cuInit", 0)
        device, count = c.c_int(), c.c_int()
        call("cuDeviceGetCount", c.byref(count))
        if count.value != 1:
            raise RuntimeError("The holder must see exactly one CUDA device.")
        call("cuDeviceGet", c.byref(device), 0)
        call("cuCtxCreate_v2", c.byref(context), 0, device)
        free, total = c.c_size_t(), c.c_size_t()
        call("cuMemGetInfo_v2", c.byref(free), c.byref(total))
        allocation = (free.value - args.leave_mib * 1024**2) // (2 * 1024**2) * (2 * 1024**2)
        if allocation <= 0:
            raise RuntimeError("The requested free capacity does not require a pressure allocation.")
        call("cuMemAlloc_v2", c.byref(pointer), c.c_size_t(allocation))
        call("cuMemsetD32_v2", pointer, c.c_uint(42), c.c_size_t(1))
        call("cuCtxSynchronize")
        answer = c.c_uint()
        call("cuMemcpyDtoH_v2", c.byref(answer), pointer, c.c_size_t(4))
        if answer.value != 42:
            raise RuntimeError("The holder CUDA memory check failed.")
        call("cuMemGetInfo_v2", c.byref(free), c.byref(total))
        current = gpu_snapshot(args.gpu)
        if len(current["pids"]) != 1 or not args.leave_mib <= free.value / 1024**2 < args.leave_mib + 4:
            raise RuntimeError("The resulting GPU occupancy does not match the pressure design.")
        record.update(
            status="ready",
            ready_at=time.time(),
            allocated_mib=allocation / 1024**2,
            free_mib=free.value / 1024**2,
            used_mib=current["used_mib"],
            gpu_pid=current["pids"][0],
            kernel_output=answer.value,
        )
        save()
        deadline = time.monotonic() + args.max_hold_s
        while not stopping and time.monotonic() < deadline and os.getppid() == args.parent_pid:
            time.sleep(0.25)
    except BaseException as exc:
        record.update(status="error", error=repr(exc))
        raise
    finally:
        if pointer.value:
            cuda.cuMemFree_v2(pointer)
        if context.value:
            cuda.cuCtxDestroy_v2(context)
        record.update(status="released" if record.get("status") == "ready" else "error", finished_at=time.time())
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--leave-mib", type=int, default=1536)
    parser.add_argument("--max-hold-s", type=int, default=1800)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    if not 768 <= args.leave_mib or not 0 < args.max_hold_s <= 1800:
        raise ValueError("The holder requires at least 768 MiB free and at most 1800 seconds.")
    hold(args)
