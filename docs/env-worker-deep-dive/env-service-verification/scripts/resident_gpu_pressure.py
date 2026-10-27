"""
Hold bounded, owned CUDA memory outside task containers for a pressure comparison.
"""

import argparse
import ctypes as c
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class ResidentPressure:
    """
    Require confirmed allocation and stop only the child created by this controller.
    """

    def __init__(self, gpu, gib, output, lifetime):
        self.gpu, self.gib, self.output, self.lifetime = gpu, gib, Path(output), lifetime
        self.process = None
        self.log = None

    def start(self):
        ready = self.output / "resident-ready.json"
        if ready.exists():
            raise RuntimeError("Resident evidence already exists. Use a fresh arm directory.")
        self.log = (self.output / "resident.log").open("w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu))
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--gib",
                str(self.gib),
                "--lifetime",
                str(self.lifetime),
                "--parent",
                str(os.getpid()),
                "--ready",
                str(ready),
            ],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Resident allocation failed. Inspect resident.log.")
            if ready.exists():
                return json.loads(ready.read_text())
            time.sleep(0.1)
        raise TimeoutError("Resident allocation did not become ready within 45 seconds.")

    def stop(self):
        if self.process is None:
            return {"started": False}
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log:
            self.log.close()
        return {"pid": self.process.pid, "exit_code": self.process.returncode}


def hold(args):
    cuda = c.CDLL("libcuda.so.1")
    context, pointer = c.c_void_p(), c.c_uint64()
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    def call(name, *values):
        code = getattr(cuda, name)(*values)
        if code:
            raise RuntimeError(f"{name} returned CUDA error {code}.")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        call("cuInit", 0)
        device, count = c.c_int(), c.c_int()
        call("cuDeviceGetCount", c.byref(count))
        if count.value != 1:
            raise RuntimeError("Resident pressure requires exactly one visible GPU.")
        call("cuDeviceGet", c.byref(device), 0)
        gpu_uuid = c.create_string_buffer(16)
        call("cuDeviceGetUuid_v2", gpu_uuid, device)
        call("cuCtxCreate_v2", c.byref(context), 0, device)
        call("cuMemAlloc_v2", c.byref(pointer), c.c_size_t(args.gib * 1024**3))
        call("cuMemsetD32_v2", pointer, c.c_uint(42), c.c_size_t(1))
        call("cuCtxSynchronize")
        value = c.c_uint()
        call("cuMemcpyDtoH_v2", c.byref(value), pointer, c.c_size_t(4))
        if value.value != 42:
            raise RuntimeError("Resident CUDA readback failed.")
        ready = {
            "pid": os.getpid(),
            "parent": args.parent,
            "gpu_uuid_hex": gpu_uuid.raw.hex(),
            "allocated_bytes": args.gib * 1024**3,
            "ready_at": time.time(),
            "lifetime_seconds": args.lifetime,
            "readback": value.value,
        }
        temporary = args.ready.with_suffix(".tmp")
        temporary.write_text(json.dumps(ready, indent=2))
        temporary.replace(args.ready)
        deadline = time.monotonic() + args.lifetime
        while not stopped and time.monotonic() < deadline and os.getppid() == args.parent:
            time.sleep(0.2)
    finally:
        if pointer.value:
            cuda.cuMemFree_v2(pointer)
        if context.value:
            cuda.cuCtxDestroy_v2(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=int, required=True)
    parser.add_argument("--lifetime", type=float, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    if args.gib <= 0 or args.lifetime <= 0:
        parser.error("Allocation size and lifetime must be positive.")
    hold(args)
