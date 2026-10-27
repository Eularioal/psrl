"""Keep a CPU parent alive while a separately cancellable child performs CUDA work."""

import ctypes as c
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def child(root):
    cuda = c.CDLL("libcuda.so.1")

    def call(name, *args):
        result = getattr(cuda, name)(*args)
        if result:
            raise RuntimeError(f"{name} failed with CUDA error {result}.")

    c.CDLL(None).prctl(15, b"same-gpu-name", 0, 0, 0)
    device, memory = c.c_int(), c.c_uint64()
    context, module, kernel = c.c_void_p(), c.c_void_p(), c.c_void_p()
    try:
        call("cuInit", 0)
        call("cuDeviceGet", c.byref(device), 0)
        call("cuCtxCreate_v2", c.byref(context), 0, device)
        call("cuMemAlloc_v2", c.byref(memory), c.c_size_t(64 * 1024**2))
        ptx = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry work(.param .u64 output) {
 .reg .u64 %address;
 .reg .u32 %start, %now, %elapsed;
 .reg .pred %again;
 ld.param.u64 %address, [output];
 mov.u32 %start, %clock;
loop:
 mov.u32 %now, %clock;
 sub.u32 %elapsed, %now, %start;
 setp.lt.u32 %again, %elapsed, 1000000;
 @%again bra loop;
 st.global.u32 [%address], 42;
 ret;
}
"""
        call("cuModuleLoadData", c.byref(module), c.c_char_p(ptx))
        call("cuModuleGetFunction", c.byref(kernel), module, b"work")
        params = (c.c_void_p * 1)(c.addressof(memory))
        call("cuLaunchKernel", kernel, 512, 1, 1, 128, 1, 1, 0, None, params, None)
        call("cuCtxSynchronize")
        answer = c.c_uint32()
        call("cuMemcpyDtoH_v2", c.byref(answer), memory, c.c_size_t(4))
        if answer.value != 42:
            raise RuntimeError("The CUDA output must be 42.")
        save(root / "cuda.json", {"pid": os.getpid(), "output": answer.value, "name": "same-gpu-name"})
        deadline = time.monotonic() + 20
        while not (root / "finish").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("The CUDA child exceeded its time budget.")
            call("cuLaunchKernel", kernel, 512, 1, 1, 128, 1, 1, 0, None, params, None)
            call("cuCtxSynchronize")
    finally:
        if memory.value:
            cuda.cuMemFree_v2(memory)
        if context.value:
            cuda.cuCtxDestroy_v2(context)


def main():
    root = Path(sys.argv[1])
    if len(sys.argv) == 3:
        child(root)
        return
    save(root / "parent.json", {"pid": os.getpid()})
    deadline = time.monotonic() + 55
    while not (root / "start").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("The parent did not receive a GPU grant.")
        time.sleep(0.05)
    process = subprocess.Popen([sys.executable, __file__, str(root), "child"])
    try:
        while process.poll() is None:
            if (root / "cancel").exists():
                process.send_signal(signal.SIGTERM)
            if time.monotonic() >= deadline:
                raise TimeoutError("The workload exceeded its time budget.")
            time.sleep(0.05)
        save(root / "child-done.json", {"exit_code": process.returncode})
        while not (root / "stop-parent").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("The parent cleanup exceeded its time budget.")
            time.sleep(0.05)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


if __name__ == "__main__":
    main()
