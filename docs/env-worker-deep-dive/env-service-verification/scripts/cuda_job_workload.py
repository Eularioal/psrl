"""Run bounded real CUDA kernels for process attribution and job overlap checks."""

import argparse
import ctypes as c
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=6)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        raise ValueError("CUDA probe duration must be at most 60 seconds.")
    cuda = c.CDLL("libcuda.so.1")

    def call(name, *values):
        code = getattr(cuda, name)(*values)
        if code:
            raise RuntimeError(f"{name} returned CUDA error {code}.")

    context, module, kernel = c.c_void_p(), c.c_void_p(), c.c_void_p()
    pointer, device = c.c_uint64(), c.c_int()
    try:
        call("cuInit", 0)
        call("cuDeviceGet", c.byref(device), 0)
        call("cuCtxCreate_v2", c.byref(context), 0, device)
        call("cuMemAlloc_v2", c.byref(pointer), c.c_size_t(256 * 1024**2))
        ptx = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry busy(.param .u64 output) {
 .reg .u64 %base, %offset, %address;
 .reg .u32 %start, %now, %elapsed, %block, %thread, %index;
 .reg .pred %again;
 ld.param.u64 %base, [output];
 mov.u32 %block, %ctaid.x;
 mov.u32 %thread, %tid.x;
 mad.lo.u32 %index, %block, 256, %thread;
 mul.wide.u32 %offset, %index, 4;
 add.u64 %address, %base, %offset;
 mov.u32 %start, %clock;
loop:
 mov.u32 %now, %clock;
 sub.u32 %elapsed, %now, %start;
 setp.lt.u32 %again, %elapsed, 2000000;
 @%again bra loop;
 st.global.u32 [%address], 42;
 ret;
}
"""
        call("cuModuleLoadData", c.byref(module), c.c_char_p(ptx))
        call("cuModuleGetFunction", c.byref(kernel), module, b"busy")
        parameters = (c.c_void_p * 1)(c.addressof(pointer))
        started = time.monotonic()
        count = 0
        while time.monotonic() - started < args.seconds:
            call("cuLaunchKernel", kernel, 512, 1, 1, 256, 1, 1, 0, None, parameters, None)
            call("cuCtxSynchronize")
            count += 1
        answer = c.c_uint32()
        call("cuMemcpyDtoH_v2", c.byref(answer), pointer, c.c_size_t(4))
        assert answer.value == 42, "CUDA kernel output must match the expected value."
        print(json.dumps({"kernel_output": answer.value, "launches": count, "seconds": time.monotonic() - started}))
    finally:
        if pointer.value:
            cuda.cuMemFree_v2(pointer)
        if context.value:
            cuda.cuCtxDestroy_v2(context)


if __name__ == "__main__":
    main()
