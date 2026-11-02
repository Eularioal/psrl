"""Run bounded CUDA work and publish identity for the MPS integration probe."""

import argparse
import ctypes as c
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--mib", type=int, default=256)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60 or not 0 < args.mib <= 512:
        raise ValueError("Probe resource budget exceeded.")
    cuda = c.CDLL("libcuda.so.1")
    record = {
        "name": args.name,
        "pid": os.getpid(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "launches": 0,
        "mib": args.mib,
    }
    context, module, kernel = c.c_void_p(), c.c_void_p(), c.c_void_p()
    pointer, device = c.c_uint64(), c.c_int()

    def publish():
        path = args.output / f"{args.name}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record))
        temporary.replace(path)

    def call(name, *values):
        code = getattr(cuda, name)(*values)
        if code:
            record.update(failed_call=name, cuda_error=code)
            raise RuntimeError(f"{name} returned CUDA error {code}.")

    try:
        call("cuInit", 0)
        call("cuDeviceGet", c.byref(device), 0)
        call("cuCtxCreate_v2", c.byref(context), 0, device)
        sms = c.c_int()
        call("cuDeviceGetAttribute", c.byref(sms), 16, device)
        record["visible_sms"] = sms.value
        call("cuMemAlloc_v2", c.byref(pointer), c.c_size_t(args.mib * 1024**2))
        ptx = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry busy(.param .u64 output) {
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
        call("cuModuleGetFunction", c.byref(kernel), module, b"busy")
        parameters = (c.c_void_p * 1)(c.addressof(pointer))
        started = last_publish = time.monotonic()
        record.update(status="running", started_at=time.time())
        publish()
        while time.monotonic() - started < args.seconds:
            if (args.output / f"stop-{args.name}").exists():
                break
            call("cuLaunchKernel", kernel, 1, 1, 1, 1, 1, 1, 0, None, parameters, None)
            call("cuCtxSynchronize")
            record["launches"] += 1
            if time.monotonic() - last_publish > 0.2:
                publish()
                last_publish = time.monotonic()
        answer = c.c_uint32()
        call("cuMemcpyDtoH_v2", c.byref(answer), pointer, c.c_size_t(4))
        if answer.value != 42:
            raise RuntimeError("CUDA kernel output does not match.")
        record.update(status="ok", answer=answer.value, elapsed_s=time.monotonic() - started)
    except Exception as exc:
        record.update(status="error", error=repr(exc))
    finally:
        if pointer.value:
            record["free_code"] = cuda.cuMemFree_v2(pointer)
        if context.value:
            record["destroy_code"] = cuda.cuCtxDestroy_v2(context)
        publish()
        print(json.dumps(record), flush=True)
    raise SystemExit(0 if record["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
