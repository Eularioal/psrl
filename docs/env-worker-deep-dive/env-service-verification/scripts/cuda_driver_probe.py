"""Run a PTX kernel through the CUDA driver API without a CUDA Python package."""

import ctypes as c
import json
import os
import subprocess
import time
from pathlib import Path

cuda = c.CDLL("libcuda.so.1")


def call(name, *args):
    code = getattr(cuda, name)(*args)
    if code:
        raise RuntimeError(f"{name} returned CUDA error {code}")


call("cuInit", 0)
count = c.c_int()
call("cuDeviceGetCount", c.byref(count))
assert count.value == 1, f"Expected one accessible GPU, got {count.value}."
device = c.c_int()
call("cuDeviceGet", c.byref(device), 0)
name = c.create_string_buffer(128)
call("cuDeviceGetName", name, 128, device)
gpu_uuid = c.create_string_buffer(16)
call("cuDeviceGetUuid_v2", gpu_uuid, device)
context = c.c_void_p()
call("cuCtxCreate_v2", c.byref(context), 0, device)
memory = c.c_uint64()
call("cuMemAlloc_v2", c.byref(memory), c.c_size_t(64 * 1024 * 1024))
ptx = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry write_answer(.param .u64 output) {
 .reg .u64 %rd1;
 .reg .u32 %r1;
 ld.param.u64 %rd1, [output];
 mov.u32 %r1, 42;
 st.global.u32 [%rd1], %r1;
 ret;
}
"""
module, kernel = c.c_void_p(), c.c_void_p()
call("cuModuleLoadData", c.byref(module), c.c_char_p(ptx))
call("cuModuleGetFunction", c.byref(kernel), module, b"write_answer")
params = (c.c_void_p * 1)(c.addressof(memory))
call("cuLaunchKernel", kernel, 1, 1, 1, 1, 1, 1, 0, None, params, None)
call("cuCtxSynchronize")
answer = c.c_uint32()
call("cuMemcpyDtoH_v2", c.byref(answer), memory, c.c_size_t(4))
assert answer.value == 42, f"Wrong kernel output {answer.value}."
result = dict(
    device_count=count.value,
    device_name=name.value.decode(),
    uuid_hex=gpu_uuid.raw.hex(),
    kernel_output=answer.value,
    allocation_bytes=64 * 1024 * 1024,
    visible_devices=sorted(str(p) for p in Path("/dev").glob("nvidia[0-9]*")),
    nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True),
)
print(json.dumps(result), flush=True)
Path("/logs/artifacts/cuda-probe.json").write_text(json.dumps(result, indent=2))
hold_s = float(os.environ.get("PROBE_HOLD_SECONDS", "10"))
if os.environ.get("PROBE_BUSY") == "1":
    busy_ptx = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry busy(.param .u64 output) {
 .reg .pred %p;
 .reg .u32 %r, %block, %thread, %index;
 .reg .u64 %base, %offset, %address;
 ld.param.u64 %base, [output];
 mov.u32 %block, %ctaid.x;
 mov.u32 %thread, %tid.x;
 mad.lo.u32 %index, %block, 256, %thread;
 mul.wide.u32 %offset, %index, 4;
 add.u64 %address, %base, %offset;
 mov.u32 %r, 0;
loop:
 add.u32 %r, %r, 1;
 st.volatile.global.u32 [%address], %r;
 setp.lt.u32 %p, %r, 10000;
 @%p bra loop;
 ret;
}
"""
    busy_module, busy_kernel = c.c_void_p(), c.c_void_p()
    call("cuModuleLoadData", c.byref(busy_module), c.c_char_p(busy_ptx))
    call("cuModuleGetFunction", c.byref(busy_kernel), busy_module, b"busy")
    until = time.monotonic() + hold_s
    while time.monotonic() < until:
        call("cuLaunchKernel", busy_kernel, 256, 1, 1, 256, 1, 1, 0, None, params, None)
        call("cuCtxSynchronize")
else:
    time.sleep(hold_s)
call("cuMemFree_v2", memory)
call("cuCtxDestroy_v2", context)
