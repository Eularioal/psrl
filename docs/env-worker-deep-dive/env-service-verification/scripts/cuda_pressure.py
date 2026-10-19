"""Controlled real CUDA allocation for an explicitly constructed pressure experiment."""

import argparse
import ctypes as c
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=int, default=50)
    parser.add_argument("--hold", type=float, default=45)
    parser.add_argument("--output", default="/logs/artifacts/gpu-pressure.json")
    args = parser.parse_args()
    result = {
        "started_at": time.time(),
        "requested_bytes": args.gib * 1024**3,
        "hold_seconds": args.hold,
        "scope": "Injected CUDA pressure, not native LAPS demand",
    }
    cuda = c.CDLL("libcuda.so.1")
    context = c.c_void_p()
    pointer = c.c_uint64()

    def call(name, *values):
        code = getattr(cuda, name)(*values)
        if code:
            result.update(
                status="cuda_oom" if code == 2 and name == "cuMemAlloc_v2" else "cuda_error",
                cuda_error=code,
                failed_call=name,
            )
            raise RuntimeError(f"{name} returned CUDA error {code}")

    exit_code = 0
    try:
        call("cuInit", 0)
        count, device = c.c_int(), c.c_int()
        call("cuDeviceGetCount", c.byref(count))
        if count.value != 1:
            raise RuntimeError("Expected exactly one container-visible CUDA device.")
        call("cuDeviceGet", c.byref(device), 0)
        gpu_uuid = c.create_string_buffer(16)
        call("cuDeviceGetUuid_v2", gpu_uuid, device)
        result["gpu_uuid_hex"] = gpu_uuid.raw.hex()
        call("cuCtxCreate_v2", c.byref(context), 0, device)
        free, total = c.c_size_t(), c.c_size_t()
        call("cuMemGetInfo_v2", c.byref(free), c.byref(total))
        result.update(free_before=free.value, total_bytes=total.value)
        call("cuMemAlloc_v2", c.byref(pointer), c.c_size_t(result["requested_bytes"]))
        result["allocated_at"] = time.time()
        call("cuMemsetD32_v2", pointer, c.c_uint(42), c.c_size_t(1))
        call("cuCtxSynchronize")
        answer = c.c_uint()
        call("cuMemcpyDtoH_v2", c.byref(answer), pointer, c.c_size_t(4))
        if answer.value != 42:
            raise RuntimeError("CUDA memory operation produced the wrong value.")
        result["kernel_output"] = answer.value
        time.sleep(args.hold)
        result["status"] = "ok"
    except Exception as exc:
        result.setdefault("status", "probe_error")
        result["error"] = repr(exc)
        exit_code = 42 if result["status"] == "cuda_oom" else 1
    finally:
        if pointer.value:
            cuda.cuMemFree_v2(pointer)
        if context.value:
            cuda.cuCtxDestroy_v2(context)
        result["finished_at"] = time.time()
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(out)
        print(json.dumps(result), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
