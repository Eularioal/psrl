"""Time the same allocation workload with monotonic clocks and per-call diagnostics."""

import argparse
import ctypes as c
import json
import time
from pathlib import Path


def cpu_state():
    """Read container CPU accounting without changing resource limits."""
    result = {}
    for path in (
        "/sys/fs/cgroup/cpu/cpu.stat",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.stat",
        "/sys/fs/cgroup/cpu.stat",
        "/proc/self/schedstat",
    ):
        try:
            result[path] = Path(path).read_text().strip()
        except OSError:
            pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=int, default=1)
    parser.add_argument("--hold", type=float, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "started_at": time.time(),
        "started_mono": time.monotonic(),
        "requested_bytes": args.gib * 1024**3,
        "hold_seconds": args.hold,
        "scope": "Instrumented CUDA probe with unchanged work, no model",
        "calls": [],
        "cpu_before": cpu_state(),
    }

    def measured(name, fn, *values):
        span = {"name": name, "started_at": time.time(), "started_mono": time.monotonic()}
        cpu = time.process_time()
        try:
            return fn(*values)
        finally:
            span.update(duration_s=time.monotonic() - span["started_mono"], cpu_s=time.process_time() - cpu)
            result["calls"].append(span)

    cuda = measured("load_libcuda", c.CDLL, "libcuda.so.1")
    context = c.c_void_p()
    pointer = c.c_uint64()

    def call(name, *values):
        code = measured(name, getattr(cuda, name), *values)
        if code:
            result.update(
                status="cuda_oom" if code == 2 and name == "cuMemAlloc_v2" else "cuda_error",
                cuda_error=code,
                failed_call=name,
            )
            raise RuntimeError(f"{name} returned CUDA error {code!r}.")

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
        result["allocated_mono"] = time.monotonic()
        call("cuMemsetD32_v2", pointer, c.c_uint(42), c.c_size_t(1))
        call("cuCtxSynchronize")
        answer = c.c_uint()
        call("cuMemcpyDtoH_v2", c.byref(answer), pointer, c.c_size_t(4))
        if answer.value != 42:
            raise RuntimeError("CUDA memory operation produced the wrong value.")
        result["kernel_output"] = answer.value
        measured("hold_sleep", time.sleep, args.hold)
        result["status"] = "ok"
    except Exception as exc:
        result.setdefault("status", "probe_error")
        result["error"] = repr(exc)
        exit_code = 42 if result["status"] == "cuda_oom" else 1
    finally:
        if pointer.value:
            measured("cuMemFree_v2", cuda.cuMemFree_v2, pointer)
        if context.value:
            measured("cuCtxDestroy_v2", cuda.cuCtxDestroy_v2, context)
        result.update(finished_at=time.time(), finished_mono=time.monotonic(), cpu_after=cpu_state())
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(out)
        print(json.dumps(result), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
