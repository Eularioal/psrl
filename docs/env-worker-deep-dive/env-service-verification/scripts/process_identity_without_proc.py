"""Check a child pidfd without reading procfs or starting a GPU workload."""

import argparse
import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import uuid
from pathlib import Path


def pidfd_syscall(number, *arguments):
    """Support Python without pidfd wrappers on the two checked Linux ABIs."""
    if os.uname().machine not in {"x86_64", "aarch64"} or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OSError("The syscall fallback requires native x86_64 or aarch64 Linux.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def effective_capabilities():
    """Read capability bits using capget without loading a BPF program."""

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]

    libc = ctypes.CDLL(None, use_errno=True)
    header, data = Header(0x20080522, 0), (Data * 2)()
    if libc.capget(ctypes.byref(header), ctypes.byref(data)) != 0:
        return {"error_errno": ctypes.get_errno()}
    mask = data[0].effective | data[1].effective << 32
    names = {"SYS_ADMIN": 21, "SYS_PTRACE": 19, "PERFMON": 38, "BPF": 39}
    return {name: bool(mask & (1 << bit)) for name, bit in names.items()}


def probe():
    report = {
        "scope": "Real CPU child and pidfd. No procfs reads, BPF attachment, Docker or CUDA workload.",
        "kernel": os.uname().release,
        "python": sys.version,
        "pidfd_open_backend": "python" if hasattr(os, "pidfd_open") else "linux_syscall",
        "supervisor_pid": os.getpid(),
        "job_id": uuid.uuid4().hex,
        "effective_capabilities": effective_capabilities(),
        "host_nvml_pid_mapping_verified": False,
    }
    descriptor = None
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"], stdin=subprocess.PIPE)
    report["child_local_pid"] = child.pid
    try:
        # [1] The blocked child stays alive while its supervisor obtains a stable handle.
        descriptor = os.pidfd_open(child.pid, 0) if hasattr(os, "pidfd_open") else pidfd_syscall(434, child.pid, 0)
        report["pidfd_local_descriptor_number"] = descriptor
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        report["exit_event_while_alive"] = bool(poller.poll(0))
        # [2] Release only this child and wait for its exit through the existing parent.
        child.stdin.write(b"x")
        child.stdin.flush()
        report["child_exit_code"] = child.wait(timeout=5)
        report["exit_event_after_exit"] = bool(poller.poll(1000))
        try:
            if hasattr(signal, "pidfd_send_signal"):
                signal.pidfd_send_signal(descriptor, 0)
            else:
                pidfd_syscall(424, descriptor, 0, ctypes.c_void_p(), 0)
            report["exited_handle_rejects_signal"] = False
        except ProcessLookupError:
            report["exited_handle_rejects_signal"] = True
        assert not report["exit_event_while_alive"], "The child must remain alive until released."
        assert report["exit_event_after_exit"], "The handle must report child termination."
        assert report["exited_handle_rejects_signal"], "The exited handle must not target another process."
        report["status"] = "passed"
    except (OSError, AttributeError, AssertionError, subprocess.TimeoutExpired) as exc:
        report["status"] = "unavailable_or_failed"
        report["error"] = repr(exc)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        if descriptor is not None:
            os.close(descriptor)
        report["child_reaped"] = True
        report["pidfd_closed"] = descriptor is not None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered, end="")
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
