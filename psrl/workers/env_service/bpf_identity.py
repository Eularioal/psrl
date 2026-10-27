"""Maintain trusted process identities from registered cgroup v1 kernel events."""

import ctypes as c
import ctypes.util
import errno
import os
import secrets
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from psrl.workers.env_service.bpf_layout import KernelTypes


class Record(c.Structure):
    _fields_ = [("birth_ns", c.c_uint64), ("cgroup", c.c_uint64), ("cookie", c.c_uint64)]


def library():
    """Use libbpf for ELF relocation, verification and map operations."""
    name = ctypes.util.find_library("bpf")
    if not name:
        raise RuntimeError("The eBPF GPU identity backend requires libbpf.")
    lib = c.CDLL(name, use_errno=True)
    signatures = {
        "bpf_object__open_file": (c.c_void_p, [c.c_char_p, c.c_void_p]),
        "libbpf_get_error": (c.c_long, [c.c_void_p]),
        "bpf_object__load": (c.c_int, [c.c_void_p]),
        "bpf_object__close": (None, [c.c_void_p]),
        "bpf_object__find_map_fd_by_name": (c.c_int, [c.c_void_p, c.c_char_p]),
        "bpf_object__find_program_by_name": (c.c_void_p, [c.c_void_p, c.c_char_p]),
        "bpf_program__fd": (c.c_int, [c.c_void_p]),
        "bpf_raw_tracepoint_open": (c.c_int, [c.c_char_p, c.c_int]),
        "bpf_map_lookup_elem": (c.c_int, [c.c_int, c.c_void_p, c.c_void_p]),
        "bpf_map_update_elem": (c.c_int, [c.c_int, c.c_void_p, c.c_void_p, c.c_uint64]),
        "bpf_map_delete_elem": (c.c_int, [c.c_int, c.c_void_p]),
    }
    for symbol, (result, arguments) in signatures.items():
        function = getattr(lib, symbol)
        function.restype, function.argtypes = result, arguments
    return lib


class BpfIdentity:
    """Own one node collector and disjoint command registrations until explicit close."""

    name = "ebpf"

    def __init__(self):
        if c.sizeof(c.c_void_p) != 8 or os.uname().machine not in {"x86_64", "aarch64"}:
            raise RuntimeError("GPU eBPF identity supports native 64 bit x86_64 or aarch64 Linux.")
        self.lock = threading.RLock()
        self.lib = library()
        self.object = None
        self.links = []
        self.bound = {}
        self.closed = False
        try:
            self._load()
        except BaseException:
            self.close()
            raise

    def _load(self):
        defines = KernelTypes().identity_defines()
        defines.update(IOCTL_NR=16, GETPID_NR=39) if os.uname().machine == "x86_64" else defines.update(
            IOCTL_NR=29, GETPID_NR=172
        )
        with tempfile.TemporaryDirectory(prefix="psrl-identity-", dir="/tmp") as directory:
            root = Path(directory)
            (root / "identity_offsets.h").write_text(
                "".join(f"#define {key} {value}\n" for key, value in defines.items())
            )
            build = subprocess.run(
                [
                    "clang",
                    "-target",
                    "bpf",
                    "-mcpu=v1",
                    "-O2",
                    "-g",
                    "-Wall",
                    "-Werror",
                    "-I",
                    directory,
                    "-c",
                    str(Path(__file__).with_name("bpf_identity.bpf.c")),
                    "-o",
                    str(root / "identity.o"),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if build.returncode:
                raise RuntimeError(f"GPU identity BPF compilation failed: {build.stderr[-4000:]}.")
            obj = self.lib.bpf_object__open_file(os.fsencode(root / "identity.o"), None)
            if not obj or self.lib.libbpf_get_error(obj):
                raise RuntimeError("libbpf could not open the GPU identity object.")
            self.object = obj
            if self.lib.bpf_object__load(obj) < 0:
                raise RuntimeError(f"GPU identity BPF loading failed with errno {c.get_errno()}.")
        self.maps = {}
        for name in ("registrations", "identities", "health"):
            fd = self.lib.bpf_object__find_map_fd_by_name(self.object, name.encode())
            if fd < 0:
                raise RuntimeError(f"GPU identity BPF map {name!r} is unavailable.")
            self.maps[name] = fd
        events = {
            "on_syscall": "sys_enter",
            "on_fork": "sched_process_fork",
            "on_exec": "sched_process_exec",
            "on_exit": "sched_process_exit",
            "on_move": "cgroup_attach_task",
            "on_transfer": "cgroup_transfer_tasks",
        }
        for function, event in events.items():
            program = self.lib.bpf_object__find_program_by_name(self.object, function.encode())
            if not program:
                raise RuntimeError(f"GPU identity BPF program {function!r} is unavailable.")
            fd = self.lib.bpf_raw_tracepoint_open(event.encode(), self.lib.bpf_program__fd(program))
            if fd < 0:
                raise RuntimeError(f"GPU identity tracing cannot attach {event!r}, errno {c.get_errno()}.")
            self.links.append(fd)

    def lookup(self, name, key, value):
        result = self.lib.bpf_map_lookup_elem(self.maps[name], c.byref(key), c.byref(value))
        if result < 0:
            error = c.get_errno()
            if error == errno.ENOENT:
                return None
            raise OSError(error, os.strerror(error))
        return value

    def check_support(self):
        with self.lock:
            if self.closed or len(self.links) != 6:
                raise RuntimeError("GPU identity collector is closed or not fully attached.")
            value = self.lookup("health", c.c_uint32(0), c.c_uint64())
            if value is None or value.value:
                raise RuntimeError("GPU identity collector lost records. Reservations must remain charged.")

    def bind(self, fence):
        with self.lock:
            self.check_support()
            path = fence.path.resolve(strict=True)
            if any(
                path == item.path or path in item.path.parents or item.path in path.parents
                for item in self.bound.values()
            ):
                raise RuntimeError("GPU identity registrations must have disjoint cgroup subtrees.")
            owner = BpfOwner(self, path)
            key, value = c.c_uint64(owner.inode), c.c_uint64(owner.cookie)
            if self.lib.bpf_map_update_elem(self.maps["registrations"], c.byref(key), c.byref(value), 1) < 0:
                os.close(owner.directory_fd)
                raise RuntimeError(f"GPU cgroup registration failed with errno {c.get_errno()}.")
            self.bound[owner.cookie] = owner
            return owner

    def close(self):
        with self.lock:
            if self.closed:
                return
            for owner in tuple(self.bound.values()):
                owner.close()
            for fd in reversed(self.links):
                os.close(fd)
            self.links.clear()
            if self.object is not None:
                self.lib.bpf_object__close(self.object)
                self.object = None
            self.closed = True


class BpfOwner:
    """Bind a command generation to a pinned device cgroup directory."""

    def __init__(self, backend, path):
        self.backend, self.path = backend, path
        self.directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.inode = os.fstat(self.directory_fd).st_ino
        self.cookie = secrets.randbits(63) or 1
        self.closed = False

    def owners(self, pids):
        with self.backend.lock:
            self.backend.check_support()
            if self.closed or self.path.stat().st_ino != self.inode:
                raise RuntimeError("GPU task cgroup registration is no longer valid.")
            current = self.backend.lookup("registrations", c.c_uint64(self.inode), c.c_uint64())
            if current is None or current.value != self.cookie:
                raise RuntimeError("GPU task cgroup registration has been replaced.")
            offset_us = int((time.time() - time.monotonic()) * 1e6)
            result = {}
            for pid in pids:
                record = self.backend.lookup("identities", c.c_uint32(pid), Record())
                if record is not None and record.cookie == self.cookie and record.cgroup == self.inode:
                    result[pid] = (record.birth_ns, offset_us + record.birth_ns // 1000)
            self.backend.check_support()
            return result

    def close(self):
        with self.backend.lock:
            if self.closed:
                return
            key = c.c_uint64(self.inode)
            current = self.backend.lookup("registrations", key, c.c_uint64())
            if current is not None and current.value == self.cookie:
                if self.backend.lib.bpf_map_delete_elem(self.backend.maps["registrations"], c.byref(key)) < 0:
                    raise OSError(c.get_errno(), "Cannot remove the GPU cgroup registration.")
            self.backend.bound.pop(self.cookie, None)
            os.close(self.directory_fd)
            self.closed = True
