"""Validate MPS client identities before cooperative managed cancellation."""

import math
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


class MpsControlChannel:
    """Serialize administrative connections to one daemon without blocking CUDA work."""

    def __init__(self, container: str, timeout_s: float = 10):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("MPS control timeout must be finite and positive.")
        self.container = container
        self.timeout_s = timeout_s
        self._lock = threading.Lock()

    def __call__(self, command: str) -> str:
        deadline = time.monotonic() + self.timeout_s
        if not self._lock.acquire(timeout=self.timeout_s):
            raise TimeoutError(f"MPS control queue deadline expired for {command!r} on {self.container!r}.")
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"MPS control deadline expired for {command!r} on {self.container!r}.")
            result = subprocess.run(
                ["docker", "exec", "-i", self.container, "nvidia-cuda-mps-control"],
                input=command + "\n",
                capture_output=True,
                text=True,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"MPS control execution deadline expired for {command!r} on {self.container!r}."
            ) from exc
        finally:
            self._lock.release()
        if result.returncode:
            raise RuntimeError(
                f"MPS control {command!r} failed on {self.container!r} with exit code {result.returncode!r}. "
                f"stdout={result.stdout!r}, stderr={result.stderr!r}."
            )
        return result.stdout.strip()


@dataclass(frozen=True)
class MpsClient:
    pid: int
    client_id: int
    server_pid: int
    device: str
    namespace: int


def parse_clients(output):
    """Reject incomplete control views instead of interpreting PID zero as an owner."""
    clients = []
    for line in output.splitlines():
        fields = line.split()
        if not fields or fields[0] == "PID":
            continue
        if len(fields) < 6:
            raise RuntimeError("Incomplete MPS client table.")
        try:
            client = MpsClient(int(fields[0]), int(fields[1]), int(fields[2]), fields[3], int(fields[4]))
        except ValueError as exc:
            raise RuntimeError(f"MPS client identity is unavailable: {fields[:5]!r}.") from exc
        if min(client.pid, client.server_pid, client.namespace) <= 0:
            raise RuntimeError(f"MPS client identity is unavailable: {fields[:5]!r}.")
        clients.append(client)
    return clients


def local_identity(pid, root=Path("/proc")):
    """Read birth ticks and namespace in the same PID view as the control service."""
    proc = root / str(pid)
    birth = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
    return birth, (proc / "ns/pid").stat().st_ino


@dataclass(frozen=True)
class MpsBinding:
    owner: tuple[str, int]
    client: MpsClient
    birth: tuple[int, int]
    server_birth: tuple[int, int]


class MpsClientController:
    """Control trusted clients whose admission is serialized by the caller."""

    def __init__(self, control, identity=local_identity):
        self.control = control
        self.identity = identity

    def bind(self, job_id, epoch, pid):
        """Bind an authoritative local PID, never a PID supplied by task stdout."""
        if not job_id or epoch < 1:
            raise ValueError("MPS binding requires an owner and a positive epoch.")
        before = self.identity(pid)
        matches = [row for row in parse_clients(self.control("ps")) if row.pid == pid]
        if len(matches) != 1 or matches[0].namespace != before[1] or self.identity(pid) != before:
            raise RuntimeError("MPS client ownership cannot be established.")
        client = matches[0]
        binding = MpsBinding((job_id, epoch), client, before, self.identity(client.server_pid))
        self.validate(binding)
        return binding

    def validate(self, binding):
        """Reject stale process births, server restarts and changed client connections."""
        if self.identity(binding.client.server_pid) != binding.server_birth:
            raise RuntimeError("MPS server generation changed.")
        if self.identity(binding.client.pid) != binding.birth:
            raise RuntimeError("MPS client process changed.")
        matches = [row for row in parse_clients(self.control("ps")) if row.pid == binding.client.pid]
        if matches != [binding.client]:
            raise RuntimeError("MPS client connection changed.")

    def terminate(self, binding, job_id, epoch):
        """Acknowledge CUDA teardown before the caller kills processes or releases capacity."""
        if binding.owner != (job_id, epoch):
            raise RuntimeError("MPS cancellation has a stale owner or epoch.")
        self.validate(binding)
        response = self.control(f"terminate_client {binding.client.server_pid} {binding.client.pid}").strip()
        if response != "0":
            raise RuntimeError(f"MPS termination was not acknowledged: {response!r}.")
