"""Validate MPS client identities before cooperative managed cancellation."""

from dataclasses import dataclass
from pathlib import Path


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
