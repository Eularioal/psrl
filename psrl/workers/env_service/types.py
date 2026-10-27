"""
Plain data types for the Env Service (units, demands, leases, snapshots).

Everything here is pure Python so it can be imported by the Ray router, the agent
loop client, and CPU-only tests alike. See
`docs/psrl/2026-09-21-env-service-design.md` section 6.2 for the unit model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from psrl.workers.env_service.container_budget import ContainerBudget

GRANTED = "granted"
WAIT = "wait"
REJECT = "reject"


@dataclass(frozen=True)
class EnvUnit:
    """
    One allocatable resource bundle on one node.

    A unit owns whole GPU devices outright and shares them among every container it
    hosts, so several containers may sit on one card as long as their summed device
    memory fits `gpu_memory_mb`. Episode reservations retain this budget until cleanup.
    Command reservations charge it only while a fenced GPU command is active. Host
    resources remain reserved for the container lifetime in both modes.

    Attributes:
        unit_id (str): Stable, unique cluster-wide id.
        node_ip (str): Node whose Docker daemon hosts the unit's containers.
        cpu_quota (float): CPUs shared by this unit's containers.
        memory_mb (int): Host memory shared by this unit's containers.
        gpu_indices (tuple[int, ...]): Whole devices owned by this unit. Empty means CPU only.
        max_containers (int): Configured ceiling on concurrent containers, not derived.
        gpu_memory_mb (int): Device memory budget across `gpu_indices`. Zero for CPU units.
        gpu_reservation (str): Charge GPU capacity for the episode or explicit commands.
    """

    unit_id: str
    node_ip: str
    cpu_quota: float
    memory_mb: int
    gpu_indices: tuple[int, ...] = ()
    max_containers: int = 1
    gpu_memory_mb: int = 0
    rollout_replica_rank: int | None = None
    gpu_reservation: str = "episode"
    gpu_memory_per_device_mb: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.gpu_indices and not self.gpu_memory_per_device_mb:
            if len(self.gpu_indices) != 1:
                raise ValueError("Multi-device units require explicit gpu_memory_per_device_mb budgets.")
            object.__setattr__(self, "gpu_memory_per_device_mb", (self.gpu_memory_mb,))
        if len(self.gpu_memory_per_device_mb) != len(self.gpu_indices):
            raise ValueError("GPU budgets must match gpu_indices in order.")
        if any(m <= 0 for m in self.gpu_memory_per_device_mb):
            raise ValueError("Each GPU memory budget must be positive.")
        if sum(self.gpu_memory_per_device_mb) != self.gpu_memory_mb:
            raise ValueError("Per-device GPU budgets must sum to gpu_memory_mb.")
        if self.gpu_reservation not in ("episode", "command"):
            raise ValueError(f"Unknown GPU reservation mode {self.gpu_reservation!r}.")
        if self.gpu_reservation == "command" and (not self.gpu_indices or self.rollout_replica_rank is not None):
            raise ValueError("Command reservations require dedicated environment GPUs.")
        if self.rollout_replica_rank is not None:
            if self.rollout_replica_rank < 0 or not self.gpu_indices or self.max_containers != 1:
                raise ValueError("A borrowed unit requires one container slot, GPUs and a nonnegative replica rank.")
        if not self.unit_id:
            raise ValueError("EnvUnit.unit_id must be non-empty.")
        if not self.node_ip:
            raise ValueError(f"EnvUnit {self.unit_id!r}: node_ip must be non-empty.")
        if self.cpu_quota <= 0:
            raise ValueError(f"EnvUnit {self.unit_id!r}: cpu_quota must be > 0, got {self.cpu_quota}.")
        if self.memory_mb <= 0:
            raise ValueError(f"EnvUnit {self.unit_id!r}: memory_mb must be > 0, got {self.memory_mb}.")
        if self.max_containers < 1:
            raise ValueError(f"EnvUnit {self.unit_id!r}: max_containers must be >= 1, got {self.max_containers}.")
        if self.gpu_indices:
            if len(set(self.gpu_indices)) != len(self.gpu_indices):
                raise ValueError(f"EnvUnit {self.unit_id!r}: duplicate gpu_indices {self.gpu_indices}.")
            if any(i < 0 for i in self.gpu_indices):
                raise ValueError(f"EnvUnit {self.unit_id!r}: negative gpu index in {self.gpu_indices}.")
            if self.gpu_memory_mb <= 0:
                raise ValueError(f"EnvUnit {self.unit_id!r}: a GPU unit needs gpu_memory_mb > 0.")
        elif self.gpu_memory_mb != 0:
            raise ValueError(f"EnvUnit {self.unit_id!r}: gpu_memory_mb must be 0 for a CPU-only unit.")

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_indices)

    @classmethod
    def from_mapping(cls, raw: Mapping) -> EnvUnit:
        """Build a unit from a plain mapping (for example a parsed YAML entry)."""
        return cls(
            unit_id=str(raw["unit_id"]),
            node_ip=str(raw["node_ip"]),
            cpu_quota=float(raw["cpu_quota"]),
            memory_mb=int(raw["memory_mb"]),
            gpu_indices=tuple(int(i) for i in (raw.get("gpu_indices") or ())),
            max_containers=int(raw.get("max_containers", 1)),
            gpu_memory_mb=int(raw.get("gpu_memory_mb", 0)),
            rollout_replica_rank=(
                int(raw["rollout_replica_rank"]) if raw.get("rollout_replica_rank") is not None else None
            ),
            gpu_reservation=str(raw.get("gpu_reservation", "episode")),
            gpu_memory_per_device_mb=tuple(int(m) for m in raw.get("gpu_memory_per_device_mb", ())),
        )


def validate_units(units: list[EnvUnit]) -> None:
    """
    Enforce the unit invariants that a typo would otherwise silently break.

    Raises:
        ValueError: On a duplicate `unit_id`, or when two units on one node name the
            same device index (units never overlap on a device).
    """
    seen_ids: set[str] = set()
    owner: dict[tuple[str, int], str] = {}
    for unit in units:
        if unit.unit_id in seen_ids:
            raise ValueError(f"Duplicate unit_id {unit.unit_id!r}.")
        seen_ids.add(unit.unit_id)
        for index in unit.gpu_indices:
            key = (unit.node_ip, index)
            if key in owner:
                raise ValueError(
                    f"Device {index} on node {unit.node_ip} is claimed by both "
                    f"{owner[key]!r} and {unit.unit_id!r}. Units must never overlap on a device."
                )
            owner[key] = unit.unit_id


@dataclass(frozen=True)
class EnvDemand:
    """
    Resources one episode's container set needs while it runs.

    Attributes:
        cpu (float): CPUs charged against the unit.
        memory_mb (int): Host memory charged. Zero means "not accounted".
        needs_gpu (bool): Whether the episode needs a GPU device.
        gpu_memory_mb (int): Device memory charged. Zero on a GPU demand means "use the
            router's configured default", because Harbor declares no such field.
    """

    cpu: float = 1.0
    memory_mb: int = 0
    needs_gpu: bool = False
    gpu_memory_mb: int = 0
    gpu_memory_per_device_mb: tuple[int, ...] = ()
    container_budget: ContainerBudget | None = None

    def __post_init__(self):
        if self.container_budget and (
            self.cpu != self.container_budget.cpu or self.memory_mb != self.container_budget.memory_mb
        ):
            raise ValueError("Host demand must match the complete container budget.")
        if self.gpu_memory_per_device_mb and (
            not self.needs_gpu
            or any(m <= 0 for m in self.gpu_memory_per_device_mb)
            or sum(self.gpu_memory_per_device_mb) != self.gpu_memory_mb
        ):
            raise ValueError("GPU demand must declare positive per-device budgets summing to gpu_memory_mb.")

    def device_memory(self, unit: EnvUnit) -> tuple[int, ...]:
        """Assume an unpartitioned request can concentrate its entire budget on any device."""
        return self.gpu_memory_per_device_mb or (self.gpu_memory_mb,) * len(unit.gpu_indices)

    def with_default_gpu_memory(self, default_mb: int) -> EnvDemand:
        """Return a copy whose GPU memory is filled in when it is a GPU demand left at zero."""
        if self.needs_gpu and self.gpu_memory_mb <= 0:
            return replace(self, gpu_memory_mb=int(default_mb))
        return self


@dataclass
class EnvLease:
    """
    One admitted episode's hold on a unit.

    Attributes:
        lease_id (str): Unique id. Stamped on the containers as the `psrl.env_lease` label.
        session_id (str): The episode's session id.
        unit_id (str): The unit that hosts the episode.
        node_ip (str): The unit's node.
        gpu_indices (tuple[int, ...]): Devices the containers get. Shared with other leases of the unit.
        demand (EnvDemand): What was charged.
        granted_at (float): Ledger clock when granted.
        last_seen (float): Ledger clock of the latest heartbeat. Drives TTL reclaim.
        shared (bool): Whether the unit may host several containers on these devices. Gate B
            then judges idleness on compute utilization alone, because a neighbour's resident
            model weights hold device memory even while it is idle.
    """

    lease_id: str
    session_id: str
    unit_id: str
    node_ip: str
    gpu_indices: tuple[int, ...]
    demand: EnvDemand
    granted_at: float
    last_seen: float
    shared: bool = False
    node_service_url: str = ""
    gpu_lending: bool = False
    gpu_reservation: str = "episode"
    gpu_active: bool = True
    gpu_closing: bool = False


@dataclass(frozen=True)
class Admission:
    """
    Outcome of one admission attempt.

    Attributes:
        status (str): `granted`, `wait` (capacity is full, retry later), or `reject`
            (the demand can never be placed, retrying is pointless).
        lease (EnvLease | None): Set when granted.
        reason (str): Human-readable explanation for `wait` and `reject`.
    """

    status: str
    lease: EnvLease | None = None
    reason: str = ""


@dataclass(frozen=True)
class NodeCapacity:
    """
    What one node can still take, as seen by unit-aware dispatch.

    Attributes:
        cpu_free (int): Free container slots on units that accept CPU-only episodes.
        gpu_free (int): Free container slots on units that accept GPU episodes.
        cpu_eligible (bool): Whether any unit on this node may ever host a CPU episode.
        gpu_eligible (bool): Whether any unit on this node may ever host a GPU episode.
    """

    cpu_free: int = 0
    gpu_free: int = 0
    cpu_eligible: bool = False
    gpu_eligible: bool = False
    available: bool = True
    reason: str = ""
    units: tuple[UnitCapacity, ...] = ()
    node_load_per_cpu: float | None = None


@dataclass(frozen=True)
class UnitCapacity:
    """Expose resource headroom and policy eligibility for one placement candidate."""

    unit: EnvUnit
    containers_free: int
    cpu_free: float
    memory_free_mb: int
    gpu_free_mb: tuple[float, ...]
    gpu_total_free_mb: float
    cpu_eligible: bool
    gpu_eligible: bool


@dataclass(frozen=True)
class UnitSnapshot:
    """Observable state of one unit, mirroring vLLM's `EngineStats` for env."""

    unit_id: str
    node_ip: str
    leases_active: int
    max_containers: int
    cpu_used: float
    cpu_quota: float
    memory_used_mb: int
    memory_mb: int
    declared_gpu_memory_mb: int
    measured_gpu_memory_mb: float | None
    gpu_memory_mb: int
    sessions: tuple[str, ...] = field(default_factory=tuple)
    sessions_waiting: int = 0
    gpu_commands_active: int = 0
    gpu_commands_waiting: int = 0
    containers_running: int | None = None
    containers_total: int | None = None
    measured_memory_mb: float | None = None
    measured_cpu_cores: float | None = None
    node_cpu_load_1m: float | None = None
    node_cpu_count: int | None = None
    resource_sample_age_s: float | None = None
    resource_error: str = "No physical resource sample received."
