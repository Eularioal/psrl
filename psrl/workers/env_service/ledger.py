"""
The unit ledger: Gate A's admission arithmetic, with no Ray and no I/O.

This is a pure state machine. The Ray actor in `router.py` serializes calls into it, and
the tests drive it directly with a fake clock, so every branch is checkable on a laptop.

What it decides, in the vocabulary of `docs/psrl/2026-09-21-env-service-design.md`:

* **Gate A** (section 6.3): resource-keyed admission. An episode is granted only when one
  unit can hold it right now, summing CPU, host memory, container count and device memory.
  Otherwise the caller waits at no cost, because no container exists yet.
* **One card, several containers** (section 5.3): a unit owns whole devices and shares
  them among its containers. SM time is time-sliced by the driver, but device memory adds
  exactly, so memory is the ceiling and it is summed rather than queued per card.
* **Memory promises**: measured device usage is charged along with each reservation's
  unallocated remainder. Only verified current ownership can offset that reservation.
* **TTL reclaim** (section 8.1): a lease that stops heartbeating is dropped and queued for
  node-local container cleanup, because the session layer never cleans up a crashed agent.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
import uuid
from collections.abc import Callable, Iterable, Mapping

from psrl.workers.env_service.gpu_memory import accounted_gpu_memory_mb
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.types import (
    GRANTED,
    REJECT,
    WAIT,
    Admission,
    EnvDemand,
    EnvLease,
    EnvUnit,
    NodeCapacity,
    UnitCapacity,
    UnitSnapshot,
    validate_units,
)

_EPS = 1e-9


class EnvUnitLedger:
    """
    Admission control over a static set of units.

    Not thread-safe by design: the router actor is the single writer.

    Args:
        units (Iterable[EnvUnit]): The units, validated for unique ids and no device overlap.
        rules (NodeRules): Which nodes may host which demands.
        default_gpu_memory_mb (int): Device memory charged to a GPU demand that declares none.
        lease_ttl_s (float): A lease with no heartbeat for this long is reclaimed.
        measurement_ttl_s (float): A reported measurement older than this is ignored, so a
            stale high reading cannot block a unit forever.
        wait_ttl_s (float): A waiter not seen for this long stops counting as waiting.
        clock (Callable[[], float]): Monotonic clock, replaced in tests.
        id_factory (Callable[[], str] | None): Lease id suffix source, replaced in tests.
    """

    def __init__(
        self,
        units: Iterable[EnvUnit],
        rules: NodeRules,
        *,
        default_gpu_memory_mb: int = 24000,
        lease_ttl_s: float = 1800.0,
        measurement_ttl_s: float = 60.0,
        wait_ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] | None = None,
    ):
        self._units: dict[str, EnvUnit] = {}
        unit_list = list(units)
        validate_units(unit_list)
        for unit in unit_list:
            self._units[unit.unit_id] = unit
        self._rules = rules
        self._default_gpu_memory_mb = int(default_gpu_memory_mb)
        self._lease_ttl_s = float(lease_ttl_s)
        self._measurement_ttl_s = float(measurement_ttl_s)
        self._wait_ttl_s = float(wait_ttl_s)
        self._clock = clock
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex[:8])
        self._counter = itertools.count(1)

        self._blocked_units = {
            unit.unit_id
            for unit in unit_list
            if unit.rollout_replica_rank is not None or unit.shared_rollout_replica_rank is not None
        }
        self._leases: dict[str, EnvLease] = {}
        self._by_session: dict[str, str] = {}
        # node_ip -> (reported_at, {device_index: used_mb})
        self._measured: dict[str, tuple[float, dict[int, float]]] = {}
        # node_ip -> leases reclaimed by TTL, awaiting node-local container cleanup
        self._reaped: dict[str, list[EnvLease]] = {}
        # session_id -> last time it asked and had to wait
        self._waiting: dict[str, float] = {}
        self._gpu_cleanup_pending: dict[str, str] = {}
        self._gpu_command_waiters: dict[str, str] = {}

    # ------------------------------------------------------------------ admission

    def try_admit(
        self,
        session_id: str,
        demand: EnvDemand,
        node_ip: str | None = None,
        measured_gpu_used_mb: Mapping[int, float] | None = None,
        *,
        reap: bool = True,
        available_node_ips=None,
    ) -> Admission:
        """
        Try to place one episode right now.

        Args:
            session_id (str): The episode's session id. Admitting the same id twice returns the
                existing lease, so a retry after a lost reply cannot double-charge a unit.
            demand (EnvDemand): What the episode needs.
            node_ip (str | None): When set, only units on this node are considered. A container
                always lands on the node of the agent loop worker that starts it, so the caller
                passes its own node.
            measured_gpu_used_mb (Mapping[int, float] | None): Per-device memory in use on
                `node_ip`, read by the caller from `nvidia-smi`.

        Returns:
            Admission: `granted` with a lease, `wait` when every eligible unit is full, or
            `reject` when no unit could ever host the demand.
        """
        now = self._clock()
        if reap:
            self._reap(now)
        if measured_gpu_used_mb is not None and node_ip:
            self.report_measurement(node_ip, measured_gpu_used_mb)

        existing_id = self._by_session.get(session_id)
        if existing_id is not None:
            lease = self._leases[existing_id]
            if available_node_ips is not None and lease.node_ip not in available_node_ips:
                return Admission(WAIT, reason="the existing lease's node is unavailable")
            lease.last_seen = now
            return Admission(GRANTED, dataclasses.replace(lease))

        demand = demand.with_default_gpu_memory(self._default_gpu_memory_mb)
        eligible: list[tuple[int, EnvUnit]] = []
        for unit in self._units.values():
            if available_node_ips is not None and unit.node_ip not in available_node_ips:
                continue
            if node_ip is not None and unit.node_ip != node_ip:
                continue
            tier = self._rules.tier(unit, demand)
            if tier is not None and self._demand_matches_kind(unit, demand):
                eligible.append((tier, unit))
        if not eligible:
            self._waiting.pop(session_id, None)
            return Admission(REJECT, reason=self._explain_no_eligible(demand, node_ip))
        if not any(self._could_ever_fit(unit, demand) for _, unit in eligible):
            self._waiting.pop(session_id, None)
            return Admission(REJECT, reason=self._explain_too_big(demand, [u for _, u in eligible]))

        fitting = [(tier, unit) for tier, unit in eligible if self._fits_now(unit, demand, now)]
        if fitting:
            # Command sessions may queue, but usable capacity elsewhere takes precedence.
            _, unit = min(
                fitting,
                key=lambda item: (
                    not self._gpu_fits_now(item[1], demand, now),
                    item[0],
                    item[1].has_gpu and not demand.needs_gpu,
                    self._load(item[1], demand, now),
                    item[1].unit_id,
                ),
            )
            lease = EnvLease(
                lease_id=f"{unit.unit_id}-{next(self._counter)}-{self._id_factory()}",
                session_id=session_id,
                unit_id=unit.unit_id,
                node_ip=unit.node_ip,
                gpu_indices=unit.gpu_indices if demand.needs_gpu else (),
                demand=demand,
                granted_at=now,
                last_seen=now,
                shared=unit.has_gpu and unit.max_containers > 1,
                gpu_lending=unit.rollout_replica_rank is not None and demand.needs_gpu,
                gpu_reservation=unit.gpu_reservation,
                gpu_active=unit.gpu_reservation == "episode",
            )
            self._leases[lease.lease_id] = lease
            self._by_session[session_id] = lease.lease_id
            self._waiting.pop(session_id, None)
            return Admission(GRANTED, dataclasses.replace(lease))

        self._waiting[session_id] = now
        return Admission(WAIT, reason="every eligible unit is at capacity")

    def cancel_wait(self, session_id: str) -> None:
        """Forget a waiter that gave up, so it stops counting in `sessions_waiting`."""
        self._waiting.pop(session_id, None)

    # ------------------------------------------------------------------ lease lifecycle

    def heartbeat(self, lease_id: str) -> bool:
        """
        Refresh a lease's TTL.

        Returns:
            bool: `False` when the lease is unknown, most likely because it was already
            reclaimed after a missed heartbeat window.
        """
        lease = self._leases.get(lease_id)
        if lease is None:
            return False
        lease.last_seen = self._clock()
        return True

    def release(self, lease_id: str) -> bool:
        """Release a lease. Idempotent: returns `False` when it was already gone."""
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            return False
        if self._by_session.get(lease.session_id) == lease_id:
            del self._by_session[lease.session_id]
        return True

    def release_session(self, session_id: str) -> bool:
        """Release whatever lease a session holds."""
        lease_id = self._by_session.get(session_id)
        return self.release(lease_id) if lease_id is not None else False

    def reap(self) -> list[EnvLease]:
        """Reclaim every lease past its TTL. Returns the reclaimed leases."""
        return self._reap(self._clock())

    def pending_reaped(self, node_ip: str) -> list[EnvLease]:
        """Redeliver cleanup obligations until the node acknowledges physical removal."""
        return [dataclasses.replace(lease) for lease in self._reaped.get(node_ip, ())]

    def acknowledge_reaped(self, node_ip: str, lease_id: str) -> None:
        """Retire only the acknowledged cleanup obligation, leaving concurrent expiries intact."""
        pending = [lease for lease in self._reaped.get(node_ip, ()) if lease.lease_id != lease_id]
        if pending:
            self._reaped[node_ip] = pending
        else:
            self._reaped.pop(node_ip, None)

    def retire(self, lease_id: str) -> bool:
        """Remove a live reservation while retaining its physical cleanup obligation."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return False
        if lease.gpu_reservation == "command":
            self.block_gpu_cleanup(lease.lease_id, lease.unit_id)
        self.release(lease_id)
        self._reaped.setdefault(lease.node_ip, []).append(lease)
        return True

    def _reap(self, now: float) -> list[EnvLease]:
        expired = [lease for lease in self._leases.values() if now - lease.last_seen > self._lease_ttl_s]
        for lease in expired:
            self.retire(lease.lease_id)
        for sid in [s for s, ts in self._waiting.items() if now - ts > self._wait_ttl_s]:
            del self._waiting[sid]
        return expired

    # ------------------------------------------------------------------ telemetry

    def report_measurement(self, node_ip: str, used_mb_by_index: Mapping[int, float]) -> None:
        """Record per-device memory in use on a node, as read by the node's own `nvidia-smi`."""
        self._measured[node_ip] = (self._clock(), {int(i): float(v) for i, v in used_mb_by_index.items()})

    def snapshot(self) -> list[UnitSnapshot]:
        """Per-unit observable state, the env counterpart of `EngineStats`."""
        now = self._clock()
        waiting = len(self._waiting)
        result: list[UnitSnapshot] = []
        for unit in self._units.values():
            leases = self._unit_leases(unit.unit_id)
            result.append(
                UnitSnapshot(
                    unit_id=unit.unit_id,
                    node_ip=unit.node_ip,
                    leases_active=len(leases),
                    max_containers=unit.max_containers,
                    cpu_used=sum(lease.demand.cpu for lease in leases),
                    cpu_quota=unit.cpu_quota,
                    memory_used_mb=sum(lease.demand.memory_mb for lease in leases),
                    memory_mb=unit.memory_mb,
                    declared_gpu_memory_mb=self._declared_gpu_mb(unit),
                    measured_gpu_memory_mb=self._measured_gpu_mb(unit, now),
                    gpu_memory_mb=unit.gpu_memory_mb,
                    sessions=tuple(lease.session_id for lease in leases),
                    sessions_waiting=waiting,
                    gpu_commands_active=sum(
                        lease.gpu_active for lease in leases if lease.gpu_reservation == "command"
                    ),
                    gpu_commands_waiting=sum(u == unit.unit_id for u in self._gpu_command_waiters.values()),
                )
            )
        return result

    def node_capacity(self) -> dict[str, NodeCapacity]:
        """
        Free container slots per node, for unit-aware dispatch.

        A GPU slot is bounded by both the container ceiling and by how many default-sized
        GPU tasks still fit in device memory.
        """
        now = self._clock()
        cpu_probe = EnvDemand(cpu=0.0)
        gpu_probe = EnvDemand(cpu=0.0, needs_gpu=True, gpu_memory_mb=self._default_gpu_memory_mb)
        per_node: dict[str, dict[str, int | bool]] = {}
        resources = {}
        for unit in self._units.values():
            entry = per_node.setdefault(
                unit.node_ip, {"cpu_free": 0, "gpu_free": 0, "cpu_eligible": False, "gpu_eligible": False}
            )
            if unit.unit_id in self._blocked_units:
                continue
            leases = self._unit_leases(unit.unit_id)
            resources.setdefault(unit.node_ip, []).append(
                UnitCapacity(
                    unit=unit,
                    containers_free=unit.max_containers - len(leases),
                    cpu_free=unit.cpu_quota - sum(lease.demand.cpu for lease in leases),
                    memory_free_mb=unit.memory_mb - sum(lease.demand.memory_mb for lease in leases),
                    gpu_free_mb=tuple(
                        budget - used
                        for budget, used in zip(
                            unit.gpu_memory_per_device_mb, self._effective_gpu_by_device(unit, now)
                        )
                    ),
                    gpu_total_free_mb=unit.gpu_memory_mb - self._effective_gpu_mb(unit, now),
                    cpu_eligible=self._rules.tier(unit, cpu_probe) is not None,
                    gpu_eligible=unit.has_gpu and self._rules.tier(unit, gpu_probe) is not None,
                    cpu_priority=self._rules.tier(unit, cpu_probe) or 0,
                    gpu_priority=self._rules.tier(unit, gpu_probe) or 0,
                )
            )
            containers_free = max(0, unit.max_containers - len(self._unit_leases(unit.unit_id)))
            if self._rules.tier(unit, cpu_probe) is not None:
                entry["cpu_eligible"] = True
                entry["cpu_free"] += containers_free
            if self._rules.tier(unit, gpu_probe) is not None and unit.has_gpu:
                entry["gpu_eligible"] = True
                headroom = unit.gpu_memory_mb - self._effective_gpu_mb(unit, now)
                by_memory = int(max(0.0, headroom) // max(1, self._default_gpu_memory_mb))
                entry["gpu_free"] += (
                    containers_free if unit.gpu_reservation == "command" else min(containers_free, by_memory)
                )
        return {
            ip: NodeCapacity(
                **vals,
                units=tuple(resources.get(ip, ())),
                cpu_priority=min((c.cpu_priority for c in resources.get(ip, ()) if c.cpu_eligible), default=0),
                gpu_priority=min((c.gpu_priority for c in resources.get(ip, ()) if c.gpu_eligible), default=0),
            )
            for ip, vals in per_node.items()
        }

    # ------------------------------------------------------------------ arithmetic

    @staticmethod
    def _demand_matches_kind(unit: EnvUnit, demand: EnvDemand) -> bool:
        """A GPU demand needs a unit that owns devices. A CPU demand fits any unit."""
        return unit.has_gpu or not demand.needs_gpu

    def _unit_leases(self, unit_id: str) -> list[EnvLease]:
        return [lease for lease in self._leases.values() if lease.unit_id == unit_id]

    def _declared_gpu_mb(self, unit: EnvUnit) -> int:
        return sum(
            lease.demand.gpu_memory_mb
            for lease in self._unit_leases(unit.unit_id)
            if lease.demand.needs_gpu and lease.gpu_active
        )

    def block_gpu_cleanup(self, lease_id: str, unit_id: str) -> None:
        """Keep expired or uncertain GPU access fenced until node cleanup acknowledges it."""
        self._gpu_cleanup_pending[lease_id] = unit_id
        self._blocked_units.add(unit_id)

    def confirm_gpu_cleanup(self, lease_id: str, unit_id: str) -> None:
        """Unblock only after every uncertain lease on this unit has been removed."""
        self._gpu_cleanup_pending.pop(lease_id, None)
        if unit_id not in self._gpu_cleanup_pending.values():
            self._blocked_units.discard(unit_id)

    def _measured_gpu_mb(self, unit: EnvUnit, now: float) -> float | None:
        """Summed measured memory over the unit's devices, or `None` when unknown or stale."""
        if not unit.has_gpu:
            return None
        reading = self._measured.get(unit.node_ip)
        if reading is None:
            return None
        reported_at, used = reading
        if now - reported_at > self._measurement_ttl_s:
            return None
        if any(index not in used for index in unit.gpu_indices):
            return None
        return sum(used[index] for index in unit.gpu_indices)

    def _effective_gpu_mb(self, unit: EnvUnit, now: float, owned_mb: Mapping[str, float] | None = None) -> float:
        """
        Charge measured usage plus reservations not covered by verified current ownership.

        Unattributed usage cannot pay down a reservation. Lending uses its separate
        physical handoff protocol and only charges the borrowed task's reservation.
        """
        if unit.rollout_replica_rank is not None:
            return float(self._declared_gpu_mb(unit))
        measured = self._measured_gpu_mb(unit, now)
        reservations = {
            lease.lease_id: lease.demand.gpu_memory_mb
            for lease in self._unit_leases(unit.unit_id)
            if lease.demand.needs_gpu and lease.gpu_active
        }
        return accounted_gpu_memory_mb(
            measured or 0.0, reservations, owned_mb if measured is not None else None
        ) + unit.model_headroom_mb * len(unit.gpu_indices)

    def _could_ever_fit(self, unit: EnvUnit, demand: EnvDemand) -> bool:
        """Whether the demand fits an empty unit. If not, waiting would be forever."""
        if demand.cpu > unit.cpu_quota + _EPS or demand.memory_mb > unit.memory_mb:
            return False
        if demand.needs_gpu:
            per_device = demand.device_memory(unit)
            return (
                len(per_device) == len(unit.gpu_indices)
                and demand.gpu_memory_mb <= unit.gpu_memory_mb
                and all(m <= capacity for m, capacity in zip(per_device, unit.gpu_memory_per_device_mb))
            )
        return True

    def _effective_gpu_by_device(self, unit: EnvUnit, now: float) -> tuple[float, ...]:
        """Account separately on each device without inventing per-device ownership."""
        measured = {}
        reading = self._measured.get(unit.node_ip)
        if unit.rollout_replica_rank is None and reading and now - reading[0] <= self._measurement_ttl_s:
            measured = reading[1]
        reservations = [
            lease.demand.device_memory(unit)
            for lease in self._unit_leases(unit.unit_id)
            if lease.demand.needs_gpu and lease.gpu_active
        ]
        return tuple(
            measured.get(index, 0.0) + sum(values[position] for values in reservations) + unit.model_headroom_mb
            for position, index in enumerate(unit.gpu_indices)
        )

    def _fits_now(self, unit: EnvUnit, demand: EnvDemand, now: float) -> bool:
        if not self._could_ever_fit(unit, demand):
            return False
        if unit.unit_id in self._blocked_units:
            return False
        leases = self._unit_leases(unit.unit_id)
        if len(leases) + 1 > unit.max_containers:
            return False
        if sum(lease.demand.cpu for lease in leases) + demand.cpu > unit.cpu_quota + _EPS:
            return False
        if sum(lease.demand.memory_mb for lease in leases) + demand.memory_mb > unit.memory_mb:
            return False
        if unit.gpu_reservation == "command":
            return self._could_ever_fit(unit, demand)
        return self._gpu_fits_now(unit, demand, now)

    def _gpu_fits_now(self, unit: EnvUnit, demand: EnvDemand, now: float) -> bool:
        """Compare device headroom even when a command session may enter a queue."""
        if demand.needs_gpu and any(
            used + requested > capacity + _EPS
            for used, requested, capacity in zip(
                self._effective_gpu_by_device(unit, now),
                demand.device_memory(unit),
                unit.gpu_memory_per_device_mb,
            )
        ):
            return False
        return not (
            demand.needs_gpu and self._effective_gpu_mb(unit, now) + demand.gpu_memory_mb > unit.gpu_memory_mb + _EPS
        )

    def _load(self, unit: EnvUnit, demand: EnvDemand, now: float) -> float:
        """
        Resource-weighted occupancy after placing the demand, in 0..1.

        A container count is a poor load proxy because env calls run from seconds to hours
        (section 4.5), so the most constrained resource decides.
        """
        leases = self._unit_leases(unit.unit_id)
        ratios = [
            (len(leases) + 1) / unit.max_containers,
            (sum(lease.demand.cpu for lease in leases) + demand.cpu) / unit.cpu_quota,
            (sum(lease.demand.memory_mb for lease in leases) + demand.memory_mb) / unit.memory_mb,
        ]
        if demand.needs_gpu and unit.gpu_memory_mb > 0 and unit.gpu_reservation == "episode":
            ratios.append((self._effective_gpu_mb(unit, now) + demand.gpu_memory_mb) / unit.gpu_memory_mb)
        return max(ratios)

    def _explain_no_eligible(self, demand: EnvDemand, node_ip: str | None) -> str:
        kind = "GPU" if demand.needs_gpu else "CPU"
        where = f" on node {node_ip}" if node_ip else ""
        return (
            f"no unit{where} may host a {kind} episode under policy={self._rules.policy.value!r} "
            f"(units: {sorted(self._units)}). Declare a unit on that node, or relax the node policy."
        )

    @staticmethod
    def _explain_too_big(demand: EnvDemand, units: list[EnvUnit]) -> str:
        biggest = max(units, key=lambda u: (u.gpu_memory_mb, u.memory_mb, u.cpu_quota))
        return (
            f"demand (cpu={demand.cpu}, memory_mb={demand.memory_mb}, gpu_memory_mb={demand.gpu_memory_mb}) exceeds "
            f"even an empty unit; the largest eligible unit {biggest.unit_id!r} has cpu={biggest.cpu_quota}, "
            f"memory_mb={biggest.memory_mb}, gpu_memory_mb={biggest.gpu_memory_mb}."
        )
