"""
Unit-aware dispatch: send an episode to an agent loop worker whose node can host it.

A container always lands on the node of the agent loop worker that starts it, because
Harbor's `DockerEnvironment` shells out to the nested `dockerd` of that worker's pod.
So "host this episode on node X's unit" is the same as "dispatch it to the worker on
node X" (design document section 6.4). Without this, plain round robin can send a GPU
episode to a worker whose node has no GPU unit, or to a node whose units are all full.

Pure functions only: the manager owns the Ray calls and feeds snapshots in.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import replace

from psrl.workers.env_service.types import EnvDemand, NodeCapacity


class EnvPlacementUnavailable(RuntimeError):
    """No known healthy worker can host this demand within the placement budget."""


def demand_from_extra_info(extra_info, default_gpu_memory_mb=24000):
    """Use the same Harbor resource plan at dispatch and at container admission."""
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    info = extra_info if isinstance(extra_info, dict) else {}
    if info.get("task_path"):
        from psrl.workers.env_service.harbor_resources import demand_from_task

        demand = demand_from_task(
            info["task_path"],
            needs_gpu_hint=needs_gpu_from_extra_info(info),
            gpu_memory_mb_override=int(info.get("gpu_memory_mb", 0)),
        )
    else:
        demand = EnvDemand(
            cpu=float(info.get("cpus", 1)),
            memory_mb=int(info.get("memory_mb", 0)),
            needs_gpu=needs_gpu_from_extra_info(info),
            gpu_memory_mb=int(info.get("gpu_memory_mb", 0)),
        )
    return demand.with_default_gpu_memory(default_gpu_memory_mb)


def needs_gpu_from_extra_info(extra_info: object) -> bool:
    """
    Whether a dataset row's `extra_info` asks for a GPU.

    Accepts a dict, a JSON string (how it arrives from parquet), or anything else, which
    counts as no GPU. Never raises, because dispatch must not fail on a malformed row.
    """
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except (json.JSONDecodeError, TypeError):
            return False
    if not isinstance(extra_info, dict):
        return False
    try:
        return int(extra_info.get("gpus", 0)) > 0
    except (TypeError, ValueError):
        return False


class EnvAwareDispatcher:
    """
    Pick a worker per prompt group using a snapshot of per-node free capacity.

    Missing or stale health raises a placement error for the manager's bounded retry.

    Args:
        worker_node_ips (Sequence[str]): Node IP of each agent loop worker, by worker index.
        max_snapshot_age_s (float): A snapshot older than this is ignored.
        clock (Callable[[], float]): Monotonic clock, replaced in tests.
    """

    def __init__(
        self,
        worker_node_ips: Sequence[str],
        *,
        max_snapshot_age_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not worker_node_ips:
            raise ValueError("EnvAwareDispatcher needs at least one worker.")
        self._worker_nodes = list(worker_node_ips)
        self._max_age = float(max_snapshot_age_s)
        self._clock = clock
        self._capacity: dict[str, NodeCapacity] = {}
        self._captured_at: float | None = None

    @property
    def snapshot_age_s(self) -> float | None:
        """Seconds since the last `update`, or `None` before the first one."""
        return None if self._captured_at is None else self._clock() - self._captured_at

    def update(self, capacity: dict[str, NodeCapacity]) -> None:
        """Install a fresh per-node capacity snapshot."""
        self._capacity = dict(capacity)
        self._captured_at = self._clock()

    def is_fresh(self) -> bool:
        age = self.snapshot_age_s
        return age is not None and age <= self._max_age

    def assign(self, group_needs_gpu: Sequence[bool], rr_start: int, *, demands=None, group_sizes=None) -> list[int]:
        """
        Choose a worker index for each prompt group, in order.

        Siblings stay together while their full resource demand is charged to the chosen
        node. Admission remains authoritative because snapshots can race with other calls.

        Args:
            group_needs_gpu (Sequence[bool]): Whether each group needs a GPU.
            rr_start (int): The manager's round-robin cursor.

        Returns:
            list[int]: A worker index per group.
        """
        n = len(self._worker_nodes)
        if not self.is_fresh():
            raise EnvPlacementUnavailable("A fresh environment capacity snapshot is required.")

        if demands is not None and any(cap.units for cap in self._capacity.values()):
            return self._assign_resources(demands, group_sizes, rr_start)

        free = {
            ip: {"cpu": cap.cpu_free, "gpu": cap.gpu_free, "cpu_ok": cap.cpu_eligible, "gpu_ok": cap.gpu_eligible}
            for ip, cap in self._capacity.items()
            if cap.available
        }
        chosen: list[int] = []
        for k, needs_gpu in enumerate(group_needs_gpu):
            kind = "gpu" if needs_gpu else "cpu"
            order = [(rr_start + k + j) % n for j in range(n)]
            pick = next((w for w in order if free.get(self._worker_nodes[w], {}).get(kind, 0) > 0), None)
            if pick is None:
                # Nothing has room right now. Prefer a node that could host it once a slot
                # frees, because Gate A will simply wait there, over one that never can.
                pick = next(
                    (w for w in order if free.get(self._worker_nodes[w], {}).get(f"{kind}_ok", False)),
                    None,
                )
            if pick is None:
                raise EnvPlacementUnavailable("No healthy node is eligible for this environment demand.")
            free[self._worker_nodes[pick]][kind] -= group_sizes[k] if group_sizes else 1
            chosen.append(pick)
        return chosen

    @staticmethod
    def _eligible(cap, demand):
        unit = cap.unit
        if not (cap.gpu_eligible if demand.needs_gpu else cap.cpu_eligible):
            return False
        if demand.cpu > unit.cpu_quota or demand.memory_mb > unit.memory_mb:
            return False
        if demand.needs_gpu:
            values = demand.device_memory(unit)
            return (
                len(values) == len(unit.gpu_indices)
                and demand.gpu_memory_mb <= unit.gpu_memory_mb
                and all(m <= budget for m, budget in zip(values, unit.gpu_memory_per_device_mb))
            )
        return True

    @staticmethod
    def _room(cap, demand):
        room = cap.containers_free
        if demand.cpu > 0:
            room = min(room, int(cap.cpu_free // demand.cpu))
        if demand.memory_mb > 0:
            room = min(room, cap.memory_free_mb // demand.memory_mb)
        if demand.needs_gpu and cap.unit.gpu_reservation == "episode":
            room = min(room, int(cap.gpu_total_free_mb // demand.gpu_memory_mb))
            for free, requested in zip(cap.gpu_free_mb, demand.device_memory(cap.unit)):
                room = min(room, int(free // requested))
        return max(0, room)

    def _assign_resources(self, demands, group_sizes, rr_start):
        units = {ip: list(cap.units) for ip, cap in self._capacity.items() if cap.available}
        chosen = []
        for k, demand in enumerate(demands):
            count = group_sizes[k] if group_sizes else 1
            order = [(rr_start + k + j) % len(self._worker_nodes) for j in range(len(self._worker_nodes))]
            candidates = {}
            for worker in order:
                caps = [c for c in units.get(self._worker_nodes[worker], ()) if self._eligible(c, demand)]
                if caps:
                    candidates[worker] = sum(self._room(c, demand) for c in caps)
            if not candidates:
                raise EnvPlacementUnavailable("No healthy unit can accommodate this request's resource shape.")

            def load_key(worker):
                load = self._capacity[self._worker_nodes[worker]].node_load_per_cpu
                return (load is None, load if load is not None else 0.0)

            fitting = [w for w, room in candidates.items() if room >= count]
            pick = min(fitting, key=load_key) if fitting else None
            if pick is None:
                pick = min(candidates, key=lambda w: (-candidates[w], *load_key(w)))
            caps = units[self._worker_nodes[pick]]
            for _ in range(count):
                index = max(
                    (i for i, c in enumerate(caps) if self._eligible(c, demand)),
                    key=lambda i: self._room(caps[i], demand),
                )
                cap = caps[index]
                charge_gpu = demand.needs_gpu and cap.unit.gpu_reservation == "episode"
                caps[index] = replace(
                    cap,
                    containers_free=cap.containers_free - 1,
                    cpu_free=cap.cpu_free - demand.cpu,
                    memory_free_mb=cap.memory_free_mb - demand.memory_mb,
                    gpu_total_free_mb=cap.gpu_total_free_mb - (demand.gpu_memory_mb if charge_gpu else 0),
                    gpu_free_mb=tuple(
                        free - needed for free, needed in zip(cap.gpu_free_mb, demand.device_memory(cap.unit))
                    )
                    if charge_gpu
                    else cap.gpu_free_mb,
                )
            chosen.append(pick)
        return chosen
