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

from psrl.workers.env_service.types import NodeCapacity


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

    Falls back to plain round robin whenever the snapshot is missing or stale, so a dead
    router can slow placement down but never stop dispatch.

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

    def assign(self, group_needs_gpu: Sequence[bool], rr_start: int) -> list[int]:
        """
        Choose a worker index for each prompt group, in order.

        Each group is assigned one slot on its node, so a batch spreads across nodes
        instead of piling onto the first one with room. Siblings of a group stay together
        because the caller assigns per group, not per child.

        Args:
            group_needs_gpu (Sequence[bool]): Whether each group needs a GPU.
            rr_start (int): The manager's round-robin cursor.

        Returns:
            list[int]: A worker index per group.
        """
        n = len(self._worker_nodes)
        if not self.is_fresh():
            return [(rr_start + k) % n for k in range(len(group_needs_gpu))]

        free = {
            ip: {"cpu": cap.cpu_free, "gpu": cap.gpu_free, "cpu_ok": cap.cpu_eligible, "gpu_ok": cap.gpu_eligible}
            for ip, cap in self._capacity.items()
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
                    order[0],
                )
            else:
                free[self._worker_nodes[pick]][kind] -= 1
            chosen.append(pick)
        return chosen
