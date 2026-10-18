"""
Whether episode containers may run on nodes that host training or rollout GPUs.

Harbor containers are invisible to Ray, so nothing stops a container from opening a
device a vLLM engine is generating on. At the default `gpu_memory_utilization=0.85` on a
96 GB card only about 14 GB stay free, so such a collision is an intermittent silent OOM
rather than a clean failure. This module is the knob that decides how close env
containers may come to the train and rollout pools.

A container always lands on the node of the agent loop worker that starts it, so the
policy acts in two places: it filters which nodes may host agent loop workers, and it
filters which units the router may grant.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from psrl.workers.env_service.types import EnvDemand, EnvUnit


class NodePolicy(str, Enum):
    """How env containers treat nodes that host train or rollout GPUs."""

    # No restriction. Matches the behaviour before the env service existed.
    ANY = "any"
    # Train and rollout nodes may host CPU-only episodes but never GPU episodes.
    CPU_ONLY = "cpu_only"
    # Never place any container on a train or rollout node.
    AVOID = "avoid"
    # Soft `AVOID`: use other nodes first and spill onto train and rollout nodes only
    # when no other node can take the episode right now.
    PREFER_OTHER = "prefer_other"


def parse_policy(value: str | NodePolicy) -> NodePolicy:
    """Parse a policy name, listing the valid choices on a typo."""
    if isinstance(value, NodePolicy):
        return value
    try:
        return NodePolicy(str(value).strip().lower())
    except ValueError as exc:
        choices = ", ".join(p.value for p in NodePolicy)
        raise ValueError(f"Invalid env_service node policy {value!r}; expected one of: {choices}.") from exc


@dataclass(frozen=True)
class NodeRules:
    """
    Node eligibility rules for episode containers.

    Attributes:
        policy (NodePolicy): How train and rollout nodes are treated.
        train_rollout_node_ips (frozenset[str]): Nodes that host train, rollout, validation
            or reward-model GPUs.
        allow_node_ips (frozenset[str]): When non-empty, only these nodes may host containers.
        deny_node_ips (frozenset[str]): These nodes never host containers (for example a node
            whose Docker daemon is wedged).
    """

    policy: NodePolicy = NodePolicy.CPU_ONLY
    train_rollout_node_ips: frozenset[str] = field(default_factory=frozenset)
    allow_node_ips: frozenset[str] = field(default_factory=frozenset)
    deny_node_ips: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def build(
        cls,
        policy: str | NodePolicy = NodePolicy.CPU_ONLY,
        train_rollout_node_ips: Iterable[str] = (),
        allow_node_ips: Iterable[str] = (),
        deny_node_ips: Iterable[str] = (),
    ) -> NodeRules:
        return cls(
            policy=parse_policy(policy),
            train_rollout_node_ips=frozenset(train_rollout_node_ips),
            allow_node_ips=frozenset(allow_node_ips),
            deny_node_ips=frozenset(deny_node_ips),
        )

    def is_train_rollout(self, node_ip: str) -> bool:
        return node_ip in self.train_rollout_node_ips

    def node_allowed(self, node_ip: str) -> bool:
        """Whether the allow and deny lists permit this node at all."""
        if node_ip in self.deny_node_ips:
            return False
        return not self.allow_node_ips or node_ip in self.allow_node_ips

    def tier(self, unit: EnvUnit, demand: EnvDemand) -> int | None:
        """
        Rank a unit for a demand.

        Returns:
            int | None: `None` when the unit may never host the demand, `0` for a preferred
            unit, `1` for a fallback that is only used when no tier-0 unit fits right now.
        """
        if not self.node_allowed(unit.node_ip):
            return None
        if not self.is_train_rollout(unit.node_ip):
            return 0
        if self.policy is NodePolicy.ANY:
            return 0
        if self.policy is NodePolicy.CPU_ONLY:
            return None if demand.needs_gpu and unit.rollout_replica_rank is None else 0
        if self.policy is NodePolicy.AVOID:
            return None
        return 1  # PREFER_OTHER


def lint_units(units: Sequence[EnvUnit], rules: NodeRules) -> list[str]:
    """
    Describe unit declarations that contradict the node policy.

    A GPU unit on a train or rollout node claims devices those pools own, which is exactly
    the collision this module exists to prevent. These are warnings, not errors, because
    a node can legitimately be listed in `train_rollout_node_ips` by mistake.
    """
    warnings: list[str] = []
    for unit in units:
        if not rules.is_train_rollout(unit.node_ip):
            continue
        if unit.rollout_replica_rank is not None:
            continue
        if unit.has_gpu and rules.policy in (NodePolicy.CPU_ONLY, NodePolicy.AVOID):
            warnings.append(
                f"Unit {unit.unit_id!r} owns GPU(s) {list(unit.gpu_indices)} on {unit.node_ip}, a train/rollout "
                f"node, but policy={rules.policy.value!r} never schedules GPU episodes there, so it can only "
                f"host CPU episodes (or none under 'avoid')."
            )
        elif unit.has_gpu and rules.policy is NodePolicy.ANY:
            warnings.append(
                f"Unit {unit.unit_id!r} owns GPU(s) {list(unit.gpu_indices)} on {unit.node_ip}, a train/rollout "
                f"node, with policy='any'. Nothing prevents a container from opening a device a vLLM engine "
                f"or trainer is using."
            )
        if unit.has_gpu and rules.policy is NodePolicy.PREFER_OTHER:
            warnings.append(
                f"Unit {unit.unit_id!r} owns GPU(s) {list(unit.gpu_indices)} on train/rollout node {unit.node_ip}; "
                f"under 'prefer_other' it is used only as a fallback and can still collide with a vLLM engine."
            )
    return warnings


def select_worker_nodes(
    candidates: Sequence[tuple[str, str]],
    rules: NodeRules,
) -> tuple[list[tuple[str, str]], str]:
    """
    Choose the nodes that may host agent loop workers, hence episode containers.

    Args:
        candidates (Sequence[tuple[str, str]]): `(node_id, node_ip)` pairs, already narrowed
            by `agent.node_ips` when that is set.
        rules (NodeRules): The node rules.

    Returns:
        tuple[list[tuple[str, str]], str]: The selected pairs and a note for the log (empty
        when the policy changed nothing).

    Raises:
        ValueError: When the policy leaves no node.
    """
    permitted = [(nid, ip) for nid, ip in candidates if rules.node_allowed(ip)]
    if not permitted:
        raise ValueError(
            f"env_service allow/deny lists leave no node. Candidates: {sorted(ip for _, ip in candidates)}, "
            f"allow={sorted(rules.allow_node_ips)}, deny={sorted(rules.deny_node_ips)}."
        )
    others = [(nid, ip) for nid, ip in permitted if not rules.is_train_rollout(ip)]
    if rules.policy is NodePolicy.AVOID:
        if not others:
            raise ValueError(
                "env_service node policy 'avoid' leaves no node: every candidate hosts train or rollout GPUs "
                f"({sorted(rules.train_rollout_node_ips)}). Add a dedicated env node or use 'cpu_only'."
            )
        return others, f"policy=avoid removed {len(permitted) - len(others)} train/rollout node(s)"
    if rules.policy is NodePolicy.PREFER_OTHER:
        if others:
            return others, f"policy=prefer_other kept {len(others)} non-train/rollout node(s)"
        return permitted, "policy=prefer_other found no other node, falling back to train/rollout nodes"
    return permitted, ""


def restrict_to_unit_nodes(
    candidates: Sequence[tuple[str, str]],
    unit_node_ips: Iterable[str],
) -> tuple[list[tuple[str, str]], str]:
    """
    Keep only the nodes that have at least one declared env unit.

    A container is admitted only to a declared unit on the node of the worker that starts
    it. A worker on a node with no unit would have every episode refused, and the manager
    would then refill each failed group, so such a worker is never placed.

    Args:
        candidates (Sequence[tuple[str, str]]): `(node_id, node_ip)` pairs.
        unit_node_ips (Iterable[str]): Nodes that declare at least one unit.

    Returns:
        tuple[list[tuple[str, str]], str]: The kept pairs and a log note (empty when nothing
        was dropped).

    Raises:
        ValueError: When no candidate node has a unit.
    """
    unit_nodes = frozenset(unit_node_ips)
    kept = [(nid, ip) for nid, ip in candidates if ip in unit_nodes]
    if not kept:
        raise ValueError(
            f"No candidate node declares an env unit. Candidates: {sorted(ip for _, ip in candidates)}, "
            f"nodes with units: {sorted(unit_nodes)}. Declare a unit on a node that may host containers, "
            f"or relax psrl.env_service.node_policy."
        )
    dropped = sorted(ip for _, ip in candidates if ip not in unit_nodes)
    note = f"no agent loop worker on {dropped}: those nodes declare no env unit" if dropped else ""
    return kept, note


def node_ips_from_worker_groups(
    worker_groups: dict[str, object],
    placement_group_table: Callable[[object], dict],
    node_id_to_ip: dict[str, str],
    skip_prefixes: tuple[str, ...] = ("ps",),
) -> frozenset[str]:
    """
    Collect the IPs of every node that hosts a train, rollout, validation or reward-model GPU.

    Reads placement-group metadata rather than calling actors, so it works before the
    workers finish initializing.

    Args:
        worker_groups (dict[str, object]): Named Ray worker groups, each exposing
            `.resource_pool.pgs`.
        placement_group_table (Callable[[object], dict]): Maps a placement group to its GCS
            table, e.g. `lambda pg: ray._private.state.state.placement_group_table(pg.id)`.
        node_id_to_ip (dict[str, str]): Ray node id to node IP.
        skip_prefixes (tuple[str, ...]): Worker group names to ignore (the PS pool is CPU only).

    Returns:
        frozenset[str]: Node IPs. Unknown node ids are skipped rather than guessed.
    """
    ips: set[str] = set()
    for name, group in worker_groups.items():
        if any(name.startswith(prefix) for prefix in skip_prefixes):
            continue
        pool = getattr(group, "resource_pool", None)
        pgs = getattr(pool, "pgs", None)
        if not pgs:
            continue
        for pg in pgs:
            bundles_to_node = placement_group_table(pg).get("bundles_to_node_id", {})
            for node_id in bundles_to_node.values():
                ip = node_id_to_ip.get(node_id)
                if ip:
                    ips.add(ip)
    return frozenset(ips)
