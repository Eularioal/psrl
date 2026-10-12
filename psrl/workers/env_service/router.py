"""
The EnvRouter: a single Ray actor that owns the unit ledger.

It is the control plane only. It grants and releases leases and reaps dead ones, but it
never sits on the exec path: containers are started by Harbor inside the agent loop worker,
exactly as before. This is the same relationship SMG has with a vLLM engine (design
document section 6.1).

`EnvRouterCore` is plain Python so tests can drive it without Ray. `EnvRouter` is the same
class wrapped as an actor, and is `None` when Ray is not installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable, Mapping

from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules, lint_units
from psrl.workers.env_service.types import (
    Admission,
    EnvDemand,
    EnvLease,
    EnvUnit,
    NodeCapacity,
    UnitSnapshot,
)

psrl_logger = logging.getLogger("psrl.env_service.router")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTER_NAME = "psrl_env_router"


class EnvRouterCore:
    """
    Serializes access to an `EnvUnitLedger` and runs the TTL reaper.

    A Ray async actor runs every method on one event loop, so ledger mutations never
    interleave. Each method here is synchronous inside, with no `await` between reading
    and writing ledger state.

    Args:
        config (Mapping): The `psrl.env_service` section as a plain dict. Reads `units`,
            `default_gpu_memory_mb`, `slot_ttl_s`, `node_policy`, `allow_node_ips`,
            `deny_node_ips`.
        train_rollout_node_ips (Iterable[str]): Nodes that host train, rollout, validation or
            reward-model GPUs.
    """

    def __init__(self, config: Mapping, train_rollout_node_ips: Iterable[str] = ()):
        units = [EnvUnit.from_mapping(raw) for raw in (config.get("units") or ())]
        rules = NodeRules.build(
            policy=config.get("node_policy", "cpu_only"),
            train_rollout_node_ips=train_rollout_node_ips,
            allow_node_ips=config.get("allow_node_ips") or (),
            deny_node_ips=config.get("deny_node_ips") or (),
        )
        self._ledger = EnvUnitLedger(
            units,
            rules,
            default_gpu_memory_mb=int(config.get("default_gpu_memory_mb", 24000)),
            lease_ttl_s=float(config.get("slot_ttl_s", 1800.0)),
        )
        self._warnings = lint_units(units, rules)
        for warning in self._warnings:
            psrl_logger.warning("env_service: %s", warning)
        psrl_logger.info(
            "EnvRouter ready: %d unit(s), policy=%s, train/rollout nodes=%s.",
            len(units),
            rules.policy.value,
            sorted(rules.train_rollout_node_ips),
        )
        self._reap_interval_s = max(1.0, float(config.get("slot_ttl_s", 1800.0)) / 10.0)
        self._reaper: asyncio.Task | None = None

    def _ensure_reaper(self) -> None:
        """Start the reaper on the actor's own loop the first time any method runs."""
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.get_running_loop().create_task(self._reap_forever())

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(self._reap_interval_s)
            try:
                for lease in self._ledger.reap():
                    psrl_logger.warning(
                        "Reclaimed lease %s of session %s on unit %s after missed heartbeats.",
                        lease.lease_id,
                        lease.session_id,
                        lease.unit_id,
                    )
            except Exception:  # noqa: BLE001
                # One bad pass must not kill the control plane for good.
                psrl_logger.exception("Lease reaper pass failed; continuing.")

    async def try_admit(
        self,
        session_id: str,
        demand: EnvDemand,
        node_ip: str | None = None,
        measured_gpu_used_mb: Mapping[int, float] | None = None,
    ) -> Admission:
        self._ensure_reaper()
        return self._ledger.try_admit(session_id, demand, node_ip, measured_gpu_used_mb)

    async def heartbeat(self, lease_id: str) -> bool:
        self._ensure_reaper()
        return self._ledger.heartbeat(lease_id)

    async def release(self, lease_id: str) -> bool:
        self._ensure_reaper()
        return self._ledger.release(lease_id)

    async def release_session(self, session_id: str) -> bool:
        self._ensure_reaper()
        return self._ledger.release_session(session_id)

    async def cancel_wait(self, session_id: str) -> None:
        self._ledger.cancel_wait(session_id)

    async def pop_reaped(self, node_ip: str) -> list[EnvLease]:
        return self._ledger.pop_reaped(node_ip)

    async def snapshot(self) -> list[UnitSnapshot]:
        return self._ledger.snapshot()

    async def node_capacity(self) -> dict[str, NodeCapacity]:
        return self._ledger.node_capacity()

    async def warnings(self) -> list[str]:
        return list(self._warnings)

    async def ping(self) -> bool:
        return True


try:
    import ray

    # No Ray CPU reservation: this actor does bookkeeping, and reserving a core could
    # fail to schedule on a fully packed cluster.
    EnvRouter = ray.remote(num_cpus=0)(EnvRouterCore)
except ImportError:  # pragma: no cover - exercised only on a box without Ray
    EnvRouter = None


def create_env_router(config: Mapping, train_rollout_node_ips: Iterable[str] = ()):
    """
    Create (or reuse) the named router actor and wait until it answers.

    Args:
        config (Mapping): The `psrl.env_service` section as a plain dict.
        train_rollout_node_ips (Iterable[str]): Nodes hosting train or rollout GPUs.

    Returns:
        ray.actor.ActorHandle: A handle to the router.
    """
    import ray

    if EnvRouter is None:  # pragma: no cover
        raise RuntimeError("Ray is required to create the EnvRouter actor.")
    name = str(config.get("router_name", DEFAULT_ROUTER_NAME))
    handle = EnvRouter.options(name=name, get_if_exists=True).remote(dict(config), tuple(train_rollout_node_ips))
    ray.get(handle.ping.remote())
    return handle
