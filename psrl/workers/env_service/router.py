"""
The EnvRouter: a single Ray actor that owns the unit ledger.

It is the control plane only. It grants and releases leases and reaps dead ones, but it
never sits on the exec path: containers are started by Harbor inside the agent loop worker,
exactly as before. This is the same relationship SMG has with a vLLM engine (design
document section 6.1).

`EnvRouterCore` is plain Python so tests can drive it without Ray. `EnvRouter` adds
node services for telemetry and cleanup, and is `None` when Ray is not installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace

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
        self._lending_config = dict(config.get("gpu_lending") or {})
        self._gpu_bindings = {}
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

    def _lending_unavailable(self, demand, node_ip):
        """Reject a permanently quarantined pool instead of polling it forever."""
        if not demand.needs_gpu:
            return False
        eligible = [
            unit
            for unit in self._ledger._units.values()
            if (node_ip is None or unit.node_ip == node_ip)
            and unit.has_gpu
            and self._ledger._rules.tier(unit, demand) is not None
        ]
        return bool(eligible) and all(
            unit.unit_id in self._gpu_bindings and self._gpu_bindings[unit.unit_id].phase == "quarantined"
            for unit in eligible
        )

    async def try_admit(
        self,
        session_id: str,
        demand: EnvDemand,
        node_ip: str | None = None,
        measured_gpu_used_mb: Mapping[int, float] | None = None,
    ) -> Admission:
        self._ensure_reaper()
        if self._lending_unavailable(demand, node_ip):
            return Admission(status="reject", reason="Eligible GPU lending units are quarantined.")
        return self._ledger.try_admit(session_id, demand, node_ip, measured_gpu_used_mb)

    async def set_gpu_phase(self, lease_id: str, phase: str, sequence: int):
        """Switch only a live lease whose unit has a verified rollout binding."""
        lease = self._ledger._leases.get(lease_id)
        if lease is None or not lease.gpu_lending or lease.unit_id not in self._gpu_bindings:
            raise RuntimeError("GPU handoff requires a live, configured borrowing lease.")
        binding = self._gpu_bindings[lease.unit_id]
        try:
            await binding.set_phase(
                lease, phase, sequence, is_live=lambda: self._ledger._leases.get(lease_id) is lease
            )
        except BaseException:
            if binding.phase == "quarantined":
                self._ledger._blocked_units.add(lease.unit_id)
                try:
                    await self.finish_gpu_lease(lease_id, lease.unit_id)
                    self._ledger.release(lease_id)
                except BaseException:
                    psrl_logger.exception("GPU handoff cleanup is unconfirmed. The unit remains blocked.")
            raise

    async def begin_gpu_container(self, lease_id: str, token: str):
        """Fence container creation against GPU return and expired leases."""
        lease = self._ledger._leases.get(lease_id)
        if lease is None or not lease.gpu_lending:
            raise RuntimeError("Container start requires a live GPU lending lease.")
        binding = self._gpu_bindings[lease.unit_id]
        if binding.lock.locked() or binding.owner != lease_id or binding.phase != "env":
            raise RuntimeError("Container start requires the environment GPU phase.")
        binding.starting.add(token)

    async def fail_gpu_container(self, lease_id: str, token: str):
        """Keep an ambiguous Docker start fenced until its outcome is reconciled."""
        for unit_id, binding in self._gpu_bindings.items():
            if binding.owner == lease_id and token in binding.starting:
                binding.phase = "quarantined"
                self._ledger._blocked_units.add(unit_id)
                return

    async def end_gpu_container(self, lease_id: str, token: str):
        """Acknowledge even a late container start so physical cleanup can proceed."""
        for binding in self._gpu_bindings.values():
            if binding.owner == lease_id:
                binding.starting.discard(token)
                return

    async def finish_gpu_lease(self, lease_id: str, unit_id: str):
        """Acknowledge physical cleanup before making a borrowed unit available."""
        binding = self._gpu_bindings.get(unit_id)
        if binding is not None:
            self._ledger._blocked_units.add(unit_id)
            await binding.finish(lease_id)
            self._ledger._blocked_units.discard(unit_id)

    async def reclaim_gpu_phases(self, node_ip: str):
        """Bound an environment phase even when the episode keeps heartbeating."""
        for unit_id, binding in self._gpu_bindings.items():
            if (
                binding.unit.node_ip == node_ip
                and binding.owner is not None
                and binding.deadline is not None
                and time.monotonic() >= binding.deadline
            ):
                lease_id = binding.owner
                try:
                    await self.finish_gpu_lease(lease_id, unit_id)
                    self._ledger.release(lease_id)
                except Exception:
                    psrl_logger.exception("Expired GPU phase remains quarantined after cleanup failure.")

    async def heartbeat(self, lease_id: str) -> bool:
        self._ensure_reaper()
        return self._ledger.heartbeat(lease_id)

    async def release(self, lease_id: str) -> bool:
        self._ensure_reaper()
        lease = self._ledger._leases.get(lease_id)
        if lease is not None and lease.gpu_lending:
            await self.finish_gpu_lease(lease_id, lease.unit_id)
        return self._ledger.release(lease_id)

    async def release_session(self, session_id: str) -> bool:
        self._ensure_reaper()
        lease_id = self._ledger._by_session.get(session_id)
        return await self.release(lease_id) if lease_id is not None else False

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
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    class RayEnvRouter(EnvRouterCore):
        """Own node services so cleanup survives the death of an episode worker."""

        def __init__(self, config, train_rollout_node_ips=()):
            super().__init__(config, train_rollout_node_ips)
            self._node_services = {}

        async def _start_node(self, node_ip):
            from psrl.workers.env_service.node import EnvNodeService

            nodes = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes() if node["Alive"]}
            service = (
                ray.remote(num_cpus=0)(EnvNodeService)
                .options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[node_ip], soft=False))
                .remote(ray.get_runtime_context().current_actor, node_ip)
            )
            url = await service.ready.remote()
            return service, url

        async def bind_rollout_gpu(self, unit_id, server, coordinator):
            """Bind a verified replica before this unit can admit an episode."""
            from psrl.workers.env_service.lending import GpuLendingBinding

            unit = self._ledger._units[unit_id]
            await server.validate_gpu_lending.remote()
            if unit_id in self._gpu_bindings or unit.rollout_replica_rank is None:
                raise ValueError("GPU lending binding is duplicate or not configured.")
            if unit.node_ip not in self._node_services:
                self._node_services[unit.node_ip] = asyncio.create_task(self._start_node(unit.node_ip))
            node, _ = await self._node_services[unit.node_ip]
            await node.gpu_lending_snapshot.remote(unit.gpu_indices)
            self._gpu_bindings[unit_id] = GpuLendingBinding(server, node, unit, self._lending_config, coordinator)
            self._ledger._blocked_units.discard(unit_id)

        async def try_admit(self, session_id, demand, node_ip=None, measured_gpu_used_mb=None):
            if self._lending_unavailable(demand, node_ip):
                return Admission(status="reject", reason="Eligible GPU lending units are quarantined.")
            self._ensure_reaper()
            self._ledger.reap()
            node_ips = [node_ip] if node_ip else list({unit.node_ip for unit in self._ledger._units.values()})
            urls = {}
            for ip in node_ips:
                if ip not in self._node_services:
                    self._node_services[ip] = asyncio.create_task(self._start_node(ip))
                service, urls[ip] = await self._node_services[ip]
                if await service.cleanup.remote():
                    return Admission(status="wait", reason="Expired containers are still being removed.")
            if any(self._ledger._reaped.get(ip) for ip in node_ips):
                return Admission(status="wait", reason="Expired containers are awaiting cleanup.")
            admission = self._ledger.try_admit(session_id, demand, node_ip, measured_gpu_used_mb, reap=False)
            if admission.lease is not None:
                admission = replace(
                    admission, lease=replace(admission.lease, node_service_url=urls[admission.lease.node_ip])
                )
            return admission

    # No Ray CPU reservation: this actor does bookkeeping, and reserving a core could
    # fail to schedule on a fully packed cluster.
    EnvRouter = ray.remote(num_cpus=0)(RayEnvRouter)
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
