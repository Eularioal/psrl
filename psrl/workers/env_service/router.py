"""Own environment leases and GPU grants while executors retain command payloads and output."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace

from psrl.workers.env_service.command_gpu import GpuCommandScheduler
from psrl.workers.env_service.health import NodeHealth
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node_policy import NodeRules, lint_units
from psrl.workers.env_service.process_identity import validate_identity_backend
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
        self._gpu_identity_backend = validate_identity_backend(config.get("gpu_identity_backend", "procfs"))
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
        self._gpu_commands = GpuCommandScheduler(self._ledger, self._gpu_command_node)
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
        telemetry = dict(config.get("telemetry") or {})
        self._resource_interval_s = float(telemetry.get("interval_s", 5))
        self._resource_timeout_s = float(telemetry.get("timeout_s", 8))
        self._resource_max_age_s = float(telemetry.get("max_age_s", 15))
        if min(self._resource_interval_s, self._resource_timeout_s, self._resource_max_age_s) <= 0:
            raise ValueError("Resource telemetry interval, timeout and freshness bound must be positive.")
        self._resource_samples = {}
        self._resource_errors = {}

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
                psrl_logger.exception("Lease reaper pass failed. Continuing.")

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
        if lease is not None and lease.gpu_reservation == "command":
            await self.finish_gpu_command_lease(lease_id, lease.unit_id)
        if lease is not None and not lease.gpu_lending and lease.gpu_reservation != "command":
            return self._ledger.retire(lease_id)
        return self._ledger.release(lease_id)

    async def _gpu_command_node(self, node_ip):
        raise RuntimeError("GPU command reservations require an attached node service.")

    async def acquire_gpu_command(self, lease_id, container_id, request_id, tag=""):
        self._ensure_reaper()
        return await self._gpu_commands.acquire(lease_id, container_id, request_id, tag)

    async def gpu_job_reports(self, lease_id):
        return self._gpu_commands.reports(lease_id)

    async def gpu_job_telemetry(self, lease_id, request_id):
        return await self._gpu_commands.telemetry(lease_id, request_id)

    async def require_gpu_attribution(self, lease_id):
        lease = self._gpu_commands._lease(lease_id)
        node = await self._gpu_command_node(lease.node_ip)
        return await node.check_gpu_attribution_support.remote(lease.gpu_indices)

    async def release_gpu_command(self, lease_id, request_id, epoch):
        return await self._gpu_commands.release(lease_id, request_id, epoch)

    async def begin_gpu_command_return(self, lease_id, request_id, epoch):
        """Own command cleanup before allowing its caller to continue model planning."""
        self._ensure_reaper()
        return await self._gpu_commands.begin_return(lease_id, request_id, epoch)

    async def gpu_command_return_status(self, lease_id, request_id, epoch):
        return self._gpu_commands.return_status(lease_id, request_id, epoch)

    async def wait_gpu_command_returns(self, lease_id):
        await self._gpu_commands.wait_returns(lease_id)

    async def cancel_gpu_command(self, lease_id, request_id):
        await self._gpu_commands.cancel(lease_id, request_id)

    async def finish_gpu_command_lease(self, lease_id, unit_id):
        await self._gpu_commands.finish(lease_id, unit_id)

    async def release_session(self, session_id: str) -> bool:
        self._ensure_reaper()
        lease_id = self._ledger._by_session.get(session_id)
        return await self.release(lease_id) if lease_id is not None else False

    async def cancel_wait(self, session_id: str) -> None:
        self._ledger.cancel_wait(session_id)

    async def pending_reaped(self, node_ip: str) -> list[EnvLease]:
        return self._ledger.pending_reaped(node_ip)

    async def acknowledge_reaped(self, node_ip: str, lease_id: str):
        self._ledger.acknowledge_reaped(node_ip, lease_id)

    async def snapshot(self) -> list[UnitSnapshot]:
        snapshots = []
        for snapshot in self._ledger.snapshot():
            sample, age, error = self._resource_state(snapshot.node_ip)
            measured = sample.get("units", {}).get(snapshot.unit_id, {})
            snapshots.append(
                replace(
                    snapshot,
                    containers_running=measured.get("containers_running"),
                    containers_total=measured.get("containers_total"),
                    measured_memory_mb=measured.get("measured_memory_mb"),
                    measured_cpu_cores=measured.get("measured_cpu_cores"),
                    node_cpu_load_1m=sample.get("cpu_load_1m"),
                    node_cpu_count=sample.get("cpu_count"),
                    resource_sample_age_s=age,
                    resource_error=error or measured.get("error", ""),
                )
            )
        return snapshots

    def _resource_state(self, node_ip):
        reading = self._resource_samples.get(node_ip)
        age = None if reading is None else time.monotonic() - reading[0]
        error = self._resource_errors.get(node_ip, "")
        if reading is None:
            return {}, age, error or "No physical resource sample received."
        if error or age > self._resource_max_age_s:
            return {}, age, error or "Physical resource sample is stale."
        return reading[1], age, ""

    async def node_capacity(self) -> dict[str, NodeCapacity]:
        capacity = self._ledger.node_capacity()
        for ip, cap in capacity.items():
            sample, _, _ = self._resource_state(ip)
            load, cpus = sample.get("cpu_load_1m"), sample.get("cpu_count")
            capacity[ip] = replace(cap, node_load_per_cpu=load / cpus if load is not None and cpus else None)
        return capacity

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
            self._node_handles = {}
            self._nodes_closing = False
            prewarm = dict(config.get("prewarm") or {})
            self._node_start_timeout_s = float(prewarm.get("timeout_s", 30))
            concurrency = int(prewarm.get("max_concurrency", 4))
            if self._node_start_timeout_s <= 0 or concurrency <= 0:
                raise ValueError("Node startup timeout and concurrency must be positive.")
            self._node_start_slots = asyncio.Semaphore(concurrency)
            health = dict(config.get("health") or {})
            self._health_interval_s = float(health.get("interval_s", 5))
            self._health_timeout_s = float(health.get("timeout_s", 30))
            self._health_max_failures = int(health.get("max_failures", 3))
            if min(self._health_interval_s, self._health_timeout_s, self._health_max_failures) <= 0:
                raise ValueError("Health probe interval, timeout and retry budget must be positive.")
            self._node_health = {u.node_ip: NodeHealth() for u in self._ledger._units.values()}
            self._health_tasks = {}
            self._resource_tasks = {}

        async def _poll_node_resources(self, node_ip):
            unit_ids = tuple(u.unit_id for u in self._ledger._units.values() if u.node_ip == node_ip)
            while not self._nodes_closing:
                try:
                    async with asyncio.timeout(self._resource_timeout_s):
                        service, _ = await asyncio.shield(self._node_task(node_ip))
                        sample = await service.resource_snapshot.remote(unit_ids, self._resource_timeout_s / 2)
                    self._resource_samples[node_ip] = (time.monotonic(), sample)
                    self._resource_errors.pop(node_ip, None)
                except Exception as exc:
                    self._resource_errors[node_ip] = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(self._resource_interval_s)

        def _create_node(self, node_ip):
            from psrl.workers.env_service.node import EnvNodeService

            nodes = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes() if node["Alive"]}
            service = (
                ray.remote(num_cpus=0)(EnvNodeService)
                .options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[node_ip], soft=False))
                .remote(ray.get_runtime_context().current_actor, node_ip, self._gpu_identity_backend)
            )
            return service

        async def _start_node(self, node_ip):
            service = None
            try:
                async with asyncio.timeout(self._node_start_timeout_s):
                    async with self._node_start_slots:
                        service = self._create_node(node_ip)
                        self._node_handles[node_ip] = service
                        url = await service.ready.remote()
                        units = [u for u in self._ledger._units.values() if u.node_ip == node_ip]
                        if any(u.gpu_reservation == "command" for u in units):
                            await service.check_gpu_command_support.remote()
                        indices = tuple(sorted({i for u in units for i in u.gpu_indices}))
                        readings = await service.prepare.remote(indices)
                        self._validate_device_budgets(units, readings)
                        return service, url
            except BaseException:
                if service is not None:
                    ray.kill(service, no_restart=True)
                    self._node_handles.pop(node_ip, None)
                raise

        @staticmethod
        def _validate_device_budgets(units, readings):
            for unit in units:
                for index, budget in zip(unit.gpu_indices, unit.gpu_memory_per_device_mb):
                    if budget > readings[index]["total_mb"]:
                        raise ValueError(f"GPU {index} budget exceeds physical device capacity.")

        async def _check_node(self, node_ip):
            state = self._node_health[node_ip]
            try:
                async with asyncio.timeout(self._health_timeout_s):
                    task = self._node_services.get(node_ip)
                    if task is not None and task.done() and not task.cancelled() and task.exception():
                        self._node_services.pop(node_ip)
                    service, _ = await asyncio.shield(self._node_task(node_ip))
                    units = [u for u in self._ledger._units.values() if u.node_ip == node_ip]
                    indices = tuple(sorted({i for u in units for i in u.gpu_indices}))
                    readings = await service.prepare.remote(indices)
                    self._validate_device_budgets(units, readings)
                    if readings:
                        self._ledger.report_measurement(node_ip, {i: r["used_mb"] for i, r in readings.items()})
                    pending = await service.cleanup.remote()
                    state.success(time.monotonic(), bool(pending))
            except Exception as exc:
                state.fail(time.monotonic(), exc)
                if isinstance(exc, ray.exceptions.RayActorError):
                    active = any(lease.node_ip == node_ip for lease in self._ledger._leases.values())
                    bound = any(binding.unit.node_ip == node_ip for binding in self._gpu_bindings.values())
                    gpu_cleanup = any(lease.gpu_indices for lease in self._ledger.pending_reaped(node_ip))
                    if not active and not bound and not gpu_cleanup:
                        self._node_services.pop(node_ip, None)
                        self._node_handles.pop(node_ip, None)
                        self._resource_errors[node_ip] = "Node service died. Awaiting fresh resource measurements."
            return state

        def _health_task(self, node_ip, *, force=False):
            if self._nodes_closing:
                return None
            state = self._node_health[node_ip]
            task = self._health_tasks.get(node_ip)
            if task is not None and not task.done():
                return task
            if state.failures >= self._health_max_failures:
                return None
            if force or time.monotonic() - state.checked_at >= self._health_interval_s:
                task = asyncio.create_task(self._check_node(node_ip))
                self._health_tasks[node_ip] = task
                return task
            return None

        async def node_capacity(self):
            capacity = await super().node_capacity()
            for ip, cap in capacity.items():
                self._health_task(ip)
                state = self._node_health[ip]
                capacity[ip] = replace(cap, available=state.available, reason=state.error)
            return capacity

        def _node_task(self, node_ip):
            if self._nodes_closing:
                raise RuntimeError("Node services are closing.")
            if node_ip not in {u.node_ip for u in self._ledger._units.values()}:
                raise ValueError(f"Node {node_ip!r} has no configured environment unit.")
            if node_ip not in self._node_services:
                task = asyncio.create_task(self._start_node(node_ip))
                # Observe background failures while retaining them for the first consumer.
                task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                self._node_services[node_ip] = task
            if node_ip not in self._resource_tasks:
                self._resource_tasks[node_ip] = asyncio.create_task(self._poll_node_resources(node_ip))
            return self._node_services[node_ip]

        async def begin_prewarm(self):
            """Submit one bounded startup per configured node without waiting for readiness."""
            nodes = sorted({u.node_ip for u in self._ledger._units.values()})
            for node_ip in nodes:
                self._node_task(node_ip)
            return nodes

        async def wait_prewarm(self):
            """Join the same startup tasks used by admission without creating leases."""
            nodes = await self.begin_prewarm()
            await asyncio.gather(*(asyncio.shield(self._node_task(ip)) for ip in nodes))
            return nodes

        async def shutdown_nodes(self):
            """Stop owned startup tasks and services after all leases have been released."""
            if self._ledger._leases:
                raise RuntimeError("Cannot close node services with active environment leases.")
            for node_ip in tuple(self._ledger._reaped):
                async with asyncio.timeout(self._health_timeout_s):
                    service, _ = await asyncio.shield(self._node_task(node_ip))
                    await service.cleanup.remote()
                if self._ledger._reaped.get(node_ip):
                    raise RuntimeError("Cannot close node services before physical cleanup is acknowledged.")
            self._nodes_closing = True
            for task in self._resource_tasks.values():
                task.cancel()
            await asyncio.gather(*self._resource_tasks.values(), return_exceptions=True)
            for task in self._health_tasks.values():
                task.cancel()
            await asyncio.gather(*self._health_tasks.values(), return_exceptions=True)
            tasks = list(self._node_services.values())
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            errors = []
            for service in list(self._node_handles.values()):
                try:
                    await asyncio.wait_for(service.close.remote(), timeout=5)
                except Exception as exc:
                    errors.append(repr(exc))
                finally:
                    ray.kill(service, no_restart=True)
            self._node_handles.clear()
            if errors:
                raise RuntimeError(f"Node service shutdown errors: {errors!r}.")

        async def _gpu_command_node(self, node_ip):
            node, _ = await asyncio.shield(self._node_task(node_ip))
            return node

        async def bind_rollout_gpu(self, unit_id, server, coordinator):
            """Bind a verified replica before this unit can admit an episode."""
            from psrl.workers.env_service.lending import GpuLendingBinding

            unit = self._ledger._units[unit_id]
            await server.validate_gpu_lending.remote()
            if unit_id in self._gpu_bindings or unit.rollout_replica_rank is None:
                raise ValueError("GPU lending binding is duplicate or not configured.")
            node, _ = await asyncio.shield(self._node_task(unit.node_ip))
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
                if ip not in self._node_health:
                    return Admission(status="reject", reason="Node has no configured environment unit.")
                state = self._node_health[ip]
                task = self._health_task(ip, force=state.available)
                if task is not None:
                    await asyncio.shield(task)
                if state.available and not self._ledger._reaped.get(ip):
                    _, urls[ip] = await asyncio.shield(self._node_task(ip))
            for ip in urls:
                # Node preparation is newer than telemetry sampled before this RPC.
                admission = self._ledger.try_admit(session_id, demand, ip, reap=False)
                if admission.lease is not None:
                    return replace(admission, lease=replace(admission.lease, node_service_url=urls[ip]))
                if node_ip:
                    return admission
            if all(self._node_health[ip].failures >= self._health_max_failures for ip in node_ips):
                return Admission(status="reject", reason="Node health retry budget exhausted.")
            return Admission(status="wait", reason="Eligible nodes are full, unhealthy or awaiting cleanup.")

    # No Ray CPU reservation: this actor does bookkeeping, and reserving a core could
    # fail to schedule on a fully packed cluster.
    EnvRouter = ray.remote(num_cpus=0)(RayEnvRouter)
except ImportError:  # pragma: no cover
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
    if (config.get("prewarm") or {}).get("enable", True):
        ray.get(handle.begin_prewarm.remote())
    return handle
