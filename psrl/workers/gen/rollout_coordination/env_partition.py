"""Adapt rollout replica capacity while preserving command and version ownership."""

import asyncio
import math
import time
from collections import deque

from psrl.workers.gen.smg_adapter import build_pause_resume_payload, build_weight_version_updates


class PartitionPolicy:
    """Require sustained pressure and spare request capacity before changing roles."""

    def __init__(self, config):
        self.config = config
        self.low_since = None
        self.high_since = None
        self.last_change = -math.inf
        self.last_sample = None

    def observe(self, now, active, borrowed, load, backlog, pressure):
        """Choose one transition using request counts from the same observation."""
        c = self.config
        if self.last_sample is not None and now - self.last_sample > 3 * c["interval_s"]:
            self.low_since = self.high_since = None
        self.last_sample = now
        low = pressure and not backlog and active > c["min_replicas"]
        low = low and load <= (active - 1) * c["request_capacity"] * c["shrink_fraction"]
        high = borrowed and (backlog > 0 or load > active * c["request_capacity"] * c["restore_fraction"])
        self.low_since = (self.low_since if self.low_since is not None else now) if low else None
        self.high_since = (self.high_since if self.high_since is not None else now) if high else None
        if self.high_since is not None and now - self.high_since >= c["restore_window_s"]:
            return "restore"
        if now - self.last_change < c["cooldown_s"]:
            return None
        if self.low_since is not None and now - self.low_since >= c["shrink_window_s"]:
            return "borrow"
        return None

    def changed(self):
        """Discard observations gathered under the previous allocation."""
        self.last_change = time.monotonic()
        self.low_since = self.high_since = None


class EnvPartitionMixin:
    """Coordinate SMG admission, sleeping model capacity and EnvRouter roles."""

    async def configure_env_partition(self, router, units):
        """Validate the supported topology before starting automatic transitions."""
        c = dict(self.config.psrl.env_service.elastic_partition)
        if not c["enable"]:
            return
        if hasattr(self, "_env_partition_units"):
            raise ValueError("Environment partition controller is already configured.")
        if self.config.psrl.ps_mode != "nixl_cpu" or self.config.psrl.profile.fix_weight:
            raise ValueError("Elastic environment partitions require mutable nixl_cpu policy weights.")
        if self.config.psrl.deployment.elastic_rm.enable or self._thunder_agent_enabled:
            raise ValueError("Environment partitions cannot share control with elastic RM or pinned sessions.")
        if self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.enable:
            raise ValueError("Environment partitions require request migration to be disabled.")
        if not self.config.psrl.status_collection.enable:
            raise ValueError("Environment partitions require live engine statistics.")
        positive = (
            "request_capacity",
            "interval_s",
            "shrink_window_s",
            "restore_window_s",
            "cooldown_s",
            "transition_timeout_s",
            "stats_max_age_s",
            "borrowed_max_gpu_commands",
            "sleep_memory_mb",
        )
        if any(not math.isfinite(c[key]) or c[key] <= 0 for key in positive):
            raise ValueError("Environment partition bounds must be finite and positive.")
        if not 0 < c["shrink_fraction"] < c["restore_fraction"] <= 1:
            raise ValueError("Partition thresholds require hysteresis within unit capacity.")
        mapped = {}
        for unit in units:
            replica_id = self.replica_idx_to_replica_id[unit["rank"]]
            instances = [item for item in self.instance_ids if item[0] == replica_id]
            if len(instances) != 1 or len(unit["gpus"]) != 1 or not unit["pool"]:
                raise ValueError("Environment partitions require complete single GPU replicas in a command pool.")
            if replica_id in mapped:
                raise ValueError("Each rollout replica must have one environment partition unit.")
            await self.server_handles[replica_id].validate_gpu_lending.remote()
            mapped[replica_id] = unit["unit_id"]
        if set(mapped) != self.tag_to_replica_ids["rollout"] or not 1 <= c["min_replicas"] <= len(mapped):
            raise ValueError("All rollout replicas must be explicitly bound and at least one must stay active.")
        self._env_partition_router = router
        self._env_partition_config = c
        self._env_partition_units = mapped
        self._env_partition_states = dict.fromkeys(mapped, "resident")
        self._env_partition_policy = PartitionPolicy(c)
        self._env_partition_events = deque(maxlen=1024)
        await router.configure_partition_commands.remote(c["borrowed_max_gpu_commands"])
        self._env_partition_task = asyncio.create_task(self._env_partition_loop())

    def env_partition_status(self):
        """Expose decisions and acknowledged roles for experiment audits."""
        return {
            "states": dict(getattr(self, "_env_partition_states", {})),
            "events": list(getattr(self, "_env_partition_events", ())),
        }

    async def _partition_worker_control(self, replica_id, action):
        instances = self._expand_replica_instance_ids([(replica_id, 0)])
        response = await self._gateway_post_json(f"/workers/{action}", payload=build_pause_resume_payload(instances))
        if response.get("rejected", 0) or any(x.get("status") == "rejected" for x in response.get("results", [])):
            raise RuntimeError(f"Gateway rejected partition action {action!r}: {response!r}.")

    async def _partition_role(self, replica_id, state):
        unit = self._env_partition_units[replica_id]
        await self._env_partition_router.set_partition_unit_state.remote(unit, state)
        self._env_partition_states[replica_id] = state
        self._env_partition_events.append({"at": time.time(), "replica": replica_id, "state": state})

    async def _borrow_env_replica(self, replica_id):
        server = self.server_handles[replica_id]
        overlap = self._env_partition_config.get("handover_overlap", False)
        # Synchronization selects replicas under the same lock before launching its pull.
        async with self._env_gpu_transition_lock:
            await self._wait_for_replica_syncs([replica_id])
            self._env_gpu_reservations[replica_id] = "partition"
            state = "draining_shared" if overlap else "draining"
            await self._partition_role(replica_id, state)
            await server.set_env_partition_state.remote(state)
        # The global barrier only covers dispatch handoff, not request execution or environment return.
        async with self._routing_pause_controller().pause():
            await self._partition_worker_control(replica_id, "pause")
        self._env_partition_events.append({"at": time.time(), "replica": replica_id, "phase": "gateway_paused"})
        await server.wait_for_requests_to_drain.remote()
        self._env_partition_events.append({"at": time.time(), "replica": replica_id, "phase": "engine_drained"})
        if overlap:
            # Freeze admissions while existing resident commands keep their ownership.
            await self._partition_role(replica_id, "draining")
            await server.set_env_partition_state.remote("retiring")
        await server.pause_generation.remote(mode="wait", wait_for_inflight_requests=True, clear_cache=False)
        await server.close_grpc_generate_admission.remote()
        await server.nixl_sleep.remote(level=1)
        memory = await self._env_partition_router.verify_partition_sleep.remote(
            self._env_partition_units[replica_id], self._env_partition_config["sleep_memory_mb"]
        )
        self._env_partition_events.append(
            {
                "at": time.time(),
                "replica": replica_id,
                "phase": "sleep_verified",
                "non_environment_mb": memory["non_environment_mb"],
                "owned_environment_mb": sum(memory["owned_mb"].values()),
                "device_used_mb": next(iter(memory["devices"].values()))["used_mb"],
            }
        )
        await server.set_env_partition_state.remote("sleeping")
        await self._partition_role(replica_id, "borrowed")

    async def _restore_env_replica(self, replica_id):
        server = self.server_handles[replica_id]
        await self._partition_role(replica_id, "restoring")
        await server.set_env_partition_state.remote("restoring")
        # Waking waits for command release, which is acknowledged only after physical cleanup.
        await server.nixl_wake_up.remote()
        await server.pause_for_sync.remote()
        actual = await server.pull_model_for_sync.remote(self.ps_model_version)
        instances = self._expand_replica_instance_ids([(replica_id, 0)])
        async with self._env_gpu_transition_lock:
            async with self._routing_pause_controller().pause():
                await self._publish_weight_version_updates(build_weight_version_updates(instances, actual))
                for item in instances:
                    self.set_rollout_instance_model_version(item, actual)
                    self.instance_to_version_after_sync[item] = actual
                await server.resume_after_sync.remote()
                await self._partition_worker_control(replica_id, "resume")
                await server.set_env_partition_state.remote("resident")
                await self._partition_role(replica_id, "resident")
                self._env_gpu_reservations.pop(replica_id)
        self._env_partition_events.append({"at": time.time(), "replica": replica_id, "restored_version": actual})

    async def _partition_transition(self, replica_id, action):
        try:
            async with asyncio.timeout(self._env_partition_config["transition_timeout_s"]):
                if action == "borrow":
                    await self._borrow_env_replica(replica_id)
                else:
                    await self._restore_env_replica(replica_id)
        except BaseException as exc:
            # A timed out RPC may still finish remotely, so neither owner may reuse this device.
            self._env_gpu_reservations[replica_id] = "partition"
            self._env_partition_states[replica_id] = "quarantined"
            self._env_partition_events.append({"at": time.time(), "replica": replica_id, "error": repr(exc)})
            async with asyncio.timeout(10):
                await self._env_partition_router.set_partition_unit_state.remote(
                    self._env_partition_units[replica_id], "quarantined"
                )
                await self.server_handles[replica_id].set_env_partition_state.remote("quarantined")
                await self._partition_worker_control(replica_id, "pause")
            raise
        self._env_partition_policy.changed()

    async def _env_partition_tick(self):
        c = self._env_partition_config
        started = time.monotonic()
        active = [r for r, state in self._env_partition_states.items() if state == "resident"]
        borrowed = [r for r, state in self._env_partition_states.items() if state == "borrowed"]
        async with asyncio.timeout(c["stats_max_age_s"]):
            routing = await self._gateway_get_json("/routing_loop/status")
        reason = None
        if routing.get("paused") or self.replica_sync_tasks:
            reason = "routing_paused_or_sync_active"
        elif any(self.instance_to_model_version[(r, 0)] != self.ps_model_version for r in active):
            reason = "weight_version_mismatch"
        if reason:
            self._env_partition_policy.low_since = self._env_partition_policy.high_since = None
            self._env_partition_events.append({"at": time.time(), "skipped": reason})
            return
        # Query the live frontend even when idle: iteration logs are not heartbeats.
        try:
            async with asyncio.timeout(c["stats_max_age_s"]):
                loads = await asyncio.gather(*(self.server_handles[r].env_partition_load.remote() for r in active))
            counts = dict(zip(active, loads, strict=True))
            if any(type(n) is not int or n < 0 for n in loads):
                raise ValueError("Invalid live request count.")
        except (TimeoutError, RuntimeError, ValueError) as exc:
            self._env_partition_policy.low_since = self._env_partition_policy.high_since = None
            self._env_partition_events.append(
                {"at": time.time(), "skipped": "live_load_unavailable", "error": repr(exc)}
            )
            return
        backlog = routing.get("pending_request_num", routing.get("queue_len"))
        if backlog is None:
            raise RuntimeError("Gateway does not expose pending request count for partition decisions.")
        async with asyncio.timeout(c["stats_max_age_s"]):
            pressure = await self._env_partition_router.partition_snapshot.remote()
        need_env = any(x["waiting"] or x["shared_active"] for x in pressure.values())
        now = time.monotonic()
        policy = self._env_partition_policy
        previous = policy.last_sample
        action = policy.observe(now, len(active), len(borrowed), sum(counts.values()), backlog, need_env)
        self._env_partition_events.append(
            {
                "at": time.time(),
                "load": sum(counts.values()),
                "replica_loads": counts,
                "poll_s": now - started,
                "sample_gap_s": None if previous is None else now - previous,
                "low_window_s": None if policy.low_since is None else now - policy.low_since,
                "backlog": backlog,
                "active": len(active),
                "borrowed": len(borrowed),
                "env_pressure": need_env,
                "action": action,
            }
        )
        if action:
            replica_id = min(active, key=lambda r: (counts[r], r)) if action == "borrow" else borrowed[0]
            await self._partition_transition(replica_id, action)

    async def _env_partition_loop(self):
        try:
            while not self.stop_sync_and_migrate:
                await asyncio.sleep(self._env_partition_config["interval_s"])
                await self._env_partition_tick()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._env_partition_events.append({"at": time.time(), "controller_error": repr(exc)})
            # Stop automatic changes after failure, preserving every ownership fence.
            self._env_partition_failed = repr(exc)

    async def stop_env_partition(self):
        """Stop decisions without pretending borrowed devices have physically returned."""
        task = getattr(self, "_env_partition_task", None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
