"""Instrument the real command protocol in probe actors without changing admission rules."""

import asyncio
import functools
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar

import ray
from psrl.workers.env_service import command_gpu_node, lending_node
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.probe import NvmlProbe
from psrl.workers.env_service.router import EnvRouter
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

_scope = ContextVar("gpu_trace_scope", default="background")


class Trace:
    """Keep bounded in-memory spans so measurement does not write to shared storage."""

    def __init__(self):
        self.events = deque(maxlen=10000)

    @contextmanager
    def span(self, label):
        started = time.time()
        monotonic = time.perf_counter()
        event = {"label": label, "scope": _scope.get(), "started_at": started}
        try:
            yield event
        except BaseException as exc:
            event["error"] = type(exc).__name__
            raise
        finally:
            event["duration_s"] = time.perf_counter() - monotonic
            self.events.append(event)

    def async_wrap(self, obj, name, identify=None):
        original = getattr(obj, name)

        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            token = _scope.set(identify(args)) if identify else None
            try:
                with self.span(name) as event:
                    result = await original(*args, **kwargs)
                    if isinstance(result, dict) and "status" in result:
                        event["status"] = result["status"]
                    return result
            finally:
                if token is not None:
                    _scope.reset(token)

        setattr(obj, name, wrapped)

    def sync_wrap(self, obj, name):
        original = getattr(obj, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.span(name):
                return original(*args, **kwargs)

        setattr(obj, name, wrapped)

    def dump(self):
        return {"pid": os.getpid(), "events": list(self.events)}


class TracedLock:
    """Measure lock waits separately from work done while the lock is held."""

    def __init__(self, trace):
        self.lock = asyncio.Lock()
        self.trace = trace

    async def __aenter__(self):
        with self.trace.span("unit_lock_wait"):
            await self.lock.acquire()
        self.held = self.trace.span("unit_lock_held")
        self.held.__enter__()
        return self

    async def __aexit__(self, *exc):
        self.held.__exit__(*exc)
        self.lock.release()


class TracedNode(EnvNodeService):
    """Measure the existing subprocess, device and return paths in the node process."""

    def __init__(self, router, node_ip):
        super().__init__(router, node_ip)
        self.trace = Trace()
        original = lending_node.run_command

        async def command(*args):
            with self.trace.span("subprocess:" + " ".join(args[:3])):
                return await original(*args)

        lending_node.run_command = command
        command_gpu_node.run_command = command
        self.trace.sync_wrap(NvmlProbe, "read_gpu_devices")
        self.trace.sync_wrap(NvmlProbe, "read_gpu_snapshot")
        self.trace.async_wrap(command_gpu_node, "inspect_container")
        self.trace.sync_wrap(command_gpu_node.DeviceCgroupFence, "set_access")
        for name in ("gpu_lending_snapshot", "_command_container"):
            self.trace.async_wrap(self, name)
        for name in ("grant_gpu_command", "revoke_gpu_command"):
            self.trace.async_wrap(self, name, lambda args, name=name: f"{name}:{args[0]}:{args[3]}")

    async def export_trace(self):
        return self.trace.dump()


class TracedRouter(EnvRouter.__ray_metadata__.modified_class):
    """Reuse the production router and substitute only instrumented node services."""

    def __init__(self, config, train_rollout_node_ips=()):
        super().__init__(config, train_rollout_node_ips)
        self.trace = Trace()
        self._gpu_commands.locks = defaultdict(lambda: TracedLock(self.trace))
        self.trace.async_wrap(self._gpu_commands, "acquire", lambda args: args[2])
        self.trace.async_wrap(self._gpu_commands, "_return", lambda args: f"return:{args[0].lease_id}")
        self.trace.async_wrap(self, "_start_node")
        self.trace.async_wrap(self, "try_admit", lambda args: args[0])

    def _create_node(self, node_ip):
        nodes = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes() if node["Alive"]}
        service = (
            ray.remote(num_cpus=0)(TracedNode)
            .options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[node_ip], soft=False))
            .remote(ray.get_runtime_context().current_actor, node_ip)
        )
        return service

    async def export_trace(self):
        nodes = {}
        for ip, pending in self._node_services.items():
            node, _ = await pending
            nodes[ip] = await node.export_trace.remote()
        return {"router": self.trace.dump(), "nodes": nodes}
