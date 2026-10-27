"""Serialize GPU command reservations while retaining host resource leases."""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CommandRequest:
    """Identify one command across retries and delayed acknowledgements."""

    lease_id: str
    container_id: str
    unit_id: str
    state: str = "waiting"
    epoch: int = 0
    error: str | None = None
    returned_at: float | None = None
    tag: str = ""
    telemetry: dict | None = None
    admission_memory: dict | None = None


class GpuCommandScheduler:
    """Reserve before enabling devices and revoke devices before releasing capacity."""

    def __init__(self, ledger, get_node):
        self.ledger = ledger
        self.get_node = get_node
        self.locks = defaultdict(asyncio.Lock)
        self.requests: dict[str, CommandRequest] = {}
        self.epochs: dict[str, int] = defaultdict(int)
        self.returns: dict[str, asyncio.Task] = {}
        self.return_timeout_s = 30.0

    def _lease(self, lease_id):
        lease = self.ledger._leases.get(lease_id)
        if lease is None or lease.gpu_reservation != "command" or not lease.gpu_indices:
            raise RuntimeError("GPU commands require a live command reservation lease.")
        if lease.gpu_closing:
            raise RuntimeError("GPU command lease is closing for cleanup.")
        return lease

    def _node_indices(self, node_ip):
        return tuple(
            sorted({i for unit in self.ledger._units.values() if unit.node_ip == node_ip for i in unit.gpu_indices})
        )

    async def acquire(self, lease_id, container_id, request_id, tag=""):
        """Try the FIFO queue head without consuming host resources a second time."""
        lease = self._lease(lease_id)
        async with self.locks[lease.unit_id]:
            lease = self._lease(lease_id)
            request = self.requests.get(request_id)
            if request is not None:
                if (request.lease_id, request.container_id, request.tag) != (lease_id, container_id, tag):
                    raise RuntimeError("GPU request identity does not match its owner.")
                if request.state == "active":
                    return {"status": "granted", "epoch": request.epoch}
                if request.state != "waiting":
                    raise RuntimeError("GPU command request is retired or quarantined.")
            else:
                if any(r.lease_id == lease_id and r.state in ("waiting", "active") for r in self.requests.values()):
                    raise RuntimeError("A lease may have only one pending GPU command.")
                request = CommandRequest(lease_id, container_id, lease.unit_id)
                request.tag = tag
                self.requests[request_id] = request
                self.ledger._gpu_command_waiters[request_id] = lease.unit_id
            if lease.unit_id in self.ledger._blocked_units:
                raise RuntimeError("GPU command unit awaits verified cleanup.")
            if any(r.lease_id == lease_id and r.state == "returning" for r in self.requests.values()):
                return {"status": "wait"}
            head = next(k for k, r in self.requests.items() if r.unit_id == lease.unit_id and r.state == "waiting")
            if head != request_id:
                return {"status": "wait"}
            unit = self.ledger._units[lease.unit_id]
            # Full declared capacity cannot be rescued by telemetry, so avoid polling the device.
            if self.ledger._declared_gpu_mb(unit) + lease.demand.gpu_memory_mb > unit.gpu_memory_mb:
                return {"status": "wait"}
            node = await self.get_node(lease.node_ip)
            commands = [
                (r.lease_id, r.container_id, r.epoch)
                for r in self.requests.values()
                if r.unit_id == lease.unit_id and r.state in ("active", "returning")
            ]
            memory = await node.gpu_command_memory_snapshot.remote(self._node_indices(lease.node_ip), commands)
            if self.ledger._leases.get(lease_id) is not lease or lease.unit_id in self.ledger._blocked_units:
                raise RuntimeError("GPU command lease expired during telemetry.")
            readings = memory["devices"]
            self.ledger.report_measurement(lease.node_ip, {i: r["used_mb"] for i, r in readings.items()})
            used = self.ledger._effective_gpu_mb(unit, self.ledger._clock(), memory["owned_mb"])
            request.admission_memory = {
                "effective_mb": used,
                "owned_mb": memory["owned_mb"],
                "attribution_errors": memory["attribution_errors"],
            }
            physical_capacity = sum(readings[index]["total_mb"] for index in lease.gpu_indices)
            if used + lease.demand.gpu_memory_mb > min(unit.gpu_memory_mb, physical_capacity):
                return {"status": "wait"}
            device_used = (
                (used,)
                if len(unit.gpu_indices) == 1
                else self.ledger._effective_gpu_by_device(unit, self.ledger._clock())
            )
            request.admission_memory["effective_per_device_mb"] = dict(zip(unit.gpu_indices, device_used))
            if any(
                occupied + requested > min(budget, readings[index]["total_mb"])
                for index, occupied, requested, budget in zip(
                    unit.gpu_indices, device_used, lease.demand.device_memory(unit), unit.gpu_memory_per_device_mb
                )
            ):
                return {"status": "wait"}
            self.epochs[lease_id] += 1
            request.epoch = self.epochs[lease_id]
            request.state = "enabling"
            lease.gpu_active = True
            try:
                await node.grant_gpu_command.remote(lease_id, container_id, lease.gpu_indices, request.epoch)
                if self.ledger._leases.get(lease_id) is not lease:
                    raise RuntimeError("GPU command lease expired during device activation.")
            except BaseException:
                request.state = "quarantined"
                self.ledger.block_gpu_cleanup(lease_id, lease.unit_id)
                raise
            request.state = "active"
            self.ledger._gpu_command_waiters.pop(request_id, None)
            return {"status": "granted", "epoch": request.epoch}

    def return_status(self, lease_id, request_id, epoch):
        """Read an owned command without allowing a stale acknowledgement to change state."""
        request = self.requests.get(request_id)
        if request is None or request.lease_id != lease_id or request.epoch != epoch:
            raise RuntimeError("GPU command release has a stale owner or epoch.")
        return {"state": request.state, "error": request.error, "returned_at": request.returned_at}

    async def begin_return(self, lease_id, request_id, epoch):
        """Acknowledge service ownership of a bounded return task while retaining capacity."""
        self.return_status(lease_id, request_id, epoch)
        request = self.requests[request_id]
        async with self.locks[request.unit_id]:
            self.return_status(lease_id, request_id, epoch)
            if request.state in ("returning", "completed", "quarantined"):
                return self.return_status(lease_id, request_id, epoch)
            if request.state != "active":
                raise RuntimeError("GPU command is not in an active phase.")
            lease = self._lease(lease_id)
            request.state = "returning"
            self.returns[request_id] = asyncio.create_task(self._return(lease, request))
            return self.return_status(lease_id, request_id, epoch)

    async def _return(self, lease, request):
        """Perform device I/O outside the unit lock and commit the acknowledgement under it."""
        try:
            async with asyncio.timeout(self.return_timeout_s):
                node = await self.get_node(lease.node_ip)
                readings = await node.revoke_gpu_command.remote(
                    lease.lease_id,
                    request.container_id,
                    lease.gpu_indices,
                    request.epoch,
                    self._node_indices(lease.node_ip),
                )
            async with self.locks[request.unit_id]:
                if self.ledger._leases.get(lease.lease_id) is not lease:
                    raise RuntimeError("GPU command lease expired during return.")
                self.ledger.report_measurement(lease.node_ip, {i: r["used_mb"] for i, r in readings.items()})
                request.telemetry = next(iter(readings.values())).get("task_telemetry")
                lease.gpu_active = False
                request.returned_at = time.time()
                request.state = "completed"
        except BaseException as exc:
            async with self.locks[request.unit_id]:
                request.error = f"{type(exc).__name__}: {exc}"
                request.state = "quarantined"
                self.ledger.block_gpu_cleanup(lease.lease_id, lease.unit_id)

    async def release(self, lease_id, request_id, epoch):
        """Join service owned return without letting a cancelled caller abandon cleanup."""
        await self.begin_return(lease_id, request_id, epoch)
        request = self.requests[request_id]
        task = self.returns.get(request_id)
        if task is not None:
            await asyncio.shield(task)
        status = {"state": request.state, "error": request.error, "returned_at": request.returned_at}
        if status["state"] != "completed":
            raise RuntimeError(f"GPU command return failed: {status['error']}")
        return status

    async def cancel(self, lease_id, request_id):
        """Retire a queued request or return an acknowledged active command."""
        request = self.requests.get(request_id)
        if request is None:
            lease = self._lease(lease_id)
            async with self.locks[lease.unit_id]:
                request = self.requests.get(request_id)
                if request is None:
                    self.requests[request_id] = CommandRequest(lease_id, "", lease.unit_id, "cancelled")
                    return
        if request.lease_id != lease_id:
            raise RuntimeError("Cannot cancel another lease's GPU command.")
        async with self.locks[request.unit_id]:
            if request.state == "waiting":
                request.state = "cancelled"
                self.ledger._gpu_command_waiters.pop(request_id, None)
                return
            active = request.state in ("active", "returning", "quarantined")
        if active:
            await self.release(lease_id, request_id, request.epoch)

    async def wait_returns(self, lease_id):
        """Reject deferred command failures before an episode result is accepted."""
        lease = self._lease(lease_id)
        async with self.locks[lease.unit_id]:
            requests = [(key, r) for key, r in self.requests.items() if r.lease_id == lease_id]
            pending = [self.returns[key] for key, _ in requests if key in self.returns]
        if pending:
            await asyncio.shield(asyncio.gather(*pending))
        for _, request in requests:
            if request.state not in ("completed", "cancelled"):
                raise RuntimeError(f"GPU command is not safely completed: {request.state}, {request.error}")

    def reports(self, lease_id):
        self._lease(lease_id)
        return [
            {
                "request_id": key,
                "tag": request.tag,
                "state": request.state,
                "telemetry": request.telemetry,
                "admission_memory": request.admission_memory,
            }
            for key, request in self.requests.items()
            if request.lease_id == lease_id
        ]

    async def telemetry(self, lease_id, request_id):
        lease = self._lease(lease_id)
        request = self.requests.get(request_id)
        if request is None:
            return None
        if request.lease_id != lease_id:
            raise RuntimeError("GPU telemetry request belongs to another lease.")
        if request.state not in ("active", "returning"):
            return request.telemetry
        node = await self.get_node(lease.node_ip)
        return await node.gpu_job_telemetry.remote(lease_id, request.container_id, request.epoch)

    async def finish(self, lease_id, unit_id):
        """Remove the lease's containers before retiring uncertain command state."""
        async with self.locks[unit_id]:
            lease = self.ledger._leases.get(lease_id)
            if lease is not None:
                lease.gpu_closing = True
            self.ledger.block_gpu_cleanup(lease_id, unit_id)
            pending = [
                self.returns[k] for k, r in self.requests.items() if r.lease_id == lease_id and k in self.returns
            ]
        if pending:
            await asyncio.shield(asyncio.gather(*pending))
        async with self.locks[unit_id]:
            self.ledger.block_gpu_cleanup(lease_id, unit_id)
            unit = self.ledger._units[unit_id]
            node = await self.get_node(unit.node_ip)
            await node.remove_gpu_lease.remote(lease_id)
            readings = await node.gpu_lending_snapshot.remote(self._node_indices(unit.node_ip))
            self.ledger.report_measurement(unit.node_ip, {i: r["used_mb"] for i, r in readings.items()})
            for key in [key for key, request in self.requests.items() if request.lease_id == lease_id]:
                del self.requests[key]
                self.returns.pop(key, None)
                self.ledger._gpu_command_waiters.pop(key, None)
            self.epochs.pop(lease_id, None)
            lease = self.ledger._leases.get(lease_id)
            if lease is not None:
                lease.gpu_active = False
            self.ledger.confirm_gpu_cleanup(lease_id, unit_id)
