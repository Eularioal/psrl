"""Observe physical GPUs and freeze only containers owned by a lending lease."""

import asyncio
import json
import time

from psrl.workers.env_service.overlay import LEASE_LABEL
from psrl.workers.env_service.probe import NvmlProbe


async def run_command(*args: str) -> str:
    """Run a bounded probe and reap its process on timeout or cancellation."""
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), 10)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise
    if proc.returncode:
        raise RuntimeError(f"GPU handoff command {args[0]!r} failed: {stderr.decode(errors='replace')}.")
    return stdout.decode()


def returned_to_baseline(snapshot, baseline, tolerance_mb: float) -> bool:
    """Require stable GPU identity, no new CUDA processes and reclaimed memory."""
    if not baseline or snapshot.keys() != baseline.keys():
        return False
    return all(
        reading["uuid"] == baseline[index]["uuid"]
        and set(reading["pids"]) <= set(baseline[index]["pids"])
        and reading["used_mb"] <= baseline[index]["used_mb"] + tolerance_mb
        for index, reading in snapshot.items()
    )


class GpuLendingNodeMixin:
    """Perform node local handoff actions without touching unrelated containers."""

    def _gpu_probe(self):
        if not hasattr(self, "_nvml_probe"):
            self._nvml_probe = NvmlProbe()
        return self._nvml_probe

    def _gpu_lease_state(self, lease_id):
        # Retain retired identities so delayed RPCs cannot unfreeze a returned lease.
        if not hasattr(self, "_gpu_lease_states"):
            self._gpu_lease_states = {}
        return self._gpu_lease_states.setdefault(lease_id, {"lock": asyncio.Lock(), "retired": False})

    async def gpu_lending_snapshot(self, indices):
        return await asyncio.wait_for(asyncio.to_thread(self._gpu_probe().read_gpu_snapshot, indices), timeout=10)

    async def _lease_containers(self, lease_id):
        ids = (await run_command("docker", "ps", "-aq", "--filter", f"label={LEASE_LABEL}={lease_id}")).split()
        if not ids:
            return []
        return json.loads(await run_command("docker", "inspect", *ids))

    async def _pause_gpu_lease(self, lease_id):
        containers = await self._lease_containers(lease_id)
        for container in containers:
            if container["State"]["Running"] and not container["State"]["Paused"]:
                await run_command("docker", "pause", container["Id"])

    async def resume_gpu_lease(self, lease_id):
        state = self._gpu_lease_state(lease_id)
        async with state["lock"]:
            if state["retired"]:
                raise RuntimeError("Cannot resume a returned GPU lease.")
            await self._resume_gpu_lease(lease_id)

    async def _resume_gpu_lease(self, lease_id):
        for container in await self._lease_containers(lease_id):
            if container["State"]["Paused"]:
                await run_command("docker", "unpause", container["Id"])

    async def freeze_gpu_lease(self, lease_id, baseline, tolerance_mb, timeout_s):
        """Freeze after CUDA work exits, preserving the shell and filesystem."""
        state = self._gpu_lease_state(lease_id)
        async with state["lock"]:
            if state["retired"]:
                raise RuntimeError("Cannot freeze a returned GPU lease.")
            await self._freeze_gpu_lease(lease_id, baseline, tolerance_mb, timeout_s)

    async def _freeze_gpu_lease(self, lease_id, baseline, tolerance_mb, timeout_s):
        if not baseline:
            raise RuntimeError("GPU return needs an acknowledged sleep baseline.")
        deadline = time.monotonic() + timeout_s
        while True:
            await self._pause_gpu_lease(lease_id)
            snapshot = await self.gpu_lending_snapshot(tuple(baseline))
            if returned_to_baseline(snapshot, baseline, tolerance_mb):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Command still owns GPU resources at the model boundary.")
            await self._resume_gpu_lease(lease_id)
            await asyncio.sleep(0.1)

    async def remove_gpu_lease(self, lease_id):
        state = self._gpu_lease_state(lease_id)
        async with state["lock"]:
            state["retired"] = True
            if state.get("command_telemetry"):
                await state["command_telemetry"].stop()
            await self._remove_gpu_lease(lease_id)

    async def _remove_gpu_lease(self, lease_id):
        for container in await self._lease_containers(lease_id):
            await run_command("docker", "rm", "-f", container["Id"])
        if await self._lease_containers(lease_id):
            raise RuntimeError("GPU lease containers remain after cleanup.")

    async def confirm_gpu_return(self, indices, baseline, tolerance_mb):
        snapshot = await self.gpu_lending_snapshot(indices)
        if not returned_to_baseline(snapshot, baseline, tolerance_mb):
            raise RuntimeError("Physical GPU return is unconfirmed. Keeping the replica fenced.")
