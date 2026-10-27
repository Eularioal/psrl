"""Measure physical containers separately from promised lease resources."""

import asyncio
import os
import time

from psrl.workers.env_service.docker_api import container_stats, list_unit_containers


def cpu_usage_cores(stats):
    """Convert Docker CPU counters to used logical cores when both samples are present."""
    current = stats.get("cpu_stats", {})
    previous = stats.get("precpu_stats", {})
    cpu = current.get("cpu_usage", {}).get("total_usage")
    prior_cpu = previous.get("cpu_usage", {}).get("total_usage")
    system = current.get("system_cpu_usage")
    prior_system = previous.get("system_cpu_usage")
    online = current.get("online_cpus")
    if None in (cpu, prior_cpu, system, prior_system, online) or system <= prior_system or cpu < prior_cpu:
        return None
    return (cpu - prior_cpu) / (system - prior_system) * online


async def read_node_resources(unit_ids, *, timeout_s=4):
    """Collect bounded physical measurements, preserving missing values as unknown."""
    containers = await list_unit_containers(timeout_s=min(timeout_s, 3))
    unit_ids = set(unit_ids)
    owned = [row for row in containers if row.get("Labels", {}).get("psrl.env_unit") in unit_ids]
    active = [row for row in owned if row.get("State") in ("running", "paused", "restarting")]
    slots = asyncio.Semaphore(8)

    async def sample(row):
        async with slots:
            try:
                stats = await container_stats(row["Id"], timeout_s=min(timeout_s, 3))
                memory = stats.get("memory_stats", {}).get("usage")
                return {
                    "memory_mb": None if memory is None else memory / 1024**2,
                    "cpu_cores": cpu_usage_cores(stats),
                    "error": "" if memory is not None else "Docker memory sample is missing.",
                }
            except Exception as exc:
                return {"memory_mb": None, "cpu_cores": None, "error": f"{type(exc).__name__}: {exc}"}

    tasks = {row["Id"]: asyncio.create_task(sample(row)) for row in active}
    try:
        if tasks:
            await asyncio.wait(tasks.values(), timeout=timeout_s)
        readings = {
            container_id: task.result()
            if task.done()
            else {"memory_mb": None, "cpu_cores": None, "error": "Container stats deadline exceeded."}
            for container_id, task in tasks.items()
        }
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    units = {}
    for unit_id in unit_ids:
        all_rows = [row for row in owned if row["Labels"]["psrl.env_unit"] == unit_id]
        live_rows = [row for row in active if row["Labels"]["psrl.env_unit"] == unit_id]
        samples = [readings[row["Id"]] for row in live_rows]
        units[unit_id] = {
            "containers_running": len(live_rows),
            "containers_total": len(all_rows),
            "measured_memory_mb": (
                sum(s["memory_mb"] for s in samples) if all(s["memory_mb"] is not None for s in samples) else None
            ),
            "measured_cpu_cores": (
                sum(s["cpu_cores"] for s in samples) if all(s["cpu_cores"] is not None for s in samples) else None
            ),
            "error": " | ".join(s["error"] for s in samples if s["error"]),
        }
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = None
    return {"sample_time_s": time.time(), "cpu_load_1m": load, "cpu_count": os.cpu_count(), "units": units}
