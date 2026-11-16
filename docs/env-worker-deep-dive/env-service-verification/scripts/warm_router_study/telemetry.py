"""Collect bounded, read-only Docker CPU and memory evidence for one experiment."""

import asyncio
import json
import re
import time

_TIMEOUT_S = 3.0
_CPU_SOURCES = {
    "cgroup_v2": "/sys/fs/cgroup/cpu.stat",
    "cgroup_v1_cpu": "/sys/fs/cgroup/cpu/cpu.stat",
    "cgroup_v1_cpu_cpuacct": "/sys/fs/cgroup/cpu,cpuacct/cpu.stat",
}
_CPU_STAT_COMMAND = """for path in /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/cpu/cpu.stat \
/sys/fs/cgroup/cpu,cpuacct/cpu.stat; do
    if [ -r "$path" ]; then
        case "$path" in
            /sys/fs/cgroup/cpu.stat) source=cgroup_v2 ;;
            /sys/fs/cgroup/cpu/cpu.stat) source=cgroup_v1_cpu ;;
            /sys/fs/cgroup/cpu,cpuacct/cpu.stat) source=cgroup_v1_cpu_cpuacct ;;
        esac
        printf '%s\\n' "$source"
        cat "$path"
        exit $?
    fi
done
exit 44
"""
_HOST_FIELDS = (
    "NanoCpus",
    "CpuPeriod",
    "CpuQuota",
    "CpuShares",
    "CpusetCpus",
    "CpusetMems",
    "Memory",
    "MemorySwap",
    "MemoryReservation",
    "IpcMode",
    "ShmSize",
)


async def _docker(*arguments):
    process = None
    started = time.monotonic()
    try:
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "docker", *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            ),
            timeout=_TIMEOUT_S,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=max(0.001, _TIMEOUT_S - (time.monotonic() - started))
        )
        return {
            "status": "ok" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace")[:2000],
        }
    except (TimeoutError, OSError) as error:
        return {"status": "timeout" if isinstance(error, TimeoutError) else "unavailable", "error": str(error)}
    finally:
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.communicate(), timeout=1.0)
            except TimeoutError:
                pass


async def _cpu_stat(container_id):
    result = await _docker("exec", container_id, "sh", "-c", _CPU_STAT_COMMAND)
    result["observed_unix_s"] = time.time()
    result["source"] = None
    raw = result.pop("stdout", "")
    if result["status"] != "ok":
        if result.get("returncode") == 44:
            result["status"] = "missing"
        return result
    try:
        source, *lines = raw.splitlines()
        if source not in _CPU_SOURCES:
            raise ValueError("Unknown cgroup CPU counter source.")
        result["source"] = source
        values = {}
        for line in lines:
            key, value = line.split()
            values[key] = int(value)
        if not values:
            raise ValueError("Empty cpu.stat output.")
    except ValueError as error:
        result.update(status="invalid", error=str(error))
        return result
    result["values"] = values
    return result


async def inspect_owned(tag):
    """Inspect only containers whose actor label exactly matches the experiment tag.

    Missing known cgroup v1/v2 files and Docker failures have explicit statuses.
    CPU counters are cumulative snapshots, so callers must subtract matching IDs
    and sources between their before and after samples. Values retain kernel units.
    """
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("An explicit nonempty actor tag is required.")
    started = time.monotonic()
    report = {"tag": tag, "observed_unix_s": time.time(), "status": "ok", "containers": [], "errors": []}
    try:
        listed = await _docker("ps", "-aq", "--no-trunc", "--filter", f"label=psrl.actor_id={tag}")
        if listed["status"] != "ok":
            listed.pop("stdout", None)
            report.update(status="unavailable", errors=[dict(stage="list", **listed)])
            return report
        container_ids = listed["stdout"].split()
        if not container_ids:
            return report
        if any(re.fullmatch(r"[0-9a-f]{12,64}", value) is None for value in container_ids):
            report.update(status="unavailable", errors=[{"stage": "list", "error": "Invalid container ID output."}])
            return report
        inspected = await _docker("inspect", *container_ids)
        raw = inspected.pop("stdout", "")
        if inspected["status"] != "ok":
            report.update(status="unavailable", errors=[dict(stage="inspect", **inspected)])
            return report
        try:
            containers = json.loads(raw)
            if not isinstance(containers, list) or any(not isinstance(value, dict) for value in containers):
                raise ValueError("Expected a Docker inspect container list.")
        except (ValueError, TypeError) as error:
            report.update(status="unavailable", errors=[{"stage": "inspect", "error": str(error)}])
            return report
        for container in containers:
            container_id = container.get("Id")
            labels = (container.get("Config") or {}).get("Labels") or {}
            if container_id not in container_ids or labels.get("psrl.actor_id") != tag:
                report["errors"].append({"stage": "ownership", "error": "Skipped an unverified container."})
                continue
            host = container.get("HostConfig") or {}
            report["containers"].append(
                {
                    "id": container_id,
                    "name": container.get("Name"),
                    "running": (container.get("State") or {}).get("Running"),
                    "actor_id": tag,
                    "host_config": {field: host.get(field) for field in _HOST_FIELDS},
                }
            )
        counters = await asyncio.gather(*(_cpu_stat(container["id"]) for container in report["containers"]))
        for container, counter in zip(report["containers"], counters, strict=True):
            container["cpu_stat"] = counter
        if report["errors"] or any(counter["status"] != "ok" for counter in counters):
            report["status"] = "partial"
        return report
    finally:
        report["elapsed_s"] = time.monotonic() - started
