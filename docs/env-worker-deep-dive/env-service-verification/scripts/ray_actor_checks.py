"""Checks on the EnvRouter as a REAL Ray actor, in an isolated throwaway local cluster."""

import asyncio
import os
import time

import ray
from psrl.workers.env_service import GRANTED, REJECT, WAIT, EnvDemand
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.router import create_env_router

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


ray.init(
    address="local",
    num_cpus=8,
    num_gpus=0,
    include_dashboard=False,
    _temp_dir="/tmp/env_verify/ray",
    logging_level="ERROR",
    namespace="envverify",
    log_to_driver=False,
)
node_ip = ray.util.get_node_ip_address()
stamp = int(time.time())


def cfg(name, ttl=1800, max_containers=4):
    return {
        "node_policy": "any",
        "default_gpu_memory_mb": 24000,
        "slot_ttl_s": ttl,
        "router_name": f"{name}_{stamp}",
        "units": [
            {
                "unit_id": "c0",
                "node_ip": node_ip,
                "cpu_quota": 64,
                "memory_mb": 1_000_000,
                "max_containers": max_containers,
            },
        ],
    }


@ray.remote(num_cpus=0.1)
def admit_from_another_process(router, session):
    status = ray.get(router.try_admit.remote(session, EnvDemand(cpu=1.0), node_ip)).status
    return status, os.getpid(), session


# 1. Atomic admission across real worker processes.
router = create_env_router(cfg("atomic"))
outs = ray.get([admit_from_another_process.remote(router, f"s{i}") for i in range(24)])
granted = sum(s == GRANTED for s, _, _ in outs)
waited = sum(s == WAIT for s, _, _ in outs)
pids = {pid for _, pid, _ in outs}
granted_session = next(sess for s, _, sess in outs if s == GRANTED)
check(
    "24 admissions from separate worker processes, unit holds 4",
    granted == 4 and waited == 20,
    f"granted={granted} wait={waited} distinct_worker_pids={len(pids)}",
)
snap = ray.get(router.snapshot.remote())
check(
    "snapshot over Ray serialises dataclasses and agrees",
    snap[0].containers_running == 4,
    f"running={snap[0].containers_running}/{snap[0].max_containers}",
)

# 2. Release frees capacity.
first_lease = ray.get(router.try_admit.remote(granted_session, EnvDemand(cpu=1.0), node_ip)).lease
check(
    "repeat admission of an already-granted session returns its lease, not a new slot",
    first_lease is not None and ray.get(router.snapshot.remote())[0].containers_running == 4,
)
check("release over Ray", ray.get(router.release.remote(first_lease.lease_id)) is True)
again = ray.get(router.try_admit.remote("late", EnvDemand(cpu=1.0), node_ip))
check("capacity is reusable after release", again.status == GRANTED, f"status={again.status}")

# 3. Reject vs wait.
rej = ray.get(router.try_admit.remote("g", EnvDemand(needs_gpu=True), node_ip))
check(
    "GPU demand on a CPU-only router is REJECT with a reason",
    rej.status == REJECT and bool(rej.reason),
    repr(rej.reason)[:80],
)

# 4. Heartbeat keeps a lease alive past the TTL; silence loses it within TTL + reaper interval.
ttl = 3
r2 = create_env_router(cfg("ttl", ttl=ttl))
lease = ray.get(r2.try_admit.remote("hb", EnvDemand(cpu=1.0), node_ip)).lease
t0 = time.time()
alive = True
t_last = t0
while time.time() - t0 < 2 * ttl:
    alive &= ray.get(r2.heartbeat.remote(lease.lease_id))
    t_last = time.time()
    time.sleep(1.0)
check(
    "heartbeating lease survives 2x TTL",
    alive and ray.get(r2.snapshot.remote())[0].containers_running == 1,
    f"elapsed={time.time() - t0:.1f}s ttl={ttl}s",
)
while ray.get(r2.snapshot.remote())[0].containers_running != 0 and time.time() - t_last < 30:
    time.sleep(0.25)
silent_s = time.time() - t_last
check(
    "silent lease is reclaimed by the actor's own reaper loop (no admission traffic)",
    ttl < silent_s < ttl + 1.0 + 0.75,
    f"reclaimed {silent_s:.2f}s after the last heartbeat (ttl={ttl}s, reaper tick=1s)",
)
reaped = ray.get(r2.pop_reaped.remote(node_ip))
check(
    "pop_reaped hands the node its orphan once",
    [l.lease_id for l in reaped] == [lease.lease_id] and ray.get(r2.pop_reaped.remote(node_ip)) == [],
)
check("heartbeat on a reclaimed lease returns False", ray.get(r2.heartbeat.remote(lease.lease_id)) is False)

# 5. Named actor reuse and client construction from config.
a = create_env_router(cfg("reuse"))
b = create_env_router(cfg("reuse"))
check("get_if_exists returns the same actor", a._actor_id == b._actor_id)
env_cfg = dict(cfg("reuse"), enable=True)
client = EnvServiceClient.from_config(env_cfg)
check("EnvServiceClient.from_config finds the named actor", client is not None and client.node_ip == node_ip)


async def via_client():
    handle = await client.acquire("via-client", EnvDemand(cpu=1.0), max_wait_s=10)
    cap = await client.router.node_capacity.remote()
    await handle.release()
    return handle.lease.unit_id, cap


unit_id, cap = asyncio.run(via_client())
check("acquire/release through the client against a real actor", unit_id == "c0", f"node_capacity={cap}")

ray.shutdown()
print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
