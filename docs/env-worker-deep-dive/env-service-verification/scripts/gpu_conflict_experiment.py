"""
Before/after conflict experiment on real H20 memory.

Each "episode" is a process that really allocates `--gib` GiB on the card it is given and
holds it for `--hold` seconds. What differs between scenarios is only HOW a card is chosen:

A static c00354c: every GPU episode mounts the static override -> always /dev/nvidia0
B gate 72ca50c: real GpuDeviceGate, one episode per whole card
C service 00db595: real EnvRouter Ray actor + EnvServiceClient, containers share a card
D1 service a stranger holds 80 GiB on the card and the client reports real nvidia-smi
D2 service same stranger, but the client reports nothing (declared-only accounting)

Run A/B with PYTHONPATH pointing at the matching source tree and C/D with the new tree.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = str(HERE / "gpu_episode.py")
CARDS = (0, 1)
TOTAL_MIB = 97871
RAY_TMP = "/tmp/env_verify/ray"


def smi_used() -> dict[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    used = {}
    for line in out.splitlines():
        idx, mem = (p.strip() for p in line.split(","))
        used[int(idx)] = int(mem)
    return used


class Sampler:
    def __init__(self):
        self.peak = {c: 0 for c in CARDS}
        self._stop = asyncio.Event()
        self._task = None

    async def _run(self):
        while not self._stop.is_set():
            used = await asyncio.to_thread(smi_used)
            for c in CARDS:
                self.peak[c] = max(self.peak[c], used.get(c, 0))
            await asyncio.sleep(0.5)

    def start(self):
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self):
        self._stop.set()
        await self._task


async def launch(label: str, card: int, gib: float, hold: float) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(card))
    proc = await asyncio.create_subprocess_exec(
        sys.executable, WORKER, str(gib), str(hold), label, env=env, stdout=asyncio.subprocess.PIPE
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=240)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"label": label, "status": "TIMEOUT", "card": str(card), "t_alloc": None, "t_end": time.time()}
    line = [ln for ln in out.decode().splitlines() if ln.startswith("{")][-1]
    rec = json.loads(line)
    rec["t_end"] = time.time()
    return rec


def peak_concurrency(records: list[dict]) -> dict[int, int]:
    result = {}
    for c in CARDS:
        spans = [(r["t_alloc"], r["t_end"]) for r in records if r["status"] == "OK" and r["card"] == str(c)]
        events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
        cur = peak = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
        result[c] = peak
    return result


def report(name: str, records: list[dict], sampler: Sampler, t0: float, extra: dict | None = None) -> dict:
    ok = sum(r["status"] == "OK" for r in records)
    oom = sum(r["status"] == "OOM" for r in records)
    summary = {
        "scenario": name,
        "episodes": len(records),
        "ok": ok,
        "oom": oom,
        "other": len(records) - ok - oom,
        "peak_concurrent_per_card": peak_concurrency(records),
        "peak_used_mib_per_card": sampler.peak,
        "card_capacity_mib": TOTAL_MIB,
        "makespan_s": round(time.time() - t0, 1),
        "episodes_by_card": {str(c): sum(r["card"] == str(c) for r in records) for c in CARDS},
    }
    summary.update(extra or {})
    return summary


async def scenario_a(n, gib, hold):
    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    # The static override names one device, so the container only ever sees card 0.
    records = await asyncio.gather(*(launch(f"ep{i}", 0, gib, hold) for i in range(n)))
    await sampler.stop()
    return report("A static override (c00354c)", list(records), sampler, t0)


async def scenario_b(n, gib, hold):
    from psrl.utils.concurrency.gpu_device_gate import GpuDeviceGate

    os.environ["PSRL_SCIACCEL_GPU_DEVICES"] = ",".join(f"/dev/nvidia{c}" for c in CARDS)
    jobs_dir = "/tmp/env_verify/jobs_B"
    os.makedirs(jobs_dir, exist_ok=True)
    gate = GpuDeviceGate(jobs_dir=jobs_dir, slot_prefix="verify_gate_b")
    sampler = Sampler()
    sampler.start()
    t0 = time.time()

    async def episode(i):
        allocated = await gate.acquire()
        try:
            return await launch(f"ep{i}", int(allocated.device[len("/dev/nvidia") :]), gib, hold)
        finally:
            GpuDeviceGate.release(allocated)

    records = await asyncio.gather(*(episode(i) for i in range(n)))
    await sampler.stop()
    return report("B GpuDeviceGate (72ca50c)", list(records), sampler, t0)


def init_ray():
    import ray

    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        _temp_dir=RAY_TMP,
        logging_level="ERROR",
        namespace="envverify",
        log_to_driver=False,
    )
    return ray, ray.util.get_node_ip_address()


def units_cfg(node_ip, cards):
    return {
        "node_policy": "any",
        "default_gpu_memory_mb": 24000,
        "slot_ttl_s": 1800,
        "router_name": "verify_router_" + "_".join(map(str, cards)) + f"_{int(time.time())}",
        "units": [
            {
                "unit_id": f"card{c}",
                "node_ip": node_ip,
                "cpu_quota": 64,
                "memory_mb": 1_000_000,
                "gpu_indices": [c],
                "max_containers": 10,  # not the binding limit: device memory is
                "gpu_memory_mb": TOTAL_MIB,
            }
            for c in cards
        ],
    }


async def service_episode(client, i, gib, hold, max_wait_s=None):
    from psrl.workers.env_service import EnvDemand
    from psrl.workers.env_service.client import EnvAdmissionTimeout

    try:
        handle = await client.acquire(f"ep{i}", EnvDemand(needs_gpu=True), max_wait_s=max_wait_s)
    except EnvAdmissionTimeout:
        return {"label": f"ep{i}", "status": "NOT_ADMITTED", "card": "-", "t_alloc": None, "t_end": time.time()}
    try:
        return await launch(f"ep{i}", handle.lease.gpu_indices[0], gib, hold)
    finally:
        await handle.release()


async def scenario_c(n, gib, hold):
    from psrl.workers.env_service.client import EnvServiceClient
    from psrl.workers.env_service.probe import read_gpu_telemetry
    from psrl.workers.env_service.router import create_env_router

    ray, node_ip = init_ray()
    try:
        cfg = units_cfg(node_ip, CARDS)
        router = create_env_router(cfg)
        client = EnvServiceClient(
            router, node_ip, poll_interval_s=0.5, max_poll_interval_s=1.0, telemetry_reader=read_gpu_telemetry
        )
        sampler = Sampler()
        sampler.start()
        t0 = time.time()
        records = await asyncio.gather(*(service_episode(client, i, gib, hold) for i in range(n)))
        await sampler.stop()
        snap = await router.snapshot.remote()
        leftover = sum(s.leases_active for s in snap)
        return report(
            "C env service, real Ray actor (00db595)", list(records), sampler, t0, {"leases_left_after": leftover}
        )
    finally:
        ray.shutdown()


async def foreign_holder(gib):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, WORKER, str(gib), "-1", "stranger", env=env, stdout=asyncio.subprocess.PIPE
    )
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=120)
    assert json.loads(line)["status"] == "OK", line
    return proc


async def scenario_d(n, gib, hold, with_telemetry):
    from psrl.workers.env_service.client import EnvServiceClient
    from psrl.workers.env_service.probe import read_gpu_telemetry
    from psrl.workers.env_service.router import create_env_router

    async def no_telemetry():
        return {}

    ray, node_ip = init_ray()
    holder = await foreign_holder(80.0)
    try:
        router = create_env_router(units_cfg(node_ip, (0,)))
        reader = read_gpu_telemetry if with_telemetry else no_telemetry
        client = EnvServiceClient(
            router, node_ip, poll_interval_s=0.5, max_poll_interval_s=1.0, telemetry_reader=reader
        )
        sampler = Sampler()
        sampler.start()
        t0 = time.time()
        records = await asyncio.gather(*(service_episode(client, i, gib, hold, max_wait_s=8) for i in range(n)))
        await sampler.stop()
        name = (
            "D1 stranger holds 80 GiB, client reports nvidia-smi"
            if with_telemetry
            else "D2 stranger holds 80 GiB, declared-only"
        )
        return report(
            name, list(records), sampler, t0, {"not_admitted": sum(r["status"] == "NOT_ADMITTED" for r in records)}
        )
    finally:
        holder.kill()
        await holder.wait()
        ray.shutdown()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["A", "B", "C", "D1", "D2"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--gib", type=float, default=22.0)
    parser.add_argument("--hold", type=float, default=12.0)
    args = parser.parse_args()

    before = smi_used()
    assert all(before[c] < 2048 for c in CARDS), f"cards not idle at start: {before}"
    runner = {
        "A": lambda: scenario_a(args.episodes, args.gib, args.hold),
        "B": lambda: scenario_b(args.episodes, args.gib, args.hold),
        "C": lambda: scenario_c(args.episodes, args.gib, args.hold),
        "D1": lambda: scenario_d(3, args.gib, args.hold, True),
        "D2": lambda: scenario_d(3, args.gib, args.hold, False),
    }[args.scenario]
    summary = await runner()
    await asyncio.sleep(3)
    summary["cards_idle_at_end_mib"] = {c: v for c, v in smi_used().items() if c in CARDS}
    print("RESULT " + json.dumps(summary))
    Path(f"/tmp/env_verify/result_{args.scenario}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
