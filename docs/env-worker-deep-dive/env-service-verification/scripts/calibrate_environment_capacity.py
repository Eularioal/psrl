"""Measure bounded PostTrain concurrency curves with isolated owned containers."""

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from psrl.workers.env_service.overlay import build_lease_overlay, read_kernel_driver_version, resolve_driver_libs
from psrl.workers.env_service.probe import NvmlProbe, read_gpu_devices
from psrl.workers.env_service.types import EnvDemand, EnvLease


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


async def execute(argv, timeout=240):
    process = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    return process.returncode, stdout.decode(), stderr.decode()


async def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tag = "capacity-" + uuid.uuid4().hex[:10]
    probe = NvmlProbe()
    before = probe.read_gpu_snapshot([args.gpu])
    if any(v["pids"] or v["used_mb"] > 64 for v in before.values()):
        raise RuntimeError("Calibration GPU is occupied.")
    subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=8)
    source = out / "source"
    source.mkdir()
    shutil.copy2(Path(__file__).with_name("partition_worker.py"), source / "partition_worker.py")
    shutil.copy2(args.repo / "examples/posttrain_bench/local_model.py", source / "task.py")
    shutil.copytree(args.repo / ".verification-deps/bootstrap", source / "bootstrap")
    lease = EnvLease(tag, tag, tag, args.node_ip, (args.gpu,), EnvDemand(2, 16384, True, 16384), 0, 0)
    overlay = build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version()))["services"]["main"]
    record = dict(
        tag=tag,
        started_at=time.time(),
        status="running",
        before=before,
        devices=read_gpu_devices(),
        trials=[],
        scope="Direct Docker calibration, no PSRL policy training.",
        limits=dict(seconds=3900, jobs_per_trial=6, command_seconds=240),
        order=[1, 4, 2, 5, 3, 4],
        calibration=[1, 2, 4, 5],
        held_out=[3],
        repeat=[4],
    )
    save(out / "evidence.json", record)
    owned = set()

    async def trial(index, cap):
        row = dict(index=index, cap=cap, started_at=time.time(), commands=[], status="running")
        record["trials"].append(row)
        semaphore = asyncio.Semaphore(cap)

        async def job(j):
            artifact = out / f"trial-{index}" / str(j)
            artifact.mkdir(parents=True)
            name = f"{tag}-{index}-{j}"
            command = [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--label",
                f"psrl.capacity={tag}",
                "--network",
                "none",
                "--cpus",
                "2",
                "--memory",
                "16g",
            ]
            for key, flag in (("devices", "--device"), ("volumes", "--volume"), ("environment", "--env")):
                for value in overlay[key]:
                    command.extend([flag, value])
            mounts = [
                f"{source}:/study:ro",
                f"{args.model}:/model:ro",
                f"{args.data}:/benchmark:ro",
                f"{artifact}:/artifact:rw",
            ]
            for prefix in dict.fromkeys(
                (Path(sys.executable).parent.parent, Path(sys.executable).resolve().parent.parent)
            ):
                mounts.append(f"{prefix}:{prefix}:ro")
            for mount in mounts:
                command.extend(["--volume", mount])
            for value in [
                "PYTHONPATH=/study/bootstrap",
                "OMP_NUM_THREADS=2",
                "MKL_NUM_THREADS=2",
                "OPENBLAS_NUM_THREADS=1",
                "HF_HUB_OFFLINE=1",
                "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            ]:
                command.extend(["--env", value])
            command.extend([args.image, "sleep", "infinity"])
            owned.add(name)
            rc, stdout, stderr = await execute(command, 30)
            if rc:
                raise RuntimeError(f"Container creation failed: {stderr!r}.")
            try:
                for mode in ("train", "verify"):
                    requested = time.time()
                    async with semaphore:
                        started = time.time()
                        argv = [
                            "docker",
                            "exec",
                            name,
                            sys.executable,
                            "/study/partition_worker.py",
                            "--mode",
                            mode,
                            "--steps",
                            str(896 if j % 3 == 2 else 448),
                            "--data-root",
                            "/benchmark/public" if mode == "train" else "/benchmark/private",
                        ]
                        rc, stdout, stderr = await execute(argv)
                        (artifact / f"{mode}.stdout").write_text(stdout)
                        (artifact / f"{mode}.stderr").write_text(stderr)
                        if rc:
                            raise RuntimeError(f"Command failed: {stderr[-1500:]!r}.")
                        value = json.loads(
                            next(
                                line.split("=", 1)[1]
                                for line in stdout.splitlines()
                                if line.startswith("PARTITION_RESULT=")
                            )
                        )
                        if value["cuda_readback"] != 528:
                            raise RuntimeError("CUDA readback failed.")
                        row["commands"].append(
                            dict(
                                job=j,
                                mode=mode,
                                requested_at=requested,
                                admitted_at=started,
                                finished_at=time.time(),
                                result=value,
                            )
                        )
                        save(out / "evidence.json", record)
            finally:
                await execute(["docker", "rm", "-f", name], 30)
                owned.discard(name)

        async with asyncio.TaskGroup() as group:
            for j in range(6):
                group.create_task(job(j))
        row.update(status="complete", finished_at=time.time())
        row["elapsed_s"] = row["finished_at"] - row["started_at"]
        save(out / "evidence.json", record)

    try:
        async with asyncio.timeout(3900):
            for i, cap in enumerate(record["order"]):
                await trial(i, cap)
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        for name in list(owned):
            await execute(["docker", "rm", "-f", name], 30)
        for _ in range(30):
            after = probe.read_gpu_snapshot([args.gpu])
            if all(not v["pids"] and v["used_mb"] < 64 for v in after.values()):
                record["physical_release_verified"] = True
                break
            await asyncio.sleep(2)
        record.update(after=after, finished_at=time.time())
        save(out / "evidence.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "output", "model", "data"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    asyncio.run(run(parser.parse_args()))
