"""Exercise real command grants, independent GPU attribution, cancellation and return."""

import argparse
import asyncio
import ctypes as c
import json
import subprocess
import time
import uuid
from pathlib import Path

from psrl.workers.env_service.bpf_identity import Record
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.overlay import CONTROL_DEVICES, LEASE_LABEL


async def command(*args):
    result = await asyncio.to_thread(subprocess.run, args, capture_output=True, text=True, timeout=10, check=True)
    return result.stdout.strip()


async def wait_file(path):
    deadline = time.monotonic() + 20
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Workload did not produce {path.name!r}.")
        await asyncio.sleep(0.1)
    return json.loads(path.read_text())


async def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    node = EnvNodeService(None, "127.0.0.1", gpu_identity_backend="ebpf")
    report = {"status": "running", "scope": "Real Docker CUDA through node grant, telemetry and return APIs."}
    owned = []
    selected = None
    try:
        report["docker"] = await command("docker", "info", "--format", "{{.ServerVersion}}")
        inventory = await asyncio.to_thread(node._gpu_probe().read_gpu_devices)
        all_readings = await node.gpu_lending_snapshot(tuple(inventory))
        selected = next(index for index, item in all_readings.items() if not item["pids"])
        report["gpu"] = all_readings[selected]
        gpu = inventory[selected]
        version = await command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        version = version.splitlines()[0]
        report["attribution_supported"] = await node.check_gpu_attribution_support((selected,))
        for label in ("a", "b"):
            output = root / label
            output.mkdir()
            lease = "bpf-backend-" + uuid.uuid4().hex
            name = lease[:45]
            create = [
                "docker",
                "create",
                "--name",
                name,
                "--label",
                f"{LEASE_LABEL}={lease}",
                "--network",
                "none",
                "--cpus",
                "1",
                "--memory",
                "512m",
                "--pids-limit",
                "32",
                "--env",
                "LD_LIBRARY_PATH=/probe-driver",
                "--env",
                "CUDA_CACHE_DISABLE=1",
                "--mount",
                f"type=bind,src={output},dst=/evidence",
                "--mount",
                f"type=bind,src={Path(__file__).with_name('bpf_backend_workload.py')},dst=/work.py,readonly",
            ]
            for device in [gpu.path, *(p for p in CONTROL_DEVICES if Path(p).exists())]:
                create.extend(["--device", f"{device}:{device}:m"])
            for library in ("libcuda", "libnvidia-ptxjitcompiler"):
                create.extend(
                    [
                        "--mount",
                        f"type=bind,src=/usr/lib64/{library}.so.{version},dst=/probe-driver/{library}.so.1,readonly",
                    ]
                )
            create.extend([args.image, "timeout", "60", "python3", "/work.py", "/evidence"])
            container = await command(*create)
            owned.append((lease, container, output))
            (root / "owned-containers.json").write_text(json.dumps([item[1] for item in owned]))
            await command("docker", "start", container)
            await wait_file(output / "parent.json")
            await node.grant_gpu_command(lease, container, (selected,), 1)
        for _, _, output in owned:
            (output / "start").touch()
        report["cuda"] = [await wait_file(output / "cuda.json") for _, _, output in owned]
        await asyncio.sleep(2)
        requests = [(lease, container, 1) for lease, container, _ in owned]
        report["memory"] = await node.gpu_command_memory_snapshot((selected,), requests)
        tasks = [node._gpu_lease_state(lease)["command_telemetry"] for lease, _, _ in owned]
        readings = await node.gpu_lending_snapshot((selected,))
        gpu_pids = readings[selected]["pids"]
        owners = [await asyncio.to_thread(task._owners, gpu_pids) for task in tasks]
        report["owners"] = owners
        if not all(len(item) == 1 for item in owners) or set(owners[0]) & set(owners[1]):
            raise RuntimeError("The two same-name GPU children need disjoint host identities.")
        if report["memory"]["attribution_errors"] or not all(
            report["memory"]["owned_mb"].get(lease, 0) >= 64 for lease, _, _ in owned
        ):
            raise RuntimeError("Both commands must receive current owned memory credit.")
        lease_a, container_a, output_a = owned[0]
        (output_a / "cancel").touch()
        report["cancelled_child"] = await wait_file(output_a / "child-done.json")
        await node.revoke_gpu_command(lease_a, container_a, (selected,), 1, (selected,))
        report["after_cancel"] = await node.gpu_command_memory_snapshot((selected,), requests[1:])
        if report["after_cancel"]["owned_mb"].get(owned[1][0], 0) < 64:
            raise RuntimeError("Cancelling A must not remove B's GPU ownership.")
        old_pid = next(iter(owners[0]))
        backend = await node._gpu_identity()
        report["exited_identity_removed"] = backend.lookup("identities", c.c_uint32(old_pid), Record()) is None
        if not report["exited_identity_removed"]:
            raise RuntimeError("The exited CUDA process identity must be invalidated.")
        lease_b, container_b, output_b = owned[1]
        (output_b / "finish").touch()
        report["completed_child"] = await wait_file(output_b / "child-done.json")
        await node.revoke_gpu_command(lease_b, container_b, (selected,), 1, (selected,))
        report["telemetry"] = [task.summary() for task in tasks]
        if any(item["errors"] for item in report["telemetry"]):
            raise RuntimeError("Task telemetry reported an identity or NVML error.")
        report["status"] = "passed"
    except Exception as exc:
        report["status"], report["error"] = "failed", repr(exc)
    finally:
        for lease, container, output in owned:
            (output / "stop-parent").touch()
            try:
                (output / "container.log").write_text(await command("docker", "logs", container))
                await node.remove_gpu_lease(lease)
            except Exception as exc:
                report.setdefault("cleanup_errors", []).append(repr(exc))
                await command("docker", "rm", "-f", container)
        report["containers_remaining"] = [
            container
            for _, container, _ in owned
            if await command("docker", "ps", "-aq", "--filter", f"id={container}")
        ]
        if selected is not None:
            for _ in range(20):
                after = (await node.gpu_lending_snapshot((selected,)))[selected]
                if not after["pids"] and after["used_mb"] <= report["gpu"]["used_mb"] + 16:
                    report["gpu_release_verified"] = True
                    break
                await asyncio.sleep(0.25)
            report["gpu_after"] = after
        await node.close()
        if hasattr(node, "_identity_source"):
            report["collector_closed"] = node._identity_source.closed and not node._identity_source.links
        if (
            report.get("containers_remaining")
            or report.get("cleanup_errors")
            or not report.get("gpu_release_verified")
        ):
            report["status"] = "cleanup_unverified"
        (root / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
