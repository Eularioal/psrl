"""Probe isolated Docker MPS clients without changing node compute mode or shared services."""

import argparse
import asyncio
import dataclasses
import json
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from psrl.workers.env_service.gpu_execution import EpisodeGpuReservation, GpuExecutionBackend
from psrl.workers.env_service.mps import MpsClientController
from psrl.workers.env_service.overlay import CONTROL_DEVICES, read_kernel_driver_version, resolve_driver_libs
from psrl.workers.env_service.probe import NvmlProbe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-pid-mode", choices=("host", "private"), default="host")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run_id = "psrl-mps-" + uuid.uuid4().hex[:10]
    result = {
        "run_id": run_id,
        "scope": "Docker CUDA integration, not science or throughput",
        "server_pid_mode": args.server_pid_mode,
        "commands": [],
    }
    containers = []
    daemon = run_id + "-server"
    local = Path(tempfile.mkdtemp(prefix=run_id + "-"))
    data, pipes, logs = (local / part for part in ("data", "pipes", "logs"))
    for directory in (data, pipes, logs):
        directory.mkdir()
    probe = NvmlProbe()

    def run(command, *, input_text=None, check=True, timeout=15):
        completed = subprocess.run(command, input=input_text, capture_output=True, text=True, timeout=timeout)
        result["commands"].append(
            {
                "argv": command,
                "input": input_text,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if check and completed.returncode:
            raise RuntimeError(f"Command failed: {command!r}: {completed.stderr}")
        return completed.stdout.strip()

    def control(command, check=True):
        return run(["docker", "exec", "-i", daemon, "nvidia-cuda-mps-control"], input_text=command + "\n", check=check)

    def wait_record(name, predicate, seconds=20):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            path = data / f"{name}.json"
            if path.exists():
                value = json.loads(path.read_text())
                if predicate(value):
                    return value
                if value.get("status") == "error":
                    raise RuntimeError(f"Client failed: {value!r}.")
            time.sleep(0.2)
        raise TimeoutError(f"Client {name!r} did not reach the expected state.")

    try:
        run(["docker", "info", "--format", "{{.ServerVersion}}"])
        result["image_id"] = run(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"])
        device = probe.read_gpu_devices()[args.gpu]
        result["device"] = vars(device)
        initial = probe.read_gpu_snapshot([args.gpu])[args.gpu]
        result["before"] = initial
        if initial["pids"] or initial["used_mb"] > 100:
            raise RuntimeError("The selected GPU is not idle.")
        libs = resolve_driver_libs(read_kernel_driver_version())
        result["driver"] = libs.version
        base = [
            "docker",
            "run",
            "-d",
            "--network",
            "none",
            "--cpus",
            "2",
            "--memory",
            "1g",
            "--label",
            f"psrl.mps_probe={run_id}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--user",
            "0",
            "--device",
            f"{device.path}:{device.path}",
            "-e",
            f"CUDA_VISIBLE_DEVICES={device.uuid}",
            "-e",
            "LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64",
            "-v",
            f"{data}:/probe-data",
            "-v",
            f"{pipes}:/mps-pipes",
            "-v",
            f"{logs}:/mps-logs",
            "-e",
            "CUDA_MPS_PIPE_DIRECTORY=/mps-pipes",
            "-e",
            "CUDA_MPS_LOG_DIRECTORY=/mps-logs",
        ]
        for path in CONTROL_DEVICES:
            if Path(path).exists():
                base += ["--device", f"{path}:{path}"]
        for host, target in libs.mounts:
            base += ["-v", f"{host}:{target}:ro"]
        script = Path(__file__).with_name("mps_cuda_client.py").resolve()
        base += ["-v", f"{script}:/probe.py:ro"]

        def client(name, *, mps=True, extra=(), seconds=40):
            container = run_id + "-" + name
            containers.append(container)
            options = ["--ipc", f"container:{daemon}"] if mps else ["-e", "CUDA_MPS_PIPE_DIRECTORY=/absent"]
            run(
                base
                + ["--name", container]
                + options
                + list(extra)
                + [
                    "--entrypoint",
                    "python3",
                    args.image,
                    "/probe.py",
                    "--name",
                    name,
                    "--output",
                    "/probe-data",
                    "--seconds",
                    str(seconds),
                ]
            )
            return container

        # Establish real container CUDA support before introducing the MPS service.
        client("direct", mps=False, seconds=0.3)
        result["direct"] = wait_record("direct", lambda row: row["status"] == "ok")
        containers.append(daemon)
        daemon_options = ["--name", daemon, "--ipc", "shareable", "--shm-size", "256m"]
        if args.server_pid_mode == "host":
            daemon_options += ["--pid", "host"]
        for binary in ("nvidia-cuda-mps-control", "nvidia-cuda-mps-server"):
            daemon_options += ["-v", f"/usr/bin/{binary}:/usr/bin/{binary}:ro"]
        run(
            base
            + daemon_options
            + ["--entrypoint", "sh", args.image, "-c", "nvidia-cuda-mps-control -d && exec sleep 180"]
        )
        deadline = time.monotonic() + 10
        while not (pipes / "control").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        result["initial_control"] = control("get_server_list")
        a = client("a")
        b = client("b", extra=["-e", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50"])
        result["a"] = wait_record("a", lambda row: row["launches"] > 20)
        result["b"] = wait_record("b", lambda row: row["launches"] > 20)
        result["mps_ps"] = control("ps")
        result["servers"] = control("get_server_list")
        result["during"] = probe.read_gpu_snapshot([args.gpu])[args.gpu]
        result["device_clients"] = control("get_device_client_list", check=False)
        rows = [line.split() for line in result["mps_ps"].splitlines() if line.strip()]
        namespace = re.search(r"\d+", result["a"]["pid_namespace"]).group()
        matches = [row for row in rows if len(row) >= 6 and row[0].isdigit() and row[4] == namespace]
        if len(matches) != 1:
            raise RuntimeError("Cannot uniquely map client A into the MPS server PID namespace.")
        target = matches[0]
        result["cancel_target"] = {"client_pid": target[0], "server_pid": target[2], "namespace": namespace}
        before_b = json.loads((data / "b.json").read_text())["launches"]
        authoritative_pid = int(run(["docker", "inspect", "--format", "{{.State.Pid}}", a]))
        controller = MpsClientController(control)
        binding = controller.bind("probe-a", 1, authoritative_pid)
        result["binding"] = dataclasses.asdict(binding)

        async def cancel_through_backend():
            async def work():
                await asyncio.sleep(60)

            async def owner():
                return a

            async def terminate():
                await asyncio.to_thread(controller.terminate, binding, "probe-a", 1)
                result["terminate_response"] = "0"

            async def reap():
                result["backend_client_exit"] = await asyncio.to_thread(run, ["docker", "wait", a])

            backend = GpuExecutionBackend(EpisodeGpuReservation("probe", run_id))
            try:
                await backend.execute(
                    work, owner, timeout_sec=0.1, wait_timeout_sec=5, before_cancel=terminate, cancel=reap
                )
            except TimeoutError:
                result["backend_timeout_handled"] = True

        asyncio.run(cancel_through_backend())
        result["a_after_cancel"] = wait_record("a", lambda row: row["status"] == "error")
        if result["a_after_cancel"].get("cuda_error") != 810:
            raise RuntimeError("Client A did not report the expected MPS termination error.")
        result["b_after_cancel"] = wait_record("b", lambda row: row["launches"] > before_b + 100)
        result["after_cancel_ps"] = control("ps")
        result["after_cancel_memory"] = probe.read_gpu_snapshot([args.gpu])[args.gpu]
        (data / "stop-b").touch()
        result["b_completed"] = wait_record("b", lambda row: row["status"] == "ok")
        result["a_exit"] = run(["docker", "wait", a])
        result["b_exit"] = run(["docker", "wait", b])
        result["after_clients_ps"] = control("ps")
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
    finally:
        # Cooperative stops allow outstanding CUDA work to synchronize before cleanup.
        for name in ("a", "b", "direct"):
            (data / f"stop-{name}").touch()
        for container in reversed(containers):
            try:
                if container == daemon:
                    control("quit", check=False)
                else:
                    run(["docker", "wait", container], check=False, timeout=65)
                run(["docker", "logs", container], check=False)
                run(["docker", "rm", "-f", container], check=False)
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(repr(exc))
        try:
            deadline = time.monotonic() + 10
            while True:
                after = probe.read_gpu_snapshot([args.gpu])[args.gpu]
                if not after["pids"] and after["used_mb"] <= result.get("before", {}).get("used_mb", 0) + 10:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("GPU memory or process cleanup remains incomplete.")
                time.sleep(0.2)
            result["after"] = after
            result["containers_remaining"] = run(["docker", "ps", "-aq", "--filter", f"label=psrl.mps_probe={run_id}"])
            if result["containers_remaining"]:
                raise RuntimeError("Probe containers remain.")
        except Exception as exc:
            result.setdefault("cleanup_errors", []).append(repr(exc))
        for source in [*data.glob("*.json"), *logs.glob("*")]:
            if source.is_file():
                (args.output / source.name).write_bytes(source.read_bytes())
        result["local_directory"] = str(local)
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        probe.close()
        print(json.dumps({key: result.get(key) for key in ("status", "error", "cleanup_errors", "after")}), flush=True)
    raise SystemExit(0 if result["status"] == "passed" and not result.get("cleanup_errors") else 1)


if __name__ == "__main__":
    main()
