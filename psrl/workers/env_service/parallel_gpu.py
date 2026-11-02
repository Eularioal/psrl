"""Own bounded MPS jobs and physical cleanup independently of callers."""

import asyncio
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from psrl.workers.env_service.gpu_memory import accounted_gpu_memory_mb
from psrl.workers.env_service.job_fence import JobFence, container_pid
from psrl.workers.env_service.mps import MpsClientController, parse_clients
from psrl.workers.env_service.overlay import CONTROL_DEVICES, read_kernel_driver_version, resolve_driver_libs
from psrl.workers.env_service.task_telemetry import TaskGpuTelemetry, read_command_memory

RUNNER = """import json, os, sys
options = json.loads(sys.argv[1])
print(json.dumps({"pid": os.getpid()}), flush=True)
if sys.stdin.readline().strip() != "go":
    sys.exit(125)
if options["cwd"]:
    os.chdir(options["cwd"])
os.execvpe("bash", ["bash", "-c", options["command"]], {**os.environ, **options["env"]})
"""


class MpsDomain:
    """Serialize admission and preserve reservations until per-job ownership disappears."""

    def __init__(self, node, unit):
        self.node, self.unit = node, unit
        self.root = Path(tempfile.mkdtemp(prefix="psrl-mps-domain-"))
        self.pipe = self.root / "pipe"
        self.pipe.mkdir()
        self.daemon = "psrl-mps-" + uuid.uuid4().hex
        self.leases, self.jobs = {}, {}
        self.lock = asyncio.Lock()
        self.started = False
        self.error = None
        self.controller = MpsClientController(self.control)

    def control(self, text):
        result = subprocess.run(
            ["docker", "exec", "-i", self.daemon, "nvidia-cuda-mps-control"],
            input=text + "\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode:
            raise RuntimeError(f"MPS control failed: {result.stderr!r}.")
        return result.stdout.strip()

    async def register(self, lease):
        async with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if not self.started:
                existing = await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "ps", "-aq", "--filter", f"label=psrl.env_mps_unit={self.unit.unit_id}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                if existing.stdout.strip():
                    raise RuntimeError("An orphan MPS domain requires operator recovery before admission.")
                inventory = await asyncio.to_thread(self.node._gpu_probe().read_gpu_devices)
                device = inventory[self.unit.gpu_indices[0]]
                libs = resolve_driver_libs(read_kernel_driver_version())
                command = [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    self.daemon,
                    "--network",
                    "none",
                    "--pid",
                    "host",
                    "--ipc",
                    "shareable",
                    "--shm-size",
                    "256m",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges:true",
                    "--memory",
                    "512m",
                    "--cpus",
                    "1",
                    "--label",
                    f"psrl.env_mps_unit={self.unit.unit_id}",
                    "--device",
                    device.path,
                    "-v",
                    f"{self.pipe}:{self.pipe}",
                    "-e",
                    f"CUDA_MPS_PIPE_DIRECTORY={self.pipe}",
                    "-e",
                    f"CUDA_VISIBLE_DEVICES={device.uuid}",
                    "-e",
                    "LD_LIBRARY_PATH=/usr/lib64",
                ]
                for path in CONTROL_DEVICES:
                    if Path(path).exists():
                        command += ["--device", path]
                for host, target in libs.mounts:
                    command += ["-v", f"{host}:{target}:ro"]
                for binary in ("nvidia-cuda-mps-control", "nvidia-cuda-mps-server"):
                    command += ["-v", f"/usr/bin/{binary}:/usr/bin/{binary}:ro"]
                command += [
                    "--entrypoint",
                    "sh",
                    self.unit.mps_image,
                    "-c",
                    "nvidia-cuda-mps-control -d && exec sleep infinity",
                ]
                self.started = True
                result = await asyncio.to_thread(
                    subprocess.run,
                    command,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode:
                    raise RuntimeError(f"MPS start failed: {result.stderr!r}.")
                self.started = True
                deadline = time.monotonic() + 10
                while not (self.pipe / "control").exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("MPS daemon did not become ready.")
                    await asyncio.sleep(0.05)
                await asyncio.to_thread(self.control, "get_server_list")
            self.leases[lease.lease_id] = lease
            return str(self.pipe), f"container:{self.daemon}"

    def submit(
        self, lease_id, container_id, job_id, command, memory_mb, timeout_sec, wait_timeout_sec, cwd, env, tag=""
    ):
        lease = self.leases[lease_id]
        if self.error or lease.gpu_closing:
            raise RuntimeError("MPS domain is unavailable for new work.")
        if job_id in self.jobs:
            raise RuntimeError("MPS job identity was already submitted.")
        if sum(r["lease_id"] == lease_id for r in self.jobs.values()) >= 128:
            raise RuntimeError("MPS lease job limit was reached.")
        if not isinstance(command, str) or not command.strip() or len(command) > 65536:
            raise ValueError("Invalid MPS command.")
        if not isinstance(memory_mb, int) or isinstance(memory_mb, bool):
            raise ValueError("MPS job memory must be an integer.")
        if not 0 < memory_mb <= lease.demand.gpu_memory_mb:
            raise ValueError("GPU job budget must fit its lease's declared demand.")
        if not 0 < timeout_sec <= 3600 or not 0 < wait_timeout_sec <= 3600:
            raise ValueError("GPU job deadlines must be bounded.")
        if env is not None and (
            not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items())
        ):
            raise ValueError("MPS environment must map strings to strings.")
        if cwd is not None and (not isinstance(cwd, str) or not cwd.startswith("/")):
            raise ValueError("MPS cwd must be absolute.")
        row = {
            "lease_id": lease_id,
            "container": container_id,
            "job_id": job_id,
            "tag": tag,
            "memory_mb": memory_mb,
            "state": "queued",
            "submitted": time.time(),
            "output": None,
            "error": None,
            "command": command,
            "cwd": cwd,
            "env": env,
            "timeout": timeout_sec,
            "wait": wait_timeout_sec,
        }
        self.jobs[job_id] = row
        row["task"] = asyncio.create_task(self.execute(row))
        return self.status(lease_id, job_id)

    def status(self, lease_id, job_id):
        row = self.jobs[job_id]
        if row["lease_id"] != lease_id:
            raise RuntimeError("MPS job belongs to another lease.")
        status = {
            key: row.get(key) for key in ("job_id", "state", "error", "output", "submitted", "started", "returned")
        }
        status["telemetry"] = row["telemetry"].summary() if "telemetry" in row else row.get("report")
        return status

    def reports(self, lease_id):
        return [
            {
                "request_id": row["job_id"],
                "tag": row["tag"],
                "state": row["state"],
                "telemetry": self.status(lease_id, row["job_id"])["telemetry"],
            }
            for row in self.jobs.values()
            if row["lease_id"] == lease_id
        ]

    async def cancel(self, lease_id, job_id):
        self.status(lease_id, job_id)
        task = self.jobs[job_id]["task"]
        if not task.done() and not task.cancelling():
            task.cancel()
        await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        if task.cancelled():
            self.jobs[job_id]["state"] = "cancelled"
        return self.status(lease_id, job_id)

    async def admit(self, row):
        deadline = time.monotonic() + row["wait"]
        while time.monotonic() < deadline:
            async with self.lock:
                if self.error or row["lease_id"] not in self.leases:
                    raise RuntimeError("MPS job lost its domain or lease.")
                first = next(r for r in self.jobs.values() if r["state"] == "queued")
                if first is row:
                    active = [
                        r
                        for r in self.jobs.values()
                        if r["state"] in ("starting", "running", "returning", "quarantined")
                    ]
                    tasks = {r["job_id"]: r["telemetry"] for r in active if "telemetry" in r}
                    reading = await asyncio.to_thread(
                        read_command_memory, self.node._gpu_probe(), tasks, self.unit.gpu_indices
                    )
                    reservations = {r["job_id"]: r["memory_mb"] for r in active}
                    device = reading["devices"][self.unit.gpu_indices[0]]
                    used = accounted_gpu_memory_mb(device["used_mb"], reservations, reading["owned_mb"])
                    within_episode = (
                        self.unit.gpu_reservation != "episode"
                        or sum(r["memory_mb"] for r in active if r["lease_id"] == row["lease_id"]) + row["memory_mb"]
                        <= self.leases[row["lease_id"]].demand.gpu_memory_mb
                    )
                    if within_episode and used + row["memory_mb"] <= min(self.unit.gpu_memory_mb, device["total_mb"]):
                        row["state"] = "starting"
                        return
            await asyncio.sleep(0.05)
        raise TimeoutError("MPS GPU capacity queue deadline expired.")

    async def execute(self, row):
        try:
            await self.admit(row)
            async with asyncio.timeout(row["timeout"]):
                row["launch_task"] = asyncio.create_task(self.launch(row))
                await asyncio.shield(row["launch_task"])
                row["output"] = await asyncio.shield(row["output_task"])
            row["state"] = "returning"
            await self.cleanup(row)
            row["state"] = "completed"
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            row["state"] = "returning"
            try:
                launching = row.get("launch_task")
                if launching is not None:
                    await asyncio.wait_for(asyncio.shield(asyncio.gather(launching, return_exceptions=True)), 40)
                task = asyncio.create_task(self.cleanup(row))
                await asyncio.shield(task)
                row["state"] = "cancelled" if cancelled else "failed"
                row["error"] = None if cancelled else repr(exc)
            except BaseException as cleanup_error:
                row["state"] = "quarantined"
                row["error"] = repr(cleanup_error)
                self.error = "MPS cleanup is unconfirmed. Domain is quarantined."
        finally:
            row["returned"] = time.time() if row["state"] != "quarantined" else None

    async def launch(self, row):
        lease = self.leases[row["lease_id"]]
        fence = await self.node._command_container(lease.lease_id, row["container"], lease.gpu_indices)
        group = JobFence("psrl-job-" + uuid.uuid4().hex, fence)
        row["fence"] = group
        identity = await self.node._gpu_identity()
        row["telemetry"] = TaskGpuTelemetry(
            self.node._gpu_probe(), group, lease.gpu_indices, identity_backend=identity
        )
        options = {
            "command": row["command"],
            "cwd": row["cwd"],
            "env": {
                **row["env"],
                "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe),
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.unit.mps_thread_percentage),
                "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": f"0={row['memory_mb']}M",
            },
        }
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            row["container"],
            "python3",
            "-c",
            RUNNER,
            json.dumps(options),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        row["process"] = proc
        ready = json.loads(await asyncio.wait_for(proc.stdout.readline(), 10))
        from psrl.workers.env_service.docker_api import inspect_container

        container = await inspect_container(row["container"])
        if container["Config"].get("User", "") not in ("", "0", "root", "0:0", "root:root"):
            raise RuntimeError("This MPS domain requires root clients matching its server UID.")
        host = container["HostConfig"]
        if "ALL" not in (host.get("CapDrop") or []) or host.get("CapAdd"):
            raise RuntimeError("MPS clients must drop all capabilities to match their control service.")
        ipc = host.get("IpcMode", "")
        service = await inspect_container(self.daemon)
        if ipc not in (f"container:{self.daemon}", f"container:{service['Id']}"):
            raise RuntimeError("MPS client does not share its assigned service IPC namespace.")
        if not any(
            m["Source"] == str(self.pipe) and m["Destination"] == str(self.pipe) for m in container.get("Mounts", [])
        ):
            raise RuntimeError("MPS client pipe mount does not match its domain.")
        namespace = (Path("/proc") / str(container["State"]["Pid"]) / "ns/pid").stat().st_ino
        pid = container_pid(fence, namespace, ready["pid"])
        group.attach(pid)
        group.set_access(True)
        row["telemetry"].start()
        if row["state"] != "starting":
            proc.stdin.close()
            raise asyncio.CancelledError()
        proc.stdin.write(b"go\n")
        await proc.stdin.drain()
        proc.stdin.close()
        row["state"], row["started"] = "running", time.time()

        async def collect(stream):
            result = b""
            while chunk := await stream.read(8192):
                result = (result + chunk)[-65536:]
            return result.decode(errors="replace")

        async def output():
            stdout, stderr = await asyncio.gather(collect(proc.stdout), collect(proc.stderr))
            return {"stdout": stdout, "stderr": stderr, "return_code": await proc.wait()}

        row["output_task"] = asyncio.create_task(output())

    async def cleanup(self, row):
        group = row.get("fence")
        if group is None:
            return
        group.set_access(False)
        # terminate_client needs a live CPU client on this driver. Freeze only to
        # stabilize ownership snapshots, then thaw for the CUDA teardown handshake.
        terminated = set()
        for _ in range(4):
            await asyncio.to_thread(group.freeze)
            try:
                pids = group.processes()
                servers = await asyncio.to_thread(self.control, "get_server_list")
                clients = await asyncio.to_thread(lambda: parse_clients(self.control("ps"))) if servers else []
                owned = [client for client in clients if client.pid in pids and client not in terminated]
                if not owned:
                    group.kill()
                    break
                bindings = [
                    await asyncio.to_thread(self.controller.bind, row["job_id"], 1, client.pid) for client in owned
                ]
            finally:
                group.thaw()
            for binding in bindings:
                await asyncio.to_thread(self.controller.terminate, binding, row["job_id"], 1)
                terminated.add(binding.client)
        else:
            raise RuntimeError("MPS client membership did not settle during cancellation.")
        if row.get("process"):
            proc = row["process"]
            if not row.get("output_task"):
                if proc.stdin:
                    proc.stdin.close()
                await asyncio.wait_for(proc.communicate(), 10)
            else:
                await asyncio.wait_for(asyncio.shield(row["output_task"]), 10)
        deadline = time.monotonic() + 10
        telemetry = row.get("telemetry")
        while True:
            readings = await self.node.gpu_lending_snapshot(self.unit.gpu_indices)
            gpu_pids = {pid for value in readings.values() for pid in value["pids"]}
            owners = await asyncio.to_thread(telemetry._owners, gpu_pids) if telemetry else {}
            if not group.processes() and not owners:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("MPS job still owns physical resources.")
            await asyncio.sleep(0.05)
        if telemetry:
            row["report"] = await telemetry.stop()
            telemetry.close_identity()
            row.pop("telemetry")
        group.close()
        row.pop("fence")

    async def drain(self, lease_id):
        if lease_id in self.leases:
            self.leases[lease_id].gpu_closing = True
        for job_id, row in list(self.jobs.items()):
            if row["lease_id"] == lease_id:
                await self.cancel(lease_id, job_id)
                if row["state"] == "quarantined":
                    await self.cleanup(row)
                    row["state"], row["returned"] = "cancelled", time.time()
        if not any(r["state"] == "quarantined" for r in self.jobs.values()):
            self.error = None

    async def release(self, lease_id):
        await self.drain(lease_id)
        self.leases.pop(lease_id, None)
        self.jobs = {key: row for key, row in self.jobs.items() if row["lease_id"] != lease_id}
        if not self.leases:
            await self.close()

    async def close(self):
        if any(row["state"] in ("starting", "running", "returning", "quarantined") for row in self.jobs.values()):
            raise RuntimeError("Cannot stop an MPS domain with unfinished jobs.")
        if self.started:
            try:
                await asyncio.to_thread(self.control, "quit")
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
            result = await asyncio.to_thread(
                subprocess.run, ["docker", "rm", "-f", self.daemon], capture_output=True, text=True, timeout=10
            )
            if result.returncode and "No such container" not in result.stderr:
                raise RuntimeError(f"MPS service removal failed: {result.stderr!r}.")
            self.started = False
        shutil.rmtree(self.root, ignore_errors=False)
