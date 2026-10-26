"""Exercise real CPU child lifetimes with a simulated exclusive GPU admission."""

import asyncio
import os
import shutil
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.env_service.prepared_gpu import execute_prepared


class Environment:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.running = asyncio.Event()
        self.gpu_entries = 0
        self.cancelled_after_stop = False
        self.last_gpu_command = None

    async def exec(self, command, timeout_sec=10):
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True
        )
        try:
            async with asyncio.timeout(timeout_sec):
                stdout, stderr = await process.communicate()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        return SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())

    async def upload_file(self, source, destination):
        shutil.copyfile(source, destination)

    async def exec_gpu(self, command, *, timeout_sec, wait_timeout_sec, _cancel_process):
        async with self.lock:
            self.gpu_entries += 1
            self.last_gpu_command = {"request_id": self.gpu_entries}
            self.running.set()
            try:
                return await self.exec(command, timeout_sec)
            except BaseException:
                await _cancel_process()
                self.cancelled_after_stop = True
                raise


@pytest.mark.asyncio
async def test_cpu_preparation_overlaps_before_serial_gpu_execution(tmp_path):
    program = tmp_path / "program.py"
    program.write_text(
        "from pathlib import Path\nimport time\n"
        "def prepare(payload):\n"
        "    root=Path(payload['root'])\n"
        "    (root/('ready-'+str(payload['id']))).touch()\n"
        "    end=time.monotonic()+3\n"
        "    while len(list(root.glob('ready-*')))<2:\n"
        "        assert time.monotonic()<end, 'CPU preparations were serialized.'\n"
        "        time.sleep(0.01)\n"
        "    def run():\n"
        "        print('gpu-output-'+str(payload['id']),flush=True)\n"
        "    return run\n"
    )
    environment = Environment()
    outputs = await asyncio.gather(
        *(
            execute_prepared(
                environment,
                str(program),
                {"root": str(tmp_path), "id": i},
                python=sys.executable,
                prepare_timeout_sec=5,
                wait_timeout_sec=5,
                timeout_sec=5,
            )
            for i in range(2)
        )
    )
    assert all(result.return_code == 0 for result in outputs), "Both GPU phases must finish."
    assert [result.stdout.strip() for result in outputs] == ["gpu-output-0", "gpu-output-1"], (
        "Each prepared program must retain its own state."
    )
    assert environment.gpu_entries == 2, "Preparation must not consume GPU admissions."
    assert {result.reservation["request_id"] for result in outputs} == {1, 2}, (
        "Cleanup must preserve each command's own reservation."
    )


@pytest.mark.asyncio
async def test_preparation_failure_never_requests_gpu(tmp_path):
    program = tmp_path / "bad.py"
    program.write_text("def prepare(payload):\n    raise RuntimeError('Invalid input.')\n")
    environment = Environment()
    with pytest.raises(RuntimeError, match="Invalid input"):
        await execute_prepared(
            environment,
            str(program),
            {},
            python=sys.executable,
            prepare_timeout_sec=5,
            wait_timeout_sec=5,
            timeout_sec=5,
        )
    assert environment.gpu_entries == 0, "Failed CPU preparation must not reserve GPU capacity."


@pytest.mark.asyncio
async def test_cancel_stops_prepared_child_before_gpu_return(tmp_path):
    program = tmp_path / "long.py"
    marker = tmp_path / "running"
    program.write_text(
        "import os,time\nfrom pathlib import Path\n"
        "def prepare(payload):\n"
        "    def run():\n"
        "        Path(payload['marker']).write_text(str(os.getpid()))\n"
        "        time.sleep(30)\n"
        "    return run\n"
    )
    environment = Environment()
    operation = asyncio.create_task(
        execute_prepared(
            environment,
            str(program),
            {"marker": str(marker)},
            python=sys.executable,
            prepare_timeout_sec=5,
            wait_timeout_sec=5,
            timeout_sec=30,
        )
    )
    async with asyncio.timeout(5):
        while not marker.exists():
            await asyncio.sleep(0.01)
    pid = int(marker.read_text())
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert environment.cancelled_after_stop, "GPU cancellation must follow process termination."
    status = Path(f"/proc/{pid}/stat")
    assert not status.exists() or status.read_text().rsplit(")", 1)[1].split()[0] == "Z", (
        "The prepared worker must no longer execute after cancellation."
    )


@pytest.mark.asyncio
async def test_managed_environment_stops_process_before_router_cancellation(monkeypatch):
    import ray
    from harbor.environments.docker.docker import DockerEnvironment
    from psrl.workers.env_service.docker import ManagedDockerEnvironment

    order = []

    async def stop():
        order.append("stopped")

    async def cancel(*args):
        assert order == ["stopped"], "GPU return must wait for prepared process termination."
        order.append("returned")

    router = SimpleNamespace(
        acquire_gpu_command=SimpleNamespace(remote=AsyncMock(return_value={"status": "granted", "epoch": 1})),
        cancel_gpu_command=SimpleNamespace(remote=cancel),
    )
    monkeypatch.setattr(ray, "get_actor", lambda name: router)
    monkeypatch.setattr(DockerEnvironment, "exec", AsyncMock(side_effect=TimeoutError("Execution expired.")))
    env = ManagedDockerEnvironment.__new__(ManagedDockerEnvironment)
    env._gpu_commands = {"router_name": "test", "lease_id": "lease"}
    env._gpu_command_lock = asyncio.Lock()
    env._pending_gpu_return = None
    env._run_docker_compose_command = AsyncMock(return_value=SimpleNamespace(stdout="container"))
    with pytest.raises(TimeoutError, match="Execution expired"):
        await env.exec_gpu("command", _cancel_process=stop)
    assert order == ["stopped", "returned"], "Both cancellation phases must complete in order."
