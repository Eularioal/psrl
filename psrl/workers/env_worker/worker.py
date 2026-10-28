"""Own sandbox containers and their persistent shells."""

from __future__ import annotations

import asyncio
import codecs
import fcntl
import logging
import os
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path

from psrl.workers.env_worker.process import ProcessResult, run_process
from psrl.workers.env_worker.sandbox import ExecResult, SandboxSpec, build_docker_run_argv
from psrl.workers.env_worker.shell import parse_sentinel, truncate_observation, wrap_command

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

SANDBOX_LABEL_KEY = "psrl.env_sandbox_id"
ACTOR_LABEL_KEY = "psrl.actor_id"

_SHELL_READ_CHUNK = 65536
_SHELL_POLL_INTERVAL_S = 0.05
_SHELL_OUTPUT_LIMIT = 8 * 1024 * 1024
_SHELL_COMMAND_LIMIT = 1024 * 1024


class ContainerShell:
    """
    Long-lived interactive shell attached to one container.

    The container is started with `docker run -i ... /bin/bash -l`, so writes to
    stdin execute in a shell whose state persists across calls.
    """

    def __init__(self, process: subprocess.Popen, container_name: str):
        self.process = process
        self.container_name = container_name
        self._buffer = ""
        self.lock = asyncio.Lock()
        # NOTE(claude): Reads must never block the event loop thread, and the shell
        # never closes, so there is no EOF to wait on. A non-blocking fd plus
        # `os.read` gives us "return whatever is available right now" semantics.
        assert self.process.stdout is not None, "Shell stdout is not available."
        self._fd = self.process.stdout.fileno()
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        assert self.process.stdin is not None, "Shell stdin is not available."
        self._write_fd = self.process.stdin.fileno()
        os.set_blocking(self._write_fd, False)

    async def write(self, text: str, deadline: float) -> bool:
        """Write bounded commands without blocking other sandboxes on a full pipe."""
        data = memoryview(text.encode())
        if len(data) > _SHELL_COMMAND_LIMIT:
            raise RuntimeError("Sandbox command exceeds the 1 MiB limit.")
        loop = asyncio.get_running_loop()
        while data:
            if loop.time() >= deadline:
                return False
            try:
                count = os.write(self._write_fd, data[:_SHELL_READ_CHUNK])
                data = data[count:]
            except BlockingIOError:
                await asyncio.sleep(min(_SHELL_POLL_INTERVAL_S, max(0, deadline - loop.time())))
        return True

    def _read_available(self) -> bytes:
        """Read whatever bytes are buffered right now, without blocking."""
        try:
            return os.read(self._fd, _SHELL_READ_CHUNK)
        except BlockingIOError:
            return b""
        except OSError as error:
            psrl_logger.warning(f"Sandbox shell read failed: {error}.")
            return b""

    def drain(self) -> str:
        """Discard output buffered between completed commands."""
        stale = self._buffer
        self._buffer = ""
        size = len(stale.encode())
        while True:
            chunk = self._read_available()
            if not chunk:
                break
            size += len(chunk)
            if size > _SHELL_OUTPUT_LIMIT:
                raise RuntimeError("Sandbox stale output exceeds the 8 MiB limit.")
            stale += chunk.decode(errors="replace")
        if stale.strip():
            psrl_logger.warning(f"Discarded stale sandbox output. Characters={len(stale)}.")
        return stale

    async def read_until(self, deadline: float, no_output_timeout_s: float, command_id: str) -> str:
        """
        Read shell output until the sentinel appears or a deadline elapses.

        Args:
            deadline (float): Monotonic loop time after which reading stops.
            no_output_timeout_s (float): Seconds of total silence after which reading
                stops, even when the overall deadline has not been reached.

        Returns:
            str: Accumulated output, which the caller parses for the sentinel.
        """
        loop = asyncio.get_running_loop()
        last_output_at = loop.time()
        size = len(self._buffer.encode())
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        while loop.time() < deadline:
            chunk = self._read_available()
            if chunk:
                size += len(chunk)
                if size > _SHELL_OUTPUT_LIMIT:
                    raise RuntimeError("Sandbox output exceeds the 8 MiB limit.")
                self._buffer += decoder.decode(chunk)
                last_output_at = loop.time()
                if parse_sentinel(self._buffer, command_id) is not None:
                    break
                await asyncio.sleep(0)
                continue

            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Sandbox shell exited with code {self.process.returncode}: {self._buffer[-1000:]!r}."
                )
            now = loop.time()
            if now >= deadline:
                break
            if now - last_output_at >= no_output_timeout_s:
                psrl_logger.warning(f"Sandbox shell produced no output for {no_output_timeout_s}s.")
                break
            await asyncio.sleep(_SHELL_POLL_INTERVAL_S)

        buffer, self._buffer = self._buffer, ""
        return buffer

    def close(self) -> None:
        """Terminate the shell process."""
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except OSError:
            pass
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        if self.process.stdout is not None:
            self.process.stdout.close()


class EnvWorker:
    """
    Own the sandbox containers scheduled onto one placement unit.

    The class is plain Python. `EnvWorkerManager` wraps it with `ray.remote` so the
    resource request can be built at runtime from configuration.
    """

    def __init__(
        self,
        worker_id: int,
        cpu_slots: int,
        gpu_slots: int,
        exec_default_timeout_s: float,
        max_observation_chars: int = 0,
    ):
        self.worker_id = worker_id
        self.cpu_slots = cpu_slots
        self.gpu_slots = gpu_slots
        self.exec_default_timeout_s = exec_default_timeout_s
        self.max_observation_chars = max_observation_chars

        # NOTE(claude): Ray sets `CUDA_VISIBLE_DEVICES` to exactly the GPUs it granted
        # this actor. Deriving sandbox devices from it makes it impossible for a
        # sandbox to touch a training GPU.
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        self.available_gpu_indices: tuple[int, ...] = (
            tuple(int(part) for part in visible.split(",") if part.strip() != "") if visible else ()
        )

        self._shells: dict[str, ContainerShell] = {}
        self._container_names: dict[str, str] = {}
        self._sandbox_gpus: dict[str, tuple[int, ...]] = {}
        self._last_used_at: dict[str, float] = {}
        self._busy_gpu_indices: set[int] = set()
        self._destroy_locks: dict[str, asyncio.Lock] = {}
        self._children: dict[str, set[str]] = {}
        self._closing: dict[str, float] = {}

    def _allocate_gpu_indices(self, count: int) -> tuple[int, ...]:
        """
        Reserve `count` physical GPU indices for one sandbox.

        Raises:
            RuntimeError: If not enough free GPUs remain on this worker.
        """
        if count == 0:
            return ()
        free = [index for index in self.available_gpu_indices if index not in self._busy_gpu_indices]
        if len(free) < count:
            raise RuntimeError(
                f"Worker {self.worker_id} has no free GPU capacity for {count} device(s), free={free!r}."
            )
        chosen = tuple(free[:count])
        self._busy_gpu_indices.update(chosen)
        return chosen

    def _build_labels(self, spec: SandboxSpec, sandbox_id: str) -> dict[str, str]:
        """Merge caller labels with the labels the reaper needs for cleanup."""
        labels = dict(spec.labels)
        labels[SANDBOX_LABEL_KEY] = sandbox_id
        actor_id = os.getenv("PSRL_ACTOR_ID", "")
        if actor_id:
            labels[ACTOR_LABEL_KEY] = actor_id
        return labels

    async def create_sandbox(self, spec: SandboxSpec, sandbox_id: str | None = None) -> str:
        """
        Start one container and its shell.

        Args:
            spec (SandboxSpec): Sandbox description.

        Returns:
            str: Sandbox id used by all later calls.

        Raises:
            RuntimeError: If the shell does not become responsive before the timeout.
        """
        now = time.monotonic()
        for closed_id, closed_at in list(self._closing.items()):
            if now - closed_at > 3600 and closed_id not in self._container_names:
                self._closing.pop(closed_id, None)
                self._destroy_locks.pop(closed_id, None)
        sandbox_id = sandbox_id or f"s-{uuid.uuid4().hex[:12]}"
        if sandbox_id in self._closing:
            raise RuntimeError("Sandbox reservation was cancelled before startup.")
        container_name = f"psrl-env-{sandbox_id}"
        gpu_indices = self._allocate_gpu_indices(spec.gpus)

        labelled_spec = replace(spec, labels=self._build_labels(spec, sandbox_id))
        try:
            argv = build_docker_run_argv(labelled_spec, container_name, gpu_indices)
            psrl_logger.info(f"Starting sandbox={sandbox_id!r} with image={spec.image!r}.")
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
        except Exception:
            for index in gpu_indices:
                self._busy_gpu_indices.discard(index)
            raise
        shell = ContainerShell(process, container_name)
        self._shells[sandbox_id] = shell
        self._container_names[sandbox_id] = container_name
        self._sandbox_gpus[sandbox_id] = gpu_indices
        self._last_used_at[sandbox_id] = time.monotonic()

        try:
            probe = await self.exec(sandbox_id, "echo ready", timeout_s=spec.startup_timeout_s)
            if probe.timed_out or probe.exit_code != 0 or "ready" not in probe.stdout:
                raise RuntimeError(f"Sandbox {sandbox_id!r} failed readiness: {probe.stdout[:200]!r}.")
        except BaseException:
            await self.destroy_sandbox(sandbox_id)
            raise
        return sandbox_id

    async def exec(
        self,
        sandbox_id: str,
        command: str,
        timeout_s: float | None = None,
        no_output_timeout_s: float | None = None,
    ) -> ExecResult:
        """
        Run one command in a sandbox's persistent shell.

        Timeouts destroy the sandbox before returning `ExecResult.timed_out`.

        Args:
            sandbox_id (str): Sandbox to run in.
            command (str): Shell command.
            timeout_s (float): Maximum seconds to wait for the sentinel.
            no_output_timeout_s (float | None): Maximum silent seconds. Defaults to
                `timeout_s`.

        Returns:
            ExecResult: Output, exit code, timeout flag, and duration.

        Raises:
            KeyError: If the sandbox is unknown.
        """
        if sandbox_id not in self._shells:
            raise KeyError(f"Unknown sandbox id {sandbox_id!r}.")

        timeout_s = self.exec_default_timeout_s if timeout_s is None else timeout_s
        shell = self._shells[sandbox_id]
        async with shell.lock:
            if sandbox_id not in self._shells or sandbox_id in self._closing:
                raise RuntimeError("Sandbox is being destroyed.")
            started_at = time.monotonic()
            command_id = uuid.uuid4().hex
            try:
                deadline = asyncio.get_running_loop().time() + timeout_s
                shell.drain()
                written = await shell.write(wrap_command(command, command_id), deadline)
                buffer = ""
                if written:
                    buffer = await shell.read_until(
                        deadline,
                        float(timeout_s if no_output_timeout_s is None else no_output_timeout_s),
                        command_id,
                    )
                parsed = parse_sentinel(buffer, command_id)
                if parsed is None:
                    # An uncertain shell must never produce another training observation.
                    await self.destroy_sandbox(sandbox_id)
                    return ExecResult(
                        stdout=truncate_observation(buffer, self.max_observation_chars),
                        exit_code=None,
                        timed_out=True,
                        duration_s=time.monotonic() - started_at,
                    )
                body, exit_code = parsed
                self._last_used_at[sandbox_id] = time.monotonic()
                return ExecResult(
                    stdout=truncate_observation(body, self.max_observation_chars),
                    exit_code=exit_code,
                    timed_out=False,
                    duration_s=time.monotonic() - started_at,
                )
            except BaseException:
                await self.destroy_sandbox(sandbox_id)
                raise

    def container_name(self, sandbox_id: str) -> str:
        """Resolve only an open sandbox owned by this worker."""
        if sandbox_id in self._closing or sandbox_id not in self._container_names:
            raise RuntimeError("Sandbox is absent or closing.")
        return self._container_names[sandbox_id]

    async def exec_process(
        self,
        sandbox_id: str,
        command: str,
        timeout_s: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
        output_callback=None,
    ) -> ProcessResult:
        """Execute in a fresh shell and destroy the sandbox on uncertain command completion."""
        argv = ["docker", "exec"]
        if cwd is not None:
            argv.extend(["--workdir", cwd])
        if user is not None:
            argv.extend(["--user", str(user)])
        for key, value in sorted((env or {}).items()):
            argv.extend(["--env", f"{key}={value}"])
        argv.extend([self.container_name(sandbox_id), "/bin/bash", "-c", command])
        self._last_used_at[sandbox_id] = time.monotonic()
        try:
            result = await run_process(
                argv, self.exec_default_timeout_s if timeout_s is None else timeout_s, output_callback=output_callback
            )
            self._last_used_at[sandbox_id] = time.monotonic()
            return result
        except BaseException:
            cleanup = asyncio.create_task(self.destroy_sandbox(sandbox_id))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise

    async def copy_path(self, sandbox_id: str, host_path: str, container_path: str, upload: bool) -> None:
        """Transfer a file or directory through the same owned container lifecycle."""
        container = self.container_name(sandbox_id)
        local = str(Path(host_path).resolve())
        if upload and host_path.endswith("/."):
            local += "/."
        remote = f"{container}:{container_path}"
        source, target = (local, remote) if upload else (remote, local)
        result = await run_process(["docker", "cp", source, target], 30)
        if result.return_code:
            raise RuntimeError(f"Sandbox file transfer failed: {result.stderr!r}.")

    async def read_file(self, sandbox_id: str, path: str) -> bytes:
        """Copy one file out of a sandbox via docker exec."""
        container_name = self._container_names[sandbox_id]
        completed = await asyncio.to_thread(
            subprocess.run,
            ["docker", "exec", container_name, "/usr/bin/head", "-c", str(64 * 1024 * 1024 + 1), "--", path],
            capture_output=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to read {path!r} from sandbox {sandbox_id!r}: "
                f"{completed.stderr.decode(errors='replace')[:200]}."
            )
        if len(completed.stdout) > 64 * 1024 * 1024:
            raise ValueError("Sandbox file exceeds the 64 MiB limit.")
        return completed.stdout

    async def write_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        """Write bytes through a bounded exec without blocking the persistent shell."""
        if sandbox_id in self._closing:
            raise RuntimeError("Sandbox is closing.")
        completed = await asyncio.to_thread(
            subprocess.run,
            ["docker", "exec", "-i", self._container_names[sandbox_id], "/usr/bin/tee", "--", path],
            input=data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if completed.returncode:
            raise RuntimeError(f"Failed to write sandbox file {path!r}.")

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """Confirm physical deletion before releasing local resource ownership."""
        self._closing[sandbox_id] = time.monotonic()
        async with self._destroy_locks.setdefault(sandbox_id, asyncio.Lock()):
            for child_id in tuple(self._children.get(sandbox_id, ())):
                await self.destroy_sandbox(child_id)
            container_name = self._container_names.get(sandbox_id)
            if container_name is not None:
                completed = await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    timeout=20,
                )
                if completed.returncode and b"No such container" not in completed.stderr:
                    raise RuntimeError(f"Container deletion failed for {sandbox_id!r}.")
            shell = self._shells.pop(sandbox_id, None)
            if shell is not None:
                shell.close()
            self._container_names.pop(sandbox_id, None)
            for index in self._sandbox_gpus.pop(sandbox_id, ()):
                self._busy_gpu_indices.discard(index)
            self._last_used_at.pop(sandbox_id, None)
            self._children.pop(sandbox_id, None)

    async def evaluate(self, sandbox_id: str, spec: SandboxSpec, command: str, timeout_s: float) -> ExecResult:
        """Grade a bounded submission snapshot in a separate, owned container."""
        if sandbox_id not in self._shells or sandbox_id in self._closing:
            raise RuntimeError("Cannot grade a closed sandbox.")
        submission = await self.read_file(sandbox_id, "/home/agent/workspace/submission.csv")
        if len(submission) > 64 * 1024 * 1024:
            raise ValueError("Submission exceeds the 64 MiB limit.")
        if sandbox_id in self._closing:
            raise RuntimeError("Sandbox was closed while its submission was copied.")
        child_id = f"v-{uuid.uuid4().hex}"
        self._children.setdefault(sandbox_id, set()).add(child_id)
        try:
            await self.create_sandbox(spec, child_id)
            await self.write_file(child_id, "/home/agent/workspace/submission.csv", submission)
            if spec.code_execution is not None:
                from psrl.workers.env_worker.isolated_code import evaluate_isolated

                return await evaluate_isolated(self, sandbox_id, child_id, spec, timeout_s)
            return await self.exec(child_id, command, timeout_s)
        finally:
            await self.destroy_sandbox(child_id)
            self._children.get(sandbox_id, set()).discard(child_id)

    async def live_sandbox_count(self) -> int:
        """Return how many sandboxes this worker currently owns."""
        return len(self._shells)

    async def node_ip(self) -> str:
        """Return the IP of the node this worker runs on."""
        import ray

        return ray.util.get_node_ip_address()
