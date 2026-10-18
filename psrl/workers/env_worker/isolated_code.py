"""Broker a private judge and disposable code containers with a bounded data protocol."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import suppress

from psrl.workers.env_worker.sandbox import ExecResult, SandboxSpec

WIRE_LIMIT = 2 * 1024 * 1024


class OutputLimitError(Exception):
    """A child process exceeded its allowed output size."""


async def read_bounded(stream, limit):
    """Drain a pipe while enforcing a hard byte budget."""
    chunks, size = [], 0
    while chunk := await stream.read(65536):
        size += len(chunk)
        if size > limit:
            raise OutputLimitError("Child output exceeded its byte limit.")
        chunks.append(chunk)
    return b"".join(chunks)


async def stop_process(process):
    """Reap the owned CLI process even if its container is already gone."""
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    await process.communicate()


class CandidateProcess:
    """Keep one disposable execution channel per candidate, never trust its control output."""

    def __init__(self, process, config):
        self.process = process
        self.config = config
        self.stderr_task = asyncio.create_task(read_bounded(process.stderr, config.output_bytes))

    @classmethod
    async def start(cls, container, config, source, function):
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "--workdir",
            "/tmp",
            container,
            config.python,
            "-I",
            "-B",
            config.runner_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=WIRE_LIMIT,
        )
        session = cls(process, config)
        try:
            async with asyncio.timeout(10):
                if json.loads(await process.stdout.readline()) != {"ready": True}:
                    raise RuntimeError("Code execution image or Docker command is unavailable.")
                process.stdin.write((json.dumps({"code": source, "fn_name": function}) + "\n").encode())
                await process.stdin.drain()
            return session
        except (ValueError, TimeoutError) as error:
            await session.close()
            raise RuntimeError("Code execution process could not become ready.") from error
        except BaseException:
            await session.close()
            raise

    async def run_case(self, inputs, timeout_s):
        """Treat every response as candidate output to be compared by the private judge."""
        line_task = None
        try:
            async with asyncio.timeout(timeout_s):
                self.process.stdin.write((json.dumps({"input": inputs}, allow_nan=False) + "\n").encode())
                await self.process.stdin.drain()
                line_task = asyncio.create_task(self.process.stdout.readline())
                done, _ = await asyncio.wait((line_task, self.stderr_task), return_when=asyncio.FIRST_COMPLETED)
                if self.stderr_task in done:
                    self.stderr_task.result()
                line = await line_task
                if len(line) > WIRE_LIMIT:
                    raise OutputLimitError("Candidate response exceeded its limit.")
                response = json.loads(line)
                if (
                    not isinstance(response, dict)
                    or type(response.get("ok")) is not bool
                    or not isinstance(response.get("stdout"), str)
                    or len(response["stdout"].encode()) > self.config.output_bytes
                ):
                    raise ValueError("Candidate returned invalid output data.")
                return {"ok": response["ok"], "stdout": response["stdout"]}
        except (TimeoutError, OutputLimitError, ValueError, BrokenPipeError, ConnectionResetError):
            return {"ok": False, "stdout": ""}
        finally:
            if line_task is not None:
                line_task.cancel()
                await asyncio.gather(line_task, return_exceptions=True)

    async def close(self):
        self.stderr_task.cancel()
        await asyncio.gather(self.stderr_task, return_exceptions=True)
        await stop_process(self.process)


def candidate_spec(verifier):
    """Mount only a public runner into a fresh namespace with no datasets or credentials."""
    config = verifier.code_execution
    return SandboxSpec(
        image=verifier.image,
        cpus=config.cpus,
        memory=config.memory,
        mounts=((config.runner_host_dir, "/opt/psrl-code", "ro"),),
        env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        network="none",
        labels={**verifier.labels, "psrl.airs_role": "candidate"},
        startup_timeout_s=verifier.startup_timeout_s,
        read_only=True,
        user="agent",
        pids_limit=64,
        tmpfs=("/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",),
    )


async def evaluate_isolated(worker, parent_id, judge_id, spec, timeout_s):
    """Let only the trusted judge produce scores and own all children under the episode."""
    started = time.monotonic()
    config = spec.code_execution
    candidate_id, source, function, candidate_deadline = None, None, None, 0
    judge = None
    candidate = None
    stderr_task = None
    try:
        async with asyncio.timeout(timeout_s):
            judge = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                "-i",
                worker._container_names[judge_id],
                config.python,
                "-I",
                "-B",
                "/opt/psrl-apps/judge.py",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=WIRE_LIMIT,
            )
            stderr_task = asyncio.create_task(read_bounded(judge.stderr, 65536))
            metrics = None
            invalid = False
            while True:
                line_task = asyncio.create_task(judge.stdout.readline())
                try:
                    done, _ = await asyncio.wait((line_task, stderr_task), return_when=asyncio.FIRST_COMPLETED)
                    if stderr_task in done:
                        stderr_task.result()
                    line = await line_task
                finally:
                    line_task.cancel()
                    await asyncio.gather(line_task, return_exceptions=True)
                if not line:
                    break
                if len(line) >= WIRE_LIMIT:
                    raise RuntimeError("Judge request exceeded its byte limit.")
                try:
                    message = json.loads(line)
                except ValueError as error:
                    raise RuntimeError("Private judge emitted invalid protocol data.") from error
                op = message["op"]
                if op == "result":
                    if candidate_id is not None or metrics is not None:
                        raise RuntimeError("Judge result arrived in an invalid protocol state.")
                    metrics = message["metrics"]
                    continue
                if op == "invalid_submission":
                    invalid = True
                    continue
                if metrics is not None or invalid:
                    raise RuntimeError("Judge emitted requests after its terminal result.")
                if op == "begin":
                    if candidate_id is not None:
                        raise RuntimeError("Judge tried to overlap candidate containers.")
                    source, function = message["code"], message["fn_name"]
                    candidate_id = f"c-{uuid.uuid4().hex}"
                    if parent_id in worker._closing:
                        raise RuntimeError("Episode closed before code execution.")
                    worker._children.setdefault(parent_id, set()).add(candidate_id)
                    await worker.create_sandbox(candidate_spec(spec), candidate_id)
                    candidate = await CandidateProcess.start(
                        worker._container_names[candidate_id], config, source, function
                    )
                    candidate_deadline = time.monotonic() + config.candidate_timeout_s
                    response = {"ok": True}
                elif op == "case":
                    if candidate_id is None:
                        raise RuntimeError("Judge requested an input without a candidate.")
                    remaining = candidate_deadline - time.monotonic()
                    response = {"ok": False, "stdout": ""}
                    if remaining > 0:
                        response = await candidate.run_case(message["input"], min(config.case_timeout_s, remaining))
                elif op == "end":
                    if candidate_id is None:
                        raise RuntimeError("Judge ended a nonexistent candidate.")
                    await candidate.close()
                    candidate = None
                    await worker.destroy_sandbox(candidate_id)
                    worker._children.get(parent_id, set()).discard(candidate_id)
                    candidate_id = None
                    response = {"ok": True}
                else:
                    raise RuntimeError("Judge requested an unknown operation.")
                judge.stdin.write((json.dumps(response) + "\n").encode())
                await judge.stdin.drain()
            code = await judge.wait()
            await stderr_task
            if code != 0 or candidate_id is not None or (metrics is None and not invalid):
                raise RuntimeError("Private judge failed without a complete result.")
            return ExecResult(
                "" if invalid else json.dumps(metrics), 2 if invalid else 0, False, time.monotonic() - started
            )
    finally:
        if stderr_task is not None:
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
        if judge is not None:
            await stop_process(judge)
        if candidate is not None:
            await candidate.close()
        if candidate_id is not None:
            await worker.destroy_sandbox(candidate_id)
            worker._children.get(parent_id, set()).discard(candidate_id)
