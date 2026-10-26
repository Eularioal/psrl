"""Keep cooperative CPU preparation outside container GPU admission and return."""

import asyncio
import json
import math
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path

from psrl.workers.env_service import gpu_job_process


@dataclass(frozen=True)
class PreparedGpuResult:
    """Keep command output and its own reservation together across cleanup awaits."""

    return_code: int
    stdout: str | None
    stderr: str | None
    reservation: dict


async def execute_prepared(
    environment,
    module_path,
    payload,
    *,
    python="python3",
    prepare_timeout_sec=180,
    timeout_sec=300,
    wait_timeout_sec=300,
):
    """Run prepare(payload) on CPU and its returned callable under exec_gpu."""
    budgets = (prepare_timeout_sec, timeout_sec, wait_timeout_sec)
    if not all(math.isfinite(value) and 0 < value <= 3600 for value in budgets):
        raise ValueError("Prepared GPU budgets must be finite and at most 3600 seconds.")
    directory = f"/tmp/psrl-prepared-{uuid.uuid4().hex}"
    helper = f"{directory}/prepared_gpu_process.py"

    def command(action, *args):
        return shlex.join([python, helper, action, directory, *(str(arg) for arg in args)])

    async def stop_process():
        stopped = await environment.exec(command("cancel"), timeout_sec=10)
        if stopped.return_code:
            raise RuntimeError(f"Prepared GPU cleanup failed: {stopped.stderr!r}.")

    setup = await environment.exec(shlex.join(["mkdir", "-m", "700", directory]), timeout_sec=10)
    if setup.return_code:
        raise RuntimeError(f"Prepared program directory creation failed: {setup.stderr!r}.")
    installed = False
    execution = None
    readiness = None
    try:
        await environment.upload_file(Path(gpu_job_process.__file__), f"{directory}/gpu_job_process.py")
        await environment.upload_file(Path(__file__).with_name("prepared_gpu_process.py"), helper)
        installed = True
        async with asyncio.timeout(sum(budgets) + 15):
            execution = asyncio.create_task(
                environment.exec(
                    command("prepare", module_path, json.dumps(payload), prepare_timeout_sec, sum(budgets)),
                    timeout_sec=sum(budgets) + 10,
                )
            )
            readiness = asyncio.create_task(
                environment.exec(
                    command("wait-ready", prepare_timeout_sec),
                    timeout_sec=prepare_timeout_sec + 10,
                )
            )
            done, _ = await asyncio.wait(
                (execution, readiness),
                return_when=asyncio.FIRST_COMPLETED,
            )
            prepared = await (readiness if readiness in done else execution)
            if prepared.return_code:
                raise RuntimeError(f"CPU preparation failed: {prepared.stderr!r}.")
            # Include local container lock waiting in the admission budget.
            async with asyncio.timeout(wait_timeout_sec + timeout_sec + 5):
                output = await environment.exec_gpu(
                    command("activate", timeout_sec),
                    timeout_sec=timeout_sec + 5,
                    wait_timeout_sec=wait_timeout_sec,
                    _cancel_process=stop_process,
                )
                return PreparedGpuResult(
                    output.return_code,
                    output.stdout,
                    output.stderr,
                    dict(environment.last_gpu_command or {}),
                )
    finally:

        async def cleanup():
            if installed:
                await stop_process()
            for task in (execution, readiness):
                if task is not None:
                    await asyncio.wait_for(asyncio.shield(task), timeout=10)
            removed = await environment.exec(shlex.join(["rm", "-rf", directory]), timeout_sec=10)
            if removed.return_code:
                raise RuntimeError(f"Prepared GPU directory cleanup failed: {removed.stderr!r}.")

        finishing = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(finishing)
        except asyncio.CancelledError:
            await finishing
            raise
