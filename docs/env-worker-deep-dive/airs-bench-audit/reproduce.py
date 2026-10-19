#!/usr/bin/env python3
"""CPU audit of AIRS shell and lifecycle contracts, without Docker or Ray startup.

Run from the workspace root:
    python docs/env-worker-deep-dive/airs-bench-audit/reproduce.py

Production definitions are compiled from their AST (Python syntax tree), with
their original filenames and line numbers. Only dependency imports are replaced.
MLGym checks replay selected real methods with minimal state, not a full episode.
The RPC concurrency check models Ray's admission limit with a semaphore.
"""

from __future__ import annotations
import __future__

import argparse
import ast
import asyncio
import concurrent.futures
import dataclasses
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent / "baseline"
LOG = logging.getLogger("airs-audit")


def load_definitions(path, names=None, injected=None):
    """Load selected real definitions without importing the training stack."""
    tree = ast.parse(path.read_text(), filename=str(path))
    if names is not None:
        tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    else:
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("psrl"))
        ]
    module_name = f"audit_{path.stem}_{len(sys.modules)}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    if injected:
        module.__dict__.update(injected)
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), module.__dict__)
    return module


def extract_methods(path, class_name, names, injected=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert len(methods) == len(names), (path, names)
    namespace = dict(injected or {})
    exec(
        compile(
            ast.Module(body=methods, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    return {name: namespace[name] for name in names}


def load_backend():
    base = REPO / "psrl/workers/env_worker"
    schema = load_definitions(
        base / "sandbox.py",
        {"SandboxSpec", "ExecResult"},
        {"dataclass": dataclasses.dataclass, "field": dataclasses.field},
    )
    shell = load_definitions(base / "shell.py")
    worker = load_definitions(
        base / "worker.py",
        injected={
            "ExecResult": schema.ExecResult,
            "SandboxSpec": schema.SandboxSpec,
            "parse_sentinel": shell.parse_sentinel,
            "wrap_command": shell.wrap_command,
            "truncate_observation": shell.truncate_observation,
        },
    )
    coordinator = load_definitions(
        base / "coordinator.py",
        injected={
            "ExecResult": schema.ExecResult,
            "SandboxSpec": schema.SandboxSpec,
        },
    )
    return schema, worker, coordinator


async def shell_checks(worker_module):
    # Diagram [5]: real EnvWorker.exec -> real ContainerShell -> a local Bash pipe.
    process = subprocess.Popen(
        ["bash", "--noprofile", "--norc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    worker = worker_module.EnvWorker(0, 1, 0, 2.0)
    worker._shells["audit"] = worker_module.ContainerShell(process, "local-bash-only")
    try:
        results = []
        for command, timeout in [
            ("echo A-start; sleep 0.6; echo A-end; true", 0.1),
            ("sleep 0.35; echo B-output; false", 2.0),
            ("echo C-output; true", 2.0),
        ]:
            result = await worker.exec("audit", command, timeout)
            results.append({"command": command, **dataclasses.asdict(result)})
        assert results[0]["timed_out"]
        assert "A-end" in results[1]["stdout"] and results[1]["exit_code"] == 0
        assert "B-output" in results[2]["stdout"] and results[2]["exit_code"] == 1
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
        process.stdin.close()
        process.stdout.close()

    # Diagram [3]: a process that has already exited still consumes the read budget.
    dead = subprocess.Popen(
        ["bash", "-c", "echo startup-failed; exit 7"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    dead.wait(timeout=3)
    try:
        shell = worker_module.ContainerShell(dead, "already-exited-local-process")
        started = time.monotonic()
        output = await shell.read_until(asyncio.get_running_loop().time() + 0.2, 0.2)
        elapsed = time.monotonic() - started
        assert elapsed >= 0.18
        return {
            "command_results": results,
            "exited_process": {
                "returncode_before_read": dead.returncode,
                "read_elapsed_s": elapsed,
                "output": output,
            },
        }
    finally:
        dead.stdin.close()
        dead.stdout.close()


async def lifecycle_checks(schema, worker_module, coordinator_module):
    # Diagram [7]: execute real idle reaper, replacing only physical container removal.
    coord = coordinator_module.EnvWorkerCoordinator()
    await coord.register_worker(None, 0, 1, 0, "local-model")
    await coord.select_worker(schema.SandboxSpec(image="unused"), 1)
    worker = worker_module.EnvWorker(0, 1, 0, 1, idle_sandbox_timeout_s=0.01)
    worker._last_used_at["orphan"] = time.monotonic() - 1
    removed = []

    async def remove_only(sandbox_id):
        removed.append(sandbox_id)
        worker._last_used_at.pop(sandbox_id, None)

    worker.destroy_sandbox = remove_only
    await worker.reap_idle_sandboxes()
    used_after_reap = coord._slots[0].used_cpu
    assert removed == ["orphan"] and used_after_reap == 1

    # Diagram [3]/[7]: semaphore models an RPC concurrency pool, not a real Ray actor.
    rpc_slots = asyncio.Semaphore(2)
    entered = 0
    all_entered = asyncio.Event()

    async def queued_create():
        nonlocal entered
        async with rpc_slots:
            entered += 1
            if entered == 2:
                all_entered.set()
            try:
                await coord.select_worker(schema.SandboxSpec(image="unused"), 0.1)
            except RuntimeError:
                return "placement_timeout"

    async def release():
        async with rpc_slots:
            await coord.release(0, 0)

    waiters = [asyncio.create_task(queued_create()) for _ in range(2)]
    await all_entered.wait()
    releasing = asyncio.create_task(release())
    await asyncio.sleep(0.03)
    release_blocked = not releasing.done()
    results = await asyncio.wait_for(asyncio.gather(*waiters), 2)
    await asyncio.wait_for(releasing, 2)
    assert release_blocked and results == ["placement_timeout", "placement_timeout"]
    return {
        "idle_reaper_removed": removed,
        "coordinator_used_cpu_after_reap": used_after_reap,
        "rpc_pool_model": {
            "release_blocked": release_blocked,
            "waiter_results": results,
            "release_eventually_completed": releasing.done(),
        },
    }


async def thread_check():
    # Diagram [4]/[7]: same executor/wait_for contract, using a bounded local callable.
    started = threading.Event()
    stop = threading.Event()
    finished = threading.Event()

    def work():
        started.set()
        stop.wait(timeout=2)
        finished.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = asyncio.get_running_loop().run_in_executor(pool, work)
        while not started.is_set():
            await asyncio.sleep(0.001)
        try:
            await asyncio.wait_for(future, timeout=0.02)
        except TimeoutError:
            running_after_timeout = not finished.is_set()
        finally:
            stop.set()
    assert running_after_timeout
    return {
        "awaited_future_cancelled": future.cancelled(),
        "thread_still_running_at_timeout": running_after_timeout,
        "thread_explicitly_stopped_before_script_exit": finished.is_set(),
    }


def mlgym_checks(path):
    if not path.exists():
        return {"status": "not_run", "reason": "configured MLGym env.py does not exist"}
    # Diagram [4]: actual inherited methods, minimal object state, no imports or Docker.
    ns = {
        "psrl_logger": LOG,
        "AgentInfo": dict,
        "re": re,
        "EDIT_PATTERN": re.compile(r"^edit\s+(\d+):(\d+)(?:\n|$|\s)"),
        "TRAIN_COMMANDS": ["torchrun", "python", "python3", "accelerate", "deepspeed"],
        "AGENT_SHORT_ACTION_TIMEOUT": 25,
        "AGENT_ACTION_NO_OUTPUT_TIMEOUT": 3600,
        "NoOutputTimeoutError": type("NoOutputTimeoutError", (TimeoutError,), {}),
    }
    inherited = extract_methods(path, "MLGymEnv", {"_reset_container", "_init_scripts", "interrupt", "step"}, ns)
    adapter = extract_methods(REPO / "psrl/environments/mlgym_env.py", "PSRLMLGymEnv", {"_init_container"}, ns)
    cls = type("MinimalMLGymContractReplay", (), {**inherited, **adapter})
    obj = cls()
    obj.container = None
    obj.container_obj = None
    obj._sandbox_handle = types.SimpleNamespace(sandbox_id="audit")
    obj.logger = LOG
    failures = {}
    try:
        obj._reset_container()
    except AssertionError as exc:
        failures["initialization"] = {
            "exception": type(exc).__name__,
            "function": exc.__traceback__.tb_next.tb_next.tb_frame.f_code.co_name,
            "line": exc.__traceback__.tb_next.tb_next.tb_lineno,
        }

    def timeout(**kwargs):
        raise TimeoutError("controlled command timeout")

    obj.current_step = 0
    obj.max_steps = 40
    obj.task = types.SimpleNamespace(args=types.SimpleNamespace(training_timeout=3600))
    obj.communicate = timeout
    try:
        obj.step("sleep 1")
    except AssertionError as exc:
        tb = exc.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        failures["step_timeout"] = {
            "exception": type(exc).__name__,
            "function": tb.tb_frame.f_code.co_name,
            "line": tb.tb_lineno,
        }
    assert set(failures) == {"initialization", "step_timeout"}
    return {
        "status": "targeted_real_method_replay",
        "failures": failures,
        "env_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlgym-repo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_tree = ast.parse((REPO / "examples/airs_bench/config.py").read_text())
    configured = next(
        ast.literal_eval(n.value)
        for n in config_tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "DEFAULT_MLGYM_REPO" for t in n.targets)
    )
    mlgym_env = (args.mlgym_repo or Path(configured)) / "mlgym/environment/env.py"
    schema, worker, coordinator = load_backend()
    result = {
        "scope": "CPU only; no Docker, GPU, Ray cluster, model, or dataset execution",
        "shell": await shell_checks(worker),
        "lifecycle": await lifecycle_checks(schema, worker, coordinator),
        "executor": await thread_check(),
        "mlgym": mlgym_checks(mlgym_env),
    }
    paths = [
        "psrl/workers/env_worker/worker.py",
        "psrl/workers/env_worker/shell.py",
        "psrl/workers/env_worker/coordinator.py",
        "psrl/environments/mlgym_env.py",
        "psrl/workers/agent_loop/loops/mlgym_agent_loop.py",
    ]
    result["source_sha256"] = {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in paths}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    asyncio.run(main())
