"""Instrument the existing bounded science probe without changing admission or GPU work."""

import argparse
import asyncio
import contextvars
import functools
import inspect
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import science_router_control_ab as driver

events = deque(maxlen=30000)
scope = contextvars.ContextVar("tail_scope", default="background")


def start(label, **details):
    return {
        "label": label,
        "scope": scope.get(),
        "started_at": time.time(),
        "started_mono": time.monotonic(),
        **details,
    }


def finish(event):
    event["duration_s"] = time.monotonic() - event["started_mono"]
    events.append(event)


def instrument():
    from harbor.environments.docker.docker import DockerEnvironment

    for name in ("exec", "_compose_exec", "_run_docker_compose_command"):
        original = getattr(DockerEnvironment, name)

        def wrap(fn, label):
            @functools.wraps(fn)
            async def wrapped(self, *args, **kwargs):
                token = scope.set(str(self.session_id))
                event = start(label, arguments=str(args[:1])[:400])
                try:
                    return await fn(self, *args, **kwargs)
                finally:
                    finish(event)
                    scope.reset(token)

            return wrapped

        setattr(DockerEnvironment, name, wrap(original, name))

    original_upload = DockerEnvironment.upload_file

    async def upload(self, source_path, target_path, *args, **kwargs):
        if Path(source_path).name == "cuda_pressure.py":
            source_path = Path(__file__).with_name("cuda_timing_probe.py")
        return await original_upload(self, source_path, target_path, *args, **kwargs)

    DockerEnvironment.upload_file = upload
    original_spawn = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        event = start("spawn", command=[str(arg) for arg in args])
        try:
            process = await original_spawn(*args, **kwargs)
            event["pid"] = process.pid
            return process
        finally:
            finish(event)

    asyncio.create_subprocess_exec = spawn
    original_communicate = asyncio.subprocess.Process.communicate

    async def communicate(self, *args, **kwargs):
        event = start("communicate", pid=self.pid)
        try:
            return await original_communicate(self, *args, **kwargs)
        finally:
            event["exit_code"] = self.returncode
            finish(event)

    asyncio.subprocess.Process.communicate = communicate
    return {"harbor_docker": inspect.getfile(DockerEnvironment), "driver": inspect.getfile(driver)}


async def run(args):
    sources = instrument()
    stop = asyncio.Event()
    last_tick = [time.monotonic()]
    stack_samples = deque(maxlen=1000)
    thread_stop = threading.Event()
    main_thread = threading.get_ident()

    def sample_stacks():
        while not thread_stop.wait(0.1):
            lag = time.monotonic() - last_tick[0]
            if lag < 0.3:
                continue
            frame = sys._current_frames().get(main_thread)
            frames = []
            while frame is not None:
                frames.append(
                    {"file": frame.f_code.co_filename, "line": frame.f_lineno, "function": frame.f_code.co_name}
                )
                frame = frame.f_back
            stack_samples.append({"time": time.time(), "mono": time.monotonic(), "lag_s": lag, "frames": frames})

    sampler = None
    if args.stack_sampling:
        sampler = threading.Thread(target=sample_stacks, name="tail-stack-sampler", daemon=True)
        sampler.start()

    async def heartbeat():
        while not stop.is_set():
            last_tick[0] = time.monotonic()
            event = start("loop_tick")
            await asyncio.sleep(0.05)
            delay = time.monotonic() - event["started_mono"] - 0.05
            if delay > 0.05:
                event["lag_s"] = delay
                finish(event)

    monitor = asyncio.create_task(heartbeat())
    try:
        await driver.run(args)
    finally:
        stop.set()
        await monitor
        thread_stop.set()
        if sampler:
            sampler.join(timeout=2)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "execution-trace.json").write_text(
            json.dumps(
                {"pid": os.getpid(), "sources": sources, "events": list(events), "stack_samples": list(stack_samples)},
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    parser.add_argument("--mode", choices=["off", "on"], required=True)
    parser.add_argument("--gib", type=int, default=1)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--wave-size", type=int, default=4)
    parser.add_argument("--budget", type=int, default=240)
    parser.add_argument("--prewarm", choices=["lazy", "serial", "overlap"], default="overlap")
    parser.add_argument("--preparation-seconds", type=float, default=8)
    parser.add_argument("--ray-for-off", action="store_true", default=True)
    parser.add_argument("--stack-sampling", action="store_true")
    asyncio.run(run(parser.parse_args()))
