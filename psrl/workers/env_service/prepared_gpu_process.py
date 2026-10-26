"""Prepare a cooperative Python program before granting its GPU phase permission."""

import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from gpu_job_process import cancel, identity


def alive(record):
    pid = record["pid"]
    if identity(pid) != record["birth"]:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def stop(directory):
    cancel(directory)
    try:
        record = json.loads((directory / "process.json").read_text())
    except FileNotFoundError:
        return
    deadline = time.monotonic() + 5
    while alive(record):
        if time.monotonic() >= deadline:
            raise TimeoutError("Prepared process did not exit after cancellation.")
        time.sleep(0.05)


def worker(directory, module_path, payload, deadline):
    code = 1
    try:
        spec = importlib.util.spec_from_file_location("gpu_program", module_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = program
        spec.loader.exec_module(program)
        execute = program.prepare(json.loads(payload))
        if not callable(execute):
            raise TypeError("Program prepare must return a callable GPU phase.")
        write_json(directory / "ready.json", {"ready_at": time.time()})
        while not (directory / "activate").exists():
            if time.monotonic() >= deadline or (directory / "cancel").exists():
                raise TimeoutError("Prepared program expired before GPU activation.")
            time.sleep(0.05)
        execute()
        code = 0
    except BaseException:
        traceback.print_exc()
    finally:
        write_json(directory / "exit.json", {"return_code": code, "finished_at": time.time()})
    return code


def prepare(directory, module_path, payload, prepare_timeout, lifetime):
    if (directory / "cancel").exists():
        return 125
    with (directory / "stdout").open("wb") as stdout, (directory / "stderr").open("wb") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "worker",
                str(directory),
                module_path,
                payload,
                str(time.monotonic() + lifetime),
            ],
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    write_json(directory / "process.json", {"pid": process.pid, "birth": identity(process.pid)})
    deadline = time.monotonic() + prepare_timeout
    while not (directory / "ready.json").exists():
        if process.poll() is not None:
            sys.stderr.write((directory / "stderr").read_text())
            return process.returncode or 1
        if time.monotonic() >= deadline or (directory / "cancel").exists():
            stop(directory)
            process.wait(timeout=5)
            raise TimeoutError("CPU preparation did not finish within its budget.")
        time.sleep(0.05)
    try:
        return process.wait(timeout=lifetime)
    except BaseException:
        stop(directory)
        process.wait(timeout=5)
        raise


def wait_ready(directory, timeout):
    deadline = time.monotonic() + timeout
    while not (directory / "ready.json").exists():
        if (directory / "exit.json").exists():
            sys.stderr.write((directory / "stderr").read_text())
            return json.loads((directory / "exit.json").read_text())["return_code"] or 1
        if time.monotonic() >= deadline or (directory / "cancel").exists():
            raise TimeoutError("CPU preparation did not become ready.")
        time.sleep(0.05)
    return 0


def activate(directory, timeout):
    record = json.loads((directory / "process.json").read_text())
    if not alive(record) or (directory / "cancel").exists():
        raise RuntimeError("Prepared process is no longer available.")
    (directory / "activate").touch()
    deadline = time.monotonic() + timeout
    while alive(record):
        if time.monotonic() >= deadline:
            stop(directory)
            raise TimeoutError("GPU phase exceeded its execution budget.")
        time.sleep(0.05)
    sys.stdout.write((directory / "stdout").read_text())
    sys.stderr.write((directory / "stderr").read_text())
    return json.loads((directory / "exit.json").read_text())["return_code"]


if __name__ == "__main__":
    action, path = sys.argv[1:3]
    directory = Path(path)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if action == "worker":
        sys.exit(worker(directory, sys.argv[3], sys.argv[4], float(sys.argv[5])))
    elif action == "prepare":
        sys.exit(prepare(directory, sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6])))
    elif action == "activate":
        sys.exit(activate(directory, float(sys.argv[3])))
    elif action == "wait-ready":
        sys.exit(wait_ready(directory, float(sys.argv[3])))
    elif action == "cancel":
        stop(directory)
    else:
        raise ValueError("Unknown prepared GPU operation.")
