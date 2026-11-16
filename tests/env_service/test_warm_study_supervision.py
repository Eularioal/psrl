"""Check owned backend termination and preservation of the original failure."""

import functools
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs/env-worker-deep-dive/env-service-verification/scripts/warm_router_study/matrix_suite.py"
)


@pytest.fixture
def supervisor(monkeypatch):
    matrix = types.ModuleType("matrix")
    matrix.plan_cells = lambda: []
    monkeypatch.setitem(sys.modules, "matrix", matrix)
    spec = importlib.util.spec_from_file_location("test_matrix_supervisor", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_backend(supervisor, monkeypatch, failure, *, cleanup_failure=None):
    state = dict(waits=0, cleanups=0, signals=[])

    class Process:
        pid = 12345
        returncode = None

        def wait(self, timeout):
            state["waits"] += 1
            if state["waits"] == 1:
                raise failure
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    def signal_group(pid, sig):
        assert pid == Process.pid
        state["signals"].append(sig)
        if sig == 0:
            raise ProcessLookupError

    def cleanup(root, item):
        state["cleanups"] += 1
        if cleanup_failure is not None:
            raise cleanup_failure
        return {"attempted": True, "physical_release_confirmed": False}

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(supervisor.os, "killpg", signal_group)
    monkeypatch.setattr(supervisor, "stop_owned", cleanup)
    return state


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("Wait failed."), OSError("Wait failed.")])
def test_wait_failure_joins_the_owned_child_once_and_preserves_original(supervisor, monkeypatch, tmp_path, failure):
    state = fake_backend(supervisor, monkeypatch, failure)
    item = {}
    with pytest.raises(type(failure)) as caught:
        supervisor.supervise_backend(["unused"], None, 1, tmp_path, item)
    assert caught.value is failure
    assert state["waits"] == 2 and state["cleanups"] == 1
    assert signal.SIGINT in state["signals"]
    assert item["watchdog"]["trigger"] == type(failure).__name__
    assert not item["watchdog"]["physical_release_confirmed"]


def test_timeout_uses_the_same_recovery_once_even_if_child_then_exits_zero(supervisor, monkeypatch, tmp_path):
    state = fake_backend(supervisor, monkeypatch, subprocess.TimeoutExpired(["unused"], 1))
    item = {}
    assert supervisor.supervise_backend(["unused"], None, 1, tmp_path, item) == 0
    assert state["cleanups"] == 1
    assert item["watchdog"]["trigger"] == "TimeoutExpired"


def test_normal_child_completion_does_not_signal_or_force_owner_cleanup(supervisor, monkeypatch, tmp_path):
    process = SimpleNamespace(wait=lambda timeout: 0, returncode=0)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: process)

    def unexpected_cleanup(*args):
        pytest.fail("A normally completed backend must use its recorded physical cleanup result.")

    monkeypatch.setattr(supervisor, "stop_backend", unexpected_cleanup)
    item = {}
    assert supervisor.supervise_backend(["unused"], None, 1, tmp_path, item) == 0
    assert "watchdog" not in item


def test_nonzero_exit_recovers_owned_descendants_even_when_the_leader_has_exited(supervisor, monkeypatch, tmp_path):
    process = SimpleNamespace(wait=lambda timeout: -signal.SIGKILL, poll=lambda: -signal.SIGKILL)
    process.pid, process.returncode = 12345, -signal.SIGKILL
    signals, cleanups = [], []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(supervisor.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(supervisor, "stop_owned", lambda *args: cleanups.append(True))
    item = {}
    assert supervisor.supervise_backend(["unused"], None, 1, tmp_path, item) == -signal.SIGKILL
    assert item["watchdog"]["trigger"] == "BackendNonzeroExit"
    assert (process.pid, signal.SIGKILL) in signals
    assert cleanups == [True]


def test_owned_cleanup_failure_does_not_replace_the_original_interrupt(supervisor, monkeypatch, tmp_path):
    original = KeyboardInterrupt()
    state = fake_backend(supervisor, monkeypatch, original, cleanup_failure=OSError("Cleanup failed."))
    item = {}
    with pytest.raises(KeyboardInterrupt) as caught:
        supervisor.supervise_backend(["unused"], None, 1, tmp_path, item)
    assert caught.value is original and state["cleanups"] == 1
    assert item["watchdog"]["errors"][0]["phase"] == "owned_cleanup"


def test_unexpected_recovery_failure_does_not_replace_wait_failure(supervisor, monkeypatch, tmp_path):
    original = RuntimeError("Wait failed.")
    fake_backend(supervisor, monkeypatch, original)

    def fail_recovery(*args):
        raise OSError("Recovery failed.")

    monkeypatch.setattr(supervisor, "stop_backend", fail_recovery)
    item = {}
    with pytest.raises(RuntimeError) as caught:
        supervisor.supervise_backend(["unused"], None, 1, tmp_path, item)
    assert caught.value is original
    assert item["watchdog"]["errors"][0]["phase"] == "recovery"


def test_final_sampler_and_manifest_errors_do_not_replace_suite_failure(supervisor, monkeypatch, tmp_path):
    original = RuntimeError("Sampler startup failed.")
    sampler_module = types.ModuleType("gpu_sampler")

    class Sampler:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise original

        def stop(self):
            raise OSError("Sampler cleanup failed.")

    sampler_module.GpuSampler = Sampler
    monkeypatch.setitem(sys.modules, "gpu_sampler", sampler_module)
    root = tmp_path / "run"

    def freeze(path):
        (path / "current-worktree.patch").write_text("")
        return path / "entry.py", {}, {}, {}

    calls = []

    def save(path, manifest):
        calls.append(dict(manifest))
        if len(calls) > 1:
            raise OSError("Final manifest write failed.")

    monkeypatch.setattr(supervisor, "freeze", freeze)
    monkeypatch.setattr(supervisor, "save", save)
    monkeypatch.setattr(supervisor.subprocess, "check_output", lambda *args, **kwargs: "image-id")
    args = SimpleNamespace(root=root, budget_s=1, image="unused", gpus=[0, 1])
    with pytest.raises(RuntimeError) as caught:
        supervisor.run(args)
    assert caught.value is original
    assert calls[-1]["status"] == "incomplete"
    assert "Sampler cleanup failed" in calls[-1]["gpu_sampler_stop"]["error"]


def test_noncooperative_real_child_is_killed_within_the_recovery_bound(supervisor, monkeypatch, tmp_path):
    original_popen = subprocess.Popen
    children = []
    cleanups = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor, "stop_owned", lambda *args: cleanups.append(True))
    monkeypatch.setattr(supervisor, "stop_backend", functools.partial(supervisor.stop_backend, grace_s=0.1, kill_s=1))
    code = "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(30)"
    item = {}
    started = time.monotonic()
    try:
        code = supervisor.supervise_backend([sys.executable, "-c", code], subprocess.DEVNULL, 0.5, tmp_path, item)
        assert code == -signal.SIGKILL
        assert time.monotonic() - started < 3
        assert cleanups == [True]
    finally:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)


def test_real_parent_interrupt_reaps_its_separate_session_child(supervisor, tmp_path):
    ready = tmp_path / "child-ready.json"
    report = tmp_path / "report.json"
    child = (
        f"import json,os,pathlib,time; pathlib.Path({str(ready)!r})"
        ".write_text(json.dumps(dict(pid=os.getpid()))); time.sleep(30)"
    )
    code = f"""
import functools, importlib.util, json, pathlib, subprocess, sys, types
matrix = types.ModuleType('matrix')
matrix.plan_cells = lambda: []
sys.modules['matrix'] = matrix
spec = importlib.util.spec_from_file_location('supervisor', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.stop_backend = functools.partial(module.stop_backend, grace_s=0.1, kill_s=1)
cleanups = []
module.stop_owned = lambda *args: cleanups.append(True)
child = {child!r}
item = {{}}
try:
    module.supervise_backend([sys.executable, '-c', child], subprocess.DEVNULL, 20, pathlib.Path('.'), item)
except KeyboardInterrupt:
    pathlib.Path(sys.argv[3]).write_text(json.dumps(dict(item=item, cleanups=cleanups)))
"""
    process = subprocess.Popen([sys.executable, "-c", code, supervisor.__file__, str(ready), str(report)])
    child_pid = None
    try:
        deadline = time.monotonic() + 4
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.01)
        assert ready.exists(), "The owned child did not become ready."
        child_pid = json.loads(ready.read_text())["pid"]
        os.kill(process.pid, signal.SIGINT)
        process.wait(timeout=4)
        raw = json.loads(report.read_text())
        assert process.returncode == 0
        assert raw["cleanups"] == [True]
        assert raw["item"]["watchdog"]["trigger"] == "KeyboardInterrupt"
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_real_cli_sigterm_reaps_owned_child_restores_handler_and_preserves_exit_code(supervisor, tmp_path):
    ready = tmp_path / "child-ready.json"
    report = tmp_path / "report.json"
    child = (
        f"import json,os,pathlib,time; pathlib.Path({str(ready)!r})"
        ".write_text(json.dumps(dict(pid=os.getpid()))); time.sleep(30)"
    )
    code = f"""
import functools, importlib.util, json, pathlib, signal, subprocess, sys, types
matrix = types.ModuleType('matrix')
matrix.plan_cells = lambda: []
sys.modules['matrix'] = matrix
spec = importlib.util.spec_from_file_location('supervisor', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.stop_backend = functools.partial(module.stop_backend, grace_s=0.1, kill_s=1)
cleanups = []
module.stop_owned = lambda *args: cleanups.append(True)
child = {child!r}
report = pathlib.Path(sys.argv[2])
item = {{}}
def run(args):
    return module.supervise_backend([sys.executable, '-c', child], subprocess.DEVNULL, 20, args.root, item)
module.run = run
sys.argv = ['matrix_suite.py', '--root', '.', '--assets', '.', '--node-ip', '127.0.0.1',
            '--image', 'unused', '--container-python', sys.executable, '--baseline-repo', '.']
previous_handler = signal.getsignal(signal.SIGTERM)
try:
    module.main()
except SystemExit as error:
    report.write_text(json.dumps(dict(item=item, cleanups=cleanups, exit_code=error.code,
                                    handler_restored=signal.getsignal(signal.SIGTERM) == previous_handler)))
    raise
"""
    process = subprocess.Popen([sys.executable, "-c", code, supervisor.__file__, str(report)])
    child_pid = None
    try:
        deadline = time.monotonic() + 4
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.01)
        assert ready.exists(), "The owned child did not become ready."
        child_pid = json.loads(ready.read_text())["pid"]
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=4)
        raw = json.loads(report.read_text())
        assert raw["exit_code"] == process.returncode == 128 + signal.SIGTERM
        assert raw["handler_restored"] is True
        assert raw["cleanups"] == [True]
        assert raw["item"]["watchdog"]["trigger"] == "SystemExit"
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def released_record():
    return dict(
        cleanup_errors=[],
        containers_remaining=[],
        cleanup_phases=dict(physical_release_confirmed=True, leases_released=True),
        gpu_after={"0": dict(pids=[], used_mb=0), "1": dict(pids=[], used_mb=0)},
        ledger_after_cleanup=[dict(leases_active=0)],
    )


@pytest.mark.parametrize("ledger", [None, [], [{}], [dict(leases_active=1)], [dict(leases_active=False)]])
def test_native_cleanup_requires_a_nonempty_explicit_zero_lease_ledger(supervisor, released_record, ledger):
    if ledger is None:
        released_record.pop("ledger_after_cleanup")
    else:
        released_record["ledger_after_cleanup"] = ledger
    assert not supervisor.cleanup_passed(released_record, "native")


def test_baseline_cleanup_uses_physical_proof_without_inventing_a_native_ledger(supervisor, released_record):
    released_record.pop("ledger_after_cleanup")
    released_record["cleanup_phases"].pop("leases_released")
    assert supervisor.cleanup_passed(released_record, "baseline")
    assert not supervisor.cleanup_passed(released_record, "native")


def test_native_cleanup_requires_the_release_phases_and_finite_physical_observations(supervisor, released_record):
    assert supervisor.cleanup_passed(released_record, "native")
    for phase in ("physical_release_confirmed", "leases_released"):
        changed = deepcopy(released_record)
        changed["cleanup_phases"].pop(phase)
        assert not supervisor.cleanup_passed(changed, "native")
    for used_mb in (None, float("nan"), float("inf"), -1, 128, False):
        changed = deepcopy(released_record)
        changed["gpu_after"]["0"]["used_mb"] = used_mb
        assert not supervisor.cleanup_passed(changed, "native")
