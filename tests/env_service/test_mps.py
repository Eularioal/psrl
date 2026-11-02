"""Check ownership and cancellation boundaries without a GPU."""

import asyncio
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from psrl.workers.env_service.gpu_execution import GpuExecutionBackend
from psrl.workers.env_service.mps import MpsClientController, MpsControlChannel, parse_clients


def test_control_serializes_one_daemon_but_allows_independent_daemons(monkeypatch):
    active, peak, calls = {}, {}, {}
    guard = threading.Lock()
    both_daemons = threading.Barrier(2)

    def transport(argv, *, input, **kwargs):
        container = argv[3]
        with guard:
            active[container] = active.get(container, 0) + 1
            calls[container] = calls.get(container, 0) + 1
            peak[container] = max(peak.get(container, 0), active[container])
            first = calls[container] == 1
        try:
            if first:
                both_daemons.wait(timeout=1)
            time.sleep(0.02)
            return SimpleNamespace(returncode=0, stdout=input, stderr="")
        finally:
            with guard:
                active[container] -= 1

    monkeypatch.setattr("psrl.workers.env_service.mps.subprocess.run", transport)
    a, b = MpsControlChannel("a"), MpsControlChannel("b")
    with ThreadPoolExecutor(max_workers=4) as executor:
        requests = [
            executor.submit(channel, command) for channel, command in ((a, "ps"), (a, "list"), (b, "ps"), (b, "list"))
        ]
        assert [request.result(timeout=2) for request in requests] == ["ps", "list", "ps", "list"]
    assert peak == {"a": 1, "b": 1}


def test_control_queue_timeout_does_not_send_or_replay_command(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    commands = []

    def transport(argv, *, input, **kwargs):
        commands.append(input)
        entered.set()
        assert release.wait(timeout=1)
        return SimpleNamespace(returncode=0, stdout="0", stderr="")

    monkeypatch.setattr("psrl.workers.env_service.mps.subprocess.run", transport)
    channel = MpsControlChannel("daemon", timeout_s=0.05)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(channel, "ps")
        try:
            assert entered.wait(timeout=1)
            with pytest.raises(TimeoutError, match="queue deadline.*terminate_client"):
                channel("terminate_client 12 17")
        finally:
            release.set()
        assert first.result(timeout=1) == "0"
    assert commands == ["ps\n"]
    assert channel("get_server_list") == "0"


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_control_preserves_errors_and_unlocks_after_failure(monkeypatch, failure):
    calls = []

    def transport(argv, *, input, timeout, **kwargs):
        calls.append(input)
        assert 0 < timeout <= 10
        if len(calls) == 1:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, timeout)
            return SimpleNamespace(returncode=7, stdout="unacknowledged", stderr="service unavailable")
        return SimpleNamespace(returncode=0, stdout="server", stderr="")

    monkeypatch.setattr("psrl.workers.env_service.mps.subprocess.run", transport)
    channel = MpsControlChannel("daemon")
    with pytest.raises(TimeoutError if failure == "timeout" else RuntimeError, match="terminate_client") as error:
        channel("terminate_client 12 17")
    if failure == "exit":
        assert "exit code 7" in str(error.value) and "service unavailable" in str(error.value)
    assert calls == ["terminate_client 12 17\n"]
    assert channel("get_server_list") == "server"


@pytest.mark.parametrize("row", ["0 1 20 GPU-a ? ?", "10 1", "0 1 20 GPU-a 99 python"])
def test_incomplete_identity_rejected(row):
    with pytest.raises(RuntimeError):
        parse_clients(row)


@pytest.mark.parametrize("change", ["epoch", "birth", "server", "connection", "response"])
def test_stale_binding_cannot_release(change):
    identities = {10: (100, 99), 20: (200, 88)}
    table = ["10 1 20 GPU-a 99 python"]
    commands = []

    def control(command):
        commands.append(command)
        return table[0] if command == "ps" else "7" if change == "response" else "0"

    controller = MpsClientController(control, identities.__getitem__)
    binding = controller.bind("job", 1, 10)
    if change == "birth":
        identities[10] = (101, 99)
    elif change == "server":
        identities[20] = (201, 88)
    elif change == "connection":
        table[0] = "10 2 20 GPU-a 99 python"
    with pytest.raises(RuntimeError):
        controller.terminate(binding, "job", 2 if change == "epoch" else 1)
    if change != "response":
        assert not any(command.startswith("terminate") for command in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "mps", "process"])
async def test_cancel_order_and_failed_cleanup_retains_capacity(failure):
    events = []

    class Policy:
        mode = "episode"

        async def acquire(self, *args):
            events.append("acquire")

        async def cancel(self, record):
            events.append("release")

    async def owner():
        return "container"

    async def run():
        await asyncio.sleep(10)

    async def mps():
        events.append("mps")
        if failure == "mps":
            raise RuntimeError("MPS failure.")

    async def process():
        events.append("process")
        if failure == "process":
            raise RuntimeError("Process failure.")

    with pytest.raises(RuntimeError if failure else TimeoutError):
        await GpuExecutionBackend(Policy()).execute(
            run, owner, timeout_sec=0.01, wait_timeout_sec=1, before_cancel=mps, cancel=process
        )
    expected = {
        None: ["acquire", "mps", "process", "release"],
        "mps": ["acquire", "mps"],
        "process": ["acquire", "mps", "process"],
    }
    assert events == expected[failure]
