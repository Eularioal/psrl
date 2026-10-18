import multiprocessing
import os
import signal
import time
from types import SimpleNamespace

import psutil
import pytest
from psrl.workers.gen.rollout_gateway import _bind_to_parent_lifetime
from psrl.workers.gen.smg_adapter import build_rollout_router_args


def _serve(ready):
    _bind_to_parent_lifetime()
    ready.send(os.getpid())
    while True:
        time.sleep(1)


def _own_server(ready):
    child = multiprocessing.get_context("fork").Process(target=_serve, args=(ready,))
    child.start()
    child.join()


@pytest.mark.parametrize("exit_signal", [signal.SIGTERM, signal.SIGKILL])
def test_router_exits_when_owner_is_killed(exit_signal):
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    owner = context.Process(target=_own_server, args=(writer,))
    owner.start()
    server = None
    try:
        assert reader.poll(15), "Router did not initialize."
        server = psutil.Process(reader.recv())
        os.kill(owner.pid, exit_signal)
        owner.join(timeout=10)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not server.is_running() or server.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.05)
        else:
            pytest.fail("Router survived its owner.")
    finally:
        if owner.is_alive():
            owner.kill()
            owner.join(timeout=5)
        if server is not None and server.is_running() and server.status() != psutil.STATUS_ZOMBIE:
            server.kill()
        reader.close()
        writer.close()


def test_metrics_port_defaults_to_kernel_allocation_and_accepts_explicit_port():
    config = SimpleNamespace(psrl=SimpleNamespace(rollout_gateway=SimpleNamespace()))
    args = build_rollout_router_args(config, "127.0.0.1", 8101, "127.0.0.1:8100")
    assert args.prometheus_port == 0, "Gateway metrics must not reserve a shared fixed port."
    config.psrl.rollout_gateway.prometheus_port = 29001
    args = build_rollout_router_args(config, "127.0.0.1", 8101, "127.0.0.1:8100")
    assert args.prometheus_port == 29001, "Explicit metrics port was not forwarded."
