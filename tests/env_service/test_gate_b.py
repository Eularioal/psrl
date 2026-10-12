"""
Tests for Gate B in `session_router.py` after it learned multi-device and shared bindings.

Gate B holds a model reply until the bound GPU looks idle. An env unit that owns several
cards binds them as one comma-joined string, and a card shared by several containers is
judged on utilization alone, because a neighbour's resident weights hold device memory
even while it is idle.

Run with `PYTHONPATH=. python3 tests/env_service/test_gate_b.py` or under pytest.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _light_stubs import load_session_router  # noqa: E402

router = load_session_router()


class _Telemetry:
    """Scripted `nvidia-smi` readings: one `{index: (util, mem)}` dict per poll, last one repeats."""

    def __init__(self, *readings):
        self._readings = list(readings)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        index = min(self.calls - 1, len(self._readings) - 1)
        return self._readings[index]


def _run_wait(device, readings, shared=False, max_wait=0.5):
    """Run `_wait_for_gpu_idle` against scripted telemetry. Returns (seconds_waited_polls, telemetry)."""
    telemetry = _Telemetry(*readings)
    saved = (router._read_gpu_utilization, router._GATE_B_POLL_INTERVAL_SEC, router._GATE_B_MAX_WAIT_SEC)
    router._read_gpu_utilization = telemetry
    router._GATE_B_POLL_INTERVAL_SEC = 0.01
    router._GATE_B_MAX_WAIT_SEC = max_wait
    try:
        asyncio.run(router._wait_for_gpu_idle(device, "sid", shared=shared))
    finally:
        router._read_gpu_utilization, router._GATE_B_POLL_INTERVAL_SEC, router._GATE_B_MAX_WAIT_SEC = saved
    return telemetry


def test_parse_gpu_indices_handles_one_many_and_junk():
    assert router._parse_gpu_indices("/dev/nvidia0") == [0]
    assert router._parse_gpu_indices("/dev/nvidia2,/dev/nvidia3") == [2, 3]
    assert router._parse_gpu_indices("/dev/nvidia2, /dev/nvidia3") == [2, 3]  # spaces tolerated
    assert router._parse_gpu_indices("/dev/nvidiactl,/dev/nvidia1") == [1]  # control devices dropped
    assert router._parse_gpu_indices("") == []
    assert router._parse_gpu_indices("garbage") == []


def test_an_idle_device_releases_the_reply_immediately():
    telemetry = _run_wait("/dev/nvidia0", [{0: (0.0, 0.0)}])
    assert telemetry.calls == 1


def test_a_busy_device_holds_the_reply_until_it_idles():
    busy, idle = {0: (97.0, 18000.0)}, {0: (0.0, 100.0)}
    telemetry = _run_wait("/dev/nvidia0", [busy, busy, idle])
    assert telemetry.calls == 3


def test_every_device_of_a_multi_card_unit_must_be_idle():
    # Card 0 is idle but card 1 is still computing, so the reply must keep waiting.
    one_busy = {0: (0.0, 0.0), 1: (90.0, 500.0)}
    all_idle = {0: (0.0, 0.0), 1: (0.0, 0.0)}
    telemetry = _run_wait("/dev/nvidia0,/dev/nvidia1", [one_busy, one_busy, all_idle])
    assert telemetry.calls == 3


def test_a_device_missing_from_telemetry_fails_open():
    # Only card 0 is reported, so the unit-wide answer is unknown and the reply is released.
    telemetry = _run_wait("/dev/nvidia0,/dev/nvidia1", [{0: (99.0, 9000.0)}])
    assert telemetry.calls == 1


def test_a_shared_card_is_judged_on_utilization_alone():
    # A neighbour keeps 18000 MB of weights resident but is not computing.
    resident_but_idle = {0: (1.0, 18000.0)}
    telemetry = _run_wait("/dev/nvidia0", [resident_but_idle], shared=True)
    assert telemetry.calls == 1


def test_an_unshared_card_still_waits_on_leftover_memory():
    # The legacy single-container rule: 18000 MB in use means something still holds the card.
    resident_but_idle = {0: (1.0, 18000.0)}
    idle = {0: (0.0, 0.0)}
    telemetry = _run_wait("/dev/nvidia0", [resident_but_idle, resident_but_idle, idle], shared=False)
    assert telemetry.calls == 3


def test_a_shared_card_still_waits_for_a_neighbours_compute():
    # Shared does not mean "never wait": two solvers must not run in the graded window together.
    computing = {0: (95.0, 18000.0)}
    done = {0: (2.0, 18000.0)}
    telemetry = _run_wait("/dev/nvidia0", [computing, computing, done], shared=True)
    assert telemetry.calls == 3


def test_the_wait_is_bounded_and_then_releases_anyway():
    # Gate B runs inside the agent's in-flight request, so it must fail open on timeout.
    telemetry = _run_wait("/dev/nvidia0", [{0: (99.0, 9000.0)}], max_wait=0.05)
    assert telemetry.calls >= 2  # it polled for a while, then gave up and returned


def test_an_unparseable_binding_is_a_no_op():
    telemetry = _run_wait("not-a-device", [{0: (99.0, 9000.0)}])
    assert telemetry.calls == 0


def test_session_state_defaults_to_unbound_and_unshared():
    state = router.SessionState()
    assert state.gpu_device is None and state.gpu_shared is False


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
