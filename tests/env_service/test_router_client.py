"""
Tests for `EnvRouterCore` driven through `EnvServiceClient`, with a local adapter standing
in for the Ray actor handle (`handle.method.remote(...)` becomes a plain coroutine call).

Run with `PYTHONPATH=. python3 tests/env_service/test_router_client.py` or under pytest.
"""

import asyncio

from psrl.workers.env_service import EnvDemand
from psrl.workers.env_service.client import (
    EnvAdmissionRejected,
    EnvAdmissionTimeout,
    EnvServiceClient,
)
from psrl.workers.env_service.router import EnvRouterCore

NODE = "10.0.0.1"
TRAIN = "10.0.0.9"

GPU_UNIT = {
    "unit_id": "g0",
    "node_ip": NODE,
    "cpu_quota": 16,
    "memory_mb": 65536,
    "gpu_indices": [0],
    "max_containers": 2,
    "gpu_memory_mb": 80000,
}
CPU_UNIT = {"unit_id": "c0", "node_ip": NODE, "cpu_quota": 16, "memory_mb": 65536, "max_containers": 4}


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)  # a coroutine, awaitable like a Ray ObjectRef


class LocalRouter:
    """Looks like a Ray actor handle over an in-process `EnvRouterCore`."""

    def __init__(self, core):
        self._core = core

    def __getattr__(self, name):
        return _Remote(getattr(self._core, name))


def _config(**overrides):
    cfg = {"units": [GPU_UNIT, CPU_UNIT], "node_policy": "any", "default_gpu_memory_mb": 24000, "slot_ttl_s": 1800}
    cfg.update(overrides)
    return cfg


async def _no_telemetry():
    return {}


def _client(core, **kwargs):
    kwargs.setdefault("telemetry_reader", _no_telemetry)
    kwargs.setdefault("poll_interval_s", 0.01)
    kwargs.setdefault("max_poll_interval_s", 0.02)
    return EnvServiceClient(LocalRouter(core), NODE, default_gpu_memory_mb=24000, **kwargs)


def _run(coro):
    return asyncio.run(coro)


def _stub_docker_utils(fake_force_remove):
    """
    Replace `psrl.utils.common.docker_utils` with a fake so no real `docker` is called.

    The real module sits under a package whose `__init__` imports aiohttp, which a bare
    CPU box may lack, so the parent package is stubbed as well. Returns a restore callable.
    """
    import sys
    import types

    names = ("psrl.utils.common", "psrl.utils.common.docker_utils")
    saved = {name: sys.modules.get(name) for name in names}
    parent = types.ModuleType(names[0])
    parent.__path__ = []  # marks it as a package so the submodule import resolves
    fake = types.ModuleType(names[1])
    fake.force_remove_containers_by_label = fake_force_remove
    fake.CLEANUP_EXECUTOR = None  # `run_in_executor(None, ...)` uses the default pool in the test
    sys.modules[names[0]] = parent
    sys.modules[names[1]] = fake

    def restore():
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return restore


GPU = EnvDemand(needs_gpu=True)


def test_acquire_grants_a_lease_and_release_frees_it():
    async def scenario():
        core = EnvRouterCore(_config())
        client = _client(core)
        handle = await client.acquire("s0", GPU)
        assert handle.lease.gpu_indices == (0,) and handle.lease.unit_id == "g0"
        assert (await core.snapshot())[0].containers_running == 1
        await handle.release()
        assert (await core.snapshot())[0].containers_running == 0
        await handle.release()  # idempotent

    _run(scenario())


def test_two_episodes_share_one_card_and_a_third_waits_until_one_leaves():
    async def scenario():
        core = EnvRouterCore(_config())
        client = _client(core)
        a = await client.acquire("a", GPU)
        b = await client.acquire("b", GPU)
        assert a.lease.gpu_indices == b.lease.gpu_indices == (0,)  # same card, two containers

        third = asyncio.ensure_future(client.acquire("c", GPU))
        await asyncio.sleep(0.1)
        assert not third.done(), "max_containers=2, so the third episode must be waiting"
        snap = (await core.snapshot())[0]
        assert snap.sessions_waiting == 1

        await a.release()
        c = await asyncio.wait_for(third, timeout=2.0)
        assert c.lease.session_id == "c"
        await b.release()
        await c.release()

    _run(scenario())


def test_an_impossible_demand_raises_instead_of_waiting_forever():
    async def scenario():
        core = EnvRouterCore(_config())
        client = _client(core)
        try:
            await client.acquire("s0", EnvDemand(needs_gpu=True, gpu_memory_mb=999999))
        except EnvAdmissionRejected as exc:
            assert "exceeds" in str(exc)
        else:
            raise AssertionError("an oversized demand should be rejected")

    _run(scenario())


def test_max_wait_gives_up_and_stops_counting_as_a_waiter():
    async def scenario():
        core = EnvRouterCore(_config())
        client = _client(core)
        held = [await client.acquire(f"h{i}", GPU) for i in range(2)]
        try:
            await client.acquire("late", GPU, max_wait_s=0.1)
        except EnvAdmissionTimeout:
            pass
        else:
            raise AssertionError("the wait should have timed out")
        assert (await core.snapshot())[0].sessions_waiting == 0
        for h in held:
            await h.release()

    _run(scenario())


def test_cancelling_a_waiting_acquire_removes_it_from_the_waiters():
    async def scenario():
        core = EnvRouterCore(_config())
        client = _client(core)
        held = [await client.acquire(f"h{i}", GPU) for i in range(2)]
        waiter = asyncio.ensure_future(client.acquire("w", GPU))
        await asyncio.sleep(0.1)
        assert (await core.snapshot())[0].sessions_waiting == 1
        waiter.cancel()
        try:
            await waiter
        except asyncio.CancelledError:
            pass
        assert (await core.snapshot())[0].sessions_waiting == 0
        for h in held:
            await h.release()

    _run(scenario())


def test_node_policy_is_enforced_through_the_router():
    async def scenario():
        cfg = _config(
            units=[
                {**GPU_UNIT, "unit_id": "g-train", "node_ip": TRAIN},
                {**CPU_UNIT, "unit_id": "c-train", "node_ip": TRAIN},
            ],
            node_policy="cpu_only",
        )
        core = EnvRouterCore(cfg, train_rollout_node_ips=[TRAIN])
        client = EnvServiceClient(LocalRouter(core), TRAIN, telemetry_reader=_no_telemetry, poll_interval_s=0.01)
        try:
            await client.acquire("gpu", GPU)
        except EnvAdmissionRejected:
            pass
        else:
            raise AssertionError("cpu_only must refuse a GPU episode on a train node")
        handle = await client.acquire("cpu", EnvDemand())
        assert handle.lease.unit_id == "c-train"
        await handle.release()

    _run(scenario())


def test_the_router_warns_about_a_gpu_unit_declared_on_a_train_node():
    core = EnvRouterCore(
        _config(units=[{**GPU_UNIT, "node_ip": TRAIN}], node_policy="cpu_only"),
        train_rollout_node_ips=[TRAIN],
    )
    warnings = _run(core.warnings())
    assert len(warnings) == 1 and "g0" in warnings[0]


def test_measured_device_memory_from_the_client_blocks_a_crowded_card():
    async def scenario():
        core = EnvRouterCore(_config())

        async def crowded():
            return {0: (50.0, 70000.0)}  # a neighbour holds 70000 of 80000 MB

        client = _client(core, telemetry_reader=crowded)
        try:
            await client.acquire("s0", GPU, max_wait_s=0.1)
        except EnvAdmissionTimeout:
            pass
        else:
            raise AssertionError("a card with 70000 MB in use cannot take 24000 MB more")
        # A CPU-only episode never reads the GPU, so it is unaffected.
        handle = await client.acquire("cpu", EnvDemand())
        await handle.release()

    _run(scenario())


def test_heartbeat_keeps_a_lease_alive_across_the_ttl():
    async def scenario():
        core = EnvRouterCore(_config(slot_ttl_s=0.3))
        client = _client(core, heartbeat_interval_s=0.05)
        handle = await client.acquire("s0", EnvDemand())
        await asyncio.sleep(0.8)  # well past the 0.3s TTL, but the heartbeat refreshed it
        assert core._ledger.reap() == []
        assert (await core.snapshot())[1].containers_running == 1
        await handle.release()

    _run(scenario())


def test_a_lease_that_stops_heartbeating_is_reclaimed_and_cleaned_up_on_its_node():
    async def scenario():
        core = EnvRouterCore(_config(slot_ttl_s=0.2))
        client = _client(core, heartbeat_interval_s=3600.0)  # effectively never beats
        handle = await client.acquire("dead", EnvDemand())
        await asyncio.sleep(0.35)
        reaped = core._ledger.reap()
        assert [r.lease_id for r in reaped] == [handle.lease.lease_id]
        assert (await core.snapshot())[1].containers_running == 0

        removed = []
        restore = _stub_docker_utils(lambda key, value: removed.append((key, value)))
        try:
            assert await client.reap_local() == 1
            assert await client.reap_local() == 0  # delivered exactly once
        finally:
            restore()
        assert removed == [("psrl.env_lease", handle.lease.lease_id)]
        handle._heartbeat_task.cancel()

    _run(scenario())


def test_from_config_returns_none_when_the_service_is_disabled():
    assert EnvServiceClient.from_config(None) is None
    assert EnvServiceClient.from_config({"enable": False}) is None
    assert EnvServiceClient.from_config({}) is None


def test_a_lost_release_does_not_raise():
    async def scenario():
        core = EnvRouterCore(_config())
        client = _client(core)
        handle = await client.acquire("s0", EnvDemand())

        class Broken:
            def remote(self, *a, **k):
                raise ConnectionError("router gone")

        client.router.release = Broken()  # type: ignore[attr-defined]
        await handle.release()  # must swallow the error: the TTL reaper is the backstop

    # LocalRouter.__getattr__ does not intercept an instance attribute, so assignment works.
    _run(scenario())


if __name__ == "__main__":
    import sys

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
