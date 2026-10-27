"""
Tests for `PSRL_AgentLoopManager.get_dispatch_plan` after it learned unit-aware dispatch.

The most important property is that, with the env service off, the new code dispatches
exactly as the old round-robin did. `_reference_round_robin` below is a verbatim copy of the
pre-change algorithm, so the equivalence test is a real differential test, not a restatement.

The manager imports Ray, verl, transfer_queue and friends. They are stubbed only when the
real ones are missing, so in a full environment this runs against the genuine modules.

Run with `PYTHONPATH=. python3 tests/env_service/test_manager_dispatch.py` or under pytest.
"""

import asyncio
import importlib
import random
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _light_stubs import _importable, stub_if_missing, stub_package_keeping_children  # noqa: E402
from psrl.workers.env_service import NodeCapacity  # noqa: E402
from psrl.workers.env_service.placement import EnvAwareDispatcher  # noqa: E402


class _LooseModule(types.ModuleType):
    """A stand-in module whose every attribute is a MagicMock, for names that are only imported."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


def _loose(name):
    if _importable(name):
        return sys.modules[name]
    module = _LooseModule(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


class _FakeTu:
    """`verl.utils.tensordict_utils.get` for a dict-backed fake batch."""

    @staticmethod
    def get(data, key):
        if key not in data.columns:
            raise KeyError(key)
        return data.columns[key]


class FakeBatch:
    """Column store whose `batch[[rows]]` returns a marker, enough for dispatch plans."""

    def __init__(self, **columns):
        self.columns = columns

    def __getitem__(self, rows):
        return ("rows", tuple(rows))


def _load_manager():
    # Real psrl packages whose `__init__` may not import on a bare box keep their real
    # submodules, so `psrl.workers.agent_loop.manager` itself loads from source.
    repo = Path(__file__).resolve().parents[2]
    for package in (
        "psrl",
        "psrl.utils",
        "psrl.workers",
        "psrl.workers.agent_loop",
        "psrl.workers.agent_loop.loops",
        "psrl.workers.gen",
        "psrl.workers.ps",
    ):
        stub_package_keeping_children(package, repo.joinpath(*package.split(".")))
    # Third-party modules and psrl leaves that the manager only imports names from.
    for name in (
        "ray",
        "transfer_queue",
        "omegaconf",
        "tensordict",
        "verl",
        "verl.utils",
        "verl.utils.config",
        "verl.workers",
        "verl.workers.config",
        "psrl.utils.dataset",
        "psrl.utils.logger",
        "psrl.utils.ray",
        "psrl.utils.transferqueue_utils",
        "psrl.workers.agent_loop.loops.utils",
        "psrl.workers.gen.utils",
        "psrl.workers.ps.request_status_tracker",
        "psrl.workers.ps.staleness_controller",
    ):
        _loose(name)
    stub_if_missing("verl.utils.tensordict_utils", get=_FakeTu.get)
    stub_package_keeping_children("psrl.utils.common", repo / "psrl" / "utils" / "common")
    stub_if_missing("psrl.utils.common.http_utils")
    module = importlib.import_module("psrl.workers.agent_loop.manager")
    return module


manager_module = _load_manager()
Manager = manager_module.PSRL_AgentLoopManager


def _reference_round_robin(prompt_ids, dispatch_idx, num_workers):
    """Verbatim copy of the pre-change `get_dispatch_plan` grouping logic (round robin)."""
    keys_by_worker: dict[int, list[int]] = {}
    prompt_to_worker: dict = {}
    for i, prompt_id in enumerate(prompt_ids):
        if prompt_id in prompt_to_worker:
            worker_index = prompt_to_worker[prompt_id]
        else:
            worker_index = (dispatch_idx + len(prompt_to_worker)) % num_workers
            prompt_to_worker[prompt_id] = worker_index
        keys_by_worker.setdefault(worker_index, []).append(i)
    new_idx = (dispatch_idx + len(prompt_to_worker)) % num_workers
    return {w: ("rows", tuple(k)) for w, k in keys_by_worker.items()}, new_idx


def _manager(num_workers=3, rollout_n=2, dispatch_idx=0, dispatcher=None):
    m = object.__new__(Manager)
    m.agent_loop_workers = list(range(num_workers))
    m.rollout_n = rollout_n
    m.val_rollout_n = rollout_n
    m._dispatch_idx = dispatch_idx
    if dispatcher is not None:
        m._env_dispatcher = dispatcher
    return m


def _plan(m, batch, is_validate=False):
    with patch.object(manager_module, "tu", _FakeTu):
        return Manager.get_dispatch_plan(m, batch, is_validate=is_validate)


# --------------------------------------------------------------------------- equivalence with the old algorithm


def test_without_the_env_service_dispatch_is_identical_to_the_old_round_robin():
    rng = random.Random(7)
    for _ in range(200):
        num_workers = rng.randint(1, 6)
        rollout_n = rng.choice([1, 2, 4, 8])
        groups = rng.randint(1, 9)
        parent_ids = [g for g in range(groups) for _ in range(rollout_n)]
        uids = list(range(len(parent_ids)))
        start = rng.randrange(num_workers)

        m = _manager(num_workers, rollout_n, start)
        plan = _plan(m, FakeBatch(parent_id=parent_ids, uid=uids))
        expected_plan, expected_idx = _reference_round_robin(parent_ids if rollout_n > 1 else uids, start, num_workers)

        assert plan == expected_plan
        assert m._dispatch_idx == expected_idx


def test_siblings_of_a_group_stay_on_one_worker():
    m = _manager(num_workers=3, rollout_n=4)
    plan = _plan(m, FakeBatch(parent_id=[10] * 4 + [11] * 4, uid=list(range(8))))
    rows_by_worker = {w: rows[1] for w, rows in plan.items()}
    assert sorted(rows_by_worker.values()) == [(0, 1, 2, 3), (4, 5, 6, 7)]


def test_the_cursor_advances_by_the_number_of_groups_not_children():
    m = _manager(num_workers=5, rollout_n=3, dispatch_idx=1)
    _plan(m, FakeBatch(parent_id=[0] * 3 + [1] * 3, uid=list(range(6))))
    assert m._dispatch_idx == (1 + 2) % 5


def test_validation_uses_the_validation_rollout_n():
    m = _manager(num_workers=2, rollout_n=8)
    m.val_rollout_n = 1  # `rollout_n == 1` groups by uid, so every row is its own group
    plan = _plan(m, FakeBatch(parent_id=[0, 0], uid=[100, 101]), is_validate=True)
    assert sorted(len(rows[1]) for rows in plan.values()) == [1, 1]


# --------------------------------------------------------------------------- unit-aware dispatch


def _dispatcher(nodes, capacity):
    d = EnvAwareDispatcher(nodes)
    d.update(capacity)
    return d


def test_gpu_groups_go_to_workers_on_nodes_with_gpu_capacity():
    d = _dispatcher(
        ["cpu-node", "gpu-node"],
        {
            "cpu-node": NodeCapacity(cpu_free=4, gpu_free=0, cpu_eligible=True, gpu_eligible=False),
            "gpu-node": NodeCapacity(cpu_free=4, gpu_free=3, cpu_eligible=True, gpu_eligible=True),
        },
    )
    m = _manager(num_workers=2, rollout_n=2, dispatcher=d)
    infos = [{"gpus": 1}] * 2 + [{"gpus": 1}] * 2
    plan = _plan(m, FakeBatch(parent_id=[0, 0, 1, 1], uid=[0, 1, 2, 3], extra_info=infos))
    assert list(plan) == [1], "both GPU groups must land on worker 1, the only one with a GPU unit"


def test_cpu_groups_ignore_the_gpu_capacity_column():
    d = _dispatcher(
        ["a", "b"],
        {
            "a": NodeCapacity(cpu_free=0, gpu_free=5, cpu_eligible=True, gpu_eligible=True),
            "b": NodeCapacity(cpu_free=5, gpu_free=0, cpu_eligible=True, gpu_eligible=False),
        },
    )
    m = _manager(num_workers=2, rollout_n=1, dispatcher=d)
    plan = _plan(m, FakeBatch(uid=[0], extra_info=[{"gpus": 0}]))
    assert list(plan) == [1]


def test_extra_info_may_arrive_as_a_json_string():
    d = _dispatcher(
        ["a", "b"],
        {
            "a": NodeCapacity(cpu_free=5, gpu_free=0, cpu_eligible=True, gpu_eligible=False),
            "b": NodeCapacity(cpu_free=5, gpu_free=5, cpu_eligible=True, gpu_eligible=True),
        },
    )
    m = _manager(num_workers=2, rollout_n=1, dispatcher=d)
    plan = _plan(m, FakeBatch(uid=[0], extra_info=['{"gpus": 1}']))
    assert list(plan) == [1]


def test_a_missing_extra_info_column_degrades_to_cpu_placement_instead_of_failing():
    d = _dispatcher(["a", "b"], {"a": NodeCapacity(5, 0, True, False), "b": NodeCapacity(0, 5, False, True)})
    m = _manager(num_workers=2, rollout_n=1, dispatcher=d)
    plan = _plan(m, FakeBatch(uid=[0]))  # no extra_info column at all
    assert list(plan) == [0]


def test_a_stale_snapshot_stops_dispatch_before_choosing_an_ineligible_worker():
    clock = [0.0]
    d = EnvAwareDispatcher(["a", "b"], clock=lambda: clock[0])
    d.update({"a": NodeCapacity(0, 0, True, False), "b": NodeCapacity(5, 5, True, True)})
    clock[0] = 999.0
    m = _manager(num_workers=2, rollout_n=1, dispatch_idx=0, dispatcher=d)
    with pytest.raises(RuntimeError, match="fresh"):
        _plan(m, FakeBatch(uid=[0], extra_info=[{"gpus": 1}]))


# --------------------------------------------------------------------------- capacity refresh


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


class _Router:
    def __init__(self, capacity=None, error=None):
        self.calls = 0
        self._capacity = capacity
        self._error = error
        self.node_capacity = _Remote(self._node_capacity)

    async def _node_capacity(self):
        self.calls += 1
        if self._error:
            raise self._error
        return self._capacity


def _refreshing_manager(router, clock=None):
    d = EnvAwareDispatcher(["a", "b"], clock=clock or (lambda: 0.0))
    m = _manager(num_workers=2, dispatcher=d)
    m.env_router_handle = router
    m._env_capacity_refresh_s = 2.0
    m._env_capacity_warned = False
    return m, d


def test_refresh_installs_the_routers_snapshot():
    router = _Router(capacity={"a": NodeCapacity(1, 1, True, True)})
    m, d = _refreshing_manager(router)
    asyncio.run(Manager._refresh_env_capacity(m))
    assert router.calls == 1 and d.is_fresh()


def test_refresh_is_throttled_while_the_snapshot_is_recent():
    router = _Router(capacity={"a": NodeCapacity(1, 1, True, True)})
    m, _ = _refreshing_manager(router)

    async def twice():
        await Manager._refresh_env_capacity(m)
        await Manager._refresh_env_capacity(m)

    asyncio.run(twice())
    assert router.calls == 1


def test_a_failing_router_never_raises_and_warns_only_once():
    router = _Router(error=ConnectionError("router down"))
    m, d = _refreshing_manager(router)

    async def thrice():
        for _ in range(3):
            await Manager._refresh_env_capacity(m)

    asyncio.run(thrice())
    assert router.calls == 3 and not d.is_fresh()
    assert m._env_capacity_warned is True


def test_refresh_is_a_no_op_without_the_env_service():
    m = _manager()
    asyncio.run(Manager._refresh_env_capacity(m))  # must not touch a missing router


def test_failed_placement_wakes_training_and_validation_waiters_before_any_handoff():
    async def run():
        m = _manager(num_workers=1, dispatcher=EnvAwareDispatcher(["dead"]))
        m._env_dispatch_attempts = 3
        m._env_dispatch_backoff_s = 0.001
        m._refresh_env_capacity = AsyncMock()
        m._refill_breaker_diagnosis = None
        train_waiter = asyncio.get_running_loop().create_future()
        val_waiter = asyncio.get_running_loop().create_future()
        m._train_buffer_waiters = {0: [train_waiter]}
        m._train_chunk_waiters = {}
        m._resolved_train_chunks = {}
        m._val_buffer_waiters = {0: [val_waiter]}
        m.ps_manager_handle = MagicMock()
        batch = FakeBatch(uid=[0], parent_id=[0], version_tag=[0], extra_info=[{}])
        with patch.object(manager_module, "tu", _FakeTu), pytest.raises(RuntimeError, match="fresh"):
            await Manager._inner_dispatch_data(m, batch)
        m.ps_manager_handle.update_request_status.remote.assert_not_called()
        assert m._refresh_env_capacity.await_count == 3, "Placement must stop at the configured retry limit."
        for future in (train_waiter, val_waiter):
            with pytest.raises(RuntimeError, match="dispatch exhausted"):
                await future

    asyncio.run(run())


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
