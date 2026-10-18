import pytest
from psrl.workers.agent_loop.loops.mlgym_agent_loop import classify_exit_status
from psrl.workers.agent_loop.loops.utils import TerminateReason


@pytest.mark.cpu_test
def test_submitted_maps_to_finished():
    assert classify_exit_status("submitted") is TerminateReason.FINISHED


@pytest.mark.cpu_test
def test_context_overflow_maps_to_max_response_length():
    assert classify_exit_status("exit_context") is TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED


@pytest.mark.cpu_test
def test_max_steps_maps_to_max_turns():
    assert classify_exit_status("max_steps") is TerminateReason.MAX_TURNS_EXCEEDED


@pytest.mark.cpu_test
def test_error_statuses_map_to_rollout_error():
    for status in ("exit_error", "exit_api", "exit_format"):
        assert classify_exit_status(status) is TerminateReason.ROLLOUT_ERROR, (
            f"Unexpected classification for status={status!r}. Expected an infrastructure error."
        )


@pytest.mark.cpu_test
def test_unknown_status_maps_to_unknown():
    assert classify_exit_status("something_new") is TerminateReason.UNKNOWN


@pytest.mark.cpu_test
def test_agent_loop_is_registered_under_mlgym_agent():
    from psrl.workers.agent_loop.loops.utils import AGENT_LOOP_REGISTRY

    assert "mlgym_agent" in AGENT_LOOP_REGISTRY, "The loop must be registered for config lookup."
    assert "MLGymAgentLoop" in AGENT_LOOP_REGISTRY["mlgym_agent"]["_target_"]


@pytest.mark.cpu_test
async def test_queue_timeout_never_creates_a_session(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from examples.airs_bench.config import AirsBenchRuntimeConfig
    from psrl.workers.agent_loop.loops import mlgym_agent_loop as module

    loop = object.__new__(module.MLGymAgentLoop)
    loop.config = None
    loop.reward_manager = loop.tokenizer = loop.processor = loop.dataset_cls = None
    loop.runtime_config = AirsBenchRuntimeConfig(episode_timeout_s=0.05)
    loop._coordinator = None
    observation = {
        "dataset_data_path": "unused",
        "task_config_path": "unused",
        "airs_task_id": "tiny",
        "episode_id": "one",
    }
    env = SimpleNamespace(reset=AsyncMock(return_value=(observation, {})), close=AsyncMock())
    monkeypatch.setattr(module.Environment, "get_environment", lambda *args, **kwargs: env)
    monkeypatch.setattr(module, "build_sandbox_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "build_verifier_spec", lambda *args: (None, None))
    loop.session_scope = lambda request: pytest.fail("A queued episode created a session.")

    async def wait(*args):
        await asyncio.sleep(1)

    monkeypatch.setattr(module, "acquire_sandbox", wait)
    result = await loop.run({"env_class": "mlgym_env"})
    assert result == (None, TerminateReason.TRAJECTORY_TIMEOUT)
    env.close.assert_awaited_once()


@pytest.mark.cpu_test
@pytest.mark.parametrize("exit_status", ["exit_error", "unexpected_status", ""])
@pytest.mark.parametrize("cleanup_failure", [False, True])
async def test_infrastructure_failure_discards_data_and_cleans_up(monkeypatch, exit_status, cleanup_failure):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from examples.airs_bench.config import AirsBenchRuntimeConfig
    from psrl.workers.agent_loop.loops import mlgym_agent_loop as module

    events = []
    loop = object.__new__(module.MLGymAgentLoop)
    loop.config = None
    loop.reward_manager = loop.tokenizer = loop.processor = loop.dataset_cls = None
    loop.runtime_config = AirsBenchRuntimeConfig(episode_timeout_s=1)
    loop._coordinator = None
    observation = {
        "dataset_data_path": "unused",
        "task_config_path": "unused",
        "airs_task_id": "tiny",
        "episode_id": "one",
    }
    env = SimpleNamespace(reset=AsyncMock(return_value=(observation, {})), close=AsyncMock())
    handle = SimpleNamespace(destroy=AsyncMock())
    if cleanup_failure:
        handle.destroy.side_effect = RuntimeError("Deletion failed.")
    monkeypatch.setattr(module.Environment, "get_environment", lambda *args, **kwargs: env)
    monkeypatch.setattr(module, "build_sandbox_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "build_verifier_spec", lambda *args: (None, None))

    async def acquire(*args):
        events.append("sandbox")
        return handle

    @asynccontextmanager
    async def session(request):
        events.append("session")
        yield "one"
        events.append("session_closed")

    monkeypatch.setattr(module, "acquire_sandbox", acquire)
    loop.session_scope = session
    loop._run_episode_logic = AsyncMock(return_value={"exit_status": exit_status})
    loop.get_training_data = AsyncMock()
    if cleanup_failure:
        with pytest.raises(RuntimeError, match="Deletion failed"):
            await loop.run({"env_class": "mlgym_env"})
    else:
        result = await loop.run({"env_class": "mlgym_env"})
        assert result == (None, TerminateReason.ROLLOUT_ERROR)
    assert events == ["sandbox", "session", "session_closed"]
    loop.get_training_data.assert_not_awaited()
    handle.destroy.assert_awaited_once()
    env.close.assert_awaited_once()
