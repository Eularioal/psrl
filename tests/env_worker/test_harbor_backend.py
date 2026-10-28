"""Exercise the shared backend's execution, cleanup and capability contracts."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_worker.harbor import EnvWorkerEnvironment
from psrl.workers.env_worker.process import ProcessResult, run_process
from psrl.workers.env_worker.worker import EnvWorker


def environment(tmp_path, **kwargs):
    root = tmp_path / "environment"
    root.mkdir(exist_ok=True)
    return EnvWorkerEnvironment(
        environment_dir=root,
        environment_name="contract",
        session_id="contract",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=kwargs.pop(
            "task_env_config", EnvironmentConfig(docker_image="local:test", cpus=1, memory_mb=64)
        ),
        network_policy=kwargs.pop("network_policy", NetworkPolicy(network_mode="no-network")),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_process_separates_streams_and_preserves_failure_status():
    observed = []

    async def callback(text, stream):
        observed.append((text, stream))

    result = await run_process(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"],
        5,
        output_callback=callback,
    )
    assert result == ProcessResult("out\n", "err\n", 7), "Harbor must receive distinct streams and the real exit code."
    assert {stream for _, stream in observed} == {"stdout", "stderr"}, "Callbacks must identify each stream."


@pytest.mark.asyncio
async def test_output_limit_is_shared_across_both_streams():
    with pytest.raises(RuntimeError, match="byte limit"):
        await run_process([sys.executable, "-c", "print('x' * 4096)"], 5, output_limit=128)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError(), asyncio.CancelledError()])
async def test_uncertain_stateless_command_destroys_the_owned_sandbox(monkeypatch, failure):
    worker = EnvWorker(0, 1, 0, 5)
    worker._container_names["s"] = "owned-container"
    cleanup = worker.destroy_sandbox = AsyncMock()
    monkeypatch.setattr("psrl.workers.env_worker.worker.run_process", AsyncMock(side_effect=failure))
    with pytest.raises(type(failure)):
        await worker.exec_process("s", "sleep 10", timeout_s=0.01)
    cleanup.assert_awaited_once_with("s")


@pytest.mark.asyncio
async def test_directory_upload_preserves_contents_semantics(tmp_path, monkeypatch):
    worker = EnvWorker(0, 1, 0, 5)
    worker._container_names["s"] = "owned-container"
    run = AsyncMock(return_value=ProcessResult("", "", 0))
    monkeypatch.setattr("psrl.workers.env_worker.worker.run_process", run)
    await worker.copy_path("s", str(tmp_path) + "/.", "/work", True)
    assert run.call_args.args[0][-2:] == [str(tmp_path.resolve()) + "/.", "owned-container:/work"], (
        "Uploading directory contents must not introduce a nested directory."
    )


@pytest.mark.parametrize("feature", ["gpu", "storage", "network", "compose", "binding"])
def test_unsupported_capabilities_fail_before_start(tmp_path, feature):
    kwargs = {}
    if feature == "gpu":
        kwargs["task_env_config"] = EnvironmentConfig(docker_image="local:test", gpus=1)
    if feature == "storage":
        kwargs["task_env_config"] = EnvironmentConfig(docker_image="local:test", storage_mb=64)
    if feature == "network":
        kwargs["network_policy"] = NetworkPolicy(network_mode="public")
    if feature == "compose":
        path = tmp_path / "environment"
        path.mkdir()
        (path / "docker-compose.yml").write_text("services: {}")
    if feature == "binding":
        kwargs["gpu_execution"] = {"mode": "command"}
    with pytest.raises(ValueError):
        environment(tmp_path, **kwargs)


@pytest.mark.asyncio
async def test_scoped_env_cwd_user_and_exit_code_cross_the_adapter(tmp_path):
    env = environment(tmp_path)
    env._started = True
    execute = AsyncMock(return_value=ProcessResult("out", "err", 9))
    env._worker = SimpleNamespace(exec_process=execute)
    with env.scoped_exec_env({"TOKEN": "scoped"}):
        result = await env.exec("false", cwd="/work", env={"A": "a b"}, user=1000, timeout_sec=2)
    assert result.return_code == 9 and result.stderr == "err", "The adapter must preserve command failure."
    assert execute.call_args.kwargs["env"] == {"TOKEN": "scoped", "A": "a b"}, "Scoped environment must propagate."
    assert execute.call_args.kwargs["cwd"] == "/work" and execute.call_args.kwargs["user"] == 1000, (
        "Execution identity and directory must reach the backend."
    )


@pytest.mark.asyncio
async def test_partial_start_failure_is_cleaned_and_cannot_restart(tmp_path):
    env = environment(tmp_path)
    cleanup = AsyncMock()
    env._worker = SimpleNamespace(
        create_sandbox=AsyncMock(side_effect=RuntimeError("start failed")), destroy_sandbox=cleanup
    )
    with pytest.raises(RuntimeError, match="start failed"):
        await env.start()
    cleanup.assert_awaited_once_with(env._sandbox_id)
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await env.start()
    await env.stop()
