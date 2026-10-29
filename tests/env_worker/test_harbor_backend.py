"""Exercise the shared backend's execution, cleanup and capability contracts."""

import asyncio
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_worker.harbor import EnvWorkerEnvironment
from psrl.workers.env_worker.process import ProcessResult, run_process
from psrl.workers.env_worker.worker import EnvWorker

from tests.env_worker.test_airs_regressions import LocalBashWorker


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


@pytest.mark.asyncio
async def test_harbor_shell_preserves_state_across_turns(tmp_path):
    env = environment(tmp_path)
    env._worker = LocalBashWorker(0, 1, 0, 2)
    await env.start()
    try:
        result = await env.exec_shell(
            f"cd {shlex.quote(str(tmp_path))}\n"
            "export CONTRACT_VALUE=retained\n"
            "contract_function() { printf '%s' function-result; }\n"
            "printf file-result > artifact"
        )
        assert result.exit_code == 0, "The first turn must establish shell state."
        result = await env.exec_shell("pwd\nprintf '%s\\n' \"$CONTRACT_VALUE\"\ncontract_function\ncat artifact")
        assert str(tmp_path) in result.stdout, "The working directory must survive the turn boundary."
        assert "retained" in result.stdout, "Exported variables must survive the turn boundary."
        assert "function-result" in result.stdout, "Shell functions must survive the turn boundary."
        assert "file-result" in result.stdout, "Files must survive the turn boundary."
        failure = await env.exec_shell("echo error >&2\nfalse")
        assert failure.exit_code == 1 and "error" in failure.stdout, "Shell output must retain failure details."
        next_turn = await env.exec_shell("printf '%s' \"$CONTRACT_VALUE\"")
        assert next_turn.stdout.strip() == "retained", "A completed nonzero command must not destroy shell state."
    finally:
        await env.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "cancel", "exit"])
async def test_harbor_shell_failure_invalidates_all_access(tmp_path, failure):
    env = environment(tmp_path)
    env._worker = LocalBashWorker(0, 1, 0, 2)
    await env.start()
    try:
        if failure == "timeout":
            result = await env.exec_shell("echo begin\nsleep 5", timeout_sec=0.1)
            assert result.timed_out and result.exit_code is None, "Timeout must not become a successful command."
        elif failure == "exit":
            with pytest.raises(RuntimeError, match="exited"):
                await env.exec_shell("exit 7")
        else:
            task = asyncio.create_task(env.exec_shell("echo begin\nsleep 5"))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not env._worker._shells, "Uncertain commands must destroy the persistent shell."
        for call in (
            lambda: env.exec_shell("echo next"),
            lambda: env.exec("echo next"),
            lambda: env.upload_file(tmp_path / "x", "/x"),
            lambda: env.download_file("/x", tmp_path / "x"),
            lambda: env.download_dir("/x", tmp_path / "x"),
        ):
            with pytest.raises(RuntimeError, match="not running"):
                await call()
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            await env.start()
    finally:
        await env.stop()


@pytest.mark.asyncio
async def test_stateless_failure_invalidates_adapter_even_when_cleanup_fails(tmp_path):
    env = environment(tmp_path)
    env._started = True
    env._worker = SimpleNamespace(
        exec_process=AsyncMock(side_effect=RuntimeError("cleanup failed")),
        destroy_sandbox=AsyncMock(),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await env.exec("sleep 5")
    with pytest.raises(RuntimeError, match="not running"):
        await env.exec_shell("echo next")
    await env.stop()
    env._worker.destroy_sandbox.assert_awaited_once_with(env._sandbox_id)


@pytest.mark.asyncio
async def test_fresh_commands_share_files_but_not_persistent_shell_state(tmp_path, monkeypatch):
    env = environment(tmp_path)
    env._worker = LocalBashWorker(0, 1, 0, 2)
    await env.start()

    async def local_process(argv, timeout_s, output_callback=None):
        assert argv[:2] == ["docker", "exec"], "Only the Docker transport is replaced in this contract test."
        return await run_process(argv[-3:], timeout_s, output_callback=output_callback)

    monkeypatch.setattr("psrl.workers.env_worker.worker.run_process", local_process)
    monkeypatch.setattr(env._worker, "container_name", lambda sandbox_id: sandbox_id)
    artifact = shlex.quote(str(tmp_path / "shared"))
    try:
        await env.exec_shell(f"export CONTRACT_PRIVATE=shell\nprintf shared-file > {artifact}")
        result = await env.exec(f"printf '%s' \"${{CONTRACT_PRIVATE-unset}}\"\ncat {artifact}")
        assert result.stdout == "unsetshared-file", "Fresh commands must share files without inheriting shell state."
        await env.exec("export CONTRACT_PRIVATE=fresh")
        result = await env.exec_shell("printf '%s' \"$CONTRACT_PRIVATE\"")
        assert result.stdout.strip() == "shell", "Fresh commands must not overwrite persistent shell state."
    finally:
        await env.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("during_start", [True, False])
async def test_failed_cleanup_blocks_reuse_and_can_be_retried(tmp_path, during_start):
    env = environment(tmp_path)
    env._worker = SimpleNamespace(
        create_sandbox=AsyncMock(side_effect=RuntimeError("startup failed")),
        destroy_sandbox=AsyncMock(side_effect=[RuntimeError("cleanup failed"), None]),
    )
    env._started = not during_start
    with pytest.raises(RuntimeError, match="cleanup failed"):
        if during_start:
            await env.start()
        else:
            await env.stop()
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await env.start()
    with pytest.raises(RuntimeError, match="not running"):
        await env.exec("echo next")
    await env.stop()
    assert env._worker.destroy_sandbox.await_count == 2, "Cleanup must remain retryable after invalidation."


@pytest.mark.asyncio
async def test_start_does_not_publish_partial_environment(tmp_path):
    env = environment(tmp_path)
    entered, proceed = asyncio.Event(), asyncio.Event()
    env._worker = SimpleNamespace(create_sandbox=AsyncMock(), destroy_sandbox=AsyncMock())

    async def upload():
        entered.set()
        await proceed.wait()

    env._upload_environment_dir_after_start = upload
    task = asyncio.create_task(env.start())
    await asyncio.wait_for(entered.wait(), 2)
    try:
        with pytest.raises(RuntimeError, match="not running"):
            await env.exec("cat input.txt")
    finally:
        proceed.set()
        await task
        await env.stop()


@pytest.mark.asyncio
async def test_failed_cleanup_keeps_retry_without_false_readiness(tmp_path):
    env = environment(tmp_path)
    env._worker = SimpleNamespace(
        create_sandbox=AsyncMock(),
        destroy_sandbox=AsyncMock(side_effect=[RuntimeError("delete failed"), None]),
    )
    env._upload_environment_dir_after_start = AsyncMock(side_effect=RuntimeError("upload failed"))
    with pytest.raises(RuntimeError, match="delete failed"):
        await env.start()
    assert env._cleanup_pending and not env._started, "Failed cleanup must retain ownership without readiness."
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await env.start()
    await env.stop()
    assert not env._cleanup_pending, "Retry must acknowledge physical cleanup."


def test_admitted_container_cannot_exceed_budget(tmp_path):
    env = environment(
        tmp_path,
        container_budget={
            "main_cpu": 1,
            "main_memory_mb": 32,
            "max_sidecars": 0,
        },
    )
    with pytest.raises(ValueError, match="exceed"):
        env._spec()
