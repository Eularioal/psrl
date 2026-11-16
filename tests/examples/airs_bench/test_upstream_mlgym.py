"""Replay the actual upstream agent with local Bash and a tiny trusted grader."""

import asyncio
import csv
import json
import os
import signal
import subprocess
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mlgym", reason="Set PYTHONPATH to the pinned MLGym checkout and install its CPU dependencies.")

from examples.airs_bench import runner
from examples.airs_bench.config import AirsBenchRuntimeConfig
from mlgym.agent.base import BaseAgent  # noqa: F401
from mlgym.environment.env import MLGymEnv  # noqa: F401
from psrl.environments.mlgym_env import build_psrl_mlgym_env_class  # noqa: F401
from psrl.workers.env_worker.sandbox import ExecResult
from psrl.workers.env_worker.worker import ContainerShell, EnvWorker

pytestmark = pytest.mark.cpu_test


def test_real_litellm_backend_calls_local_policy_and_accounts_zero_cost(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from mlgym.backend.base import ModelArguments
    from mlgym.backend.utils import get_model

    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, request, self.headers.get("x-smg-tito-trajectory-id")))
            body = json.dumps(
                {
                    "id": "local-policy-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": request["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "submit"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    payload = {
        "model": "/local/psrl-test-policy",
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "sampling_params": {"max_tokens": 32},
        "trajectory_id_strategy": "manual",
    }
    try:
        runner.register_local_policy(payload)
        model = get_model(ModelArguments(**runner.build_mlgym_model_config(payload)))
        assert model.query([{"role": "user", "content": "Test local policy routing."}]) == "submit"
        assert model.stats.api_calls == 1 and model.stats.total_cost == 0
        assert calls[0][0] == "/v1/chat/completions"
        assert calls[0][1]["model"] == payload["model"]
        assert calls[0][2] == "0"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class LocalHandle:
    """Use real Bash while substituting Docker mounts and the prebuilt conda image."""

    sandbox_id = "replay"

    def __init__(self, home, labels):
        self.home = home
        self.labels = labels
        self.worker = EnvWorker(0, 1, 0, 2)
        process = subprocess.Popen(
            ["bash", "--noprofile", "--norc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
        self.worker._shells[self.sandbox_id] = ContainerShell(process, self.sandbox_id)
        self.process = process
        self.evaluations = 0

    async def exec(self, command, timeout_s, no_output_timeout_s=None):
        command = command.replace(
            "source /home/agent/miniconda3/etc/profile.d/conda.sh && conda activate mlgym_generic && ",
            "",
        ).replace("/home/agent", str(self.home))
        return await self.worker.exec(self.sandbox_id, command, timeout_s, no_output_timeout_s)

    async def write_file(self, path, data):
        destination = Path(path.replace("/home/agent", str(self.home)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    async def evaluate(self, spec, command, timeout_s):
        self.evaluations += 1
        try:
            with (self.home / "workspace" / "submission.csv").open() as stream:
                predictions = [int(row["prediction"]) for row in csv.DictReader(stream)]
        except (KeyError, ValueError):
            return ExecResult("", 2, False, 0)
        accuracy = sum(p == label for p, label in zip(predictions, self.labels, strict=True)) / len(self.labels)
        return ExecResult(json.dumps({"Accuracy": accuracy}), 0, False, 0)

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
        self.worker._shells[self.sandbox_id].close()


@pytest.mark.parametrize("agent_config_name", ["agent_config.yaml", "acceptance_agent_config.yaml"])
@pytest.mark.parametrize("ending", ["submitted", "exec_error", "missing", "invalid"])
async def test_real_base_agent_initializes_installs_tools_validates_and_submits(
    tmp_path, monkeypatch, ending, agent_config_name
):
    import mlgym

    root = tmp_path / "airs_configs"
    (root / "tasks").mkdir(parents=True)
    task = {
        "id": "tiny",
        "name": "Tiny",
        "description": "Predict three values. {dataset_docs}",
        "use_generic_conda": False,
        "requirements_path": "data/tiny/requirements.txt",
        "evaluation_paths": ["evaluate.py"],
        "use_separate_eval_container": True,
    }
    (root / "tasks" / "tiny.yaml").write_text(yaml.safe_dump(task))
    home = tmp_path / "agent"
    (home / "workspace").mkdir(parents=True)
    handle = LocalHandle(home, labels=[0, 1, 1])
    replay = tmp_path / "replay.jsonl"
    actions = [
        "printf 'prediction\\n0\\n1\\n0\\n' > submission.csv",
        "validate",
        "submit",
    ]
    if ending in {"missing", "invalid"}:
        action = "rm submission.csv" if ending == "missing" else "printf 'prediction\\nbad\\n' > submission.csv"
        actions.insert(2, action)
    if ending == "exec_error":
        actions = ["trigger_infrastructure_failure"]
        original_exec = handle.exec

        async def injected_exec(command, *args):
            if command.strip() == "trigger_infrastructure_failure":
                raise RuntimeError("Injected remote shell failure.")
            return await original_exec(command, *args)

        handle.exec = injected_exec
    replay.write_text(
        json.dumps({"tiny": [f"DISCUSSION\nTake this step.\n```\n{action}\n```" for action in actions]}) + "\n"
    )
    monkeypatch.setattr(
        runner, "build_mlgym_model_config", lambda payload: {"model_name": "replay", "replay_path": str(replay)}
    )
    runtime = AirsBenchRuntimeConfig(
        data_root=str(tmp_path),
        mlgym_repo=str(Path(mlgym.__file__).resolve().parent.parent),
        agent_config_path=str(Path(runner.__file__).with_name(agent_config_name)),
    )
    payload = {
        "episode_dir": str(tmp_path),
        "model": "/local/replay-policy",
        "runtime_config": runtime,
        "task_config_path": "tasks/tiny.yaml",
        "max_turns": 5,
        "sandbox_handle": handle,
        "event_loop": asyncio.get_running_loop(),
        "verifier_spec": None,
        "verifier_command": "trusted",
        "metric": "Accuracy",
    }
    try:
        result = await asyncio.wait_for(asyncio.to_thread(runner.run_mlgym_agent, payload), 20)
        if ending == "exec_error":
            assert result["exit_status"] == "exit_error", "A swallowed agent error was reported as submitted."
            assert not result["airs_scores"], "An infrastructure failure produced training scores."
            return
        if ending in {"missing", "invalid"}:
            expected = "submission_not_found" if ending == "missing" else "invalid_submission"
            assert result["exit_status"] == expected, result
            assert not result["airs_scores"], "A failed final submission inherited a validation score."
            assert result["airs_validate_count"] == 1, "Validation feedback must remain available."
            return
        assert result["exit_status"] == "submitted", result
        assert result["turns"] == 3
        assert result["airs_scores"][-1]["Accuracy"] == pytest.approx(2 / 3)
        assert handle.evaluations == 2
        assert result["airs_validate_count"] == result["airs_submit_count"] == 1
    finally:
        handle.close()
