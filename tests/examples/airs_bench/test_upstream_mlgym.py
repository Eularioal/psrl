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


@pytest.mark.parametrize("ending", ["submitted", "exec_error", "missing", "invalid"])
async def test_real_base_agent_initializes_installs_tools_validates_and_submits(tmp_path, monkeypatch, ending):
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
        data_root=str(tmp_path), mlgym_repo=str(Path(mlgym.__file__).resolve().parent.parent)
    )
    payload = {
        "episode_dir": str(tmp_path),
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
