"""Exercise evidence deletion with CPU fixtures, without running CUDA or training."""

import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def audit_setup(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "docs/env-worker-deep-dive/env-service-verification/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    root = tmp_path / "pressure-audit"
    monkeypatch.setenv("SCIENCE_PRESSURE_AUDIT_DIR", str(root))
    session = tmp_path / "pool-0/harbor-jobs/session"
    trial = session / "date/task"
    logs = trial / "agent"
    logs.mkdir(parents=True)
    (trial / "config.json").write_text("{}")
    return importlib.import_module("science_pressure_audit"), session, logs


@pytest.mark.parametrize("agent_class", ["FixedWorkPressureAgent", "CommandPressureAgent"])
async def test_probe_reply_survives_agent_log_deletion(audit_setup, agent_class):
    audit, session, logs = audit_setup
    module = importlib.import_module("science_fixed_work_agent")
    pressure = {"status": "ok", "kernel_output": 42, "requested_bytes": 72 * 1024**3}
    payload = {"return_code": 0, "stdout": json.dumps(pressure), "stderr": ""}

    class Environment:
        last_gpu_command = {"state": "fixture"}

        async def exec(self, command, **kwargs):
            return SimpleNamespace(model_dump=lambda: payload)

        exec_gpu = exec

    agent = object.__new__(getattr(module, agent_class))
    agent.logs_dir = logs
    agent._pressure_environment = Environment()
    await agent._exec_pressure("fixture probe")
    shutil.rmtree(logs)
    assert audit.load_pressure_evidence(session) == pressure, "Host evidence must survive writable log deletion."


def test_oom_is_retained_and_missing_is_not_success(audit_setup):
    audit, session, logs = audit_setup
    assert audit.load_pressure_evidence(session)["status"] == "missing", "Missing evidence must fail closed."
    audit.save_pressure_evidence(logs, {"return_code": 42, "stdout": '{"status":"cuda_oom"}'})
    shutil.rmtree(logs)
    assert audit.load_pressure_evidence(session)["status"] == "cuda_oom", (
        "OOM status must remain available for refill."
    )


def test_invalid_output_keeps_raw_evidence(audit_setup):
    audit, session, logs = audit_setup
    path = audit.save_pressure_evidence(logs, {"return_code": 1, "stdout": "broken probe output"})
    assert audit.load_pressure_evidence(session)["status"] == "invalid", "Malformed output must not count as success."
    assert json.loads(path.read_text())["command"]["stdout"] == "broken probe output", "Retain the original failure."


def test_reject_audit_directory_inside_trial(audit_setup):
    audit, _, logs = audit_setup
    with pytest.raises(ValueError, match="outside the trial"):
        audit.audit_path(logs, logs.parent / "artifacts/audit")
