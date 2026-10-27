"""Keep observation mode explicit while retaining execution and cleanup failures."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples.sciaccel_rl.config import HarborConfig, SciAccelRuntimeConfig, build_runtime_config  # noqa: E402
from examples.sciaccel_rl.gpu_grading import apply_gpu_activity_gate, save_gpu_reports  # noqa: E402
from psrl.workers.env_service.router import EnvRouterCore  # noqa: E402
from test_runner_env_service import _cfg, _ledger_containers, _run_episode  # noqa: E402


@pytest.mark.parametrize("mode", ["required", "observe"])
def test_missing_procfs_policy_controls_native_science_start(tmp_path, mode):
    cfg = _cfg()
    cfg["units"][0]["gpu_reservation"] = "command"
    core = EnvRouterCore(cfg)
    check = AsyncMock(side_effect=RuntimeError("GPU attribution needs host PID procfs."))
    node = SimpleNamespace(
        check_gpu_attribution_support=SimpleNamespace(remote=check),
        remove_gpu_lease=SimpleNamespace(remote=AsyncMock()),
        gpu_lending_snapshot=SimpleNamespace(remote=AsyncMock(return_value={0: {"used_mb": 0}})),
    )
    core._gpu_command_node = AsyncMock(return_value=node)
    core._gpu_commands.get_node = AsyncMock(return_value=node)
    runtime = SciAccelRuntimeConfig(harbor=HarborConfig(jobs_dir=str(tmp_path), gpu_attribution_mode=mode))
    result, captured = _run_episode(core, needs_gpu=True, runtime=runtime, task_path="/laps-accel-cuda")
    assert captured["job_create_called"] == (mode == "observe"), "Only observation mode may skip attribution."
    assert _ledger_containers(core) == 0, "Both modes must release the episode lease."
    if mode == "required":
        check.assert_awaited_once()
        assert "host PID procfs" in result.exception, "Required mode must retain its prerequisite."
    else:
        check.assert_not_awaited()
        assert result.reward == 1 and result.rewards["gpu_attribution_enforced"] == 0
        artifacts = list(tmp_path.rglob("gpu-job-reports.json"))
        assert len(artifacts) == 1, "Observation mode must persist its attribution boundary."
        report = json.loads(artifacts[0].read_text())
        assert report["jobs"] == [], "Missing samples must remain missing."
        assert report["gpu_attribution"]["score_source"] == "task_verifier_unverified"
        assert report["gpu_attribution"]["preflight"] == "skipped_by_policy"


def test_observation_does_not_hide_physical_return_failures(tmp_path):
    cfg = _cfg()
    cfg["units"][0]["gpu_reservation"] = "command"
    core = EnvRouterCore(cfg)
    node = SimpleNamespace(
        remove_gpu_lease=SimpleNamespace(remote=AsyncMock()),
        gpu_lending_snapshot=SimpleNamespace(remote=AsyncMock(return_value={0: {"used_mb": 0}})),
    )
    core._gpu_commands.get_node = AsyncMock(return_value=node)
    core.wait_gpu_command_returns = AsyncMock(side_effect=RuntimeError("GPU handles remain."))
    runtime = SciAccelRuntimeConfig(harbor=HarborConfig(jobs_dir=str(tmp_path), gpu_attribution_mode="observe"))
    result, captured = _run_episode(core, needs_gpu=True, runtime=runtime, task_path="/laps-accel-cuda")
    assert captured["job_create_called"] and result.reward == 0
    assert result.exception == "GPU handles remain.", "Observation must not suppress execution failures."
    node.remove_gpu_lease.remote.assert_awaited_once()


def test_observation_preserves_unverified_scores_and_error_evidence(tmp_path):
    reports = [{"tag": "check", "state": "completed", "telemetry": {"errors": ["Missing host procfs."]}}]
    original = {"reward": 0.7, "reward_gpu": 0.7, "gpu_active": 1.0}
    observed = apply_gpu_activity_gate(original, reports, ["check"], mode="observe")
    assert observed == {**original, "gpu_attribution_enforced": 0.0}
    save_gpu_reports(tmp_path, reports, observed, mode="observe")
    saved = json.loads((tmp_path / "gpu-job-reports.json").read_text())
    assert saved["jobs"] == reports and not saved["gpu_attribution"]["enforced"]
    assert saved["gpu_attribution"]["risk"], "Unverified GPU rewards must carry their risk."
    with pytest.raises(RuntimeError, match="host process attribution"):
        apply_gpu_activity_gate(original, reports, ["check"])


def test_attribution_mode_is_validated_in_structured_configuration():
    configured = build_runtime_config({"harbor": {"gpu_attribution_mode": "observe"}})
    assert configured.harbor.gpu_attribution_mode == "observe"
    assert HarborConfig().gpu_attribution_mode == "required"
    with pytest.raises(ValueError, match="Unknown GPU attribution mode"):
        HarborConfig(gpu_attribution_mode="skip_everything")
    with pytest.raises(ValueError, match="Unknown GPU attribution mode"):
        apply_gpu_activity_gate({}, [], [], mode="skip_everything")
