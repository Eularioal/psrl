"""Check synchronization and ownership for the cooperative compute study."""

import importlib.util
import threading
import time
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).parents[2]
    / "docs/env-worker-deep-dive/env-service-verification/scripts/resident_sharing_study/protocol.py"
)
spec = importlib.util.spec_from_file_location("study_protocol", SOURCE)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_pause_acknowledges_gpu_synchronization_before_resume(tmp_path):
    request = {"trial_id": "owned", "epoch": 3, "pause": True}
    protocol.save(tmp_path / "request.json", request)
    calls = []
    hook = protocol.PausePoint(tmp_path, "owned", lambda: calls.append("synchronized"))
    thread = threading.Thread(target=hook, args=("updated", 7), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while not (tmp_path / "state.json").exists() and time.monotonic() < deadline:
            time.sleep(0.001)
        receipt = protocol.read(tmp_path / "state.json")
        assert calls == ["synchronized"], "Expected actual synchronization before acknowledging the pause."
        assert receipt["updates"] == 7 and receipt["epoch"] == 3, "Expected an owned progress receipt."
        assert thread.is_alive(), "Expected the tool to remain blocked while compute is borrowed."
    finally:
        protocol.save(tmp_path / "request.json", {**request, "pause": False})
        thread.join(timeout=2)
    assert not thread.is_alive(), "Expected matching compute release to resume the same tool."
    assert hook.events[-1]["state"] == "resumed", "Expected resume acknowledgement."
    assert protocol.read(tmp_path / "state.json")["state"] == "resumed", "Expected visible resume permission."


def test_another_trial_cannot_supply_compute_permission(tmp_path):
    protocol.save(tmp_path / "request.json", {"trial_id": "another", "pause": False})
    hook = protocol.PausePoint(tmp_path, "owned", lambda: None)
    with pytest.raises(RuntimeError, match="another trial"):
        hook("updated", 1)
