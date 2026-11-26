"""Check dependency replay and prevent calibration/test data leakage."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / (
    "docs/env-worker-deep-dive/env-service-verification/scripts/analyze_environment_capacity.py"
)
spec = importlib.util.spec_from_file_location("capacity_analysis", SCRIPT)
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def test_dependency_replay_accounts_for_model_turns_and_waiting():
    profiles = {"short": {1: 10, 2: 10}, "long": {1: 10, 2: 10}, "verify": {1: 5, 2: 5}}
    assert capacity.simulate_batch(profiles, 1, groups=1, model_request_s=2, devices=1) == 34
    assert capacity.simulate_batch(profiles, 2, groups=1, model_request_s=2, devices=1) == 21


def test_interpolation_refuses_unprofiled_capacity():
    assert capacity.interpolate({2: 10, 4: 20}, 3) == 15
    with pytest.raises(ValueError, match="outside"):
        capacity.interpolate({2: 10, 4: 20}, 5)


def test_holdout_and_repeat_do_not_choose_static_capacity():
    def trial(index, cap, duration):
        return {
            "index": index,
            "cap": cap,
            "elapsed_s": duration,
            "commands": [
                {
                    "mode": mode,
                    "admitted_at": 0,
                    "finished_at": duration,
                    "result": {"updates": updates, "compute_s": 1, "preparation_s": 1},
                }
                for mode, updates in (("train", 448), ("train", 896), ("verify", 0))
            ],
        }

    evidence = {
        "status": "complete",
        "physical_release_verified": True,
        "calibration": [1, 2, 4, 5],
        "held_out": [3],
        "repeat": [4],
        "trials": [trial(i, cap, 10 + cap) for i, cap in enumerate([1, 4, 2, 5, 3, 4])],
    }
    before = capacity.analyze(evidence)
    evidence["trials"][4] = trial(4, 3, 10000)
    evidence["trials"][5] = trial(5, 4, 10000)
    after = capacity.analyze(evidence)
    assert before["selected_static_cap"] == after["selected_static_cap"]
    assert before["profiles"] == after["profiles"]
    assert before["held_out_validation"] != after["held_out_validation"]
