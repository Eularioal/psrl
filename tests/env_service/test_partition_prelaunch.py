"""Check resource and dependency behavior of the offline screening model."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/env-worker-deep-dive/env-service-verification/scripts/partition_prelaunch.py"
spec = importlib.util.spec_from_file_location("partition_prelaunch", SCRIPT)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def test_environment_bottleneck_and_training_serialization():
    profiles = {phase: {"4": 100.0} for phase in ("short", "long", "verify")}
    scarce = planner.simulate(profiles, 1, 3, 4)
    ample = planner.simulate(profiles, 3, 1, 4)
    assert (
        ample["work"]
        == scarce["work"]
        == {
            "commands": 96,
            "requests": 144,
            "trajectories": 48,
            "updates": 4,
        }
    )
    assert ample["predicted_step_s"] < scarce["predicted_step_s"]
    training_bound = planner.simulate(profiles, 3, 1, 4, train_s=1000.0)
    assert training_bound["predicted_step_s"] >= 4000.0


def test_model_queue_can_change_partition_preference():
    profiles = {phase: {"4": 0.01} for phase in ("short", "long", "verify")}
    one_model = planner.simulate(profiles, 3, 1, 4, request_s=100.0, train_s=0.01)
    two_models = planner.simulate(profiles, 2, 2, 4, request_s=100.0, train_s=0.01)
    assert two_models["predicted_step_s"] < one_model["predicted_step_s"]
    assert two_models["work"] == one_model["work"]
