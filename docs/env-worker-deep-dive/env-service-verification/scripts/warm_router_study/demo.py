"""Demonstrate warm waiting and cold rebuilding with CPU-only teaching inputs."""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path


def load_policy(filename="warm_placement.py"):
    """Load the standard library policy without importing the PSRL runtime."""
    repository = Path(__file__).resolve().parents[5]
    path = repository / "psrl/workers/env_service" / filename
    specification = importlib.util.spec_from_file_location(path.stem + "_teaching", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("The local placement policy could not be loaded.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main():
    policy = load_policy()
    interval = policy.DurationInterval
    request = policy.PlacementRequest("A", 100, interval(4, 6), interval(2, 3))
    idle_cold = policy.DeviceSnapshot("cold", frozenset(), 100)
    overrun = policy.estimate_remaining([1, 2], completed_steps=9, total_steps=10, elapsed_step_s=3.1)
    scenarios = [
        ("short_wait", interval(1, 2), (), idle_cold, "wait"),
        ("long_wait", interval(10, 12), (), idle_cold, "rebuild"),
        ("unknown_command", None, (), idle_cold, "defer"),
        ("overrun", overrun.remaining, (), idle_cold, "defer"),
        ("active_plus_queue", interval(3, 5), (interval(4, 6), interval(7, 8)), idle_cold, "rebuild"),
        ("insufficient_cache", interval(10, 12), (), policy.DeviceSnapshot("cold", frozenset(), 99), "wait"),
    ]
    records = []
    for name, remaining, queued, cold, expected in scenarios:
        # Steps 3 and 4 in the accompanying diagram construct and compare snapshots.
        hot = policy.DeviceSnapshot("hot", frozenset({"A"}), 0, True, remaining, queued)
        decision = policy.choose_placement(request, [hot, cold])
        if decision.action != expected:
            raise RuntimeError(f"Unexpected action for {name!r}: {decision.action!r}.")
        records.append(dict(case=name, decision=asdict(decision)))
    progress = policy.estimate_remaining([1, 2], completed_steps=2, total_steps=5, elapsed_step_s=0.5)
    predictor = load_policy("step_predictor.py")
    # Step 3 observes three complete 16-update chunks after the warmup chunk.
    observation = dict(chunk_steps=[16, 16, 16], chunk_wall_s=[0.40, 0.416, 0.432])
    forecast = predictor.predict_remaining(**observation, completed_steps=64, total_steps=2048)
    stale = predictor.predict_remaining(**observation, completed_steps=64, total_steps=2048, elapsed_since_report_s=1)
    if forecast.estimate_s != 1984 * 0.026 or stale.status != "overrun":
        raise RuntimeError("The chunk forecast teaching example changed.")
    print(
        json.dumps(
            dict(
                scope="CPU teaching inputs, no GPU execution",
                cases=records,
                progress=asdict(progress),
                overrun=asdict(overrun),
                chunk_forecast=asdict(forecast),
                chunk_overrun=asdict(stale),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
