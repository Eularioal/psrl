"""Screen static partitions with frozen component profiles and explicit dependencies."""

import argparse
import hashlib
import heapq
import itertools
import json
import time
from collections import deque
from pathlib import Path


def simulate(profiles, env_cards, model_cards, cap, request_s=6.0, steps=4, train_s=9.0):
    """Approximate two batches in flight and ordered training with global sync pauses.

    Commands use measured residence time conditional on configured capacity.
    Requests use a constant service estimate with sixteen slots per replica.
    This screening model excludes shared execution and continuous batching.
    """
    if min(env_cards, model_cards, cap, request_s, steps) <= 0:
        raise ValueError("All resource counts and service times must be positive.")
    events, sequence = [], itertools.count()
    env_queue, model_queue = deque(), deque()
    env_active = [0] * env_cards
    model_active = 0
    finished = [0] * steps
    trained = 0
    training = False
    pause_until = 0.0
    now = 0.0
    counts = {"commands": 0, "requests": 0, "trajectories": 0, "updates": 0}

    def event(at, kind, payload):
        heapq.heappush(events, (at, next(sequence), kind, payload))

    def release(batch):
        if batch < steps:
            for trajectory in range(12):
                model_queue.append((batch, trajectory, 0))

    release(0)
    release(1)
    event(0.0, "tick", None)
    while events:
        now, _, kind, payload = heapq.heappop(events)
        if kind == "model":
            model_active -= 1
            batch, trajectory, phase = payload
            counts["requests"] += 1
            if phase < 2:
                env_queue.append((batch, trajectory, phase))
            else:
                counts["trajectories"] += 1
                finished[batch] += 1
        elif kind == "env":
            gpu, batch, trajectory, phase = payload
            env_active[gpu] -= 1
            counts["commands"] += 1
            model_queue.append((batch, trajectory, phase + 1))
        elif kind == "trained":
            training = False
            trained += 1
            counts["updates"] += 1
            # A short global pause approximates synchronization after each update.
            pause_until = now + 0.3
            event(pause_until, "tick", None)
            release(trained + 1)
        while env_queue:
            gpu = next((i for i, active in enumerate(env_active) if active < cap), None)
            if gpu is None:
                break
            batch, trajectory, phase = env_queue.popleft()
            category = "verify" if phase == 1 else ("long" if trajectory // 2 == 5 else "short")
            env_active[gpu] += 1
            event(now + profiles[category][str(cap)], "env", (gpu, batch, trajectory, phase))
        while model_queue and now >= pause_until and model_active < 16 * model_cards:
            model_active += 1
            event(now + request_s, "model", model_queue.popleft())
        if trained < steps and finished[trained] == 12 and not training:
            training = True
            event(now + train_s, "trained", None)
    expected = {"commands": 24 * steps, "requests": 36 * steps, "trajectories": 12 * steps, "updates": steps}
    if counts != expected:
        raise RuntimeError(f"Incomplete dependency replay: {counts!r}.")
    return {"predicted_step_s": now, "work": counts}


def plan(selection):
    """Enumerate measured capacities without using integrated validation outcomes."""
    profiles = selection["profiles"]
    rows = []
    for env_cards in (1, 2, 3):
        for cap in (1, 2, 4, 5):
            row = {"env_cards": env_cards, "model_cards": 4 - env_cards, "cap": cap}
            row.update(simulate(profiles, env_cards, 4 - env_cards, cap))
            row["sensitivity_s"] = [
                simulate(profiles, env_cards, 4 - env_cards, cap, request_s=value)["predicted_step_s"]
                for value in (3.0, 6.0, 12.0)
            ]
            rows.append(row)
    best = min(rows, key=lambda row: row["predicted_step_s"])
    return {
        "created_at": time.time(),
        "candidate_budget": {"training_cards": 2, "environment_plus_rollout_cards": 4},
        "rows": rows,
        "selected": best,
        "assumptions": {"request_s": 6.0, "request_slots_per_replica": 16, "training_s_per_step": 9.0},
        "validation": "Freeze before new full training runs. New 1+3 is a prospective partition check.",
        "limitations": [
            "Component calibration only. No integrated outcome is fitted by this script.",
            "The author already saw historical 2+2 and 3+1 results. These are not blind validation.",
            "Request slots and constant latency approximate SMG and omit batching interference.",
            "Two batches in flight approximate staleness control, not the exact PSRL scheduler.",
            "No learned sharing profile. Shared execution is deliberately not predicted.",
            "Sensitivity bounds are scenarios, not statistical confidence intervals.",
            "Initialization, physical cleanup, and calibration cost are excluded from predictions.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan(json.loads(args.selection.read_text()))
    result["calibration_sha256"] = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
