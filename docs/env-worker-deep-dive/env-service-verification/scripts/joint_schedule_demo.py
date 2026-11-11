"""Compare deterministic schedules and prediction errors without GPU execution."""

import itertools
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    name: str
    memory: int
    duration: int
    tail: int


def schedule(order, capacities=(80, 80)):
    """Run one job per GPU, followed by an independent non-GPU tail."""
    available = [0] * len(capacities)
    rows = []
    for job in order:
        eligible = [i for i, capacity in enumerate(capacities) if job.memory <= capacity]
        gpu = min(eligible, key=lambda i: (available[i], i))
        start = available[gpu]
        end = start + job.duration
        available[gpu] = end
        rows.append(dict(job=job.name, gpu=gpu, start=start, end=end, rollout_done=end + job.tail))
    horizon = max(row["rollout_done"] for row in rows)
    return dict(group_done=horizon, gpu_busy=sum(j.duration for j in order), rows=rows)


def main():
    # Proposal nodes 1 to 3 use known jobs and toy predictions, not future observations.
    jobs = (Job("S1", 40, 10, 0), Job("S2", 60, 10, 0), Job("L", 40, 2, 20))
    fifo = schedule(jobs)
    informed = schedule(sorted(jobs, key=lambda job: -job.tail))
    oracle = min(schedule(order)["group_done"] for order in itertools.permutations(jobs))
    wrong_tail_prediction = {"S1": 20, "S2": 20, "L": 0}
    mistaken = schedule(sorted(jobs, key=lambda job: -wrong_tail_prediction[job.name]))
    assert (fifo["group_done"], informed["group_done"], oracle) == (32, 22, 22), "Unexpected toy schedule."
    # Proposal node 3 uses memory and time together for a six-second backfill window.
    free_memory, window = 30, 6
    candidates = {"short": (20, 3), "long": (20, 9)}
    fits = {name: memory <= free_memory and duration <= window for name, (memory, duration) in candidates.items()}
    print(
        json.dumps(
            dict(
                scope="Teaching model only. One job per GPU. No setup, transfer or interference.",
                fifo=fifo,
                tail_aware=informed,
                enumerated_best_for_this_toy=oracle,
                wrong_prediction_group_done=mistaken["group_done"],
                backfill_fits=fits,
                predicted_four_actual_nine_overrun=max(0, 9 - window),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
