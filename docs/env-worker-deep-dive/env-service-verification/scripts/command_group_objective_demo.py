"""Illustrate group completion and reward sensitivity with invented CPU data."""

import json
import math


def group_completion(order):
    # Diagram 1: A has GPU 10, CPU 20, GPU 1. B has one GPU command of length 1.
    gpu_ready = 0
    first_done = {}
    for name in order:
        gpu_ready += 10 if name == "A" else 1
        first_done[name] = gpu_ready
    a_done = max(gpu_ready, first_done["A"] + 20) + 1
    completed = {"A": a_done, "B": first_done["B"]}
    return {
        "first_order": order,
        "completed_s": completed,
        "mean_completion_s": sum(completed.values()) / 2,
        "group_completion_s": max(completed.values()),
    }


def advantages(rewards):
    # Diagram 6: Use population variance for teaching, not the veRL estimator.
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    return [round((value - mean) / (std + 1e-8), 6) for value in rewards]


def feedback_sensitivity():
    # Diagrams 2 and 5: Contention changes measured runtimes without changing code.
    isolated = [10.0, 10.1]
    shared = [isolated[0] * 1.10, isolated[1] * 1.01]
    result = {}
    for label, runtimes in (("isolated", isolated), ("shared", shared)):
        rewards = [12.0 / runtime for runtime in runtimes]
        result[label] = {
            "runtime_us": runtimes,
            "reward": rewards,
            "advantage": advantages(rewards),
        }
    return result


if __name__ == "__main__":
    print("Teaching simulation. No Env, GPU, timing estimator, or scheduler is executed.")
    for order in ("BA", "AB"):
        print(json.dumps(group_completion(order)))
    print(json.dumps(feedback_sensitivity()))
