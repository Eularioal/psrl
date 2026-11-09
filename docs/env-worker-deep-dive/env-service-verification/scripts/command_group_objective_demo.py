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


def centered_reward_intervals(rewards, radii):
    # Diagram 3: Propagate assumed simultaneous bounds through the group mean.
    size = len(rewards)
    mean = sum(rewards) / size
    rows = []
    for index, (reward, radius) in enumerate(zip(rewards, radii)):
        other_radii = sum(radii) - radius
        error = (1 - 1 / size) * radius + other_radii / size
        low = reward - mean - error
        high = reward - mean + error
        sign = "positive" if low > 0 else "negative" if high < 0 else "uncertain"
        rows.append({"candidate": "ABCD"[index], "low": round(low, 6), "high": round(high, 6), "sign": sign})
    return rows


def group_measurement_example():
    # Diagrams 2 and 3: Narrow D, then B, without changing the assumed means.
    rewards = [0.80, 1.00, 1.04, 1.20]
    cases = {
        "initial": [0.01, 0.01, 0.01, 0.12],
        "refine_d": [0.01, 0.01, 0.01, 0.002],
        "refine_d_then_b": [0.01, 0.002, 0.01, 0.002],
    }
    return {name: centered_reward_intervals(rewards, radii) for name, radii in cases.items()}


def measurement_contract_examples():
    # Conceptual measurement diagram: Repetition cannot remove an assumed fixed bias.
    isolated = {"A": 8.0, "B": 10.0}
    bias = {"A": 4.0, "B": 1.0}
    repeated = {}
    for count in (2, 20, 200):
        means = {
            name: round(sum(value + bias[name] + (0.1 if i % 2 else -0.1) for i in range(count)) / count, 6)
            for name, value in isolated.items()
        }
        repeated[count] = {"observed_mean_ms": means, "observed_winner": min(means, key=means.get)}
    # Conceptual decision diagram: The same intervals can settle a deadline but not a ranking.
    intervals = {"A": (9.5, 10.5), "B": (9.6, 10.6)}
    return {
        "isolated_winner": min(isolated, key=isolated.get),
        "repeated_biased_measurements": repeated,
        "intervals_ms": intervals,
        "a_definitely_faster": intervals["A"][1] < intervals["B"][0],
        "b_definitely_faster": intervals["B"][1] < intervals["A"][0],
        "both_below_20_ms": all(high < 20 for low, high in intervals.values()),
    }


if __name__ == "__main__":
    print("Teaching simulation. No Env, GPU, timing estimator, or scheduler is executed.")
    for order in ("BA", "AB"):
        print(json.dumps(group_completion(order)))
    print(json.dumps(feedback_sensitivity()))
    print("Assumed bounds, not statistical confidence or a normalized advantage guarantee.")
    print(json.dumps(group_measurement_example()))
    print("Assumed bias and decision thresholds. No empirical coverage or performance claim.")
    print(json.dumps(measurement_contract_examples()))
