"""Plan a bounded set of submitted commands using measured load and compute costs."""

import itertools
import math

POLICIES = ("fifo", "lpt", "affinity", "joint", "cost")


def compute_cost(job, profiles):
    """Read a frozen estimate without using the command's eventual runtime."""
    value = profiles[job["model"]]["compute_s"]
    return float(value[job["data"]] if isinstance(value, dict) else value)


def simulate(queue, profiles, cache, budget, load=0.0):
    """Return completion time and LRU cache, ordered from oldest to newest."""
    retained = list(cache)
    elapsed = float(load)
    if len(retained) != len(set(retained)):
        raise ValueError("The cache must contain unique model keys.")
    if sum(profiles[key]["bytes"] for key in retained) > budget:
        raise ValueError("The initial cache exceeds its model storage budget.")
    for job in queue:
        key = job["model"]
        profile = profiles[key]
        if profile["bytes"] > budget:
            raise ValueError(f"Model {key!r} exceeds its model storage budget.")
        if key in retained:
            retained.remove(key)
        else:
            while sum(profiles[item]["bytes"] for item in retained) + profile["bytes"] > budget:
                victim = retained.pop(0)
                elapsed += float(profiles[victim].get("evict_s", 0.0))
            elapsed += float(profile["load_s"])
        retained.append(key)
        elapsed += compute_cost(job, profiles)
    return elapsed, retained


def _validate(jobs, profiles, caches, budget, loads):
    if len(caches) != 2 or len(loads) != 2:
        raise ValueError("This probe requires exactly two GPU cache and load records.")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Submitted job identifiers must be unique.")
    if budget <= 0 or any(not math.isfinite(value) or value < 0 for value in loads):
        raise ValueError("The budget must be positive and residual times finite and nonnegative.")
    for job in jobs:
        profile = profiles[job["model"]]
        values = (profile["load_s"], compute_cost(job, profiles), profile.get("evict_s", 0.0))
        if profile["bytes"] <= 0 or any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Model sizes must be positive and measured costs finite and nonnegative.")
        if profile["bytes"] > budget:
            raise ValueError("Every submitted model must fit the common model storage budget.")
    for index in range(2):
        simulate([], profiles, caches[index], budget, loads[index])


def predicted_finish(queues, profiles, caches, budget, loads=None):
    """Return per-GPU finish estimates under the common cache transition rules."""
    residual = [0.0, 0.0] if loads is None else list(loads)
    return [simulate(queues[index], profiles, caches[index], budget, residual[index])[0] for index in range(2)]


def _list_plan(jobs, profiles, loads):
    queues = [[], []]
    finish = list(loads)
    for job in jobs:
        index = min(range(2), key=lambda item: (finish[item], item))
        queues[index].append(job)
        finish[index] += float(profiles[job["model"]]["load_s"]) + compute_cost(job, profiles)
    return queues


def _affinity_plan(jobs, profiles, caches, budget, loads):
    groups = {}
    for job in jobs:
        groups.setdefault(job["model"], []).append(job)
    ordered = sorted(
        groups.values(),
        key=lambda group: -simulate(group, profiles, [], budget)[0],
    )
    queues = [[], []]
    for group in ordered:
        candidates = []
        for index in range(2):
            trial = [list(queue) for queue in queues]
            trial[index].extend(group)
            finish = predicted_finish(trial, profiles, caches, budget, loads)
            candidates.append(((max(finish), sum(finish), index), trial))
        queues = min(candidates, key=lambda item: item[0])[1]
    return queues


def plan(jobs, profiles, caches, budget, policy, loads=None):
    """Return two static queues for a frozen batch of currently submitted jobs.

    FIFO and LPT here use estimated list placement. A live FIFO comparison should
    instead dispatch the shared arrival queue to whichever device actually frees.
    """
    ready = list(jobs)
    residual = [0.0, 0.0] if loads is None else list(loads)
    _validate(ready, profiles, caches, budget, residual)
    if policy not in POLICIES:
        raise ValueError(f"Unknown planning policy {policy!r}.")
    if policy == "fifo":
        return _list_plan(ready, profiles, residual)
    if policy == "lpt":
        ordered = sorted(ready, key=lambda job: -(profiles[job["model"]]["load_s"] + compute_cost(job, profiles)))
        return _list_plan(ordered, profiles, residual)
    if policy == "affinity":
        return _affinity_plan(ready, profiles, caches, budget, residual)
    if len(ready) > 6:
        raise ValueError("Exhaustive joint planning is limited to six currently submitted jobs.")
    best, best_score = [[], []], (math.inf, math.inf)
    # Every two-device sequence is represented by one permutation and one split.
    for order in itertools.permutations(ready):
        for split in range(len(order) + 1):
            trial = [list(order[:split]), list(order[split:])]
            finish = predicted_finish(trial, profiles, caches, budget, residual)
            score = max(finish), sum(finish)
            if score < best_score:
                best, best_score = trial, score
    return best
