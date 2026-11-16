"""Replay bounded agent feedback barriers using measured model staging costs."""

import argparse
import itertools
import json
import math
import random
import statistics
import time
from pathlib import Path

from memory_policy import FAMILIES, POLICIES, Costs, Job, Scheduler


def workload(pattern, seed, gap, *, sessions=4, turns=3):
    """Generate scripted arrivals without giving their future to the scheduler."""
    rng = random.Random(seed)
    agents = []
    for session in range(sessions):
        rounds = []
        for turn in range(turns):
            if pattern == "mixed":
                families = list(FAMILIES)
                rng.shuffle(families)
            elif pattern == "alternating":
                families = [FAMILIES[(session + turn) % 2]] * 2
            elif pattern == "bursty":
                families = [FAMILIES[(session // 2 + (turn == 2)) % 2]] * 2
            elif pattern == "irregular":
                families = [rng.choice(FAMILIES) for _ in range(2)]
            else:
                raise ValueError(f"Unknown workload pattern {pattern!r}.")
            rounds.append(dict(families=families, gap_s=gap * (0.5 + rng.random())))
        agents.append(rounds)
    return agents


class Arrivals:
    """Release a new agent round only after all its current checks finish."""

    def __init__(self, agents):
        self.agents = agents
        self.ready = []
        self.due = {session: 0.0 for session in range(len(agents))}
        self.turns = dict.fromkeys(self.due, 0)
        self.outstanding, self.group_issued = {}, {}
        self.feedback = []
        self.serial = 0

    def issue(self, now):
        """Expose only rounds whose release time has already arrived."""
        issued = []
        for session, due in list(self.due.items()):
            if due > now + 1e-9:
                continue
            turn = self.turns[session]
            group = session, turn
            families = self.agents[session][turn]["families"]
            self.outstanding[group] = len(families)
            self.group_issued[group] = now
            for family in families:
                job = Job(self.serial, session, turn, family, self.serial % 2, now)
                self.serial += 1
                self.ready.append(job)
            issued.append(group)
            del self.due[session]
        return issued

    def finish(self, job, now):
        """Make the next scripted round eligible after the current feedback barrier."""
        self.ready.remove(job)
        self.outstanding[job.group] -= 1
        if self.outstanding[job.group]:
            return
        self.feedback.append(now - self.group_issued[job.group])
        gap = self.agents[job.session][job.turn]["gap_s"]
        self.turns[job.session] += 1
        if self.turns[job.session] < len(self.agents[job.session]):
            self.due[job.session] = now + gap


def simulate(agents, costs, budget, policy):
    """Use virtual time for a single serialized GPU command server."""
    arrivals = Arrivals(agents)
    scheduler = Scheduler(policy, costs, budget)
    now, cache, hits, builds, peak, idle = 0.0, frozenset(), 0, 0, 0, 0.0
    count, max_wait = 0, 0.0
    while arrivals.ready or arrivals.due:
        arrivals.issue(now)
        if not arrivals.ready:
            next_due = min(arrivals.due.values())
            idle += next_due - now
            now = next_due
            continue
        action = scheduler.choose(arrivals.ready, cache, now)
        max_wait = max(max_wait, now - action.job.issued)
        hits += action.mode == "reuse"
        builds += action.mode == "prepare"
        now += action.predicted_s
        previous = cache
        cache = action.keep
        peak = max(peak, costs.bytes(cache))
        scheduler.observe(action, previous)
        if policy == "stage" and action.job.family in cache:
            now += costs.evict[action.job.family]
            cache = frozenset()
        arrivals.finish(action.job, now)
        count += 1
    expected = sum(len(turn["families"]) for agent in agents for turn in agent)
    if count != expected or peak > budget:
        raise RuntimeError("The replay dropped work or exceeded its memory budget.")
    return dict(
        makespan_s=now,
        mean_feedback_s=statistics.mean(arrivals.feedback),
        worst_feedback_s=max(arrivals.feedback),
        max_command_wait_s=max_wait,
        idle_s=idle,
        cache_hits=hits,
        preparations=builds,
        peak_payload_bytes=peak,
        commands=count,
        planning_cpu_s=scheduler.planning_s,
    )


def self_check():
    """Compare the planner with independent enumeration of four current requests."""
    costs = object.__new__(Costs)
    costs.size = dict(language=2, classification=1)
    costs.prepare = dict(language=3.0, classification=1.0)
    costs.direct = dict(language=6.0, classification=2.0)
    costs.warm = dict(language=1.0, classification=0.5)
    costs.evict = dict.fromkeys(FAMILIES, 0.0)
    jobs = [
        Job(i, i // 2, 0, family, 0, 0.0)
        for i, family in enumerate(("language", "classification", "language", "language"))
    ]

    def enumerate_cost(order, index, cached, now, completions):
        if index == len(order):
            return sum(completions.values())
        job = order[index]
        options = [(costs.direct[job.family], cached)]
        options.append(
            (costs.warm[job.family] + (0 if cached == job.family else costs.prepare[job.family]), job.family)
        )
        best = float("inf")
        for duration, next_cache in options:
            finished = dict(completions)
            finished[job.group] = now + duration
            best = min(best, enumerate_cost(order, index + 1, next_cache, now + duration, finished))
        return best

    optimum = min(enumerate_cost(order, 0, None, 0.0, {}) for order in itertools.permutations(jobs))
    scheduler = Scheduler("feedback-dp", costs, 2)
    ready, cache, now, completions = list(jobs), frozenset(), 0.0, {}
    while ready:
        action = scheduler.choose(ready, cache, now)
        now += action.predicted_s
        completions[action.job.group] = now
        ready.remove(action.job)
        cache = action.keep
    if not math.isclose(sum(completions.values()), optimum):
        raise RuntimeError("The planner disagreed with independent exhaustive enumeration.")
    zero = simulate(workload("mixed", 0, 0, sessions=2, turns=1), costs, 0, "feedback-dp")
    if zero["preparations"] or zero["cache_hits"] or zero["commands"] != 4:
        raise RuntimeError("The zero capacity fallback was incorrect.")
    return dict(exhaustive_feedback_optimum=optimum, zero_capacity_fallback=True)


def run(args):
    """Run a predeclared grid and stop at a bounded CPU deadline."""
    profile = json.loads(args.profile.read_text())
    costs = Costs(profile)
    started = time.monotonic()
    evidence = dict(
        status="running",
        self_check=self_check(),
        scope="Measured costs with scripted virtual arrivals. Not GPU throughput.",
        scenarios=[],
    )
    grid = itertools.product(
        ("mixed", "alternating", "bursty", "irregular"), range(4), (0.8, 1.3, 3.0, 5.0), (0, 1, 4)
    )
    for pattern, seed, gib, gap_scale in grid:
        if time.monotonic() - started > args.budget_seconds:
            evidence["status"] = "budget_reached"
            break
        gap = gap_scale * max(costs.direct.values())
        agents = workload(pattern, seed, gap)
        results = {policy: simulate(agents, costs, int(gib * 1024**3), policy) for policy in POLICIES}
        evidence["scenarios"].append(
            dict(pattern=pattern, seed=seed, budget_gib=gib, gap_scale=gap_scale, results=results)
        )
    else:
        evidence["status"] = "passed"
    evidence["elapsed_s"] = time.monotonic() - started
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            dict(status=evidence["status"], scenarios=len(evidence["scenarios"]), elapsed_s=evidence["elapsed_s"])
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 120:
        raise ValueError("The replay CPU budget must be between 1 and 120 seconds.")
    run(args)
