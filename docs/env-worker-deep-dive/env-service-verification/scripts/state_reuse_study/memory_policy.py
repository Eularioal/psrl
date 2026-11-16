"""Compare cache admission and command order using only already issued requests."""

import itertools
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache

FAMILIES = ("language", "classification")
POLICIES = ("direct", "stage", "lru", "gdsf", "affinity", "critical", "locality-dp", "feedback-dp")


@dataclass(frozen=True)
class Job:
    """Describe one current request without exposing future agent turns."""

    identifier: int
    session: int
    turn: int
    family: str
    seed: int
    issued: float

    @property
    def group(self):
        """Identify the feedback barrier that this request contributes to."""
        return self.session, self.turn


@dataclass(frozen=True)
class Action:
    """Bind command selection to one legal host state transition."""

    job: Job
    mode: str
    keep: frozenset
    predicted_s: float


class Costs:
    """Use a common measured profile for every policy."""

    def __init__(self, profile):
        self.size = profile["payload_bytes"]
        self.prepare, self.direct, self.warm = {}, {}, {}
        self.evict = {name: 0.0 for name in FAMILIES}
        self.evict.update(profile.get("evict_s", {}))
        for name, row in profile["models"].items():
            self.prepare[name] = row["prepare_s"]
            self.direct[name] = statistics.mean(r["total_s"] for r in row["commands"] if not r["prepared"])
            self.warm[name] = statistics.mean(r["total_s"] for r in row["commands"] if r["prepared"])

    def bytes(self, keys):
        """Count unique prepared payloads instead of process RSS."""
        return sum(self.size[key] for key in keys)


def transitions(job, cache, costs, budget):
    """Enumerate direct execution and legal prepared state admissions."""
    yield Action(job, "direct", frozenset(cache), costs.direct[job.family])
    if job.family in cache:
        yield Action(job, "reuse", frozenset(cache), costs.warm[job.family])
        return
    if costs.size[job.family] > budget:
        return
    keys = sorted(cache)
    for count in range(len(keys) + 1):
        for retained in itertools.combinations(keys, count):
            keep = frozenset((*retained, job.family))
            if costs.bytes(keep) <= budget:
                evicted = cache - keep
                duration = costs.prepare[job.family] + costs.warm[job.family]
                duration += sum(costs.evict[key] for key in evicted)
                yield Action(job, "prepare", keep, duration)


class Scheduler:
    """Apply identical memory and aging constraints to every cached policy."""

    def __init__(self, policy, costs, budget, *, horizon_groups=4):
        self.policy, self.costs, self.budget = policy, costs, budget
        self.horizon_groups = horizon_groups
        self.last, self.frequency, self.priority = {}, {}, {}
        self.clock, self.access = 0.0, 0
        self.max_age = 4 * max(costs.direct.values())
        self.planning_s = 0.0

    def _admit(self, job, cache):
        if job.family in cache:
            return Action(job, "reuse", frozenset(cache), self.costs.warm[job.family])
        if self.costs.size[job.family] > self.budget:
            return Action(job, "direct", frozenset(cache), self.costs.direct[job.family])
        keep = set(cache)
        while self.costs.bytes(keep | {job.family}) > self.budget:
            if self.policy in {"gdsf", "critical"}:
                victim = min(keep, key=lambda key: (self.priority.get(key, 0), key))
            else:
                victim = min(keep, key=lambda key: (self.last.get(key, -1), key))
            keep.remove(victim)
        keep.add(job.family)
        duration = self.costs.prepare[job.family] + self.costs.warm[job.family]
        duration += sum(self.costs.evict[key] for key in cache - keep)
        return Action(job, "prepare", frozenset(keep), duration)

    def choose(self, ready, cache, now):
        """Observe the current ready queue and select one action."""
        started = time.monotonic()
        ordered = sorted(ready, key=lambda job: (job.issued, job.identifier))
        oldest = ordered[0]
        forced = oldest if now - oldest.issued >= self.max_age else None
        if self.policy == "direct":
            action = Action(oldest, "direct", frozenset(cache), self.costs.direct[oldest.family])
        elif self.policy == "stage":
            action = self._admit(oldest, cache)
        elif self.policy in {"locality-dp", "feedback-dp"}:
            action = self._plan(ordered, cache, forced)
        else:
            job = oldest
            if forced is None and self.policy == "affinity":
                job = next((job for job in ordered if job.family in cache), oldest)
            if forced is None and self.policy == "critical":
                remaining = {job.group: sum(other.group == job.group for other in ready) for job in ready}
                job = min(ordered, key=lambda item: (remaining[item.group], item.issued, item.identifier))
            action = self._admit(job, cache)
        self.planning_s += time.monotonic() - started
        if self.costs.bytes(action.keep) > self.budget:
            raise RuntimeError("The scheduler exceeded its prepared state budget.")
        return action

    def _plan(self, ordered, cache, forced):
        groups = list(dict.fromkeys(job.group for job in ordered))[: self.horizon_groups]
        jobs = [job for job in ordered if job.group in groups]
        if len(jobs) > 8:
            raise ValueError("The bounded planner supports at most two pending checks in each of four groups.")
        names = FAMILIES

        def encode(keys):
            return sum(1 << names.index(key) for key in keys)

        def decode(bits):
            return frozenset(key for index, key in enumerate(names) if bits & (1 << index))

        @lru_cache(None)
        def value(mask, bits):
            if mask == 0:
                return 0.0
            active = [index for index in range(len(jobs)) if mask & (1 << index)]
            weight = len({jobs[index].group for index in active}) if self.policy == "feedback-dp" else 1
            best = float("inf")
            for index in active:
                for action in transitions(jobs[index], decode(bits), self.costs, self.budget):
                    score = weight * action.predicted_s + value(mask ^ (1 << index), encode(action.keep))
                    best = min(best, score)
            return best

        mask = (1 << len(jobs)) - 1
        weight = len(groups) if self.policy == "feedback-dp" else 1
        options = []
        for index, job in enumerate(jobs):
            if forced is not None and job.identifier != forced.identifier:
                continue
            for action in transitions(job, frozenset(cache), self.costs, self.budget):
                objective = weight * action.predicted_s + value(mask ^ (1 << index), encode(action.keep))
                options.append((objective, index, action.mode, sorted(action.keep), action))
        return min(options, key=lambda item: item[:-1])[-1]

    def observe(self, action, previous):
        """Update past use statistics after the selected request completes."""
        self.access += 1
        family = action.job.family
        self.last[family] = self.access
        self.frequency[family] = self.frequency.get(family, 0) + 1
        removed = previous - action.keep
        if removed:
            self.clock = max(self.clock, max(self.priority.get(key, 0) for key in removed))
        if family in action.keep:
            self.priority[family] = (
                self.clock + self.frequency[family] * self.costs.prepare[family] / self.costs.size[family]
            )
