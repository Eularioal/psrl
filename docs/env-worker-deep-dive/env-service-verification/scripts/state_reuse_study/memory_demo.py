"""Explain limited host state choices using standard library and virtual time only."""

from memory_policy import FAMILIES, Costs, Job, Scheduler
from memory_replay import self_check


def main():
    """Compare retaining the current model with admitting another model."""
    # M2: These small integer capacities and times are teaching assumptions.
    costs = object.__new__(Costs)
    costs.size = dict(language=3, classification=2)
    costs.prepare = dict(language=6.0, classification=4.0)
    costs.direct = dict(language=8.0, classification=5.0)
    costs.warm = dict.fromkeys(FAMILIES, 1.0)
    costs.evict = dict.fromkeys(FAMILIES, 1.0)
    # M1: One short classification command and two language commands are ready.
    jobs = [Job(i, i // 2, 0, family, 0, 0.0) for i, family in enumerate(("classification", "language", "language"))]
    for policy in ("lru", "locality-dp", "feedback-dp"):
        scheduler = Scheduler(policy, costs, 3)
        ready, cache, now = list(jobs), frozenset({"language"}), 0.0
        print(f"{policy}: Initial language state is warm. Capacity is 3 units.")
        while ready:
            # M3 and M4: The same decision selects the command and host state transition.
            action = scheduler.choose(ready, cache, now)
            previous = cache
            now += action.predicted_s
            cache = action.keep
            scheduler.observe(action, previous)
            ready.remove(action.job)
            print(f"  {action.job.family}: {action.mode}, keep={sorted(cache)}, finish={now:.1f}s")
        # M5: This example has no real scoring code, GPU or result publication.
        print(f"  Total virtual time: {now:.1f}s")
    print("Independent planner and zero capacity checks:", self_check())
    # M3: A second example isolates feedback barriers without a cache.
    costs.direct = dict(language=10.0, classification=2.0)
    families = ("classification", "language", "classification", "classification")
    jobs = [Job(i, i // 2, 0, family, 0, 0.0) for i, family in enumerate(families)]
    for policy in ("locality-dp", "feedback-dp"):
        scheduler = Scheduler(policy, costs, 0)
        ready, now, completed = list(jobs), 0.0, {}
        while ready:
            action = scheduler.choose(ready, frozenset(), now)
            now += action.predicted_s
            completed[action.job.group] = now
            ready.remove(action.job)
        print(f"{policy}: Feedback completion times={completed}, sum={sum(completed.values()):.1f}s")


if __name__ == "__main__":
    main()
