"""Compare placement heuristics using invented durations and an ideal release oracle."""

import json

from dynamic_command_placement_demo import Job


def replay(policy, switch_cost, other_ready):
    # P1: Files remain on one host. C already occupies device 1.
    available = [0.0, other_ready]
    rows = []
    for job in (Job("A1", 0, 2), Job("B1", 2, 20), Job("A2", 4, 3)):
        # P2: Compare equal candidate information, with no future arrivals exposed.
        candidates = (job.home,) if policy == "fixed" else range(2)
        options = []
        for gpu in candidates:
            setup = switch_cost if gpu != job.home else 0.0
            start = max(job.ready, available[gpu]) + setup
            finish = start + job.duration
            value = max(job.ready, available[gpu]) if policy == "earliest-free" else finish
            options.append((value, gpu != job.home, gpu, start, finish))
        _, _, gpu, start, finish = min(options)
        # P3 and P4: Reserve atomically and assume physical return at finish.
        available[gpu] = finish
        rows.append(dict(job=job.name, gpu=gpu, start=start, finish=finish))
    return dict(a_done=rows[-1]["finish"], makespan=max(available), events=rows)


def main():
    print("Synthetic CPU simulation. Known running-job release times. No EnvRouter or GPU execution.")
    for case, cost, other_ready in (("cheap", 0.2, 5), ("expensive", 20, 5), ("busy", 0.2, 40)):
        results = {policy: replay(policy, cost, other_ready) for policy in ("fixed", "earliest-free", "cost-aware")}
        print(json.dumps(dict(case=case, switch_cost=cost, results=results)))
    assert replay("earliest-free", 0.2, 5)["makespan"] == replay("cost-aware", 0.2, 5)["makespan"]
    assert replay("earliest-free", 20, 5)["makespan"] > replay("fixed", 20, 5)["makespan"]


if __name__ == "__main__":
    main()
