"""
Walk through real arrival rules and numerical CPU cache accounting without a GPU.
"""

import logging

from memory_replay import Arrivals, workload

psrl_logger = logging.getLogger(__file__)

PAYLOAD_BYTES = {"L": 3_006_529_536, "C": 1_191_555_072}


def show_rounds():
    """
    Exercise M1 and M6 using virtual one second checks and scripted waiting gaps.
    """
    agents = workload("mixed", 17, 2.0)
    arrivals = Arrivals(agents)
    arrivals.issue(0.0)
    psrl_logger.info(f"[M1/M2] Four directories A/B/C/D submit {len(arrivals.ready)} initial checks.")
    first_pair = [job for job in arrivals.ready if job.session == 0]
    arrivals.finish(first_pair[0], 1.0)
    psrl_logger.info("[M6] A finishes one check at virtual second 1. Its second round cannot start.")
    assert 0 not in arrivals.due, "A became eligible before both checks finished."
    arrivals.finish(first_pair[1], 2.0)
    psrl_logger.info(f"[M6] A finishes both at second 2. Its next round becomes due at {arrivals.due[0]:.3f}.")

    # M1 exposes only eligible rounds while M4 is replaced with virtual elapsed time.
    now, completed, groups = 2.0, 2, 4
    saw_overlap = False
    while arrivals.ready or arrivals.due:
        issued = arrivals.issue(now)
        groups += len(issued)
        if (0, 1) in issued:
            waiting = [chr(65 + session) for session in (2, 3) if arrivals.outstanding.get((session, 0), 0)]
            psrl_logger.info(f"[M1] A submits round 2 at second {now:.3f}. Round 1 still has work in {waiting!r}.")
            saw_overlap = bool(waiting)
        if not arrivals.ready:
            now = min(arrivals.due.values())
            continue
        job = arrivals.ready[0]
        now += 1.0
        arrivals.finish(job, now)
        completed += 1
    assert saw_overlap, "The walkthrough did not expose independent session progress."
    assert completed == 24 and groups == 12, "The walkthrough did not finish the declared work."
    psrl_logger.info(f"[M6/M7] Finished {groups} rounds and {completed} checks. No real waiting occurred.")


def show_cache():
    """
    Illustrate M3 and M4 using measured payload sizes without allocating tensors.
    """
    psrl_logger.info("[M3] CPU converted weights occupy 2.800 GiB for L and 1.110 GiB for C.")
    for budget_gib in (3, 5):
        budget = budget_gib * 1024**3
        cached, builds = set(), 0
        for family in ("L", "C", "L"):
            actions = []
            if family in cached:
                actions.append(f"Reuse {family}")
            else:
                if sum(PAYLOAD_BYTES[key] for key in cached) + PAYLOAD_BYTES[family] > budget:
                    actions.append(f"Drop {sorted(cached)!r}")
                    cached.clear()
                cached.add(family)
                builds += 1
                actions.append(f"Prepare {family}")
            payload = sum(PAYLOAD_BYTES[key] for key in cached)
            assert payload <= budget, "The numerical cache exceeded its quota."
            psrl_logger.info(
                f"[M3/M4] CPU cache cap {budget_gib} GiB. {', '.join(actions)}. Kept {payload / 1024**3:.3f} GiB."
            )
        psrl_logger.info(f"[M7] CPU cache cap {budget_gib} GiB required {builds} preparations for L, C, L.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    psrl_logger.info("Teaching simulation only. No models, Docker, GPU or large CPU allocations are used.")
    show_rounds()
    show_cache()
