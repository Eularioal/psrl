"""Simulate offload decisions with invented timings, without CUDA or network calls."""

import json


def simulate(wait, policy):
    # Steps 1 and 2: A generates, then awaits an external result while B queues.
    generate, update, borrower = 10, 10, 15
    offload, reload = 2, 2
    switch_window = offload + borrower + reload
    lend = policy == "always" or (policy == "cost_aware" and wait >= switch_window)
    if lend:
        # Proposed step 3: physically return memory before B, then restore A.
        b_done = generate + offload + borrower
        a_done = generate + max(wait, switch_window) + update
    else:
        a_done = generate + wait + update
        b_done = a_done + borrower
    return {
        "wait": wait,
        "policy": policy,
        "lend": lend,
        "a_done": a_done,
        "b_done": b_done,
        "all_done": max(a_done, b_done),
    }


if __name__ == "__main__":
    print("Teaching simulation: exact wait is assumed known, no real GPU timing.")
    for duration in (40, 2):
        for choice in ("resident", "always", "cost_aware"):
            print(json.dumps(simulate(duration, choice)))
