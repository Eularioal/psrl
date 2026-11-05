"""Calculate command-gap timelines with invented timings and no GPU calls."""

import json


def timeline(wait, before_b, b_compute, after_b):
    # G1 and G2: A has finished a command and will be ready again at wait.
    b_start = before_b
    b_done = b_start + b_compute
    # G3 and G4: B must physically return capacity before A can use it.
    capacity_ready = b_done + after_b
    a_start = max(wait, capacity_ready)
    return {
        "wait_s": wait,
        "handoff_s": before_b + after_b,
        "b_compute_s": b_compute,
        "b_done_s": b_done,
        "a_start_s": a_start,
        "a_delay_s": a_start - wait,
        "fits_without_delaying_a": capacity_ready <= wait,
    }


if __name__ == "__main__":
    print("Teaching simulation. Common A startup costs are omitted from both alternatives.")
    for case in ((20, 2, 1, 1), (1, 0.05, 10, 0.05), (0.2, 0.5, 0.2, 0.5)):
        print(json.dumps(timeline(*case)))
