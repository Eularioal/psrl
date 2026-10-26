#!/usr/bin/env python3
"""Compare state retention and cooperative suspension using logical seconds only."""

import json


def compare(wait, suspend, borrowed, restore, *, queue=0, next_gpu=5):
    """Assume inference uses another GPU and A and B cannot fit together."""
    if min(wait, suspend, borrowed, restore, queue, next_gpu) < 0:
        raise ValueError("Durations must be nonnegative.")
    # H1/H2: The model starts immediately while A saves and releases its state.
    other_done = suspend + borrowed
    # H3/H4: Reactive restoration waits for both the instruction and returned capacity.
    reactive_ready = max(wait, other_done) + queue + restore
    # Optional prefetch restores only state independent of the next instruction.
    prefetch_ready = max(wait, other_done + queue + restore)
    return {
        "kind": "teaching_model_not_gpu_measurement",
        "input_s": dict(wait=wait, suspend=suspend, borrowed=borrowed, restore=restore, queue=queue),
        "a_ready_keep_s": wait,
        "a_ready_reactive_s": reactive_ready,
        "a_ready_prefetch_s": prefetch_ready,
        "b_done_keep_s": wait + next_gpu + borrowed,
        "b_done_yield_s": other_done,
        "both_done_keep_s": wait + next_gpu + borrowed,
        "both_done_reactive_s": max(other_done, reactive_ready + next_gpu),
        "both_done_prefetch_s": max(other_done, prefetch_ready + next_gpu),
    }


def main():
    cheap = compare(60, 3, 20, 4)
    assert cheap["a_ready_reactive_s"] == 64
    assert cheap["a_ready_prefetch_s"] == 60
    assert cheap["both_done_keep_s"] == 85
    assert cheap["both_done_reactive_s"] == 69
    assert cheap["both_done_prefetch_s"] == 65
    expensive = compare(10, 8, 5, 12)
    assert expensive["a_ready_reactive_s"] == 25
    assert expensive["both_done_keep_s"] == 20
    assert expensive["both_done_reactive_s"] == 30
    for name, result in (("cheap_switch_long_wait", cheap), ("expensive_switch_short_wait", expensive)):
        print(json.dumps({"case": name, **result}))
    queued = compare(60, 3, 20, 4, queue=50)
    assert queued["a_ready_reactive_s"] == 114
    print(json.dumps({"case": "extra_queue_delays_resume", **queued}))
    # Same-card mutual exclusion serializes environment release and model execution.
    same_card = dict(env_suspend=3, model_wake=2, generation=60, model_sleep=2, env_restore=4)
    assert sum(same_card.values()) == 71
    print(json.dumps({"case": "same_card_serial_handoff", "parts_s": same_card, "a_ready_s": 71}))


if __name__ == "__main__":
    main()
