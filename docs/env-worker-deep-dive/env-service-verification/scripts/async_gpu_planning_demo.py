"""Calculate teaching timelines for planning, confirmed return, and portable commands."""

import json


def simulate(*, asynchronous, model_s=8, other_gpu_s=2, remote_setup_s=None, portable=True, return_ok=True):
    """Use logical seconds and an idle remote GPU, without calling production services."""
    # Diagram 1 and 2: command output is immutable and return intent is acknowledged at zero.
    return_s = 3
    next_gpu_s = 4
    plan_ready = model_s if asynchronous else return_s + model_s
    result = {"plan_ready_s": plan_ready, "return_confirmed": return_ok}
    if not return_ok:
        # Diagram 3: failure keeps capacity charged and blocks dependent GPU execution.
        result.update(gpu_budget_held=True, other_gpu_start_s=None, next_gpu_start_s=None)
        return result

    # Diagram 3: B can start only after A's physical return, independent of A's planner.
    other_start = return_s
    other_end = other_start + other_gpu_s
    local_start = max(plan_ready, return_s, other_end)
    local_end = local_start + next_gpu_s
    destination, next_start, next_end = "local", local_start, local_end
    if portable and remote_setup_s is not None:
        # Diagram 4: prepare a separate executor after the plan and predecessor are ready.
        remote_start = max(plan_ready, return_s) + remote_setup_s
        remote_end = remote_start + next_gpu_s
        if remote_end < local_end:
            destination, next_start, next_end = "remote", remote_start, remote_end
    assert other_start >= return_s, "B must wait for confirmed return."
    assert next_start >= max(return_s, plan_ready), "A must wait for its plan and confirmed return."
    if destination == "local":
        assert next_start >= other_end, "Local GPU commands must not overlap."
    result.update(
        gpu_budget_held=False,
        other_gpu_start_s=other_start,
        other_gpu_end_s=other_end,
        destination=destination,
        next_gpu_start_s=next_start,
        all_done_s=max(next_end, other_end),
    )
    return result


def main():
    cases = {
        "short_queue_sync": simulate(asynchronous=False),
        "short_queue_async": simulate(asynchronous=True),
        "long_queue_sync": simulate(asynchronous=False, other_gpu_s=10),
        "long_queue_async": simulate(asynchronous=True, other_gpu_s=10),
        "cheap_remote": simulate(asynchronous=True, other_gpu_s=10, remote_setup_s=2),
        "expensive_remote": simulate(asynchronous=True, other_gpu_s=10, remote_setup_s=9),
        "nonportable": simulate(asynchronous=True, other_gpu_s=10, remote_setup_s=2, portable=False),
        "early_plan": simulate(asynchronous=True, model_s=1, other_gpu_s=0),
        "failed_return": simulate(asynchronous=True, return_ok=False),
    }
    assert cases["short_queue_sync"]["all_done_s"] == 15, "Serial teaching timeline must take 15 seconds."
    assert cases["short_queue_async"]["all_done_s"] == 12, "Overlapping planning must save 3 seconds here."
    assert cases["long_queue_sync"]["all_done_s"] == cases["long_queue_async"]["all_done_s"] == 17
    assert cases["cheap_remote"]["all_done_s"] == 14
    assert cases["expensive_remote"]["destination"] == cases["nonportable"]["destination"] == "local"
    assert cases["early_plan"]["next_gpu_start_s"] == 3
    assert cases["failed_return"]["gpu_budget_held"] and cases["failed_return"]["next_gpu_start_s"] is None
    print(json.dumps({"kind": "teaching_simulation_not_gpu_measurement", "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
