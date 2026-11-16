"""Show when mandatory calibration helps or loses to a strong prior on two GPUs."""


def finish(order, durations, *, overhead=0.0, start=0.0):
    """Model exclusive jobs on two identical GPUs without transfer overlap."""
    free = [start, start]
    for job in order:
        gpu = min(range(2), key=lambda index: free[index])
        free[gpu] += durations[job] + overhead
    return max(free)


def main():
    # Diagram 1 fixes all candidates before scheduling, with no future rollout oracle.
    order = ["A", "B", "C", "D"]
    remaining = dict(zip(order, [3, 7, 3, 7], strict=True))
    complete = {name: duration + 1 for name, duration in remaining.items()}
    # Diagram 2 spends one mandatory evaluation second per candidate on two GPUs.
    calibration_done = finish(order, dict.fromkeys(order, 1))
    calibrated_order = sorted(order, key=lambda name: -remaining[name])
    results = {
        "scope": "Teaching assumptions, not measured predictor accuracy or GPU speedup",
        "batched_fifo": finish(order, complete),
        "correct_static_prior": finish(calibrated_order, complete),
        "perfect_calibration_low_handoff": finish(calibrated_order, remaining, overhead=0.2, start=calibration_done),
        "uninformative_calibration": finish(order, remaining, overhead=0.2, start=calibration_done),
        "perfect_calibration_expensive_handoff": finish(
            calibrated_order, remaining, overhead=3, start=calibration_done
        ),
    }
    for name, value in results.items():
        print(f"{name}: {value}")
    assert results["correct_static_prior"] < results["perfect_calibration_low_handoff"], (
        "A correct cheap prior must beat unnecessary calibration handoff in this example."
    )
    assert results["perfect_calibration_expensive_handoff"] > results["batched_fifo"], (
        "Handoff cost can eliminate the scheduling gain."
    )


if __name__ == "__main__":
    main()
