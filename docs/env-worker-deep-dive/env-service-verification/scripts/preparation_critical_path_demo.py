"""Illustrate preparation overlap with virtual time and two independent CPU slots."""


def compare(prepare_s, gpu_s, control_s):
    """Model two simultaneous commands on one GPU without real sleeps or devices."""
    # S1 and S2: Baseline commands prepare serially while holding GPU admission.
    baseline_wall = 2 * (prepare_s + gpu_s)
    # S2 and S3: Both preparations finish before either prepared command starts.
    prepared_ready = prepare_s + control_s
    # S4 and S5: GPU stages serialize, then each command immediately returns.
    prepared_wall = prepared_ready + 2 * gpu_s
    return baseline_wall, prepared_wall, baseline_wall, 2 * gpu_s


def main():
    """Print teaching assumptions, never estimates of real hardware performance."""
    cases = [
        ("small preparation, no added control", 2, 30, 0),
        ("small preparation, added control", 2, 30, 2),
        ("large preparation, added control", 20, 30, 2),
    ]
    print("Teaching simulation. Two commands, two CPU slots, one GPU. All times are assumed seconds.")
    for label, prepare, gpu, control in cases:
        baseline, prepared, baseline_held, prepared_held = compare(prepare, gpu, control)
        print(f"{label}: wall {baseline} -> {prepared}, reserved GPU seconds {baseline_held} -> {prepared_held}")


if __name__ == "__main__":
    main()
