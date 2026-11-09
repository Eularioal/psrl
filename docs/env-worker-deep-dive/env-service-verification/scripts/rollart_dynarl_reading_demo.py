"""Calculate invented scheduling examples without GPUs or external services."""


def main():
    # R2: Each model turn takes one second and generation has sufficient capacity.
    env_times = (1, 5)
    rounds = 2
    synchronous = rounds * (1 + max(env_times))
    independent = tuple(rounds * (1 + delay) for delay in env_times)
    print("Trajectory barrier:", (synchronous, synchronous), "independent:", independent)
    assert independent == (4, 12) and synchronous == 12

    # R4: This example allows stale data and ignores weight transfer costs.
    batches, rollout, train = 3, 12, 4
    serial = batches * (rollout + train)
    pipeline = rollout + train + (batches - 1) * max(rollout, train)
    print("Three batches: serial =", serial, "overlapped =", pipeline)
    assert (serial, pipeline) == (48, 40)

    # D3/D4: Both stages process comparable work units with invented rates.
    for a_cards, a_rate in ((4, 10), (3, 9), (2, 6)):
        b_cards = 8 - a_cards
        print("Allocation:", (a_cards, b_cards), "bottleneck rate:", min(a_rate, b_cards))
    for work in (12, 120):
        keep = work / 4
        move = 2 + work / 6
        print("Remaining work:", work, "keep:", keep, "migrate:", move)
    assert 2 + 12 / 6 > 12 / 4
    assert 2 + 120 / 6 < 120 / 4

    # L2/L3: Preparation and queue waiting overlap only in this explicit model.
    home = 12 + 4
    local = 2 + 4
    remote = max(2, 20) + 4
    print("Placement: home =", home, "local =", local, "remote cold =", remote)
    assert (home, local, remote) == (16, 6, 24)
    print("Teaching arithmetic passed. No paper algorithm or PSRL execution was reproduced.")


if __name__ == "__main__":
    main()
