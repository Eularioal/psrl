"""Illustrate revision-dependent setup costs with invented CPU data."""


def choose(revision, policy):
    # P2: A warm process contains revision 1. No actual process is resumed here.
    nodes = [("gpu0", 20, 1), ("gpu1", 0, None)]
    choices = []
    for name, queue, cached_revision in nodes:
        valid = cached_revision == revision
        estimated_warm = cached_revision is not None if policy == "revision-blind" else valid
        estimate = queue + (2 if estimated_warm else 30) + 3
        actual = queue + (2 if valid else 30) + 3
        choices.append((estimate, name, actual))
    _, selected, actual = min(choices)
    return selected, actual


if __name__ == "__main__":
    print("Teaching simulation. No GPU, state restoration, cache validation or EnvRouter execution.")
    for revision in (1, 2):
        for policy in ("revision-blind", "revision-aware"):
            print(revision, policy, choose(revision, policy))
    assert choose(1, "revision-aware")[0] == "gpu0"
    assert choose(2, "revision-aware")[0] == "gpu1"
