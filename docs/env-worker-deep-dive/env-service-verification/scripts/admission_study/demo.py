"""Explain why unknown durations do not justify unconditional serialization."""

import json


def compare(work, shared_rate):
    serial = sum(work)
    shared = max(work) / shared_rate
    return dict(serial=serial, shared=shared, saving=1 - min(serial, shared) / max(serial, shared))


if __name__ == "__main__":
    cases = {"complementary": compare([10, 10], 0.8), "contention": compare([10, 10], 0.3)}
    assert cases["complementary"]["shared"] == 12.5, "Complementary sharing should finish in 12.5 units."
    assert cases["contention"]["shared"] > cases["contention"]["serial"], (
        "Heavy contention should favor serial execution."
    )
    print(json.dumps(dict(scope="Invented CPU teaching model, not a GPU benchmark.", cases=cases), indent=2))
