"""Teach AIRS Pass@5 with fixed trusted functions, without executing submitted code."""

import json


def main():
    # Diagram [1]/[2]: two toy problems, each with five fixed candidate programs.
    problems = [
        {
            "name": "sum",
            "tests": [(2, 3, 5), (0, 0, 0), (-4, 1, -3)],
            "candidates": [
                lambda a, b: 5,
                lambda a, b: a - b,
                lambda a, b: a + b,
                lambda a, b: a * b,
                lambda a, b: 0,
            ],
        },
        {
            "name": "maximum",
            "tests": [(7, 2, 7), (-1, -3, -1), (0, 4, 4)],
            "candidates": [
                lambda a, b: min(a, b),
                lambda a, b: a - b,
                lambda a, b: a + b,
                lambda a, b: a * b,
                lambda a, b: 0,
            ],
        },
    ]
    # Diagram [3]/[4]: one candidate must pass every test for its problem to count.
    outcomes = {
        p["name"]: [all(fn(a, b) == expected for a, b, expected in p["tests"]) for fn in p["candidates"]]
        for p in problems
    }
    score = sum(any(results) for results in outcomes.values()) / len(problems)
    assert outcomes == {"sum": [False, False, True, False, False], "maximum": [False] * 5}
    assert score == 0.5
    print(json.dumps({"candidate_passes_all_tests": outcomes, "Pass@5": score}, indent=2))


if __name__ == "__main__":
    main()
