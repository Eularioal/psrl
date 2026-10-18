"""Apply APPS output normalization only inside the trusted judge."""

import math


def integer_keys(value):
    """Restore numeric JSON keys without rejecting dictionaries with string keys."""
    try:
        return {int(key): item for key, item in value.items()}
    except (ValueError, TypeError, OverflowError):
        return value


def matches(actual, expected, function_call):
    """Match JSON return values or APPS whitespace and numeric stdout variants."""
    if function_call:
        if isinstance(expected, dict):
            expected = [integer_keys(expected)]
        if isinstance(expected, list) and expected and isinstance(expected[0], dict):
            expected = [integer_keys(expected[0])]
        if isinstance(actual, dict):
            actual = integer_keys(actual)
        return actual == expected or (isinstance(expected, list) and bool(expected) and actual == expected[0])
    expected = "\n".join(expected) if isinstance(expected, list) else expected
    if actual.strip() == expected.strip():
        return True
    observed = [line.strip() for line in actual.splitlines() if line.strip()]
    wanted = [line.strip() for line in expected.splitlines() if line.strip()]
    if observed == wanted:
        return True
    try:
        if len(observed) == len(wanted) and all(
            abs(float(a) - float(b)) <= 1e-8 + 1e-5 * abs(float(b)) for a, b in zip(observed, wanted, strict=True)
        ):
            return True
    except ValueError:
        pass
    # APPS also accepts unordered token lines and rounded numeric token sets.
    if {frozenset(line.split()) for line in observed} == {frozenset(line.split()) for line in wanted}:
        return True
    try:

        def numeric_lines(lines):
            values = [[float(token) for token in line.split()] for line in lines]
            if not all(math.isfinite(v) for row in values for v in row):
                raise ValueError("Nonfinite output is not a rounded numeric match.")
            return {frozenset(round(v, 3) for v in row) for row in values}

        return numeric_lines(observed) == numeric_lines(wanted)
    except ValueError:
        return False
