"""Compute bounded local scores from frozen-judge verdict probabilities."""

import math


def rubric_score(rubrics, probabilities):
    """Apply signed rubric weights and normalize by available positive points."""
    if len(rubrics) != len(probabilities) or not rubrics:
        raise ValueError("Every rubric requires one judge probability.")
    denominator = sum(max(0, item["points"]) for item in rubrics)
    if denominator <= 0 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Rubrics require positive points and finite judge probabilities.")
    value = sum(item["points"] * p for item, p in zip(rubrics, probabilities, strict=True)) / denominator
    return min(1.0, max(0.0, value))
