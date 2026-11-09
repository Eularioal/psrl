"""Illustrate measurement boundaries with deterministic CPU arithmetic."""

from math import ceil
from statistics import median


def drain_example():
    """Model a queued sibling operation that outlives its submitting process activity."""
    # Diagram steps 2 and 3 must precede the measurement at step 6.
    gate_closed_ms = 2.0
    sibling_gpu_finished_ms = 5.0
    candidate_ms = 1.0
    unsafe_start = gate_closed_ms
    safe_start = max(gate_closed_ms, sibling_gpu_finished_ms)
    print("Queued GPU work survives a CPU freeze:")
    print(f"  Freeze-only window: [{unsafe_start:.1f}, {unsafe_start + candidate_ms:.1f}] ms, contaminated.")
    print(f"  Drained window: [{safe_start:.1f}, {safe_start + candidate_ms:.1f}] ms, uncontended in this model.")
    print(
        f"  Drain wait: {safe_start - gate_closed_ms:.1f} ms, excluded from kernel time but charged to service time."
    )


def ranking_example():
    """Separate reproducibility from accuracy using fixed interference factors."""
    native_ms = {"A": 10.0, "B": 10.2}
    slowdown = {"A": 1.10, "B": 1.01}
    shared_samples = {name: [time * slowdown[name]] * 100 for name, time in native_ms.items()}
    shared_medians = {name: median(values) for name, values in shared_samples.items()}
    native_winner = min(native_ms, key=native_ms.get)
    shared_winner = min(shared_medians, key=shared_medians.get)
    print("\nRepeating biased observations does not remove the bias:")
    print(f"  Native: {native_ms}, winner={native_winner}.")
    print(f"  Median of 100 identical shared samples: {shared_medians}, winner={shared_winner}.")
    print("  Zero sample variance is compatible with the wrong ranking.")


def pipeline_example():
    """Assume parallel CPU preparation and serialize only measurement regions."""
    commands = 4
    preparation_ms = 90.0
    measurement_ms = 10.0
    transition_ms = 1.0
    command_exclusive_ms = commands * (preparation_ms + measurement_ms)
    region_exclusive_ms = preparation_ms + commands * (transition_ms + measurement_ms)
    print("\nIdealized pipeline with sufficient CPU cores:")
    print(f"  Whole-command exclusivity: {command_exclusive_ms:.1f} ms.")
    print(f"  Region exclusivity including transitions: {region_exclusive_ms:.1f} ms.")
    print(f"  Illustrative speedup: {command_exclusive_ms / region_exclusive_ms:.3f}x.")
    print("  The model omits correctness GPU work, memory pressure, CPU freezing and context transport.")


def percentile_example():
    """Show why a one-sided percentile is not an absolute error bound."""
    ratios = [0.70] * 94 + [1.02] + [1.50] * 5
    ordered = sorted(ratios)
    p95 = ordered[ceil(0.95 * len(ordered)) - 1]
    within_five_percent = sum(abs(ratio - 1.0) <= 0.05 for ratio in ratios) / len(ratios)
    print("\nSynthetic ratios using the nearest-rank percentile convention:")
    print(f"  P95 inflation: {(p95 - 1.0) * 100:.1f}%.")
    print(f"  Within 5% absolute relative error: {within_five_percent * 100:.1f}%.")
    print("  A small upper-tail inflation alone can hide systematic underestimation.")


def main():
    """Print teaching examples without accessing a GPU or modifying system state."""
    print("Teaching simulation only. No KREX implementation or GPU performance reproduction.\n")
    drain_example()
    ranking_example()
    pipeline_example()
    percentile_example()


if __name__ == "__main__":
    main()
