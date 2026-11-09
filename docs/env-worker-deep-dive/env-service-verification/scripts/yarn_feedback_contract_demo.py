"""Illustrate comparison blocks and state retention using invented CPU data."""

import json
from statistics import mean


def main():
    # Diagram 1: Independent samples can compare different device conditions.
    clean = {"A": 10.0, "B": 10.2}
    scattered = {"A": clean["A"] * 1.10, "B": clean["B"]}
    # Diagram 4: ABBA cancels this assumed linear drift, not arbitrary interference.
    order = "ABBA"
    samples = [(name, clean[name] + position * 0.1) for position, name in enumerate(order)]
    paired = {name: mean(value for candidate, value in samples if candidate == name) for name in clean}
    assert min(scattered, key=scattered.get) == "B"
    assert min(paired, key=paired.get) == "A"
    # Diagram 2: Batching amortizes the assumed cost of entering a measurement window.
    transition_ms, sample_ms = 8, 1
    print(
        json.dumps(
            {
                "scope": "Teaching simulation, not GPU data or an implemented Env policy.",
                "clean_us": clean,
                "scattered_us": scattered,
                "abba_samples_us": samples,
                "abba_means_us": paired,
                "separate_windows_ms": 4 * (transition_ms + sample_ms),
                "one_block_ms": transition_ms + 4 * sample_ms,
                "retention_example": {
                    "device_gib": 96,
                    "resident_a_gib": 60,
                    "request_b_gib": 40,
                    "b_fits_while_a_resident": 60 + 40 <= 96,
                    "assumed_offload_reload_s": 6,
                    "assumed_b_execution_s": 1,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
