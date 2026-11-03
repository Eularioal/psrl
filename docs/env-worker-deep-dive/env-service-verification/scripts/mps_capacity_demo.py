"""Illustrate ideal compute sharing without invoking CUDA or simulating its scheduler."""

import json


def compare(demand, capacity=80, solo_seconds=10, jobs=2):
    """Compute a teaching upper bound assuming linear scaling and no overhead."""
    # Model execution at node 5 of the backend diagram after admission succeeds.
    allocation = min(demand, capacity / jobs)
    serial = jobs * solo_seconds
    shared = solo_seconds * demand / allocation
    return {
        "units_per_job_alone": demand,
        "serial_seconds": serial,
        "ideal_shared_seconds": shared,
        "ideal_speedup": round(serial / shared, 3),
    }


if __name__ == "__main__":
    results = [compare(demand) for demand in (20, 60, 80)]
    assert [row["ideal_shared_seconds"] for row in results] == [10, 15, 20], "Unexpected model times."
    print(json.dumps({"teaching_model_only": True, "results": results}, indent=2))
