"""Select a measured static capacity and validate interpolation on a held-out run."""

import argparse
import heapq
import json
import statistics
from pathlib import Path


def interpolate(points, capacity):
    """Interpolate within measured support, rejecting extrapolation."""
    if capacity in points:
        return points[capacity]
    lower = max((key for key in points if key < capacity), default=None)
    upper = min((key for key in points if key > capacity), default=None)
    if lower is None or upper is None:
        raise ValueError("Requested capacity is outside measured support.")
    fraction = (capacity - lower) / (upper - lower)
    return points[lower] + fraction * (points[upper] - points[lower])


def simulate_batch(profiles, capacity, groups=4, model_request_s=6.0, devices=2):
    """Replay model/train/model/verify/model dependencies with first-fit placement.

    Service profiles are conditional on the configured cap, not instantaneous
    occupancy. This approximation is checked against the integrated experiment.
    The model intentionally excludes shared GPUs without coexecution profiles.
    """
    available = [0] * devices
    events = []
    waiting = []
    completed = []
    serial = 0
    for episode in range(groups * 2):
        heapq.heappush(events, (model_request_s, serial, "request", episode, "train", -1))
        serial += 1
    while events:
        now, _, kind, episode, phase, gpu = heapq.heappop(events)
        if kind == "request":
            waiting.append((episode, phase))
        else:
            available[gpu] -= 1
            if phase == "train":
                heapq.heappush(events, (now + model_request_s, serial, "request", episode, "verify", -1))
                serial += 1
            else:
                completed.append(now + model_request_s)
        while waiting:
            gpu = next((i for i, count in enumerate(available) if count < capacity), None)
            if gpu is None:
                break
            episode, phase = waiting.pop(0)
            category = "verify" if phase == "verify" else ("long" if episode // 2 == groups - 1 else "short")
            duration = interpolate(profiles[category], capacity)
            available[gpu] += 1
            heapq.heappush(events, (now + duration, serial, "done", episode, phase, gpu))
            serial += 1
    return max(completed)


def analyze(evidence, groups=4):
    if evidence["status"] != "complete" or not evidence.get("physical_release_verified"):
        raise ValueError("Calibration must complete and release its GPU before capacity selection.")
    trials = evidence["trials"]
    profiles = {key: {} for key in ("short", "long", "verify")}
    times = {}
    summaries = []
    training_caps = set(evidence["calibration"])
    for trial in trials:
        cap = trial["cap"]
        categories = {key: [] for key in profiles}
        for row in trial["commands"]:
            category = (
                "verify" if row["mode"] == "verify" else ("long" if row["result"]["updates"] == 896 else "short")
            )
            categories[category].append(row["finished_at"] - row["admitted_at"])
        summaries.append(
            {
                "index": trial["index"],
                "cap": cap,
                "elapsed_s": trial["elapsed_s"],
                "command_service_mean_s": {key: statistics.mean(values) for key, values in categories.items()},
                "commands": len(trial["commands"]),
                "total_compute_s": sum(row["result"]["compute_s"] for row in trial["commands"]),
                "total_preparation_s": sum(row["result"]["preparation_s"] for row in trial["commands"]),
            }
        )
        # Keep the repeated run out of both fitting and candidate selection.
        if cap in training_caps and cap not in times:
            times[cap] = trial["elapsed_s"]
            for category, values in categories.items():
                profiles[category][cap] = statistics.mean(values)
    predictions = {cap: simulate_batch(profiles, cap, groups) for cap in sorted(times)}
    best = min(predictions.values())
    selected = min(cap for cap, estimate in predictions.items() if estimate <= best * 1.05)
    validation = []
    for row in summaries:
        cap = row["cap"]
        if cap not in evidence["held_out"]:
            continue
        predicted = interpolate(times, cap)
        validation.append(
            {
                "cap": cap,
                "predicted_elapsed_s": predicted,
                "observed_elapsed_s": row["elapsed_s"],
                "relative_absolute_error": abs(predicted / row["elapsed_s"] - 1),
                "service_predictions_s": {key: interpolate(values, cap) for key, values in profiles.items()},
                "service_observed_s": row["command_service_mean_s"],
            }
        )
    repeats = []
    for cap in evidence["repeat"]:
        selected_rows = [row for row in summaries if row["cap"] == cap]
        repeats.append(
            {
                "cap": cap,
                "elapsed_s": [row["elapsed_s"] for row in selected_rows],
                "relative_change": selected_rows[-1]["elapsed_s"] / selected_rows[0]["elapsed_s"] - 1,
            }
        )
    return {
        "scope": "Offline cap selection. No unmeasured multi-vLLM or shared-GPU prediction.",
        "trials": summaries,
        "profiles": profiles,
        "held_out_validation": validation,
        "drift_checks": repeats,
        "selected_static_cap": selected,
        "selection": "Smallest cap within 5% of best predicted batch time.",
        "batch_predictions_s": predictions,
        "groups": groups,
        "episodes_per_batch": groups * 2,
        "memory_filter": {"gpu_budget_mib": 92160, "command_budget_mib": 16384, "max_cap": 5},
        "shared_cap": min(selected, 2),
        "limitations": [
            "One calibration run per cap, one held-out cap, one repeated cap.",
            "Profiled service includes setup and changing occupancy within each batch.",
            "Integrated prediction omits overlapping batches and weight synchronization.",
            "5% is an engineering tolerance, not a confidence interval.",
        ],
    }


def plot(result, output):
    """Show measured batches separately from the target-batch predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    first = {}
    for row in result["trials"]:
        first.setdefault(row["cap"], row["elapsed_s"])
    axes[0].plot(sorted(first), [first[key] for key in sorted(first)], marker="o", label="Measured first run")
    for row in result["held_out_validation"]:
        axes[0].scatter([row["cap"]], [row["predicted_elapsed_s"]], marker="x", s=70, label="Held-out interpolation")
    for row in result["drift_checks"]:
        axes[0].scatter([row["cap"]], [row["elapsed_s"][-1]], marker="D", label="Repeat run")
    axes[0].set(
        xlabel="Command cap per GPU", ylabel="Seconds", xticks=range(1, 6), title="Measured: 6 tasks on one GPU"
    )
    axes[0].legend()
    values = {int(key): value for key, value in result["batch_predictions_s"].items()}
    keys = sorted(values)
    axes[1].bar(
        [str(key) for key in keys],
        [values[key] for key in keys],
        color=["#397dc4" if key == result["selected_static_cap"] else "#b6bec9" for key in keys],
    )
    axes[1].set(
        xlabel="Command cap per GPU",
        ylabel="Predicted seconds",
        title=f"Replay: {result['episodes_per_batch']} trajectories on two GPUs",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()), args.groups)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.figure:
        plot(result, args.figure)
    print(json.dumps(result, indent=2))
