"""Summarize completed reservation comparisons and causal loan forecasts."""

import argparse
import json
import statistics
from pathlib import Path


def summarize_reservations(document):
    trials = []
    for trial in document["trials"]:
        if "total_s" not in trial:
            continue
        a, b = trial["sessions"]["a"], trial["sessions"]["b"]
        start = a["commands"][0]["started_s"]
        commands = a["commands"] + b["commands"]
        intervals = []
        for command in commands:
            record = command["reservation"]
            end = record.get("released_at", record.get("execution_finished_at"))
            intervals.append((record["admitted_at"], end))
        intervals.sort()
        held_s = sum(end - begin for begin, end in intervals)
        overlaps = any(left[1] > right[0] for left, right in zip(intervals, intervals[1:], strict=False))
        work_s = max(a["completed_s"], b["completed_s"]) - start
        trials.append(
            dict(
                id=trial["id"],
                mode=trial["mode"],
                gap_s=trial["gap_s"],
                total_s=trial["total_s"],
                work_and_cleanup_s=max(a["closed_s"], b["closed_s"]) - start,
                work_s=work_s,
                a_completion_s=a["completed_s"] - start,
                gpu_command_held_s=held_s,
                outside_gpu_commands_s=work_s - held_s,
                gpu_command_intervals_overlap=overlaps,
                initial_admission_and_setup_s=a["admission_wait_s"] + a["setup_s"],
                a_next_gpu_wait_s=a["commands"][1]["reservation"]["admitted_at"]
                - a["commands"][1]["reservation"]["requested_at"],
                b_session_admission_wait_s=b["admission_wait_s"],
                chosen_modes={
                    name: session["lease"]["gpu_reservation"] for name, session in trial["sessions"].items()
                },
                train_hashes=[c["result"]["tensor_sha256"] for c in commands if c["phase"] == "train"],
                scores=[c["result"]["reward"] for c in commands if c["phase"] == "verify"],
                command_compute_s=sum(c["result"]["compute_s"] for c in commands),
                command_preparation_s=sum(c["result"]["preparation_s"] for c in commands),
            )
        )
    return dict(
        status=document["status"],
        trials=trials,
        completed_trials=len(trials),
        incomplete_trials=[t["id"] for t in document["trials"] if "total_s" not in t],
        completed_weights_agree=len({h for t in trials for h in t["train_hashes"]}) == 1,
        completed_scores_agree=len({s for t in trials for s in t["scores"]}) == 1,
        physical_release_confirmed=document.get("physical_release_confirmed"),
        boundary=document["boundary"],
        commit=document["commit"],
    )


def summarize_predictions(document):
    trials = []
    for trial in document["trials"]:
        if "sample" not in trial:
            continue
        trials.append({k: v for k, v in trial.items() if k not in ("borrower", "sleeping_gpu", "query", "key")})
    heldout = [r for r in trials if not r["calibration"] and r["forecast"] is not None and r["action"] == "lend"]
    errors = [abs(r["prediction_error_s"]) for r in heldout]
    completed = [r for r in document["trials"] if "sample" in r]
    return dict(
        status=document["status"],
        trials=trials,
        formula_max_error_s=max((abs(r["formula_error_s"]) for r in trials if "formula_error_s" in r), default=None),
        heldout_loans=len(heldout),
        heldout_mae_s=statistics.mean(errors) if errors else None,
        automatic_heldout_loans=sum(not r["forced"] for r in heldout),
        forced_heldout_loans=sum(r["forced"] for r in heldout),
        completed_cycles=len(trials),
        incomplete_cycles=[r["index"] for r in document["trials"] if "sample" not in r],
        completed_weights_agree=len({r["borrower"]["result"]["tensor_sha256"] for r in completed}) == 1,
        completed_tokens_agree=all(r["query"]["token_ids"] == document["warm_query"]["token_ids"] for r in completed),
        physical_release_confirmed=document.get("physical_release_confirmed"),
        boundary=document["boundary"],
        commit=document["commit"],
    )


def plot_summary(document, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = document["reservations"]["trials"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
    colors = ("#4275a8", "#e5a13a", "#aeb7bf")
    y = range(len(rows))
    held = [r["gpu_command_held_s"] for r in rows]
    outside = [r["outside_gpu_commands_s"] for r in rows]
    cleanup = [r["work_and_cleanup_s"] - r["work_s"] for r in rows]
    axes[0].barh(y, held, color=colors[0], label="GPU reservation held by commands")
    axes[0].barh(y, outside, left=held, color=colors[1], label="Outside GPU commands")
    axes[0].barh(
        y, cleanup, left=[a + b for a, b in zip(held, outside, strict=True)], color=colors[2], label="Final cleanup"
    )
    axes[0].set_yticks(list(y), [f"{r['gap_s']:g}s gap / {r['mode']}" for r in rows])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Seconds from A's first command", title="Real PostTrain: one pass per configuration")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, frameon=False)

    heldout = [
        r
        for r in document["predictions"]["trials"]
        if not r["calibration"] and r["forecast"] and r["action"] == "lend"
    ]
    positions = list(range(len(heldout)))
    axes[1].bar(
        positions, [r["forecast"]["point_wait_s"] for r in heldout], color=colors[0], width=0.5, label="Forecast"
    )
    axes[1].scatter(
        positions, [r["observed_extra_wait_s"] for r in heldout], color="#be3f40", zorder=3, label="Observed"
    )
    labels = [
        f"{r['profile'].capitalize()} gap\n{'forced observation' if r['forced'] else 'automatic loan'}"
        for r in heldout
    ]
    axes[1].set_xticks(positions, labels)
    axes[1].set(ylabel="A's extra waiting time (seconds)", title="Held-out loans: only two complete samples")
    axes[1].legend()
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y" if axis is axes[1] else "x", alpha=0.2)
        axis.set_axisbelow(True)
    figure.savefig(output)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservations", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = {}
    for name, path, function in (
        ("reservations", args.reservations, summarize_reservations),
        ("predictions", args.predictions, summarize_predictions),
    ):
        if path:
            result[name] = function(json.loads(path.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.figure:
        plot_summary(result, args.figure)
    print(json.dumps(result, indent=2))
