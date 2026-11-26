"""Read longer resident sharing evidence and preserve paired performance and output checks."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from analyze import metrics


def summarize(evidence):
    """Separate request patterns and retain every completed repeat."""
    rows = [row for row in evidence["trials"] if not row["warmup"]]
    groups = []
    pairs = []
    raw = []
    requests = []
    for row in rows:
        values = metrics(row)
        values["environment_without_pause_s"] = values["training_wall_s"] - values["paused_s"]
        values["updates_during_generation"] = sum(
            query["engine_started_ns"] <= timestamp <= query["engine_ended_ns"]
            for timestamp in row["training"]["update_times_ns"]
            for query in row["queries"]
        )
        values["score"] = row["verified"]["reward"]
        raw.append({"id": row["id"], "profile": row["profile"], "policy": row["policy"], **values})
        for index, query in enumerate(row["queries"]):
            requests.append(
                {
                    "trial_id": row["id"],
                    "profile": row["profile"],
                    "policy": row["policy"],
                    "repetition": row["repetition"],
                    "query_index": index,
                    "engine_s": query["engine_duration_s"],
                    "response_s": (query["ended_ns"] - query["scheduled_ns"]) / 1e9,
                    "handoff_s": (query["admitted_ns"] - query["submitted_ns"]) / 1e9,
                    "updates_during_generation": sum(
                        query["engine_started_ns"] <= timestamp <= query["engine_ended_ns"]
                        for timestamp in row["training"]["update_times_ns"]
                    ),
                }
            )
    for profile in ("spaced", "burst"):
        for policy in ("serial", "existing", "cooperative"):
            selected = [row for row in raw if row["profile"] == profile and row["policy"] == policy]
            if not selected:
                continue
            names = set(selected[0]) - {"id", "profile", "policy"}
            groups.append(
                {
                    "profile": profile,
                    "policy": policy,
                    "n": len(selected),
                    "median": {name: statistics.median(row[name] for row in selected) for name in names},
                    "range": {
                        name: [min(row[name] for row in selected), max(row[name] for row in selected)]
                        for name in names
                    },
                }
            )
        for repetition in sorted({row["repetition"] for row in rows}):
            selected = {
                row["policy"]: metrics(row)
                for row in rows
                if row["profile"] == profile and row["repetition"] == repetition
            }
            if len(selected) != 3:
                continue
            ratios = {}
            for candidate, reference in (
                ("existing", "serial"),
                ("cooperative", "serial"),
                ("cooperative", "existing"),
            ):
                ratios[f"{candidate}_over_{reference}"] = {
                    name: selected[candidate][name] / selected[reference][name]
                    for name in ("makespan_s", "mean_response_s", "mean_engine_s", "training_wall_s")
                }
            pairs.append({"profile": profile, "repetition": repetition, "ratios": ratios})
    return {
        "passed": evidence["passed"],
        "error": evidence.get("error"),
        "commit": evidence["commit"],
        "comparison_kind": evidence["comparison_kind"],
        "source_sha256": evidence["source_sha256"],
        "analysis_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("analyze.py"))
        },
        "data_sha256": evidence.get("data_sha256"),
        "plan": evidence["plan"],
        "calibration": evidence.get("calibration"),
        "correctness": evidence.get("correctness"),
        "spaced_request_interval_s": evidence.get("spaced_request_interval_s"),
        "initialization_s": evidence.get("initialization_s"),
        "elapsed_s": evidence.get("elapsed_s"),
        "model_profile": evidence.get("model_profile"),
        "model_resident": evidence.get("model_resident"),
        "train_ready": evidence.get("train_ready"),
        "verify_ready": evidence.get("verify_ready"),
        "complete_measured_trials": len(rows),
        "expected_measured_trials": 6 * evidence["plan"]["rounds"],
        "physical_release_confirmed": evidence.get("physical_release_confirmed"),
        "cleanup_errors": evidence["cleanup_errors"],
        "released_tool_batches": evidence.get("released_tool_batches", []),
        "groups": groups,
        "paired_ratios": pairs,
        "trial_metrics": raw,
        "request_metrics": requests,
        "tensor_hashes": sorted({row["training"]["tensor_sha256"] for row in rows}),
        "scores": sorted({row["verified"]["reward"] for row in rows}),
    }


def plot(result, output):
    """Show completion, model generation, and tool duration on the same workload."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    colors = ("#777777", "#276fa8", "#d97831")
    for axis, name, title in zip(
        axes,
        ("makespan_s", "mean_engine_s", "training_wall_s"),
        ("Fixed-work completion", "Generation per request", "Environment elapsed"),
        strict=True,
    ):
        for index, policy in enumerate(("serial", "existing", "cooperative")):
            selected = {group["profile"]: group for group in result["groups"] if group["policy"] == policy}
            positions = [position + (index - 1) * 0.23 for position in range(2)]
            values = [selected[profile]["median"][name] for profile in ("spaced", "burst")]
            errors = [
                [values[i] - selected[profile]["range"][name][0] for i, profile in enumerate(("spaced", "burst"))],
                [selected[profile]["range"][name][1] - values[i] for i, profile in enumerate(("spaced", "burst"))],
            ]
            axis.bar(
                positions, values, width=0.21, yerr=errors, capsize=2, color=colors[index], label=policy.capitalize()
            )
        axis.set_xticks((0, 1), ("Spaced", "Burst"))
        axis.set_ylabel("Seconds")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    arguments = parser.parse_args()
    summary = summarize(json.loads(arguments.evidence.read_text()))
    arguments.output.write_text(json.dumps(summary, indent=2) + "\n")
    if arguments.figure:
        plot(summary, arguments.figure)
    for group in summary["groups"]:
        values = group["median"]
        print(
            f"{group['profile']:6} {group['policy']:11} n={group['n']} "
            f"total={values['makespan_s']:.3f}s engine={values['mean_engine_s']:.3f}s "
            f"response={values['mean_response_s']:.3f}s tool={values['training_wall_s']:.3f}s "
            f"updates_during_generation={values['updates_during_generation']}"
        )
