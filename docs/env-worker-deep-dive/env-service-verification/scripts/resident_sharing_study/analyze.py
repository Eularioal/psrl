"""Summarize fixed-work resident sharing trials without rerunning GPU work."""

import argparse
import json
import statistics
from pathlib import Path


def metrics(row):
    """Include compute handoff and request backlog in response measurements."""
    queries = row["queries"]
    training = row["training"]
    pauses = {event["epoch"]: event for event in training["events"] if event["state"] == "paused"}
    resumed = {event["epoch"]: event for event in training["events"] if event["state"] == "resumed"}
    wait = [(query["admitted_ns"] - query["submitted_ns"]) / 1e9 for query in queries]
    return {
        "makespan_s": row["makespan_s"],
        "training_wall_s": (training["ended_ns"] - training["started_ns"]) / 1e9,
        "mean_response_s": statistics.mean((q["ended_ns"] - q["scheduled_ns"]) / 1e9 for q in queries),
        "mean_first_output_s": statistics.mean((q["first_output_ns"] - q["scheduled_ns"]) / 1e9 for q in queries),
        "mean_engine_s": statistics.mean(q["engine_duration_s"] for q in queries),
        "max_handoff_s": max(wait),
        "paused_s": sum((resumed[key]["at_ns"] - event["ack_ns"]) / 1e9 for key, event in pauses.items()),
        "pauses": len(pauses),
    }


def summarize(evidence):
    """Keep original and lengthened workloads separate and compare matching rounds."""
    trials = [row for row in evidence["trials"] if not row["warmup"]]
    rows = [{**row, "metrics": metrics(row)} for row in trials]
    groups = []
    pairs = []
    for epochs in (2, 64):
        for profile in ("spaced", "burst"):
            selected = [row for row in rows if row["epochs"] == epochs and row["profile"] == profile]
            for policy in ("serial", "existing", "cooperative"):
                members = [row for row in selected if row["policy"] == policy]
                if not members:
                    continue
                names = members[0]["metrics"]
                groups.append(
                    {
                        "updates": epochs * 4,
                        "profile": profile,
                        "policy": policy,
                        "n": len(members),
                        "median": {name: statistics.median(row["metrics"][name] for row in members) for name in names},
                        "range": {
                            name: [
                                min(row["metrics"][name] for row in members),
                                max(row["metrics"][name] for row in members),
                            ]
                            for name in names
                        },
                    }
                )
            for repetition in sorted({row["repetition"] for row in selected}):
                by_policy = {row["policy"]: row for row in selected if row["repetition"] == repetition}
                if len(by_policy) != 3:
                    continue
                comparisons = {}
                for candidate, reference in (
                    ("existing", "serial"),
                    ("cooperative", "serial"),
                    ("cooperative", "existing"),
                ):
                    comparisons[f"{candidate}_over_{reference}"] = {
                        name: by_policy[candidate]["metrics"][name] / by_policy[reference]["metrics"][name]
                        for name in ("makespan_s", "mean_response_s", "mean_engine_s", "mean_first_output_s")
                    }
                pairs.append(
                    {"updates": epochs * 4, "profile": profile, "repetition": repetition, "ratios": comparisons}
                )
    return {
        "passed": evidence["passed"],
        "physical_release_confirmed": evidence.get("physical_release_confirmed"),
        "error": evidence.get("error"),
        "complete_measured_trials": len(rows),
        "expected_measured_trials": 12 * evidence["plan"]["rounds"],
        "commit": evidence["commit"],
        "comparison_kind": evidence["comparison_kind"],
        "executed_source_sha256": evidence["source_sha256"],
        "elapsed_s": evidence.get("elapsed_s"),
        "spaced_request_interval_s": evidence["spaced_request_interval_s"],
        "model_profile": evidence["model_profile"],
        "groups": groups,
        "paired_ratios": pairs,
        "trial_metrics": [{"id": row["id"], **row["metrics"]} for row in rows],
        "tensor_hashes_by_updates": {
            str(epochs * 4): sorted({row["training"]["tensor_sha256"] for row in rows if row["epochs"] == epochs})
            for epochs in (2, 64)
        },
        "scores_by_updates": {
            str(epochs * 4): sorted({row["verified"]["reward"] for row in rows if row["epochs"] == epochs})
            for epochs in (2, 64)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.evidence.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.figure is not None:
        plot(result, args.figure)
    for group in result["groups"]:
        values = group["median"]
        print(
            f"{group['updates']:3} {group['profile']:6} {group['policy']:11} n={group['n']} "
            f"total={values['makespan_s']:.4f}s response={values['mean_response_s']:.4f}s "
            f"engine={values['mean_engine_s']:.4f}s handoff_max={1000 * values['max_handoff_s']:.2f}ms"
        )


def plot(result, path):
    """Compare paired completion and response ratios with a serial reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [(updates, profile) for updates in (8, 256) for profile in ("spaced", "burst")]
    policies = ("existing", "cooperative")
    colors = ("#276fa8", "#d97831")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), layout="constrained")
    for axis, metric, title in zip(
        axes,
        ("makespan_s", "mean_response_s"),
        ("Fixed-work completion", "Mean inference response"),
        strict=True,
    ):
        for policy_index, policy in enumerate(policies):
            medians = []
            for updates, profile in cases:
                values = [
                    row["ratios"][f"{policy}_over_serial"][metric]
                    for row in result["paired_ratios"]
                    if row["updates"] == updates and row["profile"] == profile
                ]
                medians.append(statistics.median(values))
            positions = [index + (policy_index - 0.5) * 0.35 for index in range(len(cases))]
            axis.bar(positions, medians, width=0.33, color=colors[policy_index], label=policy.capitalize())
        axis.axhline(1, color="#555555", linestyle="--", linewidth=1, label="Compute serial")
        axis.set_xticks(range(4), [f"{updates} updates\n{profile}" for updates, profile in cases])
        axis.set_ylim(0, 1.12)
        axis.set_ylabel("Paired median ratio to serial")
        axis.set_title(title + " (lower is better)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left")
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


if __name__ == "__main__":
    main()
