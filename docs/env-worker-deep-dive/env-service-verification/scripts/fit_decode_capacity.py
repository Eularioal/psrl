"""Validate a small Roofline residual model on held-out vLLM configurations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


def analyze(hardware, curve):
    config = hardware["model_config"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // heads)
    params = hardware["model_parameters"]
    rows = []
    for row in curve["rows"]:
        context = row["input_tokens"] + 255.5
        batch = row["concurrency"]
        kv_bytes = 2 * layers * config["num_key_value_heads"] * head_dim * context * 2 * batch
        flops = (2 * params + 4 * layers * heads * head_dim * context) * batch
        compute_ms = flops / (hardware["measured_tflops"] * 1e9)
        bandwidth_ms = (params * 2 + kv_bytes) / (hardware["measured_tb_s"] * 1e9)
        floor_ms = max(compute_ms, bandwidth_ms)
        kv_ms = kv_bytes / (hardware["measured_tb_s"] * 1e9)
        training = (row["input_tokens"] == 128 and batch in (1, 4, 16, 64, 256)) or (
            row["input_tokens"] == 2048 and batch in (1, 64)
        )
        rows.append(
            {
                "input_tokens": row["input_tokens"],
                "concurrency": batch,
                "training": training,
                "features": [1, floor_ms, kv_ms],
                "observed_tpot_ms": row["mean_tpot_ms"],
                "roofline_reference_ms": floor_ms,
            }
        )
    fit = [row for row in rows if row["training"]]
    weights, _ = nnls(np.array([row["features"] for row in fit]), np.array([row["observed_tpot_ms"] for row in fit]))
    short_fit = [row for row in fit if row["input_tokens"] == 128]
    short_weights, _ = nnls(
        np.array([row["features"][:2] for row in short_fit]), np.array([row["observed_tpot_ms"] for row in short_fit])
    )
    for row in rows:
        row["predicted_tpot_ms"] = float(np.dot(weights, row["features"]))
        row["short_only_tpot_ms"] = float(np.dot(short_weights, row["features"][:2]))
        row["relative_absolute_error"] = abs(row["predicted_tpot_ms"] / row["observed_tpot_ms"] - 1)
    test = [row for row in rows if not row["training"]]
    long = [row for row in rows if row["input_tokens"] == 2048]
    return {
        "formula": "TPOT_ms = overhead_ms + floor_scale * roofline_ms + kv_scale * kv_traffic_ms",
        "coefficients": dict(zip(("overhead_ms", "floor_scale", "kv_scale"), weights.tolist())),
        "training_configurations": len(fit),
        "held_out_configurations": len(test),
        "rows": rows,
        "held_out_mean_absolute_relative_error": float(np.mean([row["relative_absolute_error"] for row in test])),
        "held_out_max_absolute_relative_error": max(row["relative_absolute_error"] for row in test),
        "raw_roofline_held_out_mean_absolute_relative_error": float(
            np.mean([abs(row["roofline_reference_ms"] / row["observed_tpot_ms"] - 1) for row in test])
        ),
        "short_only_to_long_mean_absolute_relative_error": float(
            np.mean([abs(row["short_only_tpot_ms"] / row["observed_tpot_ms"] - 1) for row in long])
        ),
        "scope": "Historical isolated vLLM measurements, held out by configuration. No new inference run.",
        "limitations": [
            "No independent temporal test or multiple model families.",
            "Predicts average decode token latency, excluding prefill and SMG waiting.",
            "The overhead coefficient is a regression intercept, not measured CPU time.",
            "Co-resident environment commands and multiple engines require separate profiles.",
        ],
    }


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, length in zip(axes, (128, 2048)):
        rows = [row for row in result["rows"] if row["input_tokens"] == length]
        axis.plot(
            [row["concurrency"] for row in rows],
            [row["predicted_tpot_ms"] for row in rows],
            label="Calibrated prediction",
        )
        for train, marker, label in ((True, "o", "Fit data"), (False, "x", "Held-out data")):
            selected = [row for row in rows if row["training"] == train]
            axis.scatter(
                [row["concurrency"] for row in selected],
                [row["observed_tpot_ms"] for row in selected],
                marker=marker,
                label=label,
            )
        axis.set(
            xscale="log",
            xlabel="Concurrent requests",
            ylabel="Decode milliseconds / token",
            title=f"Input {length} tokens, output 512 tokens",
        )
        axis.legend()
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("hardware", "curve", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.hardware.read_text()), json.loads(args.curve.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.figure:
        plot(result, args.figure)
