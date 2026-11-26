"""Summarize completed inference waves and plot latency against throughput."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(source):
    """Validate equal output work and retain engine scheduling evidence."""
    evidence = json.loads((source / "evidence.json").read_text())
    completion = json.loads((source / "completion.json").read_text())
    exited = completion["exit_code"] == 0 or (
        completion["exit_code"] is None and completion.get("process_exit_observed", False)
    )
    if evidence["status"] != "passed" or not exited or completion["timed_out"] or not completion["physical_release"]:
        raise ValueError("The run did not complete and release its GPU successfully.")
    grouped = defaultdict(list)
    for wave in evidence["waves"]:
        if len(wave["requests"]) != wave["concurrency"]:
            raise ValueError("The completed request count differs from the submitted count.")
        if any(request["output_tokens"] != 512 for request in wave["requests"]):
            raise ValueError("Output lengths differ.")
        if len(wave["engine_finished"]) != wave["concurrency"]:
            raise ValueError("Engine timing does not cover every completed request.")
        grouped[wave["input_tokens"], wave["concurrency"]].append(wave)
    rows = []
    for (length, concurrency), waves in sorted(grouped.items()):
        if len(waves) != 2:
            raise ValueError("Every point requires two measured waves.")
        requests = [request for wave in waves for request in wave["requests"]]
        finished = [request for wave in waves for request in wave["engine_finished"]]
        latencies = sorted(request["latency_s"] for request in requests)
        rows.append(dict(
            input_tokens=length, concurrency=concurrency, waves=len(waves), requests=len(requests),
            output_tokens_per_s=len(requests) * 512 / sum(wave["elapsed_s"] for wave in waves),
            mean_latency_s=statistics.mean(latencies),
            p95_latency_s=latencies[int(0.95 * (len(latencies) - 1))],
            mean_ttft_s=statistics.mean(request["ttft_s"] for request in requests),
            mean_tpot_ms=1000 * statistics.mean(request["tpot_s"] for request in requests),
            wave_mean_latency_min_s=min(wave["mean_latency_s"] for wave in waves),
            wave_mean_latency_max_s=max(wave["mean_latency_s"] for wave in waves),
            mean_engine_queue_s=statistics.mean(request["queued_time"] for request in finished),
            mean_engine_prefill_s=statistics.mean(request["prefill_time"] for request in finished),
            mean_engine_decode_s=statistics.mean(request["decode_time"] for request in finished),
            preemptions=sum(wave["counters"]["preemptions"] for wave in waves),
            max_running=max(wave["counters"]["max_running"] for wave in waves),
            max_waiting=max(wave["counters"]["max_waiting"] for wave in waves),
            max_kv_usage=max(wave["counters"]["max_kv_usage"] for wave in waves),
        ))
    for row in rows:
        baseline = next(item for item in rows if item["input_tokens"] == row["input_tokens"])
        row["latency_vs_single"] = row["mean_latency_s"] / baseline["mean_latency_s"]
        row["tpot_vs_single"] = row["mean_tpot_ms"] / baseline["mean_tpot_ms"]
    return dict(
        status="passed", model="Qwen2.5-3B-Instruct", gpu="H20",
        engine_options={**evidence["engine_options"], "model": "<MODEL_PATH>"},
        versions=evidence["versions"], initialization_s=evidence["initialization_s"],
        elapsed_s=completion["elapsed_s"], physical_release=completion["physical_release"], rows=rows,
        exit_code=completion["exit_code"],
        boundary="Single engine burst arrivals. Synthetic token inputs. No Env commands, SM counters or PPO training.",
    )


def plot(summary, destination):
    """Render the measured service curve without claiming hardware roofline counters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for length in sorted({row["input_tokens"] for row in summary["rows"]}):
        rows = [row for row in summary["rows"] if row["input_tokens"] == length]
        x = [row["concurrency"] for row in rows]
        for axis, key, label in zip(
            axes, ["output_tokens_per_s", "mean_latency_s", "mean_tpot_ms"],
            ["Output throughput (tokens/s)", "Mean request latency (s)", "Mean decode TPOT (ms)"], strict=True,
        ):
            axis.plot(x, [row[key] for row in rows], "o-", label=f"{length} input tokens")
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Simultaneously submitted requests")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            axis.legend()
    figure.suptitle("H20 / Qwen2.5-3B / 512 output tokens / TP=1 / eager / two waves per point")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".svg"))
    figure.savefig(destination.with_suffix(".png"), dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.output.with_suffix(""))
