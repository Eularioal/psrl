"""Prepare every available operator with bounded original random input workloads."""

import argparse
import json
from pathlib import Path

from examples.flashinfer_bench.prepare import prepare
from examples.gpu_benchmarks.mix import prepare as prepare_mix
from examples.gpu_benchmarks.prepare import provenance


def prepare_suite(args):
    record = provenance(args.trace)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest, skipped = [], []
    for definition_path in sorted((args.trace / "definitions").rglob("*.json")):
        name = definition_path.stem
        workload_path = args.trace / "workloads" / definition_path.relative_to(args.trace / "definitions")
        workload_path = workload_path.with_suffix(".jsonl")
        rows = (
            [json.loads(line) for line in workload_path.read_text().splitlines() if line]
            if workload_path.is_file()
            else []
        )
        rows = [row for row in rows if all(v.get("type") == "random" for v in row["workload"]["inputs"].values())]
        if not rows:
            skipped.append({"definition": name, "reason": "No original random-only workload available."})
            continue
        rows.sort(key=lambda row: sum(v for v in row["workload"]["axes"].values() if isinstance(v, (int, float))))
        rows = rows[: args.workloads]
        subset = output / "subsets" / name
        for folder in ("definitions", "workloads", "solutions"):
            (subset / folder).mkdir(parents=True)
        definition = json.loads(definition_path.read_text())
        (subset / "definitions" / definition_path.name).write_text(json.dumps(definition))
        (subset / "workloads" / workload_path.name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        solution = {
            "name": "public_reference_control",
            "definition": name,
            "author": "local-reference-control",
            "description": "Official mathematical reference for bounded integration verification.",
            "spec": {
                "language": "python",
                "target_hardware": ["NVIDIA H20"],
                "entry_point": "main.py::run",
                "dependencies": ["torch"],
                "destination_passing_style": False,
            },
            "sources": [{"path": "main.py", "content": definition["reference"]}],
        }
        (subset / "solutions/reference.json").write_text(json.dumps(solution))
        config = argparse.Namespace(**vars(args), subset=subset)
        config.output = output / "tasks" / name
        prepare(config)
        manifest.append(
            {
                "task_path": str(config.output),
                "data_source": "flashinfer_bench",
                "environment_backend": "docker",
                "gpu_memory_mb": 8192,
            }
        )
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    prepare_mix(path, output / "dataset")
    (output / "inventory.json").write_text(
        json.dumps(
            {
                **record,
                "prepared": len(manifest),
                "skipped": skipped,
                "protocol": "bounded original random-only workloads",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "trace", "cuda", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--workloads", type=int, default=1)
    args = parser.parse_args()
    if args.workloads < 1:
        parser.error("The workload count must be positive.")
    prepare_suite(args)


if __name__ == "__main__":
    main()
