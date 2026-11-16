"""Prepare every available operator with bounded original random input workloads."""

import argparse
import json
import shutil
from pathlib import Path

from examples.flashinfer_bench.prepare import prepare
from examples.gpu_benchmarks.mix import prepare as prepare_mix
from examples.gpu_benchmarks.prepare import provenance


def prepare_suite(args):
    record = provenance(args.trace)
    trace = getattr(args, "local_trace", None) or args.trace
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest, skipped = [], []
    for definition_path in sorted((trace / "definitions").rglob("*.json")):
        name = definition_path.stem
        workload_path = trace / "workloads" / definition_path.relative_to(trace / "definitions")
        workload_path = workload_path.with_suffix(".jsonl")
        rows = (
            [json.loads(line) for line in workload_path.read_text().splitlines() if line]
            if workload_path.is_file()
            else []
        )
        assets = getattr(args, "assets", None)
        usable = []
        for row in rows:
            valid = True
            for value in row["workload"]["inputs"].values():
                if value.get("type") in ("random", "scalar"):
                    continue
                if value.get("type") != "safetensors" or assets is None:
                    valid = False
                    break
                path = (assets / value["path"]).resolve()
                if not path.is_relative_to(assets.resolve()) or not path.is_file():
                    valid = False
                    break
                with path.open("rb") as stream:
                    if stream.read(40).startswith(b"version https://git-lfs"):
                        valid = False
                        break
            if valid:
                usable.append(row)
        rows = usable
        if not rows:
            skipped.append(
                {"definition": name, "reason": "No original workload with all required input assets available."}
            )
            continue
        rows.sort(key=lambda row: sum(v for v in row["workload"]["axes"].values() if isinstance(v, (int, float))))
        rows = rows[: args.workloads]
        subset = output / "subsets" / name
        for folder in ("definitions", "workloads", "solutions"):
            (subset / folder).mkdir(parents=True)
        for row in rows:
            for value in row["workload"]["inputs"].values():
                if value["type"] == "safetensors":
                    relative = Path(value["path"])
                    destination = subset / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(assets / relative, destination)
        definition = json.loads(definition_path.read_text())
        (subset / "definitions" / definition_path.name).write_text(json.dumps(definition))
        (subset / "workloads" / workload_path.name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        completed_reference = not bool(definition.get("reference"))
        control = definition.get("reference")
        if control is None and name == "top_k_top_p_sampling_from_probs_v248320":
            template = next((trace / "definitions").rglob("top_k_top_p_sampling_from_probs_v202048.json"))
            control = json.loads(template.read_text())["reference"].replace(
                "vocab_size == 202048", "vocab_size == 248320"
            )
        if definition.get("reference") is None and control is not None:
            definition["reference"] = control
            (subset / "definitions" / definition_path.name).write_text(json.dumps(definition))
        if control is None:
            raise ValueError(f"No public reference control for {name!r}.")
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
            "sources": [{"path": "main.py", "content": control}],
        }
        (subset / "solutions/reference.json").write_text(json.dumps(solution))
        config = argparse.Namespace(
            **vars(args), subset=subset, local_profile=bool(getattr(args, "local_trace", None)) or completed_reference
        )
        config.output = output / "tasks" / name
        prepare(config)
        manifest.append(
            {
                "task_path": str(config.output),
                "data_source": "flashinfer_bench_local" if config.local_profile else "flashinfer_bench",
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
                "protocol": "bounded original workloads with materialized input assets",
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
    parser.add_argument("--local-trace", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--workloads", type=int, default=1)
    args = parser.parse_args()
    if args.workloads < 1:
        parser.error("The workload count must be positive.")
    prepare_suite(args)


if __name__ == "__main__":
    main()
