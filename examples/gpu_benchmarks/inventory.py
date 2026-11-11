"""Count local pinned task libraries without downloading assets or running GPU work."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def inventory(sources, assets, kernel_suite):
    definitions = list((sources / "flashinfer-trace/definitions").rglob("*.json"))
    workload_files = list((sources / "flashinfer-trace/workloads").rglob("*.jsonl"))
    workloads = [json.loads(line) for path in workload_files for line in path.read_text().splitlines() if line]
    blobs = {
        item["path"]
        for row in workloads
        for item in row["workload"]["inputs"].values()
        if item.get("type") == "safetensors"
    }
    missing, pointers = [], []
    for name in sorted(blobs):
        path = sources / "flashinfer-trace" / name
        if not path.is_file():
            missing.append(name)
        else:
            with path.open("rb") as stream:
                if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                    pointers.append(name)
    result = {
        "scope": "Local pinned source and prepared artifact inventory. No new GPU verification.",
        "revisions": {
            name: revision(sources / name)
            for name in ["KernelBench", "flashinfer-trace", "mle-bench", "RE-Bench", "InferenceBench", "autoresearch"]
        },
        "kernel_bench": {
            "source_by_level": {
                f"level{level}": len(list((sources / f"KernelBench/KernelBench/level{level}").glob("*.py")))
                for level in range(1, 5)
            },
        },
        "mle_bench": {
            "official_split75": len((sources / "mle-bench/experiments/splits/split75.txt").read_text().splitlines()),
            "registry_configs": len(list((sources / "mle-bench/mlebench/competitions").glob("*/config.yaml"))),
            "prepared_official_competitions": [
                path.parent.name
                for path in assets.rglob("prepared")
                if (path / "public").is_dir() and (path / "private").is_dir()
            ],
        },
        "re_bench": {"source_tasks": sorted(p.name for p in (sources / "RE-Bench").glob("ai_rd_*") if p.is_dir())},
        "inference_bench": {
            "source_scenarios": sorted(
                p.parent.name for p in (sources / "InferenceBench/src/eval/tasks").glob("*/scenario.json")
            )
        },
        "flashinfer_bench": {
            "definitions": len(definitions),
            "workload_files": len(workload_files),
            "workload_rows": len(workloads),
            "definitions_with_workloads": len({row["definition"] for row in workloads}),
            "operations": dict(
                Counter(
                    str(p.relative_to(sources / "flashinfer-trace/definitions")).split("/")[0] for p in definitions
                )
            ),
            "referenced_blob_files": len(blobs),
            "missing_blob_files": len(missing),
            "lfs_pointer_blob_files": len(pointers),
        },
        "autoresearch": {
            "fixed_training_task": 1,
            "cached_training_shards": len(
                [
                    p
                    for p in (assets / "autoresearch-data/data").glob("shard_*.parquet")
                    if p.name != "shard_06542.parquet"
                ]
            ),
        },
    }
    prepared = json.loads((kernel_suite / "inventory.json").read_text())
    result["kernel_bench"]["prepared_by_level"] = dict(Counter(f"level{p['level']}" for p in prepared["tasks"]))
    result["kernel_bench"]["prepared_tasks"] = prepared["tasks"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--kernel-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.sources, args.assets, args.kernel_suite)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
