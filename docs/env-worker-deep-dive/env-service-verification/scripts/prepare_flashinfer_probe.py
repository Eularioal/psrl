"""Prepare a bounded official FlashInfer subset without downloading large tensors."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def revision(path):
    if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip():
        raise ValueError("Use a clean pinned upstream checkout.")
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def prepare(args):
    root = args.output.resolve()
    record = {"upstream_head": revision(args.upstream), "trace_head": revision(args.trace)}
    definition = "rmsnorm_h2048"
    definition_path = args.trace / "definitions/rmsnorm" / f"{definition}.json"
    solution_path = args.trace / "solutions/baseline/rmsnorm" / definition / "flashinfer_wrapper_0af255.json"
    workload_path = args.trace / "workloads/rmsnorm" / f"{definition}.jsonl"
    solution = json.loads(solution_path.read_text())
    if "NVIDIA H20" not in solution["spec"]["target_hardware"]:
        raise ValueError("This pilot expects the official H20-compatible baseline.")
    rows = [
        line
        for line in workload_path.read_text().splitlines()
        if json.loads(line)["workload"]["axes"]["batch_size"] in (1, 64, 16254)
    ]
    if len(rows) != 3:
        raise ValueError("Expected three distinct official workloads.")
    root.mkdir(parents=True, exist_ok=False)
    for name in ("definitions", "solutions", "workloads"):
        (root / "subset" / name).mkdir(parents=True)
    shutil.copy2(definition_path, root / "subset/definitions" / definition_path.name)
    shutil.copy2(solution_path, root / "subset/solutions" / solution_path.name)
    (root / "subset/workloads" / workload_path.name).write_text("\n".join(rows) + "\n")
    shutil.copytree(args.upstream / "flashinfer_bench", root / "upstream/flashinfer_bench")
    for name in ("flashinfer_official_eval.py", "flashinfer_env_probe.py"):
        shutil.copy2(Path(__file__).with_name(name), root / name)
    record["input_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "subset").rglob("*")
        if path.is_file()
    }
    record["subset_uuids"] = [json.loads(line)["workload"]["uuid"] for line in rows]
    (root / "manifest.json").write_text(json.dumps(record, indent=2))
    print(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
