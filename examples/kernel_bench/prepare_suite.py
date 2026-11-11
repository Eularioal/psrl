"""Prepare a pinned KernelBench task library and one managed PSRL dataset."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from examples.gpu_benchmarks.mix import prepare as prepare_mix
from examples.gpu_benchmarks.prepare import prepare, provenance


def discover(repo, levels):
    """Require the evaluator interface before publishing any task."""
    tasks = []
    for level in levels:
        paths = sorted((repo / "KernelBench" / f"level{level}").glob("*.py"))
        if not paths:
            raise ValueError(f"No tasks found in level {level!r}.")
        for path in paths:
            tree = ast.parse(path.read_text())
            names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            if not {"Model", "get_inputs", "get_init_inputs"} <= names:
                raise ValueError(f"Unsupported reference interface in {path.name!r}.")
            tasks.append((level, path))
    return tasks


def prepare_suite(args):
    """Publish task identities separately from GPU verification results."""
    revision = provenance(args.repo)
    tasks = discover(args.repo, sorted(set(args.levels)))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, inventory = [], []
    for level, reference in tasks:
        task_id = f"level{level}-{reference.stem}"
        config = argparse.Namespace(
            **vars(args),
            benchmark="kernel_bench",
            level=level,
            problem=reference.name,
            environment_backend="docker",
        )
        config.output = output / "tasks" / task_id
        task = prepare(config)
        records.append(
            {
                "task_path": str(task),
                "data_source": "kernel_bench_local" if getattr(args, "bounded_inputs", False) else "kernel_bench",
                "environment_backend": "docker",
                "gpu_memory_mb": args.gpu_memory_mb,
            }
        )
        inventory.append(
            {
                "task_id": task_id,
                "level": level,
                "reference": str(reference.relative_to(args.repo)),
                "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "gpu_verified": False,
                "policy_loop_verified": False,
            }
        )
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2))
    prepare_mix(manifest, output / "dataset")
    (output / "inventory.json").write_text(
        json.dumps({**revision, "status": "prepared_not_gpu_verified", "tasks": inventory}, indent=2)
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--verifier-image")
    parser.add_argument("--python", default="/usr/local/bin/python")
    parser.add_argument("--mount-python-prefix", action="store_true")
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--bounded-inputs", action="store_true")
    parser.add_argument("--levels", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument("--backend", choices=["cuda", "triton"], default="triton")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--gpu-memory-mb", type=int, default=8192)
    args = parser.parse_args()
    if args.timeout <= 0 or args.gpu_memory_mb <= 0:
        parser.error("Timeout and GPU memory budget must be positive.")
    args.repo = args.repo.resolve()
    print(prepare_suite(args))


if __name__ == "__main__":
    main()
