"""Import every GPU task from a pinned SWE-Serve checkout without changing its verifier."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import tomllib
import yaml
from examples.gpu_benchmarks.prepare import provenance, write_dataset


def file_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def prepare(repo, output, hf_cache=None, image_map=None, gpu_memory_mb=80000, base_map=None):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    revision = provenance(repo)
    if output.exists():
        raise FileExistsError(output)
    if gpu_memory_mb <= 0:
        raise ValueError("GPU memory budget must be positive.")
    if hf_cache is not None:
        hf_cache = Path(hf_cache).resolve()
        if not (hf_cache / "hub").is_dir():
            raise ValueError("HF cache must contain a hub directory.")
    images = image_map or {}
    bases = {name: Path(path).resolve() for name, path in (base_map or {}).items()}
    with (repo / "task_list.csv").open() as stream:
        inventory = list(csv.DictReader(stream))
    selected = [row for row in inventory if int(row["gpus"]) > 0]
    names = [row["task_dir"] for row in selected]
    if len(names) != len(set(names)) or (set(images) | set(bases)) - set(names):
        raise ValueError("Duplicate tasks or unknown image overrides.")
    records = []
    # Validate the complete import before creating any runnable dataset.
    for row in selected:
        name = row["task_dir"]
        source = repo / "tasks" / name
        if source.parent != repo / "tasks" or source.is_symlink():
            raise ValueError(f"Unsafe task directory: {name!r}.")
        for filename in ("instruction.md", "task.toml", "tests/test.sh", "solution/solve.sh"):
            if not (source / filename).is_file():
                raise FileNotFoundError(source / filename)
        config = tomllib.loads((source / "task.toml").read_text())
        if config["verifier"].get("environment_mode") != "separate":
            raise ValueError(f"Task does not use an independent verifier: {name!r}.")
        if config["environment"]["gpus"] != int(row["gpus"]):
            raise ValueError(f"GPU count disagrees with the manifest: {name!r}.")
        if any((source / part / "docker-compose.yaml").exists() for part in ("environment", "tests")):
            raise ValueError(f"Review upstream Compose overrides before importing {name!r}.")
        if name in images and (not isinstance(images[name], str) or not images[name].strip()):
            raise ValueError(f"Invalid image for {name!r}.")
        verifier_recipe = source / "tests/Dockerfile"
        needs_base = verifier_recipe.is_file() and "/base" in verifier_recipe.read_text()
        if name in images and needs_base and name not in bases:
            raise ValueError(f"Prebuilt image requires an independent pristine /base mount: {name!r}.")
        if name in bases and not bases[name].is_dir():
            raise FileNotFoundError(bases[name])
        records.append((row, source, config))
    output.mkdir(parents=True)
    manifest = []
    frames = []
    for row, source, config in records:
        name = row["task_dir"]
        target = output / "tasks" / name
        shutil.copytree(source, target)
        (target / "tests/test.sh").chmod(0o755)
        if name in images:
            image = images[name]
            if not isinstance(image, str) or not image.strip():
                raise ValueError(f"Invalid image for {name!r}.")
            text = (target / "task.toml").read_text()
            if "docker_image" in config["environment"]:
                raise ValueError("Do not overwrite an upstream image pin.")
            text = text.replace("[environment]\n", f"[environment]\ndocker_image = {json.dumps(image)}\n", 1)
            (target / "task.toml").write_text(text)
        for part in ("environment", "tests"):
            mounts = [f"{target / part}:{'/tests' if part == 'tests' else '/task'}:ro"]
            if part == "tests":
                if name in bases:
                    mounts.append(f"{bases[name]}:/base:ro")
                if (target / "tests/speed_check").is_dir():
                    mounts.append(f"{target / 'tests/speed_check'}:/speed-check:ro")
            if hf_cache is not None:
                mounts.append(f"{hf_cache}:/hf-cache:ro")
            override = {"services": {"main": {"volumes": mounts}}}
            (target / part / "docker-compose.yaml").write_text(yaml.safe_dump(override))
        record = {
            **revision,
            "benchmark": "swe_serve",
            "task": name,
            "gpus": int(row["gpus"]),
            "f2p_mode": config["verifier"].get("f2p_mode"),
            "upstream_files": file_hashes(source),
            "prebuilt_image": images.get(name),
            "pristine_base": str(bases[name]) if name in bases else None,
            "hf_cache_mounted": hf_cache is not None,
            "runtime_verified": False,
        }
        (target / "provenance.json").write_text(json.dumps(record, indent=2))
        write_dataset(target, "swe_serve", gpu_memory_mb)
        frame = pd.read_parquet(target.parent / f"{name}.parquet")
        frame.at[0, "extra_info"] = {
            **frame.iloc[0].extra_info,
            "gpus": int(row["gpus"]),
            "upstream_task_id": row["task_id"],
            "upstream_revision": revision["revision"],
        }
        frame.to_parquet(target.parent / f"{name}.parquet", index=False)
        frames.append(frame)
        manifest.append(record)
    if not frames:
        raise ValueError("No GPU tasks found.")
    dataset = pd.concat(frames, ignore_index=True)
    dataset.to_parquet(output / "tasks.parquet", index=False)
    # Stable task identities keep validation tasks out of the training split.
    validation = [int(hashlib.sha256(name.encode()).hexdigest(), 16) % 5 == 0 for name in names]
    dataset.loc[[not value for value in validation]].to_parquet(output / "train.parquet", index=False)
    dataset.loc[validation].to_parquet(output / "val.parquet", index=False)
    summary = {
        **revision,
        "upstream_total": len(inventory),
        "gpu_tasks_imported": len(manifest),
        "cpu_tasks_excluded": len(inventory) - len(manifest),
        "runtime_verified_count": 0,
        "tasks": manifest,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--image-map", type=Path)
    parser.add_argument("--base-map", type=Path, help="Task-to-pristine-source mapping for independent verifiers.")
    parser.add_argument("--gpu-memory-mb", type=int, default=80000)
    args = parser.parse_args()
    images = json.loads(args.image_map.read_text()) if args.image_map else None
    bases = json.loads(args.base_map.read_text()) if args.base_map else None
    result = prepare(args.repo, args.output, args.hf_cache, images, args.gpu_memory_mb, bases)
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
