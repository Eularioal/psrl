"""Prepare official autoresearch data and the matching kernel in private caches."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--kernel-repo", default="varunneal/flash-attention-3")
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("At least one training shard is required.")
    repo = args.repo.resolve()
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Use a clean upstream checkout.")
    os.environ["HF_HOME"] = str(args.hf_cache.resolve())
    sys.path.insert(0, str(repo))
    import prepare
    from kernels import install_kernel

    # Only relocate the official cache. Training and evaluation constants stay fixed.
    prepare.CACHE_DIR = str(args.data.resolve())
    prepare.DATA_DIR = str(args.data.resolve() / "data")
    prepare.TOKENIZER_DIR = str(args.data.resolve() / "tokenizer")
    prepare.download_data(args.num_shards, download_workers=2)
    expected = [f"shard_{index:05d}.parquet" for index in range(args.num_shards)] + [prepare.VAL_FILENAME]
    for filename in expected:
        if not (Path(prepare.DATA_DIR) / filename).is_file():
            raise FileNotFoundError(filename)
    prepare.train_tokenizer()
    package, variant = install_kernel(args.kernel_repo, revision="main")
    manifest = {
        "upstream_revision": revision,
        "base_url": prepare.BASE_URL,
        "num_train_shards": args.num_shards,
        "validation_shard": prepare.VAL_FILENAME,
        "kernel_repo": args.kernel_repo,
        "kernel_package": package,
        "kernel_variant": variant.name,
        "kernel_revision": variant.parent.parent.name,
        "sha256": {},
    }
    for path in sorted(args.data.resolve().rglob("*")):
        if path.is_file() and path.name != "assets.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest["sha256"][str(path.relative_to(args.data.resolve()))] = digest.hexdigest()
    (args.data / "assets.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
