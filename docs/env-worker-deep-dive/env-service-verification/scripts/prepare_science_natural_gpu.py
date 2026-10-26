"""Freeze an existing MLGym baseline and a bounded slice of real CIFAR images."""

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.baseline, args.output / "baseline.py")
    shutil.copy2(Path(__file__).with_name("science_natural_gpu_train.py"), args.output / "train.py")
    rows = next(pq.ParquetFile(args.parquet).iter_batches(batch_size=640)).to_pylist()
    if len(rows) != 640:
        raise ValueError(f"Expected 640 real images, got {len(rows)!r}.")
    images = np.stack([np.asarray(Image.open(io.BytesIO(row["img"]["bytes"]))) for row in rows])
    np.savez(args.output / "cifar640.npz", images=images, labels=np.array([row["label"] for row in rows]))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()}
    (args.output / "sha256.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
