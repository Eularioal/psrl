"""Prepare small, disjoint real image splits without exposing evaluation labels."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SOURCES = {
    "imageClassificationCifar10": ("uoft-cs/cifar10", "plain_text", "img"),
    "imageClassificationFMnist": ("zalando-datasets/fashion_mnist", "fashion_mnist", "image"),
    "imageCaptioningCOCO": ("yerevann/coco-karpathy", "default", "url"),
}


def fetch(url):
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=30) as stream:
                return stream.read()
        except urllib.error.HTTPError as error:
            if attempt or error.code not in (429, 500, 502, 503, 504):
                raise


def download(output, train_rows=64, test_rows=16, tasks=None):
    if not 1 <= train_rows <= 100 or not 1 <= test_rows <= 100:
        raise ValueError("Each bounded split must contain between 1 and 100 rows.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for task in tasks or SOURCES:
        dataset, config, column = SOURCES[task]
        target = output / task
        (target / "public").mkdir(parents=True)
        (target / "private").mkdir()
        record = {"dataset": dataset, "config": config, "splits": {}}
        arrays = {}
        for split, count in (("train", train_rows), ("test", test_rows)):
            url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
                dict(dataset=dataset, config=config, split=split, offset=0, length=count)
            )
            response = fetch(url)
            rows = json.loads(response)["rows"]
            if len(rows) != count:
                raise ValueError(f"Expected {count!r} rows for {task!r}/{split!r}.")

            def decode(item, column=column):
                row = item["row"]
                image_url = row[column] if column == "url" else row[column]["src"]
                if image_url.startswith("http://images.cocodataset.org/"):
                    image_url = image_url.replace(
                        "http://images.cocodataset.org/", "https://s3.amazonaws.com/images.cocodataset.org/", 1
                    )
                content = fetch(image_url)
                image = Image.open(io.BytesIO(content)).convert("RGB").resize((32, 32))
                label = row["sentences"] if column == "url" else int(row["label"])
                return np.asarray(image), label, hashlib.sha256(content).hexdigest()

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                decoded = list(pool.map(decode, rows))
            arrays[f"{split}_x"] = np.stack([item[0] for item in decoded])
            labels = [item[1] for item in decoded]
            destination = target / ("public" if split == "train" else "private") / f"{split}_labels.json"
            destination.write_text(json.dumps(labels))
            record["splits"][split] = {
                "rows": count,
                "row_response_sha256": hashlib.sha256(response).hexdigest(),
                "image_sha256": [item[2] for item in decoded],
                "row_ids": [item["row_idx"] for item in rows],
            }
        np.savez_compressed(target / "public/images.npz", **arrays)
        (target / "public/config.json").write_text(json.dumps({"task": task, "caption": column == "url"}))
        (target / "assets.json").write_text(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=64)
    parser.add_argument("--test-rows", type=int, default=16)
    parser.add_argument("--tasks", nargs="+", choices=SOURCES)
    args = parser.parse_args()
    download(args.output, args.train_rows, args.test_rows, args.tasks)


if __name__ == "__main__":
    main()
