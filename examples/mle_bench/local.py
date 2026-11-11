"""Prepare a local GPU digit competition with public inputs and hidden labels."""

import argparse
import csv
import json
import shlex
import shutil
from pathlib import Path

import numpy as np
import yaml
from examples.gpu_benchmarks.prepare import compose, write_dataset
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split


def prepare(args):
    """Keep the MLE submission contract while using an openly bundled dataset."""
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("environment", "tests", "solution", "public", "private"):
        (out / name).mkdir()
    dataset = getattr(args, "dataset", "digits")
    data = {"digits": load_digits, "wine": load_wine, "iris": load_iris, "breast-cancer": load_breast_cancer}[
        dataset
    ]()
    features, classes = data.data.shape[1], len(set(data.target))
    train, test = train_test_split(np.arange(len(data.target)), test_size=0.25, random_state=17, stratify=data.target)
    for name, ids, labels in (("train", train, True), ("test", test, False)):
        with (out / "public" / f"{name}.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", *[f"pixel_{i}" for i in range(features)], *(["label"] if labels else [])])
            for i in ids:
                writer.writerow([int(i), *data.data[i].tolist(), *([int(data.target[i])] if labels else [])])
    (out / "private/labels.json").write_text(json.dumps({str(int(i)): int(data.target[i]) for i in test}))
    shutil.copyfile(Path(__file__).with_name("local_baseline.py"), out / "public/baseline.py")
    (out / "solution/solve.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n" + shlex.join([args.python, "/data/public/baseline.py"]) + "\n"
    )
    instruction = (
        "Train a CUDA classifier on /data/public/train.csv and predict /data/public/test.csv. "
        f"The dataset is sklearn's real {dataset} data, with {features} feature columns and {classes} classes. "
        "Submit /logs/artifacts/submission.csv with exactly id,label columns and one row per test id. "
        "Use PyTorch CUDA through psrl-gpu for training. Hidden test labels are only available to the verifier. "
        "A working CUDA baseline is /data/public/baseline.py. Reward is hidden accuracy. "
        "This is a local MLE-style competition, not official Kaggle scoring."
    )
    (out / "instruction.md").write_text(instruction)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "benchmark": "mle_bench_local",
                "dataset": dataset,
                "features": features,
                "classes": classes,
                "split_seed": 17,
                "train": len(train),
                "test": len(test),
                "official_scoring": False,
            }
        )
    )
    public = [f"{out / 'public'}:/data/public:ro"]
    private = public + [f"{out / 'private'}:/protected:ro", f"{out / 'tests'}:/tests:ro"]
    for name, mounts in (("environment", public), ("tests", private)):
        (out / name / "docker-compose.yaml").write_text(yaml.safe_dump(compose(mounts, args.python, True)))
    shutil.copyfile(Path(__file__).with_name("local_grade.py"), out / "tests/grade.py")
    (out / "tests/test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n" + shlex.join([args.python, "/tests/grade.py"]) + "\n"
    )
    (out / "tests/test.sh").chmod(0o755)
    (out / "task.toml").write_text(
        'version="1.0"\n[environment]\n'
        f'docker_image={json.dumps(args.image)}\ncpus=4\nmemory_mb=8192\ngpus=1\nnetwork_mode="no-network"\n'
        '[agent]\ntimeout_sec=600\n[verifier]\ntimeout_sec=60\nenvironment_mode="separate"\n'
        f'[verifier.environment]\ndocker_image={json.dumps(args.image)}\ncpus=2\nmemory_mb=4096\ngpus=0\nnetwork_mode="no-network"\n'
    )
    write_dataset(out, "mle_bench_local", 4096)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--dataset", choices=["digits", "wine", "iris", "breast-cancer"], default="digits")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
