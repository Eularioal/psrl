"""Build all three MLGym vision tasks with isolated model scoring."""

import argparse
import json
import shlex
import shutil
from pathlib import Path

import pandas as pd
import yaml
from examples.gpu_benchmarks.acceptance import managed_baseline_guidance
from examples.gpu_benchmarks.prepare import compose, write_dataset
from examples.mlgym_bench.download_vision import SOURCES
from examples.swe_serve.prepare import file_hashes


def prepare(data, output, python, image, guided_baseline=False):
    data, output = Path(data).resolve(), Path(output).resolve()
    for name in SOURCES:
        for filename in ("public/images.npz", "public/train_labels.json", "private/test_labels.json", "assets.json"):
            if not (data / name / filename).is_file():
                raise FileNotFoundError(data / name / filename)
    output.mkdir(parents=True, exist_ok=False)
    frames, records = [], []
    for name in SOURCES:
        target = output / name
        for part in ("environment", "tests", "solution", "public"):
            (target / part).mkdir(parents=True)
        for destination in (target / "public/baseline.py", target / "tests/evaluate.py"):
            shutil.copy2(Path(__file__).with_name("vision.py"), destination)
        command = shlex.join([python, "/opt/task/baseline.py", "train"])
        instruction = (
            f"Train a CUDA model for MLGym {name} on /data/public. "
            f"Run {command} for a small working baseline. "
            "Submit /logs/artifacts/model.pt with weights matching the public model architecture. "
            "You may change the training program. The independent verifier restores your weights, "
            "runs GPU inference and computes accuracy or BLEU from private test labels. "
            "This is a local bounded protocol with resized images, not canonical benchmark scoring. "
            "Do not submit self-reported metrics. Finish after saving the checkpoint."
        )
        if guided_baseline:
            instruction += managed_baseline_guidance(command, "/opt/task")
        (target / "instruction.md").write_text(instruction)
        (target / "solution/solve.sh").write_text(f"#!/bin/bash\nset -euo pipefail\n{command}\n")
        (target / "tests/test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\n" + shlex.join([python, "/tests/evaluate.py", "verify"]) + "\n"
        )
        (target / "tests/test.sh").chmod(0o755)
        for part in ("environment", "tests"):
            mounts = [f"{target / 'public'}:/opt/task:ro", f"{data / name / 'public'}:/data/public:ro"]
            if part == "tests":
                mounts += [f"{target / 'tests'}:/tests:ro", f"{data / name / 'private'}:/data/private:ro"]
            (target / part / "docker-compose.yaml").write_text(
                yaml.safe_dump(compose(mounts, python, mount_python_prefix=True))
            )
        (target / "task.toml").write_text(
            'version = "1.0"\n[environment]\n'
            f"docker_image = {json.dumps(image)}\ncpus = 4\nmemory_mb = 16384\ngpus = 1\n"
            'network_mode = "no-network"\n[agent]\ntimeout_sec = 180\n'
            '[verifier]\ntimeout_sec = 180\nenvironment_mode = "separate"\n'
            "[verifier.environment]\n"
            f"docker_image = {json.dumps(image)}\ncpus = 4\nmemory_mb = 16384\ngpus = 1\n"
            'network_mode = "no-network"\n'
        )
        record = {
            "benchmark": "mlgym_vision_local",
            "task": name,
            "assets": json.loads((data / name / "assets.json").read_text()),
            "prepared_hashes": file_hashes(data / name),
            "protocol": "fixed_network_checkpoint_independent_inference",
            "runtime_verified": False,
            "guided_baseline": guided_baseline,
        }
        (target / "provenance.json").write_text(json.dumps(record, indent=2))
        write_dataset(target, "mlgym_vision_local", 8192)
        frames.append(pd.read_parquet(output / f"{name}.parquet"))
        records.append(record)
    pd.concat(frames, ignore_index=True).to_parquet(output / "tasks.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--guided-baseline", action="store_true")
    args = parser.parse_args()
    prepare(args.data, args.output, args.python, args.image, args.guided_baseline)


if __name__ == "__main__":
    main()
