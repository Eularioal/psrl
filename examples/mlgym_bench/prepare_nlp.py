"""Prepare both MLGym language tasks with private evaluation data."""

import argparse
import json
import shlex
import shutil
from pathlib import Path

import pandas as pd
import yaml
from examples.gpu_benchmarks.acceptance import managed_baseline_guidance
from examples.gpu_benchmarks.prepare import compose, provenance, write_dataset
from examples.swe_serve.prepare import file_hashes

TASKS = ("languageModelingFineWeb", "naturalLanguageInferenceMNLI")


def prepare(repo, assets, output, python, image, guided_baseline=False):
    repo, assets, output = Path(repo).resolve(), Path(assets).resolve(), Path(output).resolve()
    revision = provenance(repo)
    source = repo / "data/languageModelingFineWeb/baseline.py"
    if not source.is_file() or not (assets / "bert/model.safetensors").is_file():
        raise FileNotFoundError("Real upstream model code and BERT weights are required.")
    for task in TASKS:
        for kind in ("public", "private"):
            if not (assets / task / kind / "rows.json").is_file():
                raise FileNotFoundError(f"Missing {kind!r} data for {task!r}.")
    output.mkdir(parents=True, exist_ok=False)
    records, frames = [], []
    for task in TASKS:
        target = output / task
        for part in ("environment", "tests", "solution", "public"):
            (target / part).mkdir(parents=True)
        shutil.copy2(source, target / "public/upstream.py")
        for dest in (target / "public/baseline.py", target / "tests/evaluate.py"):
            shutil.copy2(Path(__file__).with_name("nlp.py"), dest)
        command = shlex.join([python, "/opt/task/baseline.py", "train", "--task", task])
        protocol = (
            "The original GPT-style model uses two 64-wide layers and byte tokens. "
            "Submit all fixed-architecture weights."
            if task == "languageModelingFineWeb"
            else (
                "Tune the classification head of the real pretrained BERT model in /model. "
                "Submit only classifier weights."
            )
        )
        instruction = (
            f"Train a CUDA model for MLGym {task}. {protocol} "
            "Submit /logs/artifacts/model.safetensors. This local bounded protocol uses real data and two updates. "
            "Independent evaluation reloads your weights and scores private data. "
            f"Run {command} for a baseline and finish after saving the checkpoint."
        )
        if guided_baseline:
            instruction += managed_baseline_guidance(command, "/opt/task")
        (target / "instruction.md").write_text(instruction)
        (target / "solution/solve.sh").write_text(f"#!/bin/bash\nset -euo pipefail\n{command}\n")
        (target / "tests/test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            + shlex.join([python, "/tests/evaluate.py", "verify", "--task", task])
            + "\n"
        )
        (target / "tests/test.sh").chmod(0o755)
        for part in ("environment", "tests"):
            mounts = [f"{target / 'public'}:/opt/task:ro", f"{assets / task / 'public'}:/data/public:ro"]
            if task == "naturalLanguageInferenceMNLI":
                mounts.append(f"{assets / 'bert'}:/model:ro")
            if part == "tests":
                mounts += [f"{target / 'tests'}:/tests:ro", f"{assets / task / 'private'}:/data/private:ro"]
            (target / part / "docker-compose.yaml").write_text(
                yaml.safe_dump(compose(mounts, python, mount_python_prefix=True))
            )
        (target / "task.toml").write_text(
            'version = "1.0"\n[environment]\n'
            f"docker_image = {json.dumps(image)}\ncpus = 4\nmemory_mb = 16384\ngpus = 1\n"
            'network_mode = "no-network"\n[agent]\ntimeout_sec = 180\n'
            '[verifier]\ntimeout_sec = 180\nenvironment_mode = "separate"\n[verifier.environment]\n'
            f"docker_image = {json.dumps(image)}\ncpus = 4\nmemory_mb = 16384\ngpus = 1\n"
            'network_mode = "no-network"\n'
        )
        record = {
            **revision,
            "benchmark": "mlgym_nlp_local",
            "task": task,
            "protocol": protocol,
            "data_hashes": file_hashes(assets / task),
            "runtime_verified": False,
            "guided_baseline": guided_baseline,
        }
        if task == "naturalLanguageInferenceMNLI":
            record["base_model"] = json.loads((assets / "model.json").read_text())
        else:
            record["upstream_files"] = file_hashes(target / "public")
        (target / "provenance.json").write_text(json.dumps(record, indent=2))
        write_dataset(target, "mlgym_nlp_local", 8192)
        records.append(record)
        frames.append(pd.read_parquet(output / f"{task}.parquet"))
    pd.concat(frames, ignore_index=True).to_parquet(output / "tasks.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "assets", "output", "python", "image"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--guided-baseline", action="store_true")
    prepare(**vars(parser.parse_args()))
