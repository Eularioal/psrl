"""Prepare credential-free PostTrainBench combinations with bounded local GPU scoring."""

import argparse
import json
import shlex
import shutil
from pathlib import Path

import pandas as pd
import yaml
from examples.gpu_benchmarks.acceptance import managed_baseline_guidance
from examples.gpu_benchmarks.heterogeneous_inventory import posttrain_inventory
from examples.gpu_benchmarks.prepare import compose, write_dataset
from examples.swe_serve.prepare import file_hashes

LIKELIHOOD_BENCHMARKS = ("aime2025", "gsm8k", "humaneval")
JUDGED_BENCHMARKS = ("arenahardwriting", "healthbench")
BENCHMARKS = LIKELIHOOD_BENCHMARKS + JUDGED_BENCHMARKS


def prepare(repo, assets, output, python, image, guided_baseline=False, benchmarks=BENCHMARKS):
    assets, output = Path(assets).resolve(), Path(output).resolve()
    inventory = posttrain_inventory(Path(repo).resolve())
    models = json.loads((assets / "models.json").read_text())
    selected = [t for t in inventory["tasks"] if t["benchmark"] in benchmarks and not t["gated_assets"]]
    if not selected:
        raise ValueError("Select at least one available model and benchmark combination.")
    judged = any(t["benchmark"] in JUDGED_BENCHMARKS for t in selected)
    judge = json.loads((assets / "judge.json").read_text()) if judged else None
    if judged:
        judge_path = Path(judge["path"])
        if not (judge_path / "config.json").is_file() or not list(judge_path.glob("*.safetensors")):
            raise FileNotFoundError(f"Missing frozen judge weights: {judge_path!s}.")
    for task in selected:
        model = Path(models[task["model"]]["path"])
        if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
            raise FileNotFoundError(f"Missing real model weights: {model!s}.")
        for split in ("public", "private"):
            if not (assets / "data" / task["benchmark"] / split / "rows.json").is_file():
                raise FileNotFoundError(f"Missing {split!r} data for {task['benchmark']!r}.")
    output.mkdir(parents=True, exist_ok=False)
    records, frames = [], []
    for task in selected:
        is_judged = task["benchmark"] in JUDGED_BENCHMARKS
        data_source = "posttrain_bench_judged_local" if is_judged else "posttrain_bench_local"
        target = output / task["id"]
        for part in ("public", "environment", "tests", "solution"):
            (target / part).mkdir(parents=True)
        for part, name in (("public", "baseline.py"), ("tests", "evaluate.py")):
            source = "judged_model.py" if is_judged else "local_model.py"
            shutil.copy2(Path(__file__).with_name(source), target / part / name)
            if is_judged:
                for helper in ("local_model.py", "judge_scoring.py"):
                    shutil.copy2(Path(__file__).with_name(helper), target / part / helper)
        extra = ["--benchmark", task["benchmark"]] if is_judged else []
        command = shlex.join([python, "/opt/task/baseline.py", "train", "--data", "/data/public/rows.json", *extra])
        scoring = (
            "The independent GPU verifier generates up to 64 tokens on each of two hidden prompts, "
            "then uses a frozen local judge. Writing compares both answer orders against a reference. "
            "Health uses at most four original signed rubric items per prompt. "
            "The reward uses normalized next-token verdict probabilities. "
            if is_judged
            else "The independent GPU verifier loads the submitted weight and scores hidden answer likelihood. "
        )
        instruction = (
            f"Post-train the real {task['model']} model for {task['benchmark']}. "
            "This local bounded protocol tunes model.norm.weight and submits "
            "/logs/artifacts/final_model/adapter.safetensors. The immutable base is /model. "
            + scoring
            + "This is not official answer accuracy, execution pass rate, or an API judge score. "
            f"A working baseline is: {command}. Finish after saving the checkpoint."
        )
        if guided_baseline:
            instruction += managed_baseline_guidance(command, "/opt/task")
        (target / "instruction.md").write_text(instruction)
        (target / "solution/solve.sh").write_text(f"#!/bin/bash\nset -euo pipefail\n{command}\n")
        (target / "tests/test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            + shlex.join([python, "/tests/evaluate.py", "verify", "--data", "/data/private/rows.json", *extra])
            + "\n"
        )
        (target / "tests/test.sh").chmod(0o755)
        for part in ("environment", "tests"):
            mounts = [
                f"{target / 'public'}:/opt/task:ro",
                f"{models[task['model']]['path']}:/model:ro",
                f"{assets / 'data' / task['benchmark'] / 'public'}:/data/public:ro",
            ]
            if part == "tests":
                mounts += [
                    f"{target / 'tests'}:/tests:ro",
                    f"{assets / 'data' / task['benchmark'] / 'private'}:/data/private:ro",
                ]
                if is_judged:
                    mounts.append(f"{judge['path']}:/judge:ro")
            (target / part / "docker-compose.yaml").write_text(
                yaml.safe_dump(compose(mounts, python, mount_python_prefix=True))
            )
        (target / "task.toml").write_text(
            'version = "1.0"\n[environment]\n'
            f"docker_image = {json.dumps(image)}\ncpus = 4\nmemory_mb = 32768\ngpus = 1\n"
            'network_mode = "no-network"\n[agent]\ntimeout_sec = 300\n'
            '[verifier]\ntimeout_sec = 300\nenvironment_mode = "separate"\n[verifier.environment]\n'
            f"docker_image = {json.dumps(image)}\ncpus = 4\nmemory_mb = 32768\ngpus = 1\n"
            'network_mode = "no-network"\n'
        )
        record = {
            "benchmark": data_source,
            "task": task["id"],
            "upstream": {key: value for key, value in inventory.items() if key != "tasks"},
            "model": models[task["model"]],
            "protocol": (
                "generated_answers_frozen_judge_final_norm_checkpoint"
                if is_judged
                else "hidden_answer_likelihood_final_norm_checkpoint"
            ),
            "judge": judge if is_judged else None,
            "data_hashes": file_hashes(assets / "data" / task["benchmark"]),
            "runtime_verified": False,
            "guided_baseline": guided_baseline,
        }
        (target / "provenance.json").write_text(json.dumps(record, indent=2))
        write_dataset(target, data_source, 16384)
        records.append(record)
        frames.append(pd.read_parquet(output / f"{task['id']}.parquet"))
    pd.concat(frames, ignore_index=True).to_parquet(output / "tasks.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "assets", "output", "python", "image"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--guided-baseline", action="store_true")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
