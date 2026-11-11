"""Build Harbor tasks and PSRL Parquet rows from pinned upstream assets."""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import tomllib
import yaml


def provenance(repo):
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Pin a clean upstream checkout before preparing tasks.")
    return {"revision": revision}


def compose(mounts, python, mount_python_prefix=False):
    prefix = Path(python).parent.parent
    python_mounts = []
    if mount_python_prefix:
        for path in dict.fromkeys((prefix, Path(python).resolve().parent.parent)):
            python_mounts.append(f"{path}:{path}:ro")
    return {
        "services": {
            "main": {
                "volumes": python_mounts + mounts,
                "environment": {
                    "PATH": f"{prefix}/bin:/usr/local/bin:/usr/bin:/bin",
                    "PYTHONPATH": "/opt/upstream/src:/opt/upstream",
                    "OMP_NUM_THREADS": "1",
                    "TORCHINDUCTOR_COMPILE_THREADS": "2",
                    "HF_HUB_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                },
            }
        }
    }


def write_dataset(task, benchmark, gpu_memory_mb):
    info = {
        "task_path": str(task.resolve()),
        "reward_key": "reward",
        "gpus": 1,
        "gpu_memory_mb": gpu_memory_mb,
        "benchmark": benchmark,
    }
    row = {
        "data_source": benchmark,
        "agent_name": "gpu_benchmark",
        "prompt": [{"role": "user", "content": (task / "instruction.md").read_text()}],
        "ability": "gpu-engineering",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": info,
    }
    pd.DataFrame([row]).to_parquet(task.parent / f"{task.name}.parquet", index=False)


def prepare(args):
    repo, out = args.repo.resolve(), args.output.resolve()
    backend = getattr(args, "environment_backend", "docker")
    verifier_image = getattr(args, "verifier_image", None) or args.image
    if backend == "env_worker" and (not getattr(args, "verifier_image", None) or args.benchmark == "inference_bench"):
        raise ValueError("Native tasks require prebuilt runtime and verifier images without upstream Compose tasks.")
    record = provenance(repo)
    dataset_source = args.benchmark
    record["environment_backend"] = backend
    if out.exists():
        raise FileExistsError(out)
    if args.benchmark == "inference_bench":
        if args.harbor_task is None:
            raise ValueError("InferenceBench requires an upstream generated --harbor-task with quality baselines.")
        config = tomllib.loads((args.harbor_task / "task.toml").read_text())
        if "inferencebench" not in config.get("metadata", {}):
            raise ValueError("Expected an official InferenceBench generated task.")
        for name in ("environment/collect.sh", "tests/test.sh", "tests/ib/src/eval/inference/quality_gate.py"):
            if not (args.harbor_task / name).is_file():
                raise FileNotFoundError(args.harbor_task / name)
        shutil.copytree(args.harbor_task, out)
        # Docker bind mounts preserve the upstream hand-off without a Modal volume.
        for directory in ("environment", "tests"):
            path = out / directory / "docker-compose.yaml"
            if path.exists():
                raise ValueError("Review the upstream Compose definition before importing this task.")
            path.write_text(
                yaml.safe_dump(
                    {
                        "services": {
                            "main": {
                                "volumes": ["${HOST_ARTIFACTS_PATH}/ib_handoff:/mnt/ib_handoff"],
                            }
                        }
                    }
                )
            )
        (out / "provenance.json").write_text(json.dumps(record, indent=2))
        write_dataset(out, args.benchmark, args.gpu_memory_mb)
        return out

    public_mounts = [f"{repo}:/opt/upstream:ro"]
    hf_cache = getattr(args, "hf_cache", None)
    if hf_cache is not None:
        if not (hf_cache / "hub").is_dir():
            raise ValueError("The Hugging Face cache must contain a prepared hub directory.")
        public_mounts.append(f"{hf_cache.resolve()}:/root/.cache/huggingface:ro")
    private_mounts = []
    spec = {"benchmark": args.benchmark, "python": args.python, "timeout": args.timeout}
    if args.benchmark == "kernel_bench":
        reference = repo / "KernelBench" / f"level{args.level}" / args.problem
        if not reference.is_file():
            raise FileNotFoundError(reference)
        spec["reference"] = str(Path("/opt/upstream") / reference.relative_to(repo))
        spec["backend"] = args.backend
        record.update(benchmark="kernel_bench", level=args.level, problem=args.problem)
        if getattr(args, "bounded_inputs", False):
            from examples.kernel_bench.quick_check import bounded_source

            source, overrides = bounded_source(reference.read_text())
            dataset_source = "kernel_bench_local"
            spec["input_profile"] = "bounded_module_dimensions"
            record.update(benchmark=dataset_source, dimension_overrides=overrides)
        instruction = (
            f"Optimize {spec['reference']}. Submit /logs/artifacts/submission.py defining ModelNew. "
            "Match the reference on randomized inputs. Use psrl-gpu for GPU commands. "
            "The verifier checks correctness and reports speed. Do not modify the evaluator."
        )
        if dataset_source == "kernel_bench_local":
            instruction = instruction.replace("checks correctness and reports speed", "checks correctness only")
            instruction += f" Local bounded input dimensions: {json.dumps(overrides)}. This is not canonical scoring."
        record["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    elif args.benchmark == "mle_bench":
        if args.data is None or not args.competition:
            raise ValueError("MLE-bench requires --data and --competition from official preparation.")
        root = args.data.resolve() / args.competition / "prepared"
        for name in ("public", "private"):
            if not (root / name).is_dir():
                raise FileNotFoundError(root / name)
        public_mounts.append(f"{root / 'public'}:/data/public:ro")
        private_mounts.append(f"{args.data.resolve()}:/data/mle:ro")
        spec["competition"] = args.competition
        instruction = (repo / "mlebench/competitions" / args.competition / "description.md").read_text()
        instruction += (
            "\nPublic inputs are in /data/public. Train with psrl-gpu and submit "
            "/logs/artifacts/submission.csv. Hidden labels are available only to the verifier."
        )
    else:
        if args.data is None or not (args.data / "tokenizer").is_dir() or not (args.data / "data").is_dir():
            raise ValueError("Autoresearch requires a prepared --data cache containing tokenizer/ and data/.")
        for name in ("tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt", "data/shard_06542.parquet"):
            if not (args.data / name).is_file():
                raise FileNotFoundError(args.data / name)
        if not any(path.name != "shard_06542.parquet" for path in (args.data / "data").glob("shard_*.parquet")):
            raise ValueError("Autoresearch requires at least one official training shard.")
        manifest = args.data / "assets.json"
        if manifest.is_file():
            record["assets"] = json.loads(manifest.read_text())
        public_mounts.append(f"{args.data.resolve()}:/root/.cache/autoresearch:ro")
        instruction = (
            "Improve /opt/upstream/train.py under the unchanged prepare.py training and evaluation budget. "
            "Copy the code to a writable directory and use psrl-gpu for each bounded experiment. "
            "Submit /logs/artifacts/train.py. The verifier replays the submitted training script. "
            "At most two experiments are allowed in this episode."
        )
    for name in ("environment", "tests", "solution"):
        (out / name).mkdir(parents=True)
    (out / "instruction.md").write_text(instruction)
    (out / "provenance.json").write_text(json.dumps(record, indent=2))
    verifier_mounts = public_mounts + private_mounts + [f"{out / 'tests'}:/tests:ro"]
    for name, mounts in (("environment", public_mounts), ("tests", verifier_mounts)):
        definition = compose(mounts, args.python, getattr(args, "mount_python_prefix", False))
        if backend == "docker":
            (out / name / "docker-compose.yaml").write_text(yaml.safe_dump(definition))
    (out / "tests/spec.json").write_text(json.dumps(spec))
    shutil.copy2(Path(__file__).with_name("evaluate.py"), out / "tests/evaluate.py")
    if dataset_source == "kernel_bench_local":
        shutil.copy2(Path(__file__).parents[1] / "kernel_bench/quick_check.py", out / "tests/quick_check.py")
    if args.benchmark == "autoresearch":
        shutil.copy2(Path(__file__).with_name("autoresearch_replay.py"), out / "tests/autoresearch_replay.py")
    (out / "tests/test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n" + shlex.join([args.python, "/tests/evaluate.py"]) + "\n"
    )
    (out / "tests/test.sh").chmod(0o755)
    (out / "task.toml").write_text(
        'version = "1.0"\n[environment]\n'
        f"docker_image = {json.dumps(args.image)}\ncpus = 4\nmemory_mb = 16384\ngpus = 1\n"
        'network_mode = "no-network"\n[agent]\n'
        f"timeout_sec = {args.timeout}\n[verifier]\ntimeout_sec = {args.timeout}\n"
        'environment_mode = "separate"\n[verifier.environment]\n'
        f"docker_image = {json.dumps(verifier_image)}\ncpus = 4\nmemory_mb = 16384\n"
        f'gpus = {int(args.benchmark != "mle_bench")}\nnetwork_mode = "no-network"\n'
    )
    if dataset_source == "kernel_bench_local":
        (out / "solution/submission.py").write_text(source + "\nModelNew = Model\n")
        (out / "solution/solve.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\ncp /solution/submission.py /logs/artifacts/submission.py\n"
        )
    elif args.benchmark == "kernel_bench" and args.level == 1 and args.problem == "1_Square_matrix_multiplication_.py":
        solution = Path(__file__).parents[1] / "kernel_bench/solutions/matmul.py"
        shutil.copy2(solution, out / "solution/submission.py")
        (out / "solution/solve.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\ncp /solution/submission.py /logs/artifacts/submission.py\n"
        )
    elif args.benchmark == "autoresearch":
        (out / "solution/solve.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\ncp /opt/upstream/train.py /logs/artifacts/train.py\n"
        )
    write_dataset(out, dataset_source, args.gpu_memory_mb)
    return out


def main(benchmark=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if benchmark is None:
        parser.add_argument("benchmark", choices=["kernel_bench", "mle_bench", "inference_bench", "autoresearch"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=benchmark != "inference_bench")
    parser.add_argument("--verifier-image")
    parser.add_argument("--environment-backend", choices=["docker", "env_worker"], default="docker")
    parser.add_argument("--python", default="/usr/local/bin/python")
    parser.add_argument("--mount-python-prefix", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--competition")
    parser.add_argument("--harbor-task", type=Path)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problem", default="1_Square_matrix_multiplication_.py")
    parser.add_argument("--backend", choices=["cuda", "triton"], default="triton")
    parser.add_argument("--bounded-inputs", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--gpu-memory-mb", type=int, default=8192)
    args = parser.parse_args()
    if benchmark:
        args.benchmark = benchmark
    if args.timeout <= 0 or args.gpu_memory_mb <= 0:
        parser.error("Timeout and GPU memory budget must be positive.")
    print(prepare(args))


if __name__ == "__main__":
    main()
