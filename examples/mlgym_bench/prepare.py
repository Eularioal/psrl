"""Prepare bounded MLGym checkpoint tasks with independent GPU scoring."""

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from examples.gpu_benchmarks.acceptance import managed_baseline_guidance
from examples.gpu_benchmarks.prepare import compose, provenance, write_dataset
from examples.mlgym_bench.rl import ENVIRONMENTS
from examples.swe_serve.prepare import file_hashes


def prepare_rl(repo, output, python, image, guided_baseline=False):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    revision = provenance(repo)
    cuda_root = subprocess.check_output(
        [
            python,
            "-c",
            "import importlib.util; print(next(iter(importlib.util.find_spec('nvidia.cuda_nvcc')"
            ".submodule_search_locations)))",
        ],
        text=True,
        timeout=30,
    ).strip()
    libraries = subprocess.check_output(
        [
            python,
            "-c",
            "import glob,json,site; print(json.dumps([p for base in site.getsitepackages() "
            "for p in sorted(glob.glob(base+'/nvidia/*/lib'))]))",
        ],
        text=True,
        timeout=30,
    )
    library_path = ":".join(["/usr/local/cuda/compat", "/usr/lib64", *json.loads(libraries)])
    output.mkdir(parents=True, exist_ok=False)
    rows, records = [], []
    for task, environment in ENVIRONMENTS.items():
        source = repo / "data" / task
        target = output / task
        public = target / "public"
        shutil.copytree(source / "src", public / "src")
        shutil.copy2(Path(__file__).with_name("rl.py"), public / "baseline.py")
        (public / "python").write_text(
            f'#!/bin/sh\nexport LD_LIBRARY_PATH={shlex.quote(library_path)}\nexec {shlex.quote(python)} "$@"\n'
        )
        (public / "python").chmod(0o755)
        for part in ("environment", "tests", "solution"):
            (target / part).mkdir()
        shutil.copy2(Path(__file__).with_name("rl.py"), target / "tests/evaluate.py")
        command = shlex.join(["/opt/task/python", "/opt/task/baseline.py", "train", "--task", task])
        instruction = (
            f"Train a policy for the real Gymnax {environment} environment using CUDA. "
            "Submit /logs/artifacts/policy.msgpack with the public fixed network structure. "
            f"Start with: {command}. You may write a different training program. "
            "The independent verifier restores only weights and evaluates unseen random seeds. "
            "This local bounded protocol uses 16 parallel environments, 64 frames, and two evaluation seeds. "
            "It is not the canonical MLGym evaluation budget. Finish after saving the checkpoint."
        )
        if guided_baseline:
            instruction += managed_baseline_guidance(command, "/opt/task")
        (target / "instruction.md").write_text(instruction)
        (target / "solution/solve.sh").write_text(f"#!/bin/bash\nset -euo pipefail\n{command}\n")
        (target / "tests/test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            + shlex.join(["/opt/task/python", "/tests/evaluate.py", "verify", "--task", task])
            + "\n"
        )
        (target / "tests/test.sh").chmod(0o755)
        for part in ("environment", "tests"):
            mounts = [f"{public}:/opt/task:ro"]
            if part == "tests":
                mounts.append(f"{target / 'tests'}:/tests:ro")
            definition = compose(mounts, python, mount_python_prefix=True)
            definition["services"]["main"]["environment"].update(
                JAX_PLATFORMS="cuda",
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
                CUDA_ROOT=cuda_root,
            )
            definition["services"]["main"]["environment"]["PATH"] = (
                "/opt/task:" + definition["services"]["main"]["environment"]["PATH"]
            )
            (target / part / "docker-compose.yaml").write_text(yaml.safe_dump(definition))
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
            **revision,
            "benchmark": "mlgym_rl_local",
            "task": task,
            "upstream_files": file_hashes(source),
            "network_width": 32,
            "network_layers": 2,
            "train_updates": 2,
            "env_batch": 16,
            "frames": 64,
            "held_out_seeds": [109, 211],
            "runtime_verified": False,
            "guided_baseline": guided_baseline,
        }
        (target / "provenance.json").write_text(json.dumps(record, indent=2))
        write_dataset(target, "mlgym_rl_local", 8192)
        rows.append(pd.read_parquet(output / f"{task}.parquet"))
        records.append(record)
    pd.concat(rows, ignore_index=True).to_parquet(output / "tasks.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--guided-baseline", action="store_true")
    args = parser.parse_args()
    prepare_rl(args.repo, args.output, args.python, args.image, args.guided_baseline)


if __name__ == "__main__":
    main()
