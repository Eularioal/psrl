"""Prepare four explicitly bounded local InferenceBench scenario tasks."""

import argparse
import json
import shutil
from pathlib import Path

import yaml
from examples.gpu_benchmarks.prepare import compose, write_dataset
from examples.inference_bench.scenario_verify import PROFILES


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = ["collect"] if args.collect else list(PROFILES)
    manifest = []
    for name in names:
        task = root / name
        for directory in ["environment", "tests", "solution"]:
            (task / directory).mkdir(parents=True)
        profiles = list(PROFILES) if name == "collect" else [name]
        config = {"profiles": profiles, "collect_references": args.collect}
        (task / "tests/profile.json").write_text(json.dumps(config))
        if not args.collect:
            shutil.copy2(args.references, task / "tests/references.json")
        for file in ["scenario_verify.py", "local_verify.py"]:
            shutil.copy2(Path(__file__).with_name(file), task / "tests" / file)
        for directory, extra in [("environment", []), ("tests", [f"{task}/tests:/tests:ro"])]:
            config = compose(extra, args.python, True)
            config["services"]["main"]["environment"]["PYTHONPATH"] = "/opt/python-extra:/tests"
            (task / directory / "docker-compose.yaml").write_text(yaml.safe_dump(config))
        (task / "solution/solve.sh").write_text(
            "#!/bin/bash\nset -eu\nprintf '%s\\n' 'exec python /opt/resident/server.py --m"
            "odel /opt/model --dtype bfloat16 --max-model-len 2048' > /logs/artifacts/s"
            "erve.sh\n"
        )
        (task / "tests/test.sh").write_text(f"#!/bin/bash\nset -eu\n{args.python} /tests/scenario_verify.py\n")
        (task / "tests/test.sh").chmod(0o755)
        (task / "task.toml").write_text(
            'version="1.0"\n[environment]\n'
            + f"docker_image={json.dumps(args.image)}\n"
            + 'cpus=4\nmemory_mb=32768\ngpus=1\nnetwork_mode="no-network"\n[agent]\ntimeout_sec=120\n'
            + '[verifier]\ntimeout_sec=450\nenvironment_mode="separate"\n[verifier.environment]\n'
            + f"docker_image={json.dumps(args.verifier_image)}\n"
            + 'cpus=4\nmemory_mb=32768\ngpus=1\nnetwork_mode="no-network"\n'
        )
        (task / "instruction.md").write_text(
            f"Optimize a foreground Mistral server for local scenario {name}. Submit /logs/artifacts/serve.sh. "
            "Start from python /opt/resident/server.py --model /opt/model --dtype bfloat16 --max-model-len 2048. "
            f"Local request profile: {json.dumps({n: PROFILES[n] for n in profiles})}. "
            "The independent GPU verifier sends real MMLU-Pro requests, checks exact ou"
            "tputs against protected deterministic baselines, "
            "and rewards 1/(1+request seconds). Setup time is excluded. This bounded pr"
            "ofile represents the scenario type, "
            "not the original official workload size or judge. Use psrl-gpu for GPU experiments."
        )
        (task / "provenance.json").write_text(
            json.dumps(
                {
                    "benchmark": "inference_bench_local",
                    "profiles": profiles,
                    "official_scoring": False,
                    "collect_references": args.collect,
                }
            )
        )
        write_dataset(task, "inference_bench_local", 24576)
        manifest.append(
            {
                "task_path": str(task),
                "data_source": "inference_bench_local",
                "environment_backend": "docker",
                "gpu_memory_mb": 24576,
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--references", type=Path)
    p.add_argument("--collect", action="store_true")
    for name in ["image", "verifier-image", "python"]:
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    if not args.collect and args.references is None:
        p.error("--references is required for scored tasks.")
    prepare(args)


if __name__ == "__main__":
    main()
