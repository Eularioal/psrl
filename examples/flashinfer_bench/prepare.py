"""Prepare a single official operator subset for the PSRL GPU agent loop."""

import argparse
import json
import shlex
import shutil
from pathlib import Path

import yaml
from examples.gpu_benchmarks.prepare import compose, provenance, write_dataset


def prepare(args):
    record = provenance(args.repo)
    profile_path = args.subset / "local-protocol.json"
    profile = json.loads(profile_path.read_text()) if profile_path.is_file() else {}
    local = bool(getattr(args, "local_profile", False) or profile.get("local"))
    local = local or (args.subset / "local-sampling.json").is_file()
    data_source = "flashinfer_bench_local" if local else "flashinfer_bench"
    out = args.output.resolve()
    subset = args.subset.resolve()
    definitions = list((subset / "definitions").glob("*.json"))
    if len(definitions) != 1:
        raise ValueError("Prepare exactly one operator definition per task.")
    operator = definitions[0].stem
    for name in (f"definitions/{operator}.json", f"workloads/{operator}.jsonl"):
        if not (subset / name).is_file():
            raise FileNotFoundError(subset / name)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("environment", "tests", "solution"):
        (out / name).mkdir()
    sampling = (subset / "local-sampling.json").is_file()
    protocol = (
        "The local verifier checks sampling support and 32 probability buckets using 2048 draws per row. "
        "Support uses the upstream tolerance mask; probabilities use sequential top-k then normalized top-p. "
        "This bounded statistical check is not the original 500000-draw protocol. "
        if sampling
        else "The independent verifier runs the upstream evaluator on every provided workload. "
    )
    (out / "instruction.md").write_text(
        f"Optimize the GPU operation {operator}. Preserve the provided definition and inputs under /data/traces. "
        "Submit /logs/artifacts/solution.json in the FlashInfer Solution JSON format. "
        "A valid starting implementation is in /data/traces/solutions. Use psrl-gpu for GPU experiments. "
        + protocol
        + "Reward is correctness across the entire subset; this is not a speedup claim. "
        + ("This task uses an explicit local input or evaluation profile. " if local else "")
    )
    shutil.copyfile(Path(__file__).with_name("verify.py"), out / "tests/verify.py")
    shutil.copyfile(Path(__file__).with_name("sampling_check.py"), out / "tests/sampling_check.py")
    (out / "tests/test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n" + shlex.join([args.python, "/tests/verify.py"]) + "\n"
    )
    common = [
        f"{subset}:/data/traces:ro",
        f"{args.repo.resolve()}:/opt/fi:ro",
        f"{args.cuda.resolve()}:/usr/local/cuda:ro",
    ]
    for name, mounts in (("environment", common), ("tests", common + [f"{out / 'tests'}:/tests:ro"])):
        definition = compose(mounts, args.python, True)
        definition["services"]["main"]["environment"].update(
            PYTHONPATH="/opt/fi",
            CUDA_HOME="/usr/local/cuda",
            MAX_JOBS="2",
            PATH=f"{Path(args.python).parent}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        )
        (out / name / "docker-compose.yaml").write_text(yaml.safe_dump(definition))
    (out / "tests/test.sh").chmod(0o755)
    (out / "task.toml").write_text(
        'version="1.0"\n[environment]\n'
        f'docker_image={json.dumps(args.image)}\ncpus=4\nmemory_mb=16384\ngpus=1\nnetwork_mode="no-network"\n'
        '[agent]\ntimeout_sec=600\n[verifier]\ntimeout_sec=500\nenvironment_mode="separate"\n'
        f"[verifier.environment]\ndocker_image={json.dumps(args.image)}\n"
        'cpus=4\nmemory_mb=16384\ngpus=1\nnetwork_mode="no-network"\n'
    )
    record.update(
        benchmark=data_source,
        definition=operator,
        local_input_profile=local,
        scoring="local sampling check" if sampling else "local subset correctness, upstream evaluator",
    )
    (out / "provenance.json").write_text(json.dumps(record))
    (out / "solution/solve.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\ncp /data/traces/solutions/*.json /logs/artifacts/solution.json\n"
    )
    write_dataset(out, data_source, 8192)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--local-profile", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
