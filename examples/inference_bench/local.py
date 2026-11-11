"""Prepare deterministic local InferenceBench service optimization tasks."""

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path

import yaml
from examples.gpu_benchmarks.prepare import compose, write_dataset


def prepare(args):
    out = args.output.resolve()
    references = []
    for index in range(2):
        item = json.loads((args.references / f"request-{index}.json").read_text())
        text = item["response"]["choices"][0]["message"]["content"]
        if not text.strip():
            raise ValueError("Reference responses must be nonempty.")
        references.append(text)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("environment", "tests", "solution"):
        (out / name).mkdir()
    (out / "instruction.md").write_text(
        "Optimize a deterministic local Mistral inference service. Submit /logs/artifacts/serve.sh. "
        "It must run a foreground server compatible with /opt/resident/client.py. "
        "Starting implementation: python /opt/resident/server.py --model /opt/model "
        "--dtype bfloat16 --max-model-len 2048. "
        "The independent GPU verifier starts your service, sends two fixed real MMLU-Pro requests and stops it. "
        "Outputs must exactly match the protected baseline. Reward is 1/(1+request batch seconds). "
        "Use psrl-gpu for GPU experiments. This local fixed-output protocol is not official InferenceBench scoring."
    )
    (out / "tests/references.json").write_text(json.dumps(references))
    shutil.copyfile(Path(__file__).with_name("local_verify.py"), out / "tests/verify.py")
    (out / "tests/test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n" + shlex.join([args.python, "/tests/verify.py"]) + "\n"
    )
    for name, mounts in (("environment", []), ("tests", [f"{out / 'tests'}:/tests:ro"])):
        (out / name / "docker-compose.yaml").write_text(yaml.safe_dump(compose(mounts, args.python, True)))
    (out / "tests/test.sh").chmod(0o755)
    (out / "task.toml").write_text(
        'version="1.0"\n[environment]\n'
        f'docker_image={json.dumps(args.image)}\ncpus=4\nmemory_mb=32768\ngpus=1\nnetwork_mode="no-network"\n'
        '[agent]\ntimeout_sec=600\n[verifier]\ntimeout_sec=500\nenvironment_mode="separate"\n'
        f'[verifier.environment]\ndocker_image={json.dumps(args.image)}\ncpus=4\nmemory_mb=32768\ngpus=1\nnetwork_mode="no-network"\n'
    )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "benchmark": "inference_bench_local",
                "official_scoring": False,
                "quality_protocol": "exact deterministic baseline outputs",
                "requests": 2,
                "baseline_sha256": hashlib.sha256(json.dumps(references).encode()).hexdigest(),
            }
        )
    )
    (out / "solution/solve.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "echo 'exec python /opt/resident/server.py --model /opt/model --dtype bfloat16 --max-model-len 2048' "
        "> /logs/artifacts/serve.sh\n"
    )
    write_dataset(out, "inference_bench_local", 24576)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--python", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
