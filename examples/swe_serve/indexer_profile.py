"""Prepare bounded correctness checks for the real CUDA indexer projection."""

import argparse
import json
import shutil
from pathlib import Path

import toml
import yaml
from examples.gpu_benchmarks.prepare import write_dataset
from examples.swe_serve.patch_submission import configure

TASK = "ss-sgl23856-use-torch-torch-mm-for-deepseek-v3-2-indexer"


def prepare(source, output, image=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    provenance = json.loads((source / "provenance.json").read_text())
    if provenance.get("task") != TASK:
        raise ValueError("This bounded profile applies only to the audited indexer task.")
    benchmark = source / "tests/speed_check/bench_indexer.py"
    if "def _build_indexer():" not in benchmark.read_text():
        raise ValueError("Review the changed upstream projection construction contract.")
    shutil.copytree(source, output)
    shutil.copy2(Path(__file__).with_name("indexer_verify.py"), output / "tests/indexer_verify.py")
    (output / "tests/test.sh").rename(output / "tests/official-test.sh")
    (output / "tests/test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nsource /tests/prep.sh\npython3 -I /tests/indexer_verify.py\n"
    )
    (output / "tests/test.sh").chmod(0o755)
    for part in ("environment", "tests"):
        path = output / part / "docker-compose.yaml"
        definition = yaml.safe_load(path.read_text())
        service = definition["services"]["main"]
        service["volumes"] = [v.replace(str(source), str(output)) for v in service["volumes"]]
        path.write_text(yaml.safe_dump(definition))
    config = toml.loads((output / "task.toml").read_text())
    config["agent"]["timeout_sec"] = 120
    config["verifier"]["timeout_sec"] = 180
    if image:
        config["environment"]["docker_image"] = image
    submission = configure(output, config)
    (output / "task.toml").write_text(toml.dumps(config))
    (output / "instruction.md").write_text(
        "Local quick protocol: run the real CUDA NSA projection on 1, 8, and 32 tokens and compare "
        "with independent float32 matrix multiplication. The original public constructor and 7168-by-64 "
        "weight shape are preserved. This is functional correctness, not the full model or speedup gate. "
        "The original verifier is retained as official-test.sh.\n\n"
        + submission
        + (source / "instruction.md").read_text()
    )
    provenance.update(local_profile=True, protocol="bounded_cuda_projection", token_counts=[1, 8, 32])
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    write_dataset(output, "swe_serve_projection_local", 8192)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image")
    prepare(**vars(parser.parse_args()))
