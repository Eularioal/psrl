"""Import upstream operator identities with mandatory observed input captures."""

import argparse
import json
import shlex
import shutil
from pathlib import Path

import pandas as pd
import yaml
from examples.fastkernels_bench.runtime import materialize
from examples.gpu_benchmarks.acceptance import managed_baseline_guidance
from examples.gpu_benchmarks.heterogeneous_inventory import fastkernels_inventory
from examples.gpu_benchmarks.prepare import compose, write_dataset
from examples.swe_serve.prepare import file_hashes


def prepare(
    repo,
    captures,
    output,
    python,
    image,
    targets=None,
    guided_baseline=False,
    dependency_roots=(),
    container_python=False,
):
    repo, captures, output = Path(repo).resolve(), Path(captures).resolve(), Path(output).resolve()
    inventory = fastkernels_inventory(repo, captures)
    selected = inventory["tasks"]
    if targets is not None:
        selected = [item for item in selected if item["qualified_name"] in targets]
        if {item["qualified_name"] for item in selected} != set(targets):
            raise ValueError("Unknown operator target.")
    missing = [item["qualified_name"] for item in selected if not item["has_capture"]]
    if not selected or missing:
        raise ValueError(f"Actual forward captures are required for every selected target: {missing!r}.")
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime_patch = materialize(repo, runtime)
    frames, records = [], []
    for item in selected:
        name = f"fastkernels-l{item['level']}-{item['name']}-{item['class_name'].lower()}"
        task = output / name
        for part in ("environment", "tests", "public", "solution"):
            (task / part).mkdir(parents=True)
        source = repo / f"fastkernels/tasks/baseline/L{item['level']}/{item['name']}.py"
        shutil.copy2(source, task / "public/reference.py")
        candidate_namespace = None
        if item["level"] == 1 and item["name"] == "rms_norm":
            original = 'torch.library.Library("fastkernels_norm", "DEF")'
            reference = source.read_text()
            if reference.count(original) != 1:
                raise ValueError("Review changed upstream RMSNorm custom operator registration.")
            candidate_namespace = "psrl_candidate_fastkernels_norm"
            reference = reference.replace(original, f'torch.library.Library("{candidate_namespace}", "DEF")')
            (task / "public/reference.py").write_text(reference)
        shutil.copy2(Path(__file__).with_name("verify.py"), task / "tests/verify.py")
        (task / "tests/spec.json").write_text(json.dumps(item, indent=2))
        candidate = f"/logs/artifacts/candidate/L{item['level']}/{item['name']}.py"
        baseline = (
            f"mkdir -p {shlex.quote(str(Path(candidate).parent))}; cp /opt/task/reference.py {shlex.quote(candidate)}"
        )
        sidecar = source.with_suffix(".cu")
        if sidecar.is_file():
            shutil.copy2(sidecar, task / "public" / sidecar.name)
            baseline += (
                f"; cp {shlex.quote('/opt/task/' + sidecar.name)} "
                f"{shlex.quote(str(Path(candidate).with_suffix('.cu')))}"
            )
        (task / "solution/solve.sh").write_text(f"#!/bin/bash\nset -euo pipefail\n{baseline}\n")
        instruction = (
            f"Optimize upstream operator {item['qualified_name']} on CUDA. Submit {candidate}. "
            "Keep its class name and call signature. Imports from sibling candidate modules resolve to the "
            "immutable upstream baseline. Evaluation compares shared random weights and captured inputs. "
            "The local quick protocol uses one captured shape and two correctness rounds. "
            f"A functional starting point is: {baseline}."
        )
        if sidecar.is_file():
            instruction += (
                " The adjacent .cu file is part of the submission: keep it beside the Python module. "
                "You may edit both files; CUDA compilation runs inside the dedicated task image."
            )
        if candidate_namespace:
            instruction += f" Keep custom operator registrations in the {candidate_namespace} namespace."
        if guided_baseline:
            instruction += managed_baseline_guidance(baseline, "/opt/task")
        (task / "instruction.md").write_text(instruction)
        (task / "tests/test.sh").write_text(
            f"#!/bin/bash\nset -euo pipefail\n{shlex.quote(python)} /tests/verify.py\n"
        )
        (task / "tests/test.sh").chmod(0o755)
        for part in ("environment", "tests"):
            mounts = [f"{runtime}:/upstream:ro", f"{task / 'public'}:/opt/task:ro"]
            for root in dependency_roots:
                path = Path(root).resolve(strict=True)
                mounts.append(f"{path}:{path}:ro")
            if part == "tests":
                mounts += [f"{captures}:/captures:ro", f"{task / 'tests'}:/tests:ro"]
            definition = compose(mounts, python, mount_python_prefix=not container_python)
            definition["services"]["main"]["environment"]["PYTHONPATH"] = "/upstream"
            (task / part / "docker-compose.yaml").write_text(yaml.safe_dump(definition))
        (task / "task.toml").write_text(
            'version = "1.0"\n[environment]\n'
            f"docker_image = {json.dumps(image)}\ncpus = 4\nmemory_mb = 16384\ngpus = 1\n"
            'network_mode = "no-network"\n[agent]\ntimeout_sec = 180\n'
            '[verifier]\ntimeout_sec = 180\nenvironment_mode = "separate"\n[verifier.environment]\n'
            f"docker_image = {json.dumps(image)}\ncpus = 4\nmemory_mb = 16384\ngpus = 1\n"
            'network_mode = "no-network"\n'
        )
        record = {
            **item,
            "benchmark": "fastkernels_local",
            "task": name,
            "revision": inventory["revision"],
            "runtime_patch": runtime_patch,
            "capture_hashes": file_hashes(captures),
            "submission_sources": file_hashes(task / "public"),
            "candidate_namespace": candidate_namespace,
            "guided_baseline": guided_baseline,
        }
        (task / "provenance.json").write_text(json.dumps(record, indent=2))
        write_dataset(task, "fastkernels_local", 8192)
        frames.append(pd.read_parquet(output / f"{name}.parquet"))
        records.append(record)
    pd.concat(frames, ignore_index=True).to_parquet(output / "tasks.parquet", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "upstream_targets": inventory["operator_targets"],
                "selected_targets": len(records),
                "complete_scope": len(records) == inventory["operator_targets"],
                "tasks": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "captures", "output", "python", "image"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--guided-baseline", action="store_true")
    parser.add_argument("--dependency-roots", nargs="*", default=[])
    parser.add_argument("--container-python", action="store_true", help="Use Python and dependencies from the image.")
    prepare(**vars(parser.parse_args()))
