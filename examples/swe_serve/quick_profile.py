"""Create an explicitly local score-service profile for bounded PSRL pipeline validation."""

import argparse
import json
import shutil
from pathlib import Path

import toml
import yaml
from examples.gpu_benchmarks.acceptance import managed_baseline_guidance
from examples.gpu_benchmarks.prepare import write_dataset

TASK = "ss-sgl22544-score-multi-item-delimiters"
GROUP = "test/registered/prefill_only/test_multi_item_scoring_adapter.py::TestMultiItemScoringOptimizationAdapter"


def prepare(source, output, guided_reference=False, image=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    provenance = json.loads((source / "provenance.json").read_text())
    if provenance.get("task") != TASK:
        raise ValueError("This profile is defined only for the audited multi-item score task.")
    shutil.copytree(source, output)
    tests = output / "tests"
    (tests / "test.sh").rename(tests / "official-test.sh")
    (tests / "test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
        "python3 -I /tests/validate_vendor.py\nsource /tests/prep.sh\n"
        "python3 -I /tests/validate_model_assets.py\n"
        "python3 -I /tests/validate_upstream_e2e_package.py /logs/verifier/package.json\n"
        "set +e\n"
        f"python3 -I /tests/run_upstream_e2e_group.py {GROUP} /tests/fail_to_pass.txt "
        "/tests/pass_to_pass.txt /tests/upstream_e2e_expected_deselected.txt /logs/verifier/group.json\n"
        "status=$?\nset -e\nif [[ $status -gt 1 ]]; then exit $status; fi\n"
        "python3 -I /tests/score_local.py\n"
    )
    (tests / "test.sh").chmod(0o755)
    (tests / "score_local.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "d=json.loads(Path('/logs/verifier/group.json').read_text())\n"
        f"assert d['logical_group'] == {GROUP!r}\n"
        "assert d['all_score_eligible'] and d['verdict_code'] in (0,1)\n"
        "assert len(d['expected']) == 2 and set(d['collected']) == set(d['expected'])\n"
        "r={'reward':float(d['all_passed']),'local_evaluation':1,'eval_valid':1,'tests':len(d['expected'])}\n"
        "Path('/logs/verifier/reward.json').write_text(json.dumps(r))\n"
    )
    for part in ("environment", "tests"):
        path = output / part / "docker-compose.yaml"
        definition = yaml.safe_load(path.read_text())
        service = definition["services"]["main"]
        service["volumes"] = [item.replace(str(source), str(output)) for item in service["volumes"]]
        if guided_reference and part == "environment":
            acceptance = output / "acceptance"
            acceptance.mkdir()
            shutil.copy2(source / "solution/changes.patch", acceptance / "changes.patch")
            shutil.copy2(Path(__file__).with_name("score_probe.py"), acceptance / "probe.py")
            (acceptance / "run.sh").write_text(
                "#!/bin/bash\nset -euo pipefail\ncd /code\n"
                "git apply --check /opt/acceptance/changes.patch\ngit apply /opt/acceptance/changes.patch\n"
                "source /task/agent-prep.sh\nexport HF_HOME=/hf-cache HF_HUB_OFFLINE=1\n"
                "python3 /opt/acceptance/probe.py\n"
            )
            service["volumes"].append(f"{acceptance}:/opt/acceptance:ro")
        path.write_text(yaml.safe_dump(definition))
    config = toml.loads((output / "task.toml").read_text())
    config["agent"]["timeout_sec"] = 240
    config["verifier"]["timeout_sec"] = 240
    if image:
        config["environment"]["docker_image"] = image
    (output / "task.toml").write_text(toml.dumps(config))
    instruction = (
        "This local acceptance task checks the PSRL GPU service execution pipeline. "
        "The private verifier runs two upstream GPU tests with the original package and asset attestations. "
        "It does not reproduce the full 26-test SWE-Serve score. "
    )
    if guided_reference:
        instruction += (
            "A known reference patch is deliberately public for functional acceptance, not benchmark ranking. "
            "Run bash /opt/acceptance/run.sh once. It applies the patch and exercises Engine.score on CUDA. "
            "Then finish and submit the /code workspace. Do not run more experiments."
        )
        instruction += managed_baseline_guidance("bash /opt/acceptance/run.sh", "/code")
        (output / "solution/solve.sh").write_text("#!/bin/bash\nset -euo pipefail\nbash /opt/acceptance/run.sh\n")
    else:
        instruction += (source / "instruction.md").read_text()
    (output / "instruction.md").write_text(instruction)
    provenance.update(local_profile=True, quick_group=GROUP, guided_reference=guided_reference)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    write_dataset(output, "swe_serve_local", 80000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guided-reference", action="store_true")
    parser.add_argument("--image")
    args = parser.parse_args()
    prepare(args.source, args.output, args.guided_reference, args.image)


if __name__ == "__main__":
    main()
