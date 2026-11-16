"""Run trusted test nodes without treating collection errors or skips as validation."""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main(cuda_default_device=False):
    cases = json.loads(Path("/tests/local-cases.json").read_text())
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, case in enumerate(cases):
        junit = output / f"case-{index}.xml"
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", HF_HUB_OFFLINE="1", HF_HOME="/hf-cache")
        test_directory = str(Path(case["node"].split("::", 1)[0]).parent)
        env["PYTHONPATH"] = os.pathsep.join((test_directory, "/code/python", "/code"))
        entry = [sys.executable, "/tests/cuda_pytest.py"] if cuda_default_device else [sys.executable, "-m", "pytest"]
        cmd = entry + [
            "-c",
            "/dev/null",
            "--confcutdir=/tests",
            "--import-mode=importlib",
            "-q",
            case["node"],
            "--junitxml=" + str(junit),
        ]
        with (output / f"case-{index}.log").open("w") as stream:
            process = subprocess.run(
                cmd, cwd="/code", env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=100, check=False
            )
        suites = list(ET.parse(junit).getroot().iter("testsuite")) if junit.exists() else []
        total = sum(int(s.get("tests", "0")) for s in suites)
        errors = sum(int(s.get("errors", "0")) for s in suites)
        skipped = sum(int(s.get("skipped", "0")) for s in suites)
        failures = sum(int(s.get("failures", "0")) for s in suites)
        valid = process.returncode in (0, 1) and total > 0 and errors == 0 and skipped == 0
        results.append(
            {
                **case,
                "valid": valid,
                "tests": total,
                "failures": failures,
                "errors": errors,
                "skipped": skipped,
                "exit": process.returncode,
            }
        )
    valid = bool(results) and all(r["valid"] for r in results)
    total = sum(r["tests"] for r in results)
    passed = sum(r["tests"] - r["failures"] - r["errors"] - r["skipped"] for r in results)
    metrics = {
        "reward": passed / total if valid else 0.0,
        "eval_valid": int(valid),
        "local_evaluation": 1,
        "tests": total,
        "passed_tests": passed,
        "selected_nodes": len(cases),
    }
    if cuda_default_device:
        metrics["cuda"] = int(valid)
    (output / "local-results.json").write_text(json.dumps(results, indent=2))
    (output / "reward.json").write_text(json.dumps(metrics))
    print(json.dumps(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-default-device", action="store_true")
    main(**vars(parser.parse_args()))
