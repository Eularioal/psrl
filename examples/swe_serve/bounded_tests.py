"""Build an explicitly local subset of immutable upstream regression tests."""

import argparse
import json
import shutil
from pathlib import Path

import toml
import yaml
from examples.gpu_benchmarks.prepare import write_dataset
from examples.swe_serve.patch_submission import configure


def prepare(source, output, image, case_limit=4, hardware_profile=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not 2 <= case_limit <= 8:
        raise ValueError("The quick protocol requires between two and eight selected cases.")
    tests = source / "tests"
    hardware_mode = None
    marker = "    h100_1) expected_gpu=H100 ;;"
    if hardware_profile is not None:
        if hardware_profile != "h20_1":
            raise ValueError("The local hardware extension supports the single H20 profile only.")
        text = (tests / "prep.sh").read_text()
        if "SWE_SERVE_HARDWARE_PROFILE" in text:
            if text.count(marker) == 1:
                hardware_mode = "shell"
            elif (tests / "model_assets.json").exists():
                assets = json.loads((tests / "model_assets.json").read_text())
                if assets.get("profiles", {}).get("h100_1", {}).get("allocation_gpus") != 1:
                    raise ValueError("The local H20 profile requires a single Hopper GPU configuration.")
                hardware_mode = "manifest"
            else:
                raise ValueError("Review the changed upstream hardware selection contract.")
    lookup = {}
    for path in (tests / "postmerge_tests").rglob("*.py"):
        lookup.setdefault(path.name, []).append(path)
    selected = []
    for kind in ("fail_to_pass", "pass_to_pass"):
        path = tests / f"{kind}.txt"
        if not path.exists():
            continue
        used_files = set()
        candidates = []
        for line in path.read_text().splitlines():
            node = line.strip()
            if not node or node.startswith("#") or "::" not in node:
                continue
            filename, rest = node.split("::", 1)
            matches = lookup.get(Path(filename).name, [])
            if len(matches) != 1:
                continue
            relative = str(matches[0].relative_to(tests))
            candidates.append({"kind": kind, "upstream_node": node, "node": "/tests/" + relative + "::" + rest})
        # Cover distinct source files before additional parameterizations of one file.
        ordered = []
        for item in candidates:
            file = item["node"].split("::")[0]
            if file not in used_files:
                ordered.append(item)
                used_files.add(file)
        ordered.extend(item for item in candidates if item not in ordered)
        selected.extend(ordered[: case_limit // 2])
    if len(selected) < 2 or not any(item["kind"] == "fail_to_pass" for item in selected):
        raise ValueError("Use a task-specific protocol for non-pytest performance gates.")
    attested = (tests / "glmv_penalty_plugin.py").exists()
    if attested:
        selected = [
            next(item for item in selected if item["kind"] == kind) for kind in ("fail_to_pass", "pass_to_pass")
        ]
    shutil.copytree(source, output)
    (output / "tests/local-cases.json").write_text(json.dumps(selected, indent=2))
    shutil.copy2(Path(__file__).with_name("bounded_verify.py"), output / "tests/bounded_verify.py")
    if attested:
        shutil.copy2(Path(__file__).with_name("attested_verify.py"), output / "tests/attested_verify.py")
    (output / "tests/test.sh").rename(output / "tests/official-test.sh")
    (output / "tests/test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nsource /tests/prep.sh\npython3 -I /tests/"
        + ("attested_verify.py\n" if attested else "bounded_verify.py\n")
    )
    (output / "tests/test.sh").chmod(0o755)
    launcher = output / "environment/sglang"
    launcher.write_text("#!/usr/local/bin/python3\nfrom sglang.cli.main import main\nmain()\n")
    launcher.chmod(0o755)
    for part in ("environment", "tests"):
        path = output / part / "docker-compose.yaml"
        definition = yaml.safe_load(path.read_text())
        service = definition["services"]["main"]
        service["volumes"] = [v.replace(str(source), str(output)) for v in service["volumes"]]
        service["volumes"].append(f"{launcher}:/usr/local/bin/sglang:ro")
        path.write_text(yaml.safe_dump(definition))
    config = toml.loads((output / "task.toml").read_text())
    config["environment"]["docker_image"] = image
    config["agent"]["timeout_sec"] = 180
    config["verifier"]["timeout_sec"] = 390 if attested else 240
    if hardware_mode == "shell":
        prep = output / "tests/prep.sh"
        text = prep.read_text()
        prep.write_text(text.replace(marker, marker + "\n    h20_1) expected_gpu=H20 ;;"))
        config["verifier"].setdefault("env", {})["SWE_SERVE_HARDWARE_PROFILE"] = hardware_profile
    elif hardware_mode == "manifest":
        manifest = output / "tests/model_assets.json"
        assets = json.loads(manifest.read_text())
        assets["profiles"]["h100_1"]["gpu_name_contains"] = "H20"
        manifest.write_text(json.dumps(assets, indent=2))
        config["verifier"].setdefault("env", {})["SWE_SERVE_HARDWARE_PROFILE"] = "h100_1"
    submission = configure(output, config)
    (output / "task.toml").write_text(toml.dumps(config))
    (output / "instruction.md").write_text(
        f"Local quick protocol: {len(selected)} explicitly selected upstream regressions. "
        "The original full verifier remains available as official-test.sh. This score is not the full "
        "SWE-Serve protocol or an exhaustive regression claim.\n\n"
        + submission
        + (source / "instruction.md").read_text()
    )
    provenance = json.loads((source / "provenance.json").read_text())
    provenance.update(
        local_profile=True,
        selected_cases=selected,
        runtime_verified=False,
        hardware_profile=hardware_profile,
        hardware_adaptation=hardware_mode,
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    write_dataset(output, "swe_serve_regression_local", 80000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "output", "image"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--case-limit", type=int, default=4)
    parser.add_argument("--hardware-profile", choices=["h20_1"])
    prepare(**vars(parser.parse_args()))
