"""Run a bounded selection through the upstream attested serving-test harness."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    cases = json.loads(Path("/tests/local-cases.json").read_text())
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for case in cases:
        node = case["upstream_node"]
        groups.setdefault(node.rsplit("::", 1)[0], []).append(node)
    records = []
    for index, (group, nodes) in enumerate(groups.items()):
        expected = output / f"expected-{index}.txt"
        expected.write_text("\n".join(nodes) + "\n")
        result = output / f"group-{index}.json"
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        with (output / f"group-{index}.log").open("w") as stream:
            process = subprocess.run(
                [sys.executable, "-I", "/tests/run_upstream_e2e_group.py", group, str(expected), str(result)],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        data = json.loads(result.read_text()) if result.exists() else {}
        observed = data.get("nodes", {})
        valid = (
            process.returncode in (0, 1)
            and data.get("all_score_eligible") is True
            and data.get("trusted_identity_ok") is True
            and set(observed) == set(nodes)
        )
        records.append(
            {
                "group": group,
                "valid": valid,
                "tests": len(observed),
                "passed": sum(value.get("passed") is True for value in observed.values()),
            }
        )
    valid = bool(records) and all(record["valid"] for record in records)
    total = sum(record["tests"] for record in records)
    passed = sum(record["passed"] for record in records)
    metrics = {
        "reward": passed / total if valid else 0.0,
        "eval_valid": int(valid),
        "local_evaluation": 1,
        "tests": total,
        "passed_tests": passed,
        "selected_nodes": len(cases),
    }
    (output / "local-results.json").write_text(json.dumps(records, indent=2))
    (output / "reward.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
