"""Audit all locality pairs, phase boundaries, and explicit preparation costs."""

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

from analyze import summarize


def duration(item):
    return (datetime.fromisoformat(item["finished_at"]) - datetime.fromisoformat(item["started_at"])).total_seconds()


def analyze(root):
    raw = json.loads((root / "matrix/evidence.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if raw["policies"] != protocol["policies"]:
        raise ValueError("The execution order differs from the frozen protocol.")
    expected = set(protocol["families"])
    for block in raw["blocks"]:
        if len(block["tasks"]) != len(expected) or {t["family"] for t in block["tasks"]} != expected:
            raise ValueError("Every planned family must appear exactly once in every block.")
    result = summarize(root / "matrix/evidence.json", "local")
    result["scope"] = protocol["scope"]
    result["protocol"] = protocol
    result["source_sha256"] = json.loads((root / "source-sha256.json").read_text())
    result["analysis_source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("analyze.py", "analyze_locality.py")
    }
    if (root / "equivalence-audit.json").exists():
        result["equivalence_audit"] = json.loads((root / "equivalence-audit.json").read_text())
    result["import_probe"] = json.loads((root / "import-probe.json").read_text())
    phases = {}
    for block, public in zip(raw["blocks"], result["blocks"], strict=True):
        for task, public_task in zip(block["tasks"], public["tasks"], strict=True):
            public_task["result"] = {k: task["result"].get(k) for k in ("rewards", "exception")}
            files = list((root / "matrix/jobs" / task["tag"]).glob("*/*/result.json"))
            if len(files) != 1:
                raise ValueError("Require exactly one trial result, without silently dropping retries.")
            trial = json.loads(files[0].read_text())
            if trial["exception_info"] or trial["verifier_environment_mode"] != "separate":
                raise ValueError("Each task must finish its independent verification without exception.")
            row = {key + "_s": duration(trial[key]) for key in ("environment_setup", "agent_execution", "verifier")}
            row["other_s"] = task["completion_s"] - sum(row.values())
            public_task["phases"] = row
            phases.setdefault(task["family"], {}).setdefault(block["policy"], []).append(row)
    result["phase_means"] = {
        family: {policy: {key: mean(row[key] for row in rows) for key in rows[0]} for policy, rows in policies.items()}
        for family, policies in phases.items()
    }
    copy_s = float(re.search(r"elapsed_s=([0-9.]+)", (root / "copy.time").read_text()).group(1))
    copy_bytes = int(
        re.search(r"Total file size: ([0-9,]+) bytes", (root / "copy.log").read_text()).group(1).replace(",", "")
    )
    if (root / "copy.exit").read_text().strip() != "0":
        raise ValueError("The immutable runtime copy did not complete.")
    static = result["total"]["static_s"]
    local = result["total"]["treatment_s"]
    result["preparation"] = dict(
        copy_s=copy_s,
        bytes=copy_bytes,
        first_profitable_batch_count=math.floor(copy_s / (static - local)) + 1 if static > local else None,
        projection_scope="Extrapolation using measured warm means and copy cost only. Not a measured cold campaign.",
        projections=[
            dict(batches=n, static_s=n * static, local_with_copy_s=copy_s + n * local) for n in (1, 3, 5, 10)
        ],
    )
    result["acceptance_passed"] = result["physical_release"] and all(
        row["valid"] and row["reduction_pct"] >= 10 and min(row["paired_reduction_pct"]) > 0
        for row in result["families"].values()
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("total", "families", "preparation", "acceptance_passed")}, indent=2))
