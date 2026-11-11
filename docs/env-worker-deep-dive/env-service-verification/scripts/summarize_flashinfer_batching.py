"""Export completed batching evidence without node addresses or lease credentials."""

import argparse
import hashlib
import json
from statistics import mean


def main(args):
    root = args.root
    status = json.loads((root / "status.json").read_text())
    groups = json.loads((root / "groups.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert status["status"] == "passed" and len(groups) == 6, "Require all six completed groups."
    assert not status["cleanup_errors"] and not status["containers_remaining"], "Require confirmed cleanup."
    assert not status["gpu_after"]["pids"] and status["gpu_after"]["used_mb"] < 128, "GPU must be idle."
    exported = []
    for group in groups:
        timings = [command["timing"] for command in group["commands"]]
        rows = [row for result in group["results"] for row in result["rows"]]
        assert len(rows) == 4 and all(r["evaluation"]["status"] == "PASSED" for r in rows)
        exported.append(
            {
                "index": group["index"],
                "mode": group["mode"],
                "wall_s": group["wall_s"],
                "admission_return_s": sum(
                    t["admitted_at"] - t["requested_at"] + t["released_at"] - t["return_requested_at"] for t in timings
                ),
                "imports_s": sum(r["import_s"] for r in group["results"]),
                "worker_total_s": sum(r["worker_total_s"] for r in group["results"]),
                "rows": rows,
                "gpu_idle_after": not group["gpu_after"]["pids"] and group["gpu_after"]["used_mb"] < 128,
            }
        )
    summary = {
        mode: {
            "wall_s": [g["wall_s"] for g in exported if g["mode"] == mode],
            "mean_wall_s": mean(g["wall_s"] for g in exported if g["mode"] == mode),
            "mean_admission_return_s": mean(g["admission_return_s"] for g in exported if g["mode"] == mode),
        }
        for mode in ("separate", "one_command_fresh", "one_process")
    }
    result = {
        "date": "2026-11-09",
        "passed": True,
        "scope": manifest["scope"],
        "summary": summary,
        "groups": exported,
        "provenance": manifest,
        "cleanup": {"containers": 0, "gpu_pids": [], "gpu_used_mb": status["gpu_after"]["used_mb"]},
        "raw_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("status.json", "groups.json", "manifest.json", "job.log")
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
