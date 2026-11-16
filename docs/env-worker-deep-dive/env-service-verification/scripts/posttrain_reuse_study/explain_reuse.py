"""Explain the real cache identity helpers with tiny CPU teaching objects."""

import json
import os
import tempfile
from pathlib import Path

from bounded_worker import Slot, file_manifest


def main():
    """Demonstrate candidate replacement and same timestamp base invalidation."""
    slot, builds = Slot(), []
    with tempfile.TemporaryDirectory(prefix="posttrain-reuse-teaching-") as directory:
        root = Path(directory)
        base = root / "base.txt"
        base.write_text("base-v1")
        initial = base.stat()

        def build():
            builds.append(base.read_text())
            return {"fixed_base": base.read_text(), "candidate_parameters": None}

        def evaluate(label, parameters):
            key = file_manifest(root)["sha256"]
            model, hit = slot.get(key, build, lambda: None)
            model["candidate_parameters"] = list(parameters)
            return {
                "candidate": label,
                "base": model["fixed_base"],
                "base_reused": hit,
                "base_constructions": len(builds),
                "current_candidate_parameters": model["candidate_parameters"],
            }

        rows = [evaluate("A", [2, 3]), evaluate("B", [5, 7]), evaluate("A", [2, 3])]
        base.write_text("base-v2")
        os.utime(base, ns=(initial.st_atime_ns, initial.st_mtime_ns))
        rows.append(evaluate("A after base change", [2, 3]))
    print(
        json.dumps(
            {"scope": "CPU teaching objects, not model inference or GPU performance.", "requests": rows}, indent=2
        )
    )


if __name__ == "__main__":
    main()
