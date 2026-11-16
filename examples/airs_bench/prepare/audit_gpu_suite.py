"""Inspect prepared AIRS input boundaries on CPU without opening private answers."""

import argparse
import json
from pathlib import Path


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    seen, rows = set(), []
    for item in manifest["tasks"]:
        task = item["task"]
        if task in seen:
            raise ValueError("The manifest contains a duplicate task.")
        seen.add(task)
        data = root / "airs_prepared" / task
        test = data / ("test/queries" if item["kind"] == "retrieval" else "test")
        public = set(json.loads((test / "dataset_info.json").read_text())["features"])
        private = set(json.loads((data / "test_with_labels/dataset_info.json").read_text())["features"])
        baseline = json.loads((data / "train/psrl-baseline.json").read_text())
        if not public < private:
            raise ValueError(f"The public test schema for {task!r} does not remove private columns.")
        if any(not set(row) <= public for row in baseline["train_x"] + baseline["test_x"]):
            raise ValueError(f"Baseline features for {task!r} include nonpublic columns.")
        if len(baseline["test_x"]) != item["submission_rows"]:
            raise ValueError(f"The submission row count for {task!r} is inconsistent.")
        rows.append(
            {
                "task": task,
                "public_columns": sorted(public),
                "excluded_private_columns": sorted(private - public),
                "submission_rows": item["submission_rows"],
            }
        )
    if len(seen) != 19 or "TimeSeriesForecastingKaggleWebTrafficMASE" in seen:
        raise ValueError("Expected exactly nineteen non-Kaggle task identities.")
    return {"passed": True, "scope": "CPU schema audit, not GPU or policy execution", "tasks": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().data), indent=2))


if __name__ == "__main__":
    main()
