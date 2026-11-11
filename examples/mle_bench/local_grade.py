"""Grade local digit predictions without exposing held-out labels to the agent."""

import csv
import json
from itertools import islice
from pathlib import Path


def grade(submission: Path, labels: dict[str, int]) -> dict[str, float]:
    """Require an exact, unique prediction for every held-out example."""
    invalid = {"reward": 0.0, "accuracy": 0.0, "valid_submission": 0, "local_evaluation": 1}
    if not labels:
        raise ValueError("The protected label set must not be empty.")
    if not submission.is_file():
        return invalid
    try:
        with submission.open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream, strict=True)
            rows = list(islice(reader, len(labels) + 1))
            if reader.fieldnames != ["id", "label"] or len(rows) != len(labels):
                return invalid
    except (UnicodeError, csv.Error):
        return invalid
    if any(set(row) != {"id", "label"} for row in rows):
        return invalid
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(labels):
        return invalid
    if not all(row["label"] in {str(i) for i in range(10)} for row in rows):
        return invalid
    score = sum(int(row["label"]) == labels[row["id"]] for row in rows) / len(rows)
    return {"reward": score, "accuracy": score, "valid_submission": 1, "local_evaluation": 1}


def main():
    labels = json.loads(Path("/protected/labels.json").read_text())
    result = grade(Path("/logs/artifacts/submission.csv"), labels)
    Path("/logs/verifier/reward.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
