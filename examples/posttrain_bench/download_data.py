"""Download small disjoint public task samples for the local post-training protocol."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "question", "answer", "train", "test"),
    "aime2025": ("math-ai/aime25", "default", "problem", "answer", "test", "test"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "prompt", "canonical_solution", "test", "test"),
}


def download(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for task, (dataset, config, question, answer, train_split, test_split) in SOURCES.items():
        record = {"dataset": dataset, "config": config, "splits": {}}
        for public, split, offset, count in (
            (True, train_split, 0, 8),
            (False, test_split, 8 if train_split == test_split else 0, 4),
        ):
            url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
                dict(dataset=dataset, config=config, split=split, offset=offset, length=count)
            )
            with urllib.request.urlopen(url, timeout=45) as response:
                content = response.read()
            source = json.loads(content)["rows"]
            if len(source) != count:
                raise ValueError(f"Incomplete source rows for {task!r}.")
            rows = [
                {"prompt": str(r["row"][question]), "answer": str(r["row"][answer]), "source_id": r["row_idx"]}
                for r in source
            ]
            kind = "public" if public else "private"
            destination = output / task / kind
            destination.mkdir(parents=True, exist_ok=False)
            (destination / "rows.json").write_text(json.dumps(rows, ensure_ascii=False))
            record["splits"][kind] = {
                "source_split": split,
                "row_ids": [r["source_id"] for r in rows],
                "response_sha256": hashlib.sha256(content).hexdigest(),
            }
        (output / task / "sources.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    download(parser.parse_args().output)
