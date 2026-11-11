"""Prepare official shapes and a mathematical reference control candidate."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from prepare_flashinfer_probe import prepare


def main(args):
    prepare(args)
    root = args.output.resolve()
    workloads = root / "subset/workloads/rmsnorm_h2048.jsonl"
    rows = [json.loads(line) for line in workloads.read_text().splitlines()]
    rows = [row for row in rows if row["workload"]["axes"]["batch_size"] in (1, 16254)]
    workloads.write_text("".join(json.dumps(row) + "\n" for row in rows))
    definition = json.loads((root / "subset/definitions/rmsnorm_h2048.json").read_text())
    solution = json.loads((root / "subset/solutions/flashinfer_wrapper_0af255.json").read_text())
    solution.update(name="torch_reference", author="local-reference-control")
    solution["description"] = "Unmodified official mathematical reference packaged as a research control candidate."
    solution["spec"]["dependencies"] = ["torch"]
    solution["sources"] = [{"path": "main.py", "content": definition["reference"]}]
    (root / "subset/solutions/torch_reference.json").write_text(json.dumps(solution, indent=2))
    for name in ("flashinfer_batch_worker.py", "flashinfer_batch_experiment.py"):
        shutil.copy2(Path(__file__).with_name(name), root / name)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["subset_uuids"] = [row["workload"]["uuid"] for row in rows]
    manifest["input_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "subset").rglob("*")
        if path.is_file()
    }
    manifest["scope"] = "Two official shapes, FlashInfer and a local reference control, three policies, two rounds."
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
