"""Keep original candidate sources and hardware declarations for H20 portability checks."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from prepare_flashinfer_probe import prepare


def main(args):
    prepare(args)
    root = args.output.resolve()
    path = root / "subset/workloads/rmsnorm_h2048.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row["workload"]["axes"]["batch_size"] in (1, 16254)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    names = ["gemini-2.5-pro_triton_uc5ufi", "gpt-o3_triton_38f281", "gpt-5_triton_2f0daa"]
    for name in names:
        matches = list((args.trace / "solutions").rglob(name + ".json"))
        assert len(matches) == 1, "Expected one official solution file."
        shutil.copy2(matches[0], root / "subset/solutions" / matches[0].name)
    shutil.copy2(Path(__file__).with_name("flashinfer_racing_probe.py"), root / "flashinfer_racing_probe.py")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["subset_uuids"] = [row["workload"]["uuid"] for row in rows]
    manifest["input_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "subset").rglob("*")
        if path.is_file()
    }
    manifest["protocol"] = {
        "rounds": 13,
        "warmup_rounds": 1,
        "screening_rounds": [1, 2, 3, 4],
        "refinement_rounds": [5, 6, 7, 8],
        "holdout_rounds": [9, 10, 11, 12],
        "screening": "Keep two candidates with smallest median latency in rounds 1-4, per workload.",
        "scope": "Offline screening replay, not a deployed adaptive scheduler or error probability guarantee.",
        "portability": "Three official Triton solutions declare B200. Test unchanged on H20, without relabeling.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
