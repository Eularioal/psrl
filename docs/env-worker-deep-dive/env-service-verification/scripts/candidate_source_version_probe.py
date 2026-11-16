"""
Compare frozen and current RE source loaders using small CPU artifact versions.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


def load_verifier(
    path,
):
    """
    Import the real verifier without entering its GPU evaluation function.
    """
    spec = importlib.util.spec_from_file_location("probe_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(
    verifier,
    root,
):
    """
    Exercise actual candidate loading in the parent and two fresh child modes.
    """
    module = load_verifier(verifier)
    cases = []
    for invalidation in (py_compile.PycInvalidationMode.TIMESTAMP, py_compile.PycInvalidationMode.UNCHECKED_HASH):
        directory = root / invalidation.name
        directory.mkdir()
        source = directory / "model.py"
        source.write_text("VALUE = 1\n")
        os.utime(source, (1_700_000_000, 1_700_000_000))
        py_compile.compile(str(source), doraise=True, invalidation_mode=invalidation)
        source.write_text("VALUE = 2\n")
        os.utime(source, (1_700_000_000, 1_700_000_000))
        observed = {"same_process": module.load("candidate_model", source).VALUE}
        code = (
            "import importlib.util,sys\n"
            "s=importlib.util.spec_from_file_location('verifier',sys.argv[1])\n"
            "m=importlib.util.module_from_spec(s)\n"
            "s.loader.exec_module(m)\n"
            "print(m.load('candidate_model',sys.argv[2]).VALUE)\n"
        )
        for label, flags in (("fresh_process", []), ("fresh_process_B", ["-B"])):
            result = subprocess.check_output(
                [sys.executable, *flags, "-c", code, str(verifier), str(source)], text=True, timeout=10
            )
            observed[label] = int(result.strip())
        cases.append({"cache_mode": invalidation.name, "source_value": 2, "source_bytes": 10, **observed})
    return cases


def main():
    """
    Preserve loader identities and conditional reproduction evidence without a GPU.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, default=Path("examples/re_bench/restricted_mlm_verify.py"))
    args = parser.parse_args()
    before_bytes, current_bytes = args.before.read_bytes(), args.verifier.read_bytes()
    with tempfile.TemporaryDirectory(prefix="re-source-probe-") as directory:
        root = Path(directory)
        frozen = root / "original_verifier.py"
        frozen.write_bytes(before_bytes)
        for version in ("before", "fixed"):
            (root / version).mkdir()
        before = probe(frozen, root / "before")
        fixed = probe(args.verifier.resolve(), root / "fixed")
    labels = ("same_process", "fresh_process", "fresh_process_B")
    reproduced = all(row[label] == 1 for row in before for label in labels)
    corrected = all(row[label] == 2 for row in fixed for label in labels)
    if not reproduced or not corrected:
        raise RuntimeError("The bounded actual-loader reproduction did not satisfy its declared observations.")
    report = {
        "status": "passed",
        "scope": "CPU fault injection through actual RE restricted MLM loader functions. No GPU or model inference.",
        "boundary": "No naturally occurring benchmark misgrading was observed. Indirect imports are not covered.",
        "before_sha256": hashlib.sha256(before_bytes).hexdigest(),
        "fixed_sha256": hashlib.sha256(current_bytes).hexdigest(),
        "before": before,
        "fixed": fixed,
        "checks": {"six_old_source_observations": reproduced, "six_current_source_observations": corrected},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
