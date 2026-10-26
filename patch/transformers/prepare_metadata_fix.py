"""Create an isolated Transformers package with bounded metadata file probing."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def prepare(source, output):
    source = source.resolve()
    original = source / "utils/import_utils.py"
    before = original.read_text()
    needle = "PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()"
    if before.count(needle) != 1:
        raise ValueError("The Transformers metadata lookup does not match the supported source.")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "transformers"
    shutil.copytree(source, target, copy_function=lambda src, dst: Path(dst).symlink_to(Path(src).resolve()))
    replacement = target / "utils/import_utils.py"
    replacement.unlink()
    replacement.write_text(
        before.replace(
            needle,
            "from ._package_metadata import packages_distributions\n\n"
            "PACKAGE_DISTRIBUTION_MAPPING = packages_distributions()",
        )
    )
    shutil.copy2(Path(__file__).with_name("fast_package_metadata.py"), target / "utils/_package_metadata.py")
    manifest = {
        "source": str(source),
        "output": str(target),
        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "patched_sha256": hashlib.sha256(replacement.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
