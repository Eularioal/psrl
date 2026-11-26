"""Explain immutable preparation reuse with CPU files and invented timing units."""

import hashlib
import json
import tempfile
from pathlib import Path


def prepare(source, cache):
    """Identify the actual bytes before reusing a prepared object."""
    content = source.read_bytes()
    target = cache / hashlib.sha256(content).hexdigest()
    if not target.exists():
        target.write_bytes(content)
    return target


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "dependency.txt"
        cache = root / "cache"
        cache.mkdir()
        # Step A identifies the requested immutable dependency version.
        source.write_text("version A")
        first = prepare(source, cache)
        # Step B reuses only preparation, while both task executions still occur.
        second = prepare(source, cache)
        assert first == second, "Identical dependency bytes should reuse preparation."
        results = [path.read_text().upper() for path in (first, second)]
        # Step C keeps an old job pinned when the next job requests changed bytes.
        source.write_text("version B")
        third = prepare(source, cache)
        assert third != first, "Changed bytes require a different prepared object."
        assert first.read_text() == "version A", "The old task must retain its requested version."
        print(
            json.dumps(
                dict(
                    scope="CPU teaching example with invented costs, not GPU measurements.",
                    results=results,
                    new_result=third.read_text().upper(),
                    cold_three_jobs=3 * 5,
                    prepared_three_jobs=6 + 3 * 2,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
