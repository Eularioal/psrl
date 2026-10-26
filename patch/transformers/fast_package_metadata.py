"""Infer installed import names without checking every file in each wheel."""

import csv
import importlib.metadata
import inspect
from collections import defaultdict
from pathlib import PurePosixPath


def packages_distributions():
    """Preserve declared names and require one existing file per inferred name."""
    mapping = defaultdict(list)
    for distribution in importlib.metadata.distributions():
        names = (distribution.read_text("top_level.txt") or "").split()
        if not names:
            record = distribution.read_text("RECORD")
            if record is None:
                paths = distribution.files or []
                names = {path.parts[0] if len(path.parts) > 1 else inspect.getmodulename(path) for path in paths}
                names = {name for name in names if name and "." not in name}
            else:
                candidates = defaultdict(list)
                for row in csv.reader(record.splitlines()):
                    if not row:
                        continue
                    path = PurePosixPath(row[0])
                    name = path.parts[0] if len(path.parts) > 1 else inspect.getmodulename(str(path))
                    if name and "." not in name:
                        candidates[name].append(row[0])
                names = [
                    name
                    for name, paths in candidates.items()
                    if any(distribution.locate_file(path).exists() for path in paths)
                ]
        if names:
            distribution_name = distribution.metadata["Name"]
            for name in names:
                mapping[name].append(distribution_name)
    return dict(mapping)
