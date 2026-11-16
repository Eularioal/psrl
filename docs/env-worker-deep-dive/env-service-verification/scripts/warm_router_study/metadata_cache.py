"""Build a process-local package-name map from installed distribution manifests."""

import csv
import importlib.metadata
import inspect
from collections import defaultdict
from pathlib import PurePosixPath


def manifest_distributions():
    """Avoid statting every installed file on a shared filesystem during startup.

    The installed manifests are read without checking each RECORD entry's
    existence. Real imports still use normal find_spec and version checks.
    This helper never writes to the shared Python installation.
    """
    result = defaultdict(list)
    for distribution in importlib.metadata.distributions():
        names = (distribution.read_text("top_level.txt") or "").split()
        if not names:
            names = set()
            for row in csv.reader((distribution.read_text("RECORD") or "").splitlines()):
                if not row:
                    continue
                path = PurePosixPath(row[0])
                name = path.parts[0] if len(path.parts) > 1 else inspect.getmodulename(str(path))
                if name and "." not in name:
                    names.add(name)
        for name in names:
            result[name].append(distribution.metadata["Name"])
    return dict(result)
