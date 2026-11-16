"""Coordinate trusted edits, immutable scoring snapshots and result publication."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path

DEPENDENCIES = ("helper.py", "scorer.py", "loader.json")


def digest(value):
    """Hash a canonical JSON value rather than file timestamps."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Workspace:
    """Serialize all participating writers with result acceptance on the same lock."""

    def __init__(self, path):
        self.path = path
        self.live = path / "live"
        self.versions = path / "versions"
        self.live.mkdir(parents=True)
        self.versions.mkdir()
        self.generation = 0

    @contextlib.contextmanager
    def locked(self):
        """Share a file lock across cooperating edit and publication processes."""
        with (self.path / "workspace.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def commit(self, changes):
        """Make one logical edit while publication and snapshot creation are blocked."""
        with self.locked():
            for name, content in changes.items():
                if name not in {*DEPENDENCIES, "README.md"}:
                    raise ValueError(f"The bounded adapter does not support editing {name!r}.")
                temporary = self.live / f".{name}.tmp"
                temporary.write_text(content)
                os.replace(temporary, self.live / name)
            self.generation += 1
            (self.path / "generation.json").write_text(json.dumps(self.generation))

    def capture(self):
        """Freeze all declared content while preventing cooperating writers from editing."""
        with self.locked():
            files = {name: (self.live / name).read_text() for name in DEPENDENCIES}
            key = digest(files)
            target = self.versions / key
            if not target.exists():
                target.mkdir()
                for name, text in files.items():
                    file = target / name
                    file.write_text(text)
                    file.chmod(0o444)
                target.chmod(0o555)
            return dict(
                key=key,
                generation=self.generation,
                code=digest({name: files[name] for name in ("helper.py", "scorer.py")}),
                model=digest(json.loads(files["loader.json"])),
                snapshot=target,
            )

    def publish(self, version, result, record, *, require_current=True):
        """Accept and record a result atomically with respect to cooperating edits."""
        with self.locked():
            if result["code_identity"] != version["code"] or result["model_identity"] != version["model"]:
                return dict(status="rejected", reason="The executed state does not match the requested snapshot.")
            current = {name: (self.live / name).read_text() for name in DEPENDENCIES}
            if require_current and digest(current) != version["key"]:
                return dict(status="rejected", reason="The requested snapshot is no longer current.")
            accepted = dict(status="accepted", snapshot=version["key"], generation=version["generation"], **result)
            temporary = Path(str(record) + ".tmp")
            temporary.write_text(json.dumps(accepted, indent=2) + "\n")
            os.replace(temporary, record)
            return accepted
