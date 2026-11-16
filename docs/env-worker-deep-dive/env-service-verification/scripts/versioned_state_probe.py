"""Compare state validity policies using trusted CPU programs and real processes."""

import argparse
import hashlib
import json
import selectors
import subprocess
import sys
import tempfile
from pathlib import Path

DEPENDENCIES = ("scorer.py", "helper.py", "weights.json")
CHILD = """import json, sys
sys.dont_write_bytecode = True
from scorer import score
weights = json.loads(open('weights.json').read())
for line in sys.stdin:
    print(json.dumps(score(json.loads(line), weights)), flush=True)
"""


def capture(root, complete):
    """Capture a declared closure or the entire small teaching workspace."""
    names = DEPENDENCIES if complete else tuple(sorted(p.name for p in root.iterdir() if p.is_file()))
    return {name: (root / name).read_bytes() for name in names}


def identity(files):
    """Hash file names and contents with unambiguous JSON encoding."""
    return hashlib.sha256(json.dumps({k: v.hex() for k, v in sorted(files.items())}).encode()).hexdigest()


class Worker:
    """Load code and model data once in a private snapshot process."""

    def __init__(self, files):
        self.version = identity(files)
        self.directory = tempfile.TemporaryDirectory(prefix="version-state-")
        for name, content in files.items():
            (Path(self.directory.name) / name).write_bytes(content)
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-B", "-c", CHILD],
            cwd=self.directory.name,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def execute(self, value):
        """Run a request with a bounded response wait."""
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(timeout=5):
            raise TimeoutError("The CPU state worker did not respond.")
        return json.loads(self.process.stdout.readline())

    def close(self):
        """Reap only this worker and remove its private snapshot."""
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.directory.cleanup()


class Policy:
    """Represent four explicit teaching policies, not historical backend modes."""

    def __init__(self, name):
        self.name, self.worker, self.key, self.builds = name, None, None, 0

    def run(self, root, value, complete=True, after_snapshot=None):
        """Check a version before reuse and reject drift before returning current results."""
        files = capture(root, complete)
        version = identity(files)
        key = identity(capture(root, False)) if self.name == "whole-tree" else version
        rebuild = self.worker is None or self.name == "fresh" or not complete
        if self.name != "blind":
            rebuild |= key != self.key
        if rebuild:
            self.close()
            self.worker, self.key = Worker(files), key
            self.builds += 1
        if after_snapshot:
            after_snapshot()
        result = self.worker.execute(value)
        if self.name == "declared" and identity(capture(root, complete)) != version:
            self.close()
            return dict(status="discarded", reason="Dependency version changed after the snapshot.")
        return dict(status="accepted", value=result, requested_version=version, loaded_version=self.worker.version)

    def close(self):
        """Dispose of the retained state when its validity ends."""
        if self.worker:
            self.worker.close()
            self.worker = None


def experiment():
    """Exercise unrelated, direct and transitive edits without synthetic delays."""
    policies = [Policy(name) for name in ("fresh", "blind", "whole-tree", "declared")]
    rows = []
    with tempfile.TemporaryDirectory(prefix="version-workspace-") as directory:
        root = Path(directory)
        (root / "helper.py").write_text("OFFSET = 1\n")
        (root / "weights.json").write_text("3")
        (root / "README.md").write_text("unrelated")
        (root / "scorer.py").write_text("from helper import OFFSET\ndef score(x, w): return (x + OFFSET) * w\n")
        edits = [
            ("initial", None, None, 2, 9),
            ("input-only", None, None, 4, 15),
            ("unrelated", "README.md", "unrelated edit", 2, 9),
            ("transitive-code", "helper.py", "OFFSET = 2\n", 2, 12),
            (
                "direct-code",
                "scorer.py",
                "from helper import OFFSET\ndef score(x, w): return (x + OFFSET) * w + 1\n",
                2,
                13,
            ),
            ("weights", "weights.json", "5", 2, 21),
        ]
        try:
            for label, name, content, value, expected in edits:
                if name:
                    (root / name).write_text(content)
                outputs = {p.name: p.run(root, value) for p in policies}
                rows.append(dict(edit=label, expected=expected, outputs=outputs))
            builds = {p.name: p.builds for p in policies}
            mismatches = {
                p.name: sum(row["outputs"][p.name]["value"] != row["expected"] for row in rows) for p in policies
            }
            guarded = policies[-1]
            drift = guarded.run(root, 2, after_snapshot=lambda: (root / "helper.py").write_text("OFFSET = 3\n"))
            current = guarded.run(root, 2)
            before = guarded.builds
            guarded.run(root, 2, complete=False)
            guarded.run(root, 2, complete=False)
            unknown_rebuilds = guarded.builds - before
        finally:
            for policy in policies:
                policy.close()
    if mismatches != {"fresh": 0, "blind": 3, "whole-tree": 0, "declared": 0}:
        raise RuntimeError("Policy correctness differs from the expected CPU observations.")
    if drift["status"] != "discarded" or current["value"] != 26 or unknown_rebuilds != 2:
        raise RuntimeError("Version drift or unknown dependency handling failed.")
    return dict(
        scope="Trusted CPU teaching workload. Policies are not historical PSRL modes. No performance claim.",
        rows=rows,
        builds_for_six_commands=builds,
        mismatches=mismatches,
        drift=drift,
        current_after_retry=current,
        unknown_dependency_rebuilds=unknown_rebuilds,
        cleanup="All owned workers reaped and temporary snapshots removed.",
        limits=["Declared dependencies must be complete", "No atomic publication lock", "No untrusted code isolation"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
