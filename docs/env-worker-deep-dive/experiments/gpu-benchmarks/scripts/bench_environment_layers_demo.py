"""Simulate image layers, container writes and mounts using temporary CPU files."""

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class Mount:
    source: Path
    readonly: bool


class ContainerView:
    """Model file lookup without creating a real container or security boundary."""

    def __init__(self, image, upper, mounts):
        self.image = image
        self.upper = upper
        self.mounts = mounts
        upper.mkdir()

    def location(self, name, write=False):
        path = PurePosixPath(name)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Use an absolute simulated container path without parent traversal.")
        for target, mount in sorted(self.mounts.items(), key=lambda item: len(item[0]), reverse=True):
            relative = path.relative_to(target) if path.is_relative_to(target) else None
            if relative is not None:
                if write and mount.readonly:
                    raise PermissionError(f"Read-only simulated mount: {target}.")
                return mount.source / str(relative)
        relative = str(path.relative_to("/"))
        candidate = self.upper / relative
        return candidate if write or candidate.exists() else self.image / relative

    def read(self, name):
        return self.location(name).read_text()

    def write(self, name, content):
        path = self.location(name, write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def remove(self):
        shutil.rmtree(self.upper)


def observe(label, condition):
    if not condition:
        raise RuntimeError(f"Teaching observation failed: {label}.")
    print(f"PASS {label}")


def run(root):
    # Diagram 1 prepares a shared image and separately stored host inputs.
    image = root / "image"
    (image / "code").mkdir(parents=True)
    (image / "code/operator.txt").write_text("baseline")
    (image / "runtime").mkdir()
    (image / "runtime/package-version.txt").write_text("image-version")
    main_prefix = root / "main-python"
    main_prefix.mkdir()
    (main_prefix / "package.txt").write_text("main-version")
    hidden = root / "hidden"
    hidden.mkdir()
    (hidden / "answer.txt").write_text("candidate")
    artifacts = root / "agent-artifacts"
    artifacts.mkdir()
    verifier_artifacts = root / "verifier-artifacts"
    verifier_artifacts.mkdir()
    common = {"/python": Mount(main_prefix, readonly=True)}
    agent = ContainerView(
        image,
        root / "agent-upper",
        {**common, "/logs/artifacts": Mount(artifacts, readonly=False)},
    )
    verifier = ContainerView(
        image,
        root / "verifier-upper",
        {
            **common,
            "/logs/artifacts": Mount(verifier_artifacts, readonly=False),
            "/data/private": Mount(hidden, readonly=True),
        },
    )

    # Diagram 2 writes only the agent's upper layer unless a mount matches.
    agent.write("/code/operator.txt", "candidate")
    agent.write("/runtime/package-version.txt", "agent-version")
    observe("agent edit stays in its own upper layer", agent.read("/code/operator.txt") == "candidate")
    observe("verifier still sees clean image", verifier.read("/code/operator.txt") == "baseline")
    observe("shared image is unchanged", (image / "runtime/package-version.txt").read_text() == "image-version")
    try:
        agent.write("/python/package.txt", "changed")
    except PermissionError:
        print("PASS read-only mount rejects writes in this simulation")
    else:
        raise RuntimeError("The simulated read-only mount allowed a write.")
    observe("main Python files are unchanged", (main_prefix / "package.txt").read_text() == "main-version")
    try:
        agent.read("/data/private/answer.txt")
    except FileNotFoundError:
        print("PASS private input is absent from the agent view")
    else:
        raise RuntimeError("The agent unexpectedly read the simulated hidden input.")

    # Diagram 3 publishes an explicit artifact on a writable host mount.
    agent.write("/logs/artifacts/submission.txt", agent.read("/code/operator.txt"))
    observe("artifact reaches the host directory", (artifacts / "submission.txt").read_text() == "candidate")
    shutil.copytree(artifacts, verifier_artifacts, dirs_exist_ok=True)
    agent.write("/logs/artifacts/late.txt", "written after transfer")
    observe("verifier receives an explicit snapshot", not (verifier_artifacts / "late.txt").exists())

    # Diagram 4 explicitly replays the artifact in the clean verifier view.
    verifier.write("/code/operator.txt", verifier.read("/logs/artifacts/submission.txt"))
    observe(
        "independent verifier can score the submission",
        verifier.read("/code/operator.txt") == verifier.read("/data/private/answer.txt"),
    )
    print(
        json.dumps(
            {
                "image": "image-version",
                "agent": agent.read("/runtime/package-version.txt"),
                "verifier": verifier.read("/runtime/package-version.txt"),
            }
        )
    )

    # Diagram 5 removes only private upper layers, keeping mounted data intact.
    agent.remove()
    verifier.remove()
    observe("container removal keeps image and artifact", image.is_dir() and (artifacts / "submission.txt").is_file())


def main():
    print("Teaching simulation only. No Docker, package installation or GPU calls.")
    with tempfile.TemporaryDirectory(prefix="psrl-bench-layers-") as directory:
        root = Path(directory)
        run(root)
    observe("all demonstration files are cleaned up", not root.exists())


if __name__ == "__main__":
    main()
