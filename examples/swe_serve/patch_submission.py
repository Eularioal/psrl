"""Transfer candidate source edits as a patch and replay them in the clean verifier."""

from pathlib import Path

SUBMIT = (
    "mkdir -p /logs/artifacts && git -C /code add -A && "
    "git -C /code diff --cached --binary --no-ext-diff HEAD > /logs/artifacts/submission.patch"
)
REPLAY = (
    "test -f /logs/artifacts/submission.patch\n"
    "if [ -s /logs/artifacts/submission.patch ]; then\n"
    "  git -C /code apply --check /logs/artifacts/submission.patch\n"
    "  git -C /code apply /logs/artifacts/submission.patch\n"
    "fi\n"
)


def configure(task, config):
    task = Path(task)
    config["artifacts"] = []
    solution = task / "solution/solve.sh"
    solution.rename(task / "solution/upstream-solve.sh")
    solution.write_text("#!/bin/bash\nset -euo pipefail\nbash /solution/upstream-solve.sh\n" + SUBMIT + "\n")
    solution.chmod(0o755)
    verifier = task / "tests/test.sh"
    text = verifier.read_text()
    marker = "set -euo pipefail\n"
    if text.count(marker) != 1:
        raise ValueError("Patch replay requires the bounded verifier shell contract.")
    verifier.write_text(text.replace(marker, marker + REPLAY, 1))
    return (
        "Submit source changes as a Git patch. After editing /code, run: "
        f"`{SUBMIT}`. Only this patch is transferred to a clean independent verifier.\n\n"
    )
