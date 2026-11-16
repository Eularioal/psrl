"""Inject bounded publication faults without a model, Docker or a GPU."""

import argparse
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

from workspace import Workspace


def initial(workspace):
    """Use small files so the experiment exercises only the version contract."""
    workspace.commit(
        {
            "helper.py": "OFFSET = 0\n",
            "scorer.py": "from helper import OFFSET\n",
            "loader.json": '{"length":8}',
            "README.md": "Unrelated documentation.\n",
        }
    )


def payload(version):
    """Represent trusted execution metadata without pretending to run inference."""
    return dict(value=1, code_identity=version["code"], model_identity=version["model"])


def checks(root):
    """Follow capture, execution metadata, acceptance and a cooperating edit."""
    workspace = Workspace(root / "workspace")
    initial(workspace)
    version = workspace.capture()
    result = payload(version)
    record = root / "accepted.json"
    wrong_code = dict(result, code_identity="wrong")
    wrong_model = dict(result, model_identity="wrong")
    outcomes = {}
    outcomes["executed_code_mismatch_rejected"] = (
        workspace.publish(version, wrong_code, record)["status"] == "rejected"
    )
    outcomes["executed_model_mismatch_rejected"] = (
        workspace.publish(version, wrong_model, record)["status"] == "rejected"
    )
    outcomes["rejected_result_not_written"] = not record.exists()
    workspace.commit({"README.md": "Changed without changing declared dependencies.\n"})
    outcomes["unrelated_edit_accepted"] = workspace.publish(version, result, record)["status"] == "accepted"
    workspace.commit({"helper.py": "OFFSET = 10\n"})
    outcomes["obsolete_current_result_rejected"] = workspace.publish(version, result, record)["status"] == "rejected"
    outcomes["explicit_historical_snapshot_accepted"] = (
        workspace.publish(version, result, record, require_current=False)["status"] == "accepted"
    )
    outcomes["captured_content_unchanged"] = (version["snapshot"] / "helper.py").read_text() == "OFFSET = 0\n"
    workspace.commit({"helper.py": "OFFSET = 0\n"})
    outcomes["identical_content_after_revert_accepted"] = (
        workspace.publish(version, result, record)["status"] == "accepted"
    )

    # Pause actual record writing while the publication lock is still held.
    publication_paused = threading.Event()
    release_publication = threading.Event()
    edit_attempted = threading.Event()
    edit_finished = threading.Event()
    errors = []
    real_write = Path.write_text

    def paused_write(path, *args, **kwargs):
        if path == Path(str(record) + ".tmp"):
            publication_paused.set()
            if not release_publication.wait(2):
                raise TimeoutError("The bounded publication gate was not released.")
        return real_write(path, *args, **kwargs)

    def publish():
        try:
            accepted = workspace.publish(version, result, record)
            if accepted["status"] != "accepted":
                raise RuntimeError("The controlled publication was unexpectedly rejected.")
        except BaseException as error:
            errors.append(repr(error))

    def edit():
        try:
            edit_attempted.set()
            workspace.commit({"helper.py": "OFFSET = 20\n"})
            edit_finished.set()
        except BaseException as error:
            errors.append(repr(error))

    publisher = threading.Thread(target=publish, daemon=True)
    editor = threading.Thread(target=edit, daemon=True)
    with patch.object(Path, "write_text", paused_write):
        publisher.start()
        try:
            if not publication_paused.wait(2):
                raise TimeoutError("The publication did not reach the fault gate.")
            editor.start()
            if not edit_attempted.wait(2):
                raise TimeoutError("The cooperating writer did not start.")
            outcomes["cooperating_edit_blocked_before_record_write"] = not edit_finished.wait(0.2)
        finally:
            release_publication.set()
            publisher.join(2)
            if editor.ident is not None:
                editor.join(2)
    if publisher.is_alive() or editor.is_alive() or errors:
        raise RuntimeError(f"The bounded concurrency check failed with errors {errors!r}.")
    accepted = json.loads(record.read_text())
    outcomes["record_published_before_cooperating_edit"] = (
        accepted["snapshot"] == version["key"]
        and workspace.capture()["key"] != version["key"]
        and edit_finished.is_set()
    )
    if not all(outcomes.values()):
        raise RuntimeError(f"The publication contract failed with outcomes {outcomes!r}.")
    for directory in workspace.versions.iterdir():
        directory.chmod(0o755)
    return dict(
        status="passed",
        scope="Standard-library fault injection with synthetic execution metadata. No inference or scheduler.",
        checks=outcomes,
        boundary=(
            "All writers cooperate. Content validity at acceptance does not promise future currency or crash recovery."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="state-contract-") as temporary:
        evidence = checks(Path(temporary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
