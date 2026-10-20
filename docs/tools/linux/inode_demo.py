"""Observe inode identity, hard links and open files in a temporary directory."""

import json
import os
from pathlib import Path
import shutil
import tempfile


def snapshot(path: Path) -> dict:
    """Read filesystem identity and metadata through `stat`."""
    value = path.stat()
    return {"device": value.st_dev, "inode": value.st_ino,
            "links": value.st_nlink, "bytes": value.st_size}


def run() -> dict:
    """Exercise real filesystem calls using only this experiment's files."""
    result = {}
    with tempfile.TemporaryDirectory(prefix="inode-demo-", dir="/tmp") as directory:
        root = Path(directory)
        original = root / "a.txt"
        alias = root / "b.txt"
        copied = root / "copy.txt"
        renamed = root / "renamed.txt"
        # Step 1 creates a directory entry and a file with six bytes.
        original.write_bytes(b"hello\n")
        result["created"] = snapshot(original)
        # Step 2 gives the same inode another directory entry.
        os.link(original, alias)
        result["hard_link"] = {"a": snapshot(original), "b": snapshot(alias)}
        # Step 3 copies bytes into a distinct file and renames the original entry.
        shutil.copyfile(original, copied)
        result["copy"] = snapshot(copied)
        original.rename(renamed)
        result["renamed"] = snapshot(renamed)
        # Step 4 removes both names while an open descriptor retains the file.
        with renamed.open("rb") as handle:
            renamed.unlink()
            result["one_name_removed"] = snapshot(alias)
            alias.unlink()
            value = os.fstat(handle.fileno())
            result["no_names_but_open"] = {
                "inode": value.st_ino, "links": value.st_nlink,
                "read": handle.read().decode(),
                "names_exist": renamed.exists() or alias.exists(),
            }
        # Step 5 closes the descriptor and removes only our temporary files.
        result["descriptor_closed"] = handle.closed
    result["temporary_directory_removed"] = not root.exists()
    first = result["created"]
    second = result["copy"]
    assert result["hard_link"]["a"] == result["hard_link"]["b"], "Hard links differ."
    assert first["inode"] == result["renamed"]["inode"], "Rename changed inode identity."
    assert (first["device"], first["inode"]) != (second["device"], second["inode"]), "Copy reused identity."
    assert result["no_names_but_open"]["links"] == 0, "A directory entry remains."
    assert result["no_names_but_open"]["read"] == "hello\n", "Open file lost its contents."
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
