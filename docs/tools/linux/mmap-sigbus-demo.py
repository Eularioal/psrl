#!/usr/bin/env python3
"""Demonstrate unlink and truncation with a mapped temporary file on Linux."""

import mmap
import os
import resource
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def child(path, mode):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Diagram B: Keep the file open while changing its name or size.
    with open(path, "r+b") as stream:
        with mmap.mmap(stream.fileno(), 32768) as view:
            copied = stream.read(1)
            print("Before: copied={}, mapped={}.".format(copied[0], view[0]), flush=True)
            if mode == "unlink":
                # Diagram C1: Removing a name preserves the open file object.
                os.unlink(path)
                print("After unlink: mapped={}.".format(view[0]), flush=True)
            else:
                # Diagram C2: Truncation removes the backing data.
                os.ftruncate(stream.fileno(), 0)
                print("After truncate: copied={}.".format(copied[0]), flush=True)
                print("Reading the mapping now.", flush=True)
                print(view[0], flush=True)


def main():
    # Diagram A: The parent owns cleanup even if its child receives SIGBUS.
    with tempfile.TemporaryDirectory(prefix="mmap-sigbus-", dir="/tmp") as folder:
        for mode in ("unlink", "truncate"):
            path = Path(folder) / mode
            with path.open("wb") as stream:
                stream.write(b"A")
                stream.truncate(32768)
            result = subprocess.run(
                [sys.executable, __file__, "--child", str(path), mode],
                capture_output=True,
                text=True,
                timeout=5,
            )
            print("Case: {}.".format(mode))
            print(result.stdout, end="")
            print("Return code: {}.".format(result.returncode))
            expected = 0 if mode == "unlink" else -signal.SIGBUS
            if result.returncode != expected:
                raise RuntimeError("Unexpected result: {}.".format(result.stderr))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
