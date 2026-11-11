"""Quarantine corrupt Codex databases after all database users have exited."""

import argparse
import datetime
import json
import os
import shutil
import sqlite3
from pathlib import Path


def check(path):
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3) as db:
            return db.execute("PRAGMA integrity_check(5)").fetchall() == [("ok",)]
    except sqlite3.DatabaseError:
        return False


def users(paths):
    found = set()
    targets = {str(p.resolve()) for p in paths}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for fd in (process / "fd").iterdir():
                try:
                    if os.readlink(fd).removesuffix(" (deleted)") in targets:
                        found.add(int(process.name))
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
        except PermissionError as error:
            raise RuntimeError("Cannot verify all local database users.") from error
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-home", type=Path, required=True)
    parser.add_argument("--replacement-state", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.codex_home.resolve()
    replacement = args.replacement_state.resolve()
    names = ["state_5.sqlite", "thread_history_1.sqlite"]
    paths = [root / (name + suffix) for name in names for suffix in ["", "-wal", "-shm"]]
    holders = users(paths)
    if holders:
        raise SystemExit(f"Stop all Codex database users first. Open file PIDs: {holders}")
    if not check(replacement):
        raise SystemExit("Replacement database failed integrity validation.")
    state_healthy = check(root / names[0])
    plan = [names[1]] + ([] if state_healthy else [names[0]])
    print(json.dumps({"quarantine": plan, "replacement_state_required": not state_healthy}))
    if not args.apply:
        return
    backup = root / "backups" / ("resume-repair-apply-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    backup.mkdir(parents=True, exist_ok=False)
    if not state_healthy:
        shutil.copy2(replacement, backup / "replacement.sqlite")
        if not check(backup / "replacement.sqlite"):
            raise SystemExit("Staged replacement failed validation.")
    if users(paths):
        raise SystemExit("Database reopened during preparation. No active files changed.")
    moved = []
    installed = False
    try:
        for name in plan:
            for suffix in ["", "-wal", "-shm"]:
                path = root / (name + suffix)
                if path.exists():
                    target = backup / path.name
                    path.rename(target)
                    moved.append((path, target))
        if not state_healthy:
            (backup / "replacement.sqlite").replace(root / names[0])
            installed = True
        if not check(root / names[0]):
            raise RuntimeError("Installed state failed integrity validation.")
    except Exception:
        if installed:
            for suffix in ["", "-wal", "-shm"]:
                path = root / (names[0] + suffix)
                if path.exists():
                    path.rename(backup / ("failed-replacement" + suffix))
        for original, saved in reversed(moved):
            saved.rename(original)
        raise
    report = {"quarantined": plan, "backup": str(backup), "state_integrity": "ok"}
    (backup / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    print("Restart Codex and resume the original session. History rebuild occurs on resume.")


if __name__ == "__main__":
    main()
