"""Keep pressure evidence outside every task container's writable log mounts."""

import hashlib
import json
import os
from pathlib import Path


def audit_path(logs_dir, audit_root=None):
    """Identify a trial without placing authoritative evidence inside its mounts."""
    logs_dir = Path(logs_dir).resolve()
    root = Path(audit_root or os.environ["SCIENCE_PRESSURE_AUDIT_DIR"]).resolve()
    if root.is_relative_to(logs_dir.parent):
        raise ValueError("Pressure audit directory must be outside the trial directory.")
    name = hashlib.sha256(str(logs_dir).encode()).hexdigest()
    return root / f"{name}.json"


def save_pressure_evidence(logs_dir, command, audit_root=None):
    """Capture command output before the agent can execute its first shell command."""
    try:
        pressure = json.loads(command.get("stdout") or "")
        if not isinstance(pressure, dict) or "status" not in pressure:
            raise ValueError("Pressure output must be a status dictionary.")
    except (ValueError, TypeError) as exc:
        pressure = {"status": "invalid", "error": str(exc)}
    record = {"logs_dir": str(Path(logs_dir).resolve()), "pressure": pressure, "command": command}
    path = audit_path(logs_dir, audit_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(path)
    return path


def load_pressure_evidence(session_dir, audit_root=None):
    """Read host evidence even if the agent deleted its mounted log directories."""
    paths = [audit_path(p.parent / "agent", audit_root) for p in Path(session_dir).glob("*/*/config.json")]
    if len(paths) != 1 or not paths[0].exists():
        return {"status": "missing"}
    return json.loads(paths[0].read_text())["pressure"]
