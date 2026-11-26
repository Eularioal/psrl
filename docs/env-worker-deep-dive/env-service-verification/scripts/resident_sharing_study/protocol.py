"""Exchange owned compute pause requests without releasing GPU memory leases."""

import json
import time
from pathlib import Path


def save(path, value):
    """Replace one writer's JSON file atomically."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    """Read an optional protocol message without treating absence as an empty request."""
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


class PausePoint:
    """Acknowledge a request only after actual CUDA synchronization."""

    def __init__(self, directory, trial_id, synchronize):
        self.directory = Path(directory)
        self.trial_id = trial_id
        self.synchronize = synchronize
        self.events = []
        self.started_ns = None
        self.updates = 0
        self.update_times_ns = []

    def _request(self):
        request = read(self.directory / "request.json")
        if request and request.get("stop"):
            raise RuntimeError("The study requested worker shutdown.")
        if not request or request.get("trial_id") != self.trial_id:
            raise RuntimeError("The compute request belongs to another trial.")
        return request

    def __call__(self, stage, updates):
        self.updates = updates
        self.synchronize()
        if stage == "updated":
            self.update_times_ns.append(time.monotonic_ns())
        if stage == "prepared":
            save(self.directory / "state.json", {"trial_id": self.trial_id, "state": "prepared"})
            deadline = time.monotonic() + 60
            while not self._request().get("go"):
                if time.monotonic() >= deadline:
                    raise TimeoutError("The prepared trial did not start.")
                time.sleep(0.001)
            self.started_ns = time.monotonic_ns()
        request = self._request()
        if not request.get("pause"):
            return
        epoch = request["epoch"]
        acknowledgement = {
            "trial_id": self.trial_id,
            "state": "paused",
            "epoch": epoch,
            "updates": updates,
            "ack_ns": time.monotonic_ns(),
            "memory_retained": True,
        }
        self.events.append(acknowledgement)
        save(self.directory / "state.json", acknowledgement)
        deadline = time.monotonic() + 30
        while True:
            request = self._request()
            if request.get("epoch") == epoch and not request.get("pause"):
                resumed = {
                    "trial_id": self.trial_id,
                    "state": "resumed",
                    "epoch": epoch,
                    "at_ns": time.monotonic_ns(),
                }
                self.events.append(resumed)
                save(self.directory / "state.json", resumed)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Compute ownership was not returned to the tool.")
            time.sleep(0.001)
