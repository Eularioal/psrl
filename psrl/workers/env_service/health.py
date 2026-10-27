"""Track bounded node recovery without forgetting unresolved physical cleanup."""

from dataclasses import dataclass


@dataclass
class NodeHealth:
    failures: int = 0
    checked_at: float = float("-inf")
    error: str = ""
    pending_cleanup: bool = False

    @property
    def available(self):
        return not self.error and not self.pending_cleanup

    def success(self, now, pending_cleanup=False):
        self.failures = 0
        self.checked_at = now
        self.error = ""
        self.pending_cleanup = pending_cleanup

    def fail(self, now, error):
        self.failures += 1
        self.checked_at = now
        self.error = f"{type(error).__name__}: {error}"
