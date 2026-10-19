"""CPU-only teaching simulation of the proposed rollout/Docker GPU handoff.

No Ray, Docker, CUDA, network, subprocesses, real waits, or production imports.
Numbers are hypothetical GiB; diagram numbers refer to rollout-gpu-lending.md.
This demonstrates ordering, not production concurrency or GPU correctness.
"""

import json
from dataclasses import dataclass, field


@dataclass
class LendingDemo:
    state: str = "ROLLOUT"
    ray_gpu_reserved: int = 1
    epoch: int = 0
    model_version: int = 7
    weights_valid: bool = True
    measured_gib: int = 82
    leases: dict[str, int] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    def sleep_ack(self) -> None:
        # [1 -> 2 -> 3] Simulate closed admission, drained requests, level-2 ACK.
        self.require(self.state == "ROLLOUT", "instance is not serving")
        self.state = "DRAINING"
        self.weights_valid = False
        self.measured_gib = 2  # Hypothetical post-sleep residual, not zero.
        self.epoch += 1
        self.state = "LENDABLE"
        self.events.append("sleep acknowledged; physical=2 GiB; Ray GPU reservation=1")

    def borrow(self, lease: str, epoch: int) -> None:
        # [4 -> 5] The proposed permit is required in addition to memory capacity.
        self.require(epoch == self.epoch, "stale lending epoch")
        self.require(self.state in ("LENDABLE", "ENV"), "GPU is not lendable")
        self.require(lease not in self.leases, "duplicate lease")
        self.require(len(self.leases) < 2, "container slots exhausted")
        self.require(self.measured_gib + 24 + 8 <= 96, "memory budget exhausted")
        self.leases[lease] = 24
        self.measured_gib += 24
        self.state = "ENV"
        self.events.append(f"borrow {lease}; physical={self.measured_gib} GiB")

    def begin_return(self) -> None:
        # [1 -> 4] Revoke permits BEFORE starting cleanup; block new borrowers.
        self.require(self.state == "ENV", "nothing to return")
        self.state = "RETURNING"
        self.epoch += 1
        self.events.append("return requested; new lending is closed")

    def bookkeeping_release(self, lease: str) -> None:
        # Deliberately leaves measured memory behind to demonstrate the hazard.
        self.require(self.state == "RETURNING", "return has not started")
        del self.leases[lease]

    def wake(self) -> None:
        # [6 -> 1 -> 3] Both logical and physical acknowledgements are required.
        self.require(self.state == "RETURNING", "invalid wake transition")
        self.require(not self.leases, "Docker leases still active")
        self.require(self.measured_gib <= 2, "Docker memory has not been reclaimed")
        self.state = "WAKING"
        self.measured_gib = 82
        self.events.append("wake allocated memory; weights are still invalid")

    def serve(self) -> None:
        # [7 -> 2] Level-2 wake allocation alone does not restore model contents.
        self.require(self.weights_valid, "weights must be restored even at version 7")
        self.require(self.state == "WAKING", "instance is not ready to reopen")
        self.state = "ROLLOUT"
        self.events.append("version 7 restored; routing reopened")


def main() -> None:
    demo = LendingDemo()
    blocked = []

    def expect_blocked(name, action):
        try:
            action()
        except RuntimeError as exc:
            blocked.append({"case": name, "reason": str(exc)})
        else:
            raise AssertionError(f"unsafe action was accepted: {name}")

    expect_blocked("borrow while rollout is awake", lambda: demo.borrow("A", 0))
    demo.sleep_ack()
    token = demo.epoch
    demo.borrow("A", token)
    demo.borrow("B", token)
    expect_blocked("third container", lambda: demo.borrow("C", token))
    demo.begin_return()
    expect_blocked("late request with old permit", lambda: demo.borrow("C", token))
    expect_blocked("new request during return", lambda: demo.borrow("C", demo.epoch))
    expect_blocked("wake with active leases", demo.wake)
    demo.bookkeeping_release("A")
    demo.bookkeeping_release("B")
    expect_blocked("wake with residual Docker memory", demo.wake)
    # [6] Simulate the separate physical cleanup acknowledgement.
    demo.measured_gib = 2
    demo.wake()
    expect_blocked("serve before restoring level-2 weights", demo.serve)
    # [7] Simulate forced reload; the numeric version has NOT changed.
    demo.weights_valid = True
    demo.serve()
    assert demo.ray_gpu_reserved == 1 and not demo.leases
    print(
        json.dumps(
            {
                "kind": "CPU teaching simulation",
                "events": demo.events,
                "blocked": blocked,
                "final_state": demo.state,
                "ray_gpu_reserved": demo.ray_gpu_reserved,
                "model_version": demo.model_version,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
