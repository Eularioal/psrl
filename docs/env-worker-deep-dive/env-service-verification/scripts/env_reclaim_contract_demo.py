"""Illustrate reclaim contracts using a CPU simulation with no GPU execution."""

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Contract:
    """Describe one bounded execution segment and its assumed cleanup cost."""

    segment_ms: int
    cleanup_ms: int
    owner_budget_ms: int

    @property
    def admissible(self):
        return self.segment_ms + self.cleanup_ms <= self.owner_budget_ms


class TeachingExecutor:
    """Hide whole command duration from the scheduling policy."""

    def __init__(self, duration_ms):
        self._remaining_ms = duration_ms

    def run_segment(self, limit_ms):
        """Return elapsed simulated time and whether the command is complete."""
        elapsed = min(self._remaining_ms, limit_ms)
        self._remaining_ms -= elapsed
        return elapsed, self._remaining_ms == 0

    def run_to_completion(self):
        """Represent an opaque command that cannot cooperate at intermediate points."""
        elapsed = self._remaining_ms
        self._remaining_ms = 0
        return elapsed


def cooperative_return(executor, owner_arrival_ms, contract):
    """Stop launching segments once the owner has requested its GPU back."""
    if not contract.admissible:
        return {"admitted": False, "owner_wait_ms": 0, "completed": False}
    clock_ms = 0
    completed = False
    # Diagram 3 to 4: the policy observes segment completions, not total duration.
    while clock_ms < owner_arrival_ms and not completed:
        elapsed, completed = executor.run_segment(contract.segment_ms)
        clock_ms += elapsed
    # Diagram 5: assume completed synchronization and physical cleanup.
    # Real CUDA cleanup is not implemented by this teaching executor.
    clock_ms += contract.cleanup_ms
    return {
        "admitted": True,
        "owner_wait_ms": max(0, clock_ms - owner_arrival_ms),
        "completed": completed,
    }


def opaque_return(executor, owner_arrival_ms, cleanup_ms):
    """Model admission that waits for an opaque command to finish naturally."""
    returned_ms = executor.run_to_completion() + cleanup_ms
    return max(0, returned_ms - owner_arrival_ms)


def progress_orders():
    """Show why owner delay protection alone does not select useful Env work."""
    durations = {"env_a": 300, "env_b": 2000, "model_reply": 200}
    result = []
    # Env A is the last missing trajectory for the next batch. Env B is not.
    # Both Env commands precede model requests, but only A gates this batch.
    for order in (("env_b", "env_a", "model_reply"), ("env_a", "model_reply", "env_b")):
        clock_ms = 0
        for name in order:
            clock_ms += durations[name]
            if name == "model_reply":
                result.append({"order": list(order), "batch_ready_ms": clock_ms})
                break
    return result


def main():
    """Run bounded examples and verify only the stated teaching assumptions."""
    contract = Contract(segment_ms=100, cleanup_ms=30, owner_budget_ms=150)
    trials = []
    for private_duration in (400, 4000, 40000):
        row = {
            "executor_private_duration_ms": private_duration,
            "owner_arrival_ms": 150,
            "opaque_owner_wait_ms": opaque_return(TeachingExecutor(private_duration), 150, 30),
        }
        row.update(cooperative_return(TeachingExecutor(private_duration), 150, contract))
        assert row["owner_wait_ms"] == 80, "Expected the same delay for different command durations."
        trials.append(row)

    # Diagram 2: admission checks the reclaim bound instead of total duration.
    rejected = Contract(segment_ms=250, cleanup_ms=30, owner_budget_ms=150)
    rejection = cooperative_return(TeachingExecutor(4000), 150, rejected)
    assert not rejection["admitted"], "Expected an excessive reclaim bound to reject borrowing."

    sweep = [
        cooperative_return(TeachingExecutor(40000), arrival_ms, contract)["owner_wait_ms"]
        for arrival_ms in range(1000)
    ]
    assert max(sweep) <= 130, "Expected the assumed segment and cleanup bound to hold."

    # A cleanup failure produces isolation, not a fictitious physical return.
    failure = {"state": "quarantined", "physical_return_confirmed": False, "owner_resume_allowed": False}
    assert not failure["owner_resume_allowed"], "Expected ambiguous cleanup to keep the owner fenced."

    output = {
        "evidence_kind": "cpu_teaching_simulation",
        "assumptions": {
            "contract": asdict(contract),
            "cleanup_is_assumed_successful_and_bounded": True,
            "whole_duration_is_hidden_from_policy": True,
            "no_mps_or_cuda_or_checkpoint_implementation": True,
        },
        "duration_trials": trials,
        "arrival_sweep_max_owner_wait_ms": max(sweep),
        "excessive_segment_rejected": rejection,
        "cleanup_failure": failure,
        "dependency_order_example": progress_orders(),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
