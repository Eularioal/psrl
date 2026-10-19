"""CPU teaching simulation: phase lending, early return, and stale-command fencing.

No production scheduler, Docker, CUDA, predictor training, or measured performance.
One GPU fits one command at a time; a command is non-preemptible once started.
"""

import json


def simulate(actual_gap, predicted_gap, borrower_seconds=8, handoff_seconds=1):
    # Diagram P1: A finishes its first 4-second GPU command at t=4.
    first_end, final_command = 4, 4
    a_ready = first_end + actual_gap
    ledger_owner, epoch = "A", 1
    stale_epoch = epoch
    # Diagram P2: a confirmed CUDA exit plus execution fence permits release.
    ledger_owner = None
    epoch += 1
    fits_prediction = borrower_seconds + 2 * handoff_seconds <= predicted_gap
    events = [["A GPU", 0, first_end]]
    if fits_prediction:
        # Diagram P3: lend to B; A's old execution token must not work.
        ledger_owner = "B"
        b_start = first_end + handoff_seconds
        b_end = b_start + borrower_seconds
        assert stale_epoch != epoch
        a_resume = max(a_ready, b_end + handoff_seconds)
        events.append(["B GPU", b_start, b_end])
        # Diagram P4: early A waits; prediction error must not allow overlap.
        assert a_resume >= b_end
        a_end = a_resume + final_command
        makespan = a_end
    else:
        a_resume = a_ready
        a_end = a_resume + final_command
        b_start = a_end + handoff_seconds
        b_end = b_start + borrower_seconds
        events.append(["B GPU", b_start, b_end])
        makespan = b_end
    ledger_owner = "A"
    epoch += 1
    events.append(["A GPU", a_resume, a_end])
    ledger_owner = None
    return {
        "actual_gap": actual_gap,
        "predicted_gap": predicted_gap,
        "borrow_admitted": fits_prediction,
        "events": sorted(events, key=lambda e: e[1]),
        "a_extra_wait": a_resume - a_ready,
        "makespan": makespan,
        "stale_command_rejected": stale_epoch != epoch,
        "final_owner": ledger_owner,
    }


if __name__ == "__main__":
    result = {
        "scope": "Teaching simulation only; synthetic seconds",
        "whole_episode_reservation_makespan": 4 + 20 + 4 + 1 + 8,
        "accurate_prediction": simulate(20, 20),
        "owner_returns_early": simulate(4, 20),
        "conservative_short_gap": simulate(4, 4),
    }
    assert result["accurate_prediction"]["makespan"] == 28
    assert result["owner_returns_early"]["a_extra_wait"] == 6
    assert not result["conservative_short_gap"]["borrow_admitted"]
    print(json.dumps(result, indent=2))
