"""Illustrate ownership, queue blocking and retained history with CPU arithmetic."""

from dataclasses import dataclass, field


@dataclass
class LedgerModel:
    """Model declared reservations without physical device measurements."""

    capacity: int = 80
    reservations: dict[str, int] = field(default_factory=dict)

    def reserve(self, session, amount):
        if sum(self.reservations.values()) + amount > self.capacity:
            return False
        self.reservations[session] = amount
        return True


def main():
    # Review diagram 1 and 2 model two controllers with independent state.
    first, second = LedgerModel(), LedgerModel()
    first.reserve("A", 60)
    second.reserve("B", 60)
    promised = sum(first.reservations.values()) + sum(second.reservations.values())
    print(f"R3 independent_ledgers promised={promised} physical_capacity=80")
    restarted = LedgerModel()
    print(f"R3 reconstructed_empty_ledger recorded={sum(restarted.reservations.values())} old_reservation=60")

    # Review diagram 4 models the strict queue head rule, not a new policy.
    free = 80 - 50
    queue = [("A", 40), ("B", 10)]
    admitted = []
    for name, amount in queue:
        if amount > free:
            break
        free -= amount
        admitted.append(name)
    print(f"R5 free={free} head_request=40 next_request=10 admitted={admitted}")

    # Review diagram 3 models completed records counting toward the session limit.
    history = {f"job-{index}": "completed" for index in range(128)}
    active = sum(state != "completed" for state in history.values())
    rejected = len(history) >= 128
    print(f"R6 retained={len(history)} active={active} reject_next={rejected}")


if __name__ == "__main__":
    main()
