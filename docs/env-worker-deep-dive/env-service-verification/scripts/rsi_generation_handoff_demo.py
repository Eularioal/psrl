"""Simulate version handoff and resource yielding for a proposed improvement loop.

This CPU teaching simulation uses fixed scores and scalar memory budgets. It calls
no model, GPU, Docker, or production PSRL code and does not establish RSI gains.
"""

import logging
from dataclasses import dataclass
from threading import Lock

psrl_logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class ModelVersion:
    """Keep a candidate's immutable identity and parent lineage."""

    name: str
    parent: str | None
    checkpoint: str


@dataclass(frozen=True)
class RolloutRequest:
    """Pin an admitted request to one model version."""

    name: str
    model: ModelVersion


@dataclass(frozen=True)
class Evaluation:
    """Bind an independent evaluation outcome to the candidate and incumbent."""

    candidate: ModelVersion
    incumbent: ModelVersion
    score: float
    incumbent_score: float
    passed: bool


@dataclass(frozen=True)
class ReadySampler:
    """Bind an assumed successful sampler readiness check to an exact model version."""

    model: ModelVersion


class ActiveRouter:
    """C0: Store the active version and switch it under one control lock."""

    def __init__(self, initial: ModelVersion, initial_score: float):
        self.active = initial
        self.score = initial_score
        self.lock = Lock()

    def begin_rollout(self, name: str) -> RolloutRequest:
        """C1 and C5: Pin a new request to the currently published version."""
        with self.lock:
            return RolloutRequest(name, self.active)

    def publish(self, evaluation: Evaluation, readiness: ReadySampler | None = None) -> bool:
        """C4: Atomically publish an approved child with a matching ready sampler."""
        with self.lock:
            if (
                not evaluation.passed
                or evaluation.incumbent != self.active
                or evaluation.candidate.parent != self.active.name
                or readiness is None
                or readiness.model != evaluation.candidate
            ):
                return False
            self.active = evaluation.candidate
            self.score = evaluation.score
            return True


def train_child(parent: RolloutRequest, name: str) -> ModelVersion:
    """C2: Return a completed candidate artifact without publishing it."""
    return ModelVersion(name, parent.model.name, f"fixture-checkpoint-{name}")


def independent_evaluate(
    candidate: ModelVersion,
    incumbent: ModelVersion,
    incumbent_score: float,
    heldout_fixture_scores: dict[str, float],
    improvement_margin: float = 0.01,
) -> Evaluation:
    """C3: Apply an illustrative promotion gate using evaluator-owned score fixtures."""
    score = heldout_fixture_scores[candidate.name]
    return Evaluation(candidate, incumbent, score, incumbent_score, score > incumbent_score + improvement_margin)


class MemoryBudget:
    """R: Simulate reservations without allocating or measuring real GPU memory."""

    def __init__(self, capacity_gib: int):
        self.capacity_gib = capacity_gib
        self.owners: dict[str, int] = {}

    def acquire(self, owner: str, amount_gib: int) -> bool:
        """Admit a new owner only when its complete reservation fits."""
        if owner in self.owners or amount_gib < 0:
            raise ValueError("The teaching reservation owner or size is invalid.")
        if sum(self.owners.values()) + amount_gib > self.capacity_gib:
            return False
        self.owners[owner] = amount_gib
        return True

    def release_after_confirmation(self, owner: str) -> None:
        """Remove a reservation after an assumed successful release confirmation."""
        del self.owners[owner]


def demonstrate_resource_yield() -> None:
    """R: Expose a holding dependency cycle and simulate yielding its reservation."""
    budget = MemoryBudget(24)
    assert budget.acquire("resident_rollout", 20), "The initial resident model must fit."
    assert not budget.acquire("inner_training", 12), "The inner training must exceed the remaining budget."
    psrl_logger.info(
        "[R] Capacity=24 GiB, resident rollout=20 GiB, inner training=12 GiB. "
        "Parent waits for training while retaining memory, and training waits for memory."
    )
    # R assumes the inference process actually yields and its release is confirmed.
    budget.release_after_confirmation("resident_rollout")
    assert budget.acquire("inner_training", 12), "Yielding the parent reservation must admit training."
    budget.release_after_confirmation("inner_training")
    assert budget.acquire("resident_rollout", 20), "Inference can return after training releases its reservation."
    budget.release_after_confirmation("resident_rollout")
    assert not budget.owners, "The teaching reservation ledger must finish empty."
    psrl_logger.info("[R] Confirmed parent yield admits training, then training return permits inference restoration.")


def main() -> None:
    """Run the two-generation handoff and rejected successor with fixed teaching data."""
    g0 = ModelVersion("g0", None, "fixture-checkpoint-g0")
    router = ActiveRouter(g0, initial_score=0.60)
    heldout_fixture_scores = {"g1": 0.70, "g2": 0.65}
    old_request = router.begin_rollout("already_in_flight")
    first = router.begin_rollout("first_iteration")
    psrl_logger.info(f"[C0 C1] Active={router.active.name!r}, first rollout uses {first.model.name!r}.")

    demonstrate_resource_yield()
    g1 = train_child(first, "g1")
    assert router.active == g0, "Completing training must leave the published model unchanged."
    psrl_logger.info(f"[C2] Training completed {g1.name!r} from {g1.parent!r}. Active remains {router.active.name!r}.")

    evaluation = independent_evaluate(g1, router.active, router.score, heldout_fixture_scores)
    assert evaluation.passed, "The first teaching candidate must pass its fixture gate."
    psrl_logger.info(f"[C3] Independent fixture evaluation: incumbent=0.60, candidate={evaluation.score:.2f}.")
    assert not router.publish(evaluation), "An approved candidate must wait for its sampler readiness check."
    assert not router.publish(evaluation, ReadySampler(g0)), "Readiness for a different model must not publish g1."
    assert router.active == g0, "Failed readiness checks must preserve the active model."
    psrl_logger.info("[C4] Approval alone keeps g0 active until a sampler for the exact g1 version is ready.")
    ready_g1 = ReadySampler(g1)
    assert router.publish(evaluation, ready_g1), "The approved child and its matching ready sampler must publish."
    assert old_request.model == g0, "An in-flight request must retain its original immutable model binding."
    psrl_logger.info(
        f"[C4] Active switched atomically to {router.active.name!r}. "
        f"Existing request remains bound to {old_request.model.name!r}."
    )

    second = router.begin_rollout("second_iteration")
    assert second.model == g1, "The next rollout must use the published child."
    psrl_logger.info(f"[C5] The next rollout uses {second.model.name!r} and proposes the next training command.")
    g2 = train_child(second, "g2")
    rejected = independent_evaluate(g2, router.active, router.score, heldout_fixture_scores)
    assert not rejected.passed and not router.publish(rejected, ReadySampler(g2)), (
        "A weaker fixture candidate must not publish even when its sampler is ready."
    )
    assert router.active == g1, "Rejecting the successor must preserve the incumbent."
    assert g1.parent == "g0" and g2.parent == "g1", "Candidate lineage must identify the true training parents."
    psrl_logger.info(
        f"[C2 C3 C4] Candidate={g2.name!r}, parent={g2.parent!r}, score={rejected.score:.2f}. "
        f"Promotion rejected, active remains {router.active.name!r}."
    )
    psrl_logger.info("Teaching assertions passed. Fixed scores do not measure actual recursive improvement.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
