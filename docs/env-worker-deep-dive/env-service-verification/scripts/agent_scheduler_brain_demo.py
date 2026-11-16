"""Illustrate a proposed scheduling control loop with fixed semantic fixtures.

This CPU teaching simulation calls no model or GPU and imports no production code.
All durations and capacities are teaching assumptions, not measurements.
"""

import logging
from dataclasses import dataclass, replace

psrl_logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class Device:
    """Describe authoritative capacity and optional queue observations."""

    name: str
    free_gib: int
    wait_s: float | None
    resident_model: bool


@dataclass(frozen=True)
class Observation:
    """Bind a command version to one immutable observation epoch."""

    code_version: str
    epoch: int
    minimum_gib: int
    execution_s: float
    devices: tuple[Device, ...]


@dataclass(frozen=True)
class SemanticAdvice:
    """Represent untrusted fixed output in place of an actual model call."""

    code_version: str
    epoch: int
    expires_at_s: float
    candidate_devices: tuple[str, ...]
    claimed_free_gib: tuple[int, ...]
    cold_load_s: float | None


def observe() -> Observation:
    """O1: Return assumed observations rather than querying hardware."""
    return Observation(
        code_version="candidate-v2",
        epoch=7,
        minimum_gib=8,
        execution_s=4,
        devices=(Device("hot", 12, 12, True), Device("cold", 16, 0, False)),
    )


def semantic_fixture(
    observation: Observation,
    cold_load_s: float | None,
) -> SemanticAdvice:
    """S1: Supply fixed semantic advice without inferring or proving its accuracy."""
    return SemanticAdvice(
        code_version=observation.code_version,
        epoch=observation.epoch,
        expires_at_s=2,
        candidate_devices=("hot", "cold"),
        claimed_free_gib=(12, 16),
        cold_load_s=cold_load_s,
    )


def validate(
    observation: Observation,
    advice: SemanticAdvice,
    now_s: float = 0,
) -> None:
    """V1: Reject stale identities and capacity claims that contradict observations."""
    if advice.code_version != observation.code_version or advice.epoch != observation.epoch:
        raise ValueError("Advice version or observation epoch is stale.")
    if now_s >= advice.expires_at_s:
        raise ValueError("Advice has expired.")
    if len(advice.candidate_devices) != len(advice.claimed_free_gib):
        raise ValueError("Advice has inconsistent capacity fields.")
    known = {device.name: device for device in observation.devices}
    for name, claimed_free in zip(advice.candidate_devices, advice.claimed_free_gib, strict=True):
        if name not in known:
            raise ValueError("Advice names an unavailable device.")
        if claimed_free != known[name].free_gib:
            raise ValueError("Advice cannot redefine observed capacity.")
    if advice.cold_load_s is not None and advice.cold_load_s < 0:
        raise ValueError("Load estimates must be nonnegative or unknown.")


def choose(
    observation: Observation,
    advice: SemanticAdvice,
) -> tuple[str | None, dict[str, float | None]]:
    """P1: Compare known completion estimates with a deterministic cost rule."""
    completion = {}
    for device in observation.devices:
        if device.name not in advice.candidate_devices or device.free_gib < observation.minimum_gib:
            continue
        load = 0 if device.resident_model else advice.cold_load_s
        if device.wait_s is None or load is None:
            completion[device.name] = None
        else:
            completion[device.name] = device.wait_s + load + observation.execution_s
    known = {name: elapsed for name, elapsed in completion.items() if elapsed is not None}
    selected = min(known, key=known.get) if known else None
    return selected, completion


def execute_and_return(
    observation: Observation,
    device_name: str,
) -> tuple[int, int]:
    """E1 and R1: Simulate admission and confirmed return using a scalar ledger."""
    device = next(device for device in observation.devices if device.name == device_name)
    # E1 rechecks current capacity instead of trusting the semantic proposal.
    if observation.minimum_gib > device.free_gib:
        raise ValueError("Capacity changed before admission.")
    free_before = device.free_gib
    free_during = free_before - observation.minimum_gib
    # R1 assumes successful return confirmation. Production must verify actual release.
    free_after_confirmation = free_during + observation.minimum_gib
    assert free_after_confirmation == free_before, "Confirmed return must restore the teaching budget."
    return free_during, free_after_confirmation


def main() -> None:
    """Run bounded examples without sleeping, submitting jobs, or allocating devices."""
    observation = observe()
    for cold_load, expected, finish in ((20, "hot", 16), (2, "cold", 6)):
        advice = semantic_fixture(observation, cold_load)
        validate(observation, advice)
        selected, completion = choose(observation, advice)
        assert selected == expected, f"Unexpected placement {selected!r}."
        assert completion[selected] == finish, "The teaching completion estimate is incorrect."
        ledger = execute_and_return(observation, selected)
        psrl_logger.info(
            f"[O1 S1 V1 P1 E1 R1] Cold load={cold_load}s, completion={completion!r}, "
            f"selected={selected!r}, free budget during/after={ledger!r} GiB."
        )

    valid = semantic_fixture(observation, 2)
    invalid = {
        "fabricated_capacity": (replace(valid, claimed_free_gib=(999, 16)), 0),
        "unavailable_device": (replace(valid, candidate_devices=("missing", "cold")), 0),
        "stale_code": (replace(valid, code_version="candidate-v1"), 0),
        "stale_observation": (replace(valid, epoch=6), 0),
        "expired_advice": (valid, 3),
    }
    for label, (advice, now_s) in invalid.items():
        try:
            validate(observation, advice, now_s)
        except ValueError as exc:
            psrl_logger.info(f"[V1] Rejected {label!r}: {exc}")
        else:
            raise AssertionError(f"Invalid advice {label!r} was accepted.")

    unknown = replace(observation, devices=tuple(replace(device, wait_s=None) for device in observation.devices))
    selected, completion = choose(unknown, valid)
    assert selected is None and all(value is None for value in completion.values()), (
        "Unknown costs must remain unknown instead of becoming zero."
    )
    psrl_logger.info("[P1] All queue times are unknown, so no known fastest device can be chosen.")

    # These independent short tasks gain no useful overlap while the extra model runs.
    baseline_s, proposed_service_s, extra_model_s = 1.0, 0.8, 0.6
    proposed_total_s = proposed_service_s + extra_model_s
    assert proposed_total_s > baseline_s, "Extra inference must outweigh the assumed service saving here."
    psrl_logger.info(
        f"[Cost] Baseline={baseline_s:.1f}s, proposed service={proposed_service_s:.1f}s, "
        f"extra model={extra_model_s:.1f}s, proposed total={proposed_total_s:.1f}s."
    )
    psrl_logger.info("Teaching assertions passed. This does not evaluate a real LLM or GPU scheduler.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
