"""Teach candidate scoring and outer policy gradients with standard-library data."""

import json
import math
from statistics import mean, pstdev


def main():
    """Keep candidate weights fixed while updating a two-choice outer policy."""
    # Diagram 2 and 3 use a one-dimensional classifier instead of a trained CNN.
    examples = list(range(16))
    labels = [0] * 8 + [1] * 8
    candidate_weights = (4.0, 2.0, 0.0, 16.0)
    rewards = [
        mean(int(int(x >= threshold) == label) for x, label in zip(examples, labels, strict=True))
        for threshold in candidate_weights
    ]
    centered = [reward - mean(rewards) for reward in rewards]
    normalized = [advantage / pstdev(rewards) for advantage in centered]
    equal_group = [0.5] * 4
    # Diagram 5 marks only model actions as policy-loss positions.
    mask = [1, 1, 0, 0, 1]
    positions = ["write", "train", "tool:loss", "tool:saved", "finish"]

    # Diagram 6 uses the exact expectation gradient of a Bernoulli policy.
    theta = 0.0
    probability_before = 1 / (1 + math.exp(-theta))
    good_reward, bad_reward = 0.8, 0.4
    gradient = (good_reward - bad_reward) * probability_before * (1 - probability_before)
    theta_after = theta + gradient
    probability_after = 1 / (1 + math.exp(-theta_after))
    print(
        json.dumps(
            {
                "scope": "CPU teaching simulation, no CNN training or full DAPO optimizer",
                "candidate_weights_unchanged": candidate_weights,
                "verifier_rewards": rewards,
                "centered_advantages": centered,
                "normalized_advantages": normalized,
                "equal_group_centered": [reward - mean(equal_group) for reward in equal_group],
                "dynamic_filter_keeps_equal_group": pstdev(equal_group) > 0,
                "policy_loss_positions": [name for name, active in zip(positions, mask, strict=True) if active],
                "outer_theta_before": theta,
                "outer_theta_after": theta_after,
                "good_strategy_probability_before": probability_before,
                "good_strategy_probability_after": probability_after,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
