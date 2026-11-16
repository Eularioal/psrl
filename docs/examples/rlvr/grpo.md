# GRPO: Group Relative Policy Optimization

> **Core idea:** compare multiple answers to the same question, rather than train a separate value model. With rewards `[1, 0]`, the mean is 0.5, so the successful answer is above average and the failed answer below it. Dividing by the group standard deviation scales that relative signal. These are teaching numbers.

```{admonition} Coming Soon
:class: tip

This example is under development. Check back for a full GRPO walkthrough.
```

---

## Algorithm Overview

GRPO (Group Relative Policy Optimization) keeps PPO's clipped surrogate objective
but **eliminates the learned critic** by normalizing rewards within a group of
rollouts generated from the same prompt.

- **No critic**: advantages are computed by normalizing scores within the rollout group
- **Group normalization**: for each prompt, `n` responses are sampled, reward mean and std are computed within the group
- **Symmetric clipping**: standard PPO-style clip at `clip_ratio`
- **Efficient**: no value function training, lower memory overhead than PPO

The advantage for response $i$ in a group is

$$
\hat{A}_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}.
$$

## GRPO vs. PPO

| Aspect | [PPO](ppo.md) | GRPO |
|---|---|---|
| Critic / value model | Required (GAE) | Not used |
| Advantage estimate | $\hat{A}_t$ from GAE on critic values | Group-normalized score $(r_i-\mu)/\sigma$ |
| Memory & compute | Higher (actor + critic) | Lower (actor only) |
| Best fit | Dense / shaped rewards, learned RMs | Sparse / verifiable rewards (RLVR) |

```{seealso}
For the trainer architecture, cluster layout, and monitoring metrics shared by
all RLVR recipes, see {doc}`common`.
```
