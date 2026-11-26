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

<!-- figure:group-relative-advantages -->
这张图沿同一个场景解释：同一道题的多次尝试，怎样变成鼓励或抑制某条回答的信号？

[![GRPO：同题多次尝试，通过组内比较获得方向（1/2）](../../_static/img/document-figures/group-relative-advantages-screen.png)](../../_static/img/document-figures/group-relative-advantages.png)

[![GRPO：同题多次尝试，通过组内比较获得方向（2/2）](../../_static/img/document-figures/group-relative-advantages-screen-02.png)](../../_static/img/document-figures/group-relative-advantages.png)

教学组含四条回答，奖励为 1、1、0、0，使用总体标准差，暂不考虑额外稳定项。 组内优势表达相对好坏。全零策略梯度可能来自相同反馈，不自动说明程序崩溃。 教学简化。分组、标准化、稳定项与同奖励过滤以实际算法配置为准。 [查看矢量图](../../_static/img/document-figures/group-relative-advantages.svg)。
<!-- /figure:group-relative-advantages -->

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
