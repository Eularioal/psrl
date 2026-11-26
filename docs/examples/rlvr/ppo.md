# PPO: Proximal Policy Optimization

> **Core idea:** PPO updates the answering model using feedback relative to a learned value estimate. In a teaching example, a result scores 1 while the value model expected 0.4, giving a positive difference of 0.6. GAE combines such differences over steps. The policy is the actor; the value estimator is the critic.

```{admonition} Coming Soon
:class: tip

This example is under active development. The PPO algorithm is fully implemented in PSRL's trainer but a standalone example recipe with documentation is forthcoming.
```

---

## Algorithm Overview

PPO (Proximal Policy Optimization) is the foundational policy-gradient algorithm in
PSRL. It trains an **actor** alongside a learned **critic** (value function) and
optimizes the policy with a clipped surrogate objective. Key ingredients:

<!-- figure:ppo-clipped-objective -->
这张图沿同一个场景解释：PPO 的裁剪到底在哪一边停止奖励继续变化？

[![PPO裁剪：优势符号决定哪一侧的更新收益受到限制（1/3）](../../_static/img/document-figures/ppo-clipped-objective-screen.png)](../../_static/img/document-figures/ppo-clipped-objective.png)

[![PPO裁剪：优势符号决定哪一侧的更新收益受到限制（2/3）](../../_static/img/document-figures/ppo-clipped-objective-screen-02.png)](../../_static/img/document-figures/ppo-clipped-objective.png)

[![PPO裁剪：优势符号决定哪一侧的更新收益受到限制（3/3）](../../_static/img/document-figures/ppo-clipped-objective-screen-03.png)](../../_static/img/document-figures/ppo-clipped-objective.png)

教学条件：概率比 r，优势 A，裁剪区间 [0.8, 1.2]，图画待最大化的单项目标。 目标是 min(rA, clip(r)A)。乘上负优势后大小关系反向，不能只把 r 截断来理解。 只画策略项。实际最小化损失时有负号，还可能包含熵、KL、mask 与其他项。 [查看矢量图](../../_static/img/document-figures/ppo-clipped-objective.svg)。
<!-- /figure:ppo-clipped-objective -->

- **Generalized Advantage Estimation (GAE)** with a learned value function
- **Clipped surrogate objective** to limit the incentive for large probability-ratio changes
- **KL divergence control** (fixed or adaptive) against a reference policy

The clipped surrogate loss for token $t$ is

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[
\min\!\left(
  r_t(\theta)\,\hat{A}_t,\;
  \operatorname{clip}\!\bigl(r_t(\theta),\,1-\epsilon,\,1+\epsilon\bigr)\,\hat{A}_t
\right)
\right],
$$

where $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_{\text{old}}}(a_t\mid s_t)$
is the importance-sampling ratio and $\hat{A}_t$ is the GAE advantage.

```{seealso}
For the trainer architecture, cluster layout, and monitoring metrics shared by
all RLVR recipes, see {doc}`common`.
```
