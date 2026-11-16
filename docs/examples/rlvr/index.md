# RLVR: Reinforcement Learning from Verifiable Rewards

> **Core idea:** choose a checker that scores the intended result, then choose the training recipe. For a math question, generation produces an answer, a checker compares it with the reference, and the algorithm converts that score into a parameter update.

RLVR trains language models on tasks where correctness can be verified programmatically, math problems with numeric answers, code with test suites, or logic puzzles with deterministic solutions. A task-specific checker provides the reward signal. Its coverage and isolation still determine whether the score reflects the intended task.

---

## Supported Algorithms

Listed in order of increasing specialization, PPO is the foundational actor-critic
algorithm, GRPO removes the critic via per-prompt group normalization, and DAPO
extends GRPO with asymmetric clipping plus a dynamic sampling filter.

| Algorithm | Key Idea | Status |
|-----------|----------|--------|
| [PPO](ppo.md) | Proximal Policy Optimization with GAE and a learned critic | Standalone walkthrough pending |
| [GRPO](grpo.md) | Group Relative Policy Optimization, per-prompt advantage normalization, no critic | Standalone walkthrough pending |
| [DAPO](dapo.md) | GRPO variant with asymmetric clipping + dynamic sampling filter | Ready |

All algorithms support both FSDP and Megatron training backends, with vLLM rollout and configurable parameter transfer. The main examples use CPU-resident PS Workers with NIXL.

---

## Common Setup

Every RLVR example shares the same infrastructure. See {doc}`common` for the
full reference covering trainer architecture, cluster-layout parameters, and
monitoring metrics. At a glance:

1. **Data**: Parquet files with `prompt`, `data_source`, and `reward_model.ground_truth` fields.
2. **Reward**: A `compute_score(data_source, solution_str, ground_truth, extra_info)` function that returns `{"score": float, "acc": float}`.
3. **Training**: PSRL's decoupled trainer with configurable staleness, Token Importance Sampling (TIS), and dynamic batching.
4. **Monitoring**: WandB logging with `critic/score/mean` (raw task correctness), `critic/rewards/mean` (shaped reward), and `val-core/<data_source>/acc/mean@N` (validation accuracy) metrics.

```{toctree}
:maxdepth: 1
:hidden:

common
ppo
grpo
dapo
```
