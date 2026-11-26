# Overview

> **Start here:** PSRL separates collecting model answers from updating model parameters, then coordinates sample data and model versions between them.

第一次接触项目，可以先读[PSRL入门：从一道题到模型更新](01-PSRL入门_从一道题到模型更新.md)了解组件如何衔接，再按需查[系统与实验术语](02-PSRL入门_系统与实验术语速查.md)。这两篇默认已有机器学习与 agent 基础。

## Contents

- [What is PSRL?](#what-is-psrl)
- [Key Capabilities](#key-capabilities)
- [Supported Backends](#supported-backends)

## What is PSRL?

**PSRL** is a reinforcement learning (RL) framework for efficient large language
model (LLM) post-training. It decouples rollout, reward, and training while coordinating
them through a Parameter Server.
Generation and training progress asynchronously while model-version staleness remains
explicitly bounded.

Built on [veRL](https://github.com/volcengine/verl), PSRL focuses on the system bottlenecks that emerge in agentic, dynamic, and long-tailed RL workloads: uneven rollout latency, version-aware weight management, multi-turn KV cache reuse, and elastic resource allocation across rollout, reward, and training models.

[![PSRL System Architecture（1/3）](../_static/img/document-figures/explained-PSRL_arch-4d2b637-screen.png)](../_static/img/document-figures/explained-PSRL_arch-4d2b637.png)

[![PSRL System Architecture（2/3）](../_static/img/document-figures/explained-PSRL_arch-4d2b637-screen-02.png)](../_static/img/document-figures/explained-PSRL_arch-4d2b637.png)

[![PSRL System Architecture（3/3）](../_static/img/document-figures/explained-PSRL_arch-4d2b637-screen-03.png)](../_static/img/document-figures/explained-PSRL_arch-4d2b637.png)

Overall PSRL system architecture.

<!-- explained-source:explained-PSRL_arch-4d2b637 -->
**读图结论：**解耦让阶段有重叠机会，数据可用性和模型版本仍要显式协调。 架构概览不代表所有阶段能同时无限并发，细节见完整执行主文档。 [完整解释图（SVG）](../_static/img/document-figures/explained-PSRL_arch-4d2b637.svg) · [原始机制图或实验图](../_static/img/PSRL_arch.svg)。
<!-- /explained-source:explained-PSRL_arch-4d2b637 -->

## Key Capabilities

| Capability | Description |
|---|---|
| **Efficient RDMA-based Weight Transfer** | Uses the Parameter Server and NIXL for local or UCX/RDMA model push/pull |
| **Fine-grained Staleness Control** | Binds model versions at trajectory level and uses a global consistency protocol (Reserve/Occupy/Consume) to keep staleness bounded |
| **Flexible Rollout Coordination** | Uses SMG routing, partial rollout, redundancy, migration, and PSRL-aware worker admission |
| **Easy-to-use Agentic RL Support** | Supports native environment loops plus SessionRouter/TITO integration for OpenAI-compatible black-box agents |
| **Hierarchical KV Cache Management** | Combines SMG cache-aware routing, vLLM GPU prefix KV, LMCache offload, and cross-instance transfer |
| **Multiple Backends and Algorithms Support** | Integrates FSDP2/Megatron training, vLLM rollout, and PPO/GRPO/DAPO-style algorithms |

## Supported Backends

PSRL keeps the **generation** and **training** stacks fully decoupled, so each side
can be scaled and swapped independently.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Rollout
:class-card: psrl-feature-card

**SMG + vLLM**

The SMG gateway provides the HTTP/OpenAI ingress, the PSRL-aware routing loop, the
gRPC proxy, and TITO session capture. vLLM is the inference backend (DP + TP + PP
serving through PSRL's gRPC integration).
:::

:::{grid-item-card} Training
:class-card: psrl-feature-card

**FSDP2** &nbsp;·&nbsp; **Megatron**

A single unified engine train worker drives both backends. FSDP2 offers HSDP
sharding plus Ulysses sequence parallelism. Megatron adds DP + TP + PP + CP + EP for
large and MoE models.
:::

::::

See {doc}`../design/architecture` for the detailed architecture.

```{toctree}
:maxdepth: 1
:hidden:

01-PSRL入门_从一道题到模型更新
02-PSRL入门_系统与实验术语速查
```
