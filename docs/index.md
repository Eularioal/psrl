---
sd_hide_title: true
---

# PSRL Documentation

```{image} /_static/img/PSRL_logo_merge.png
:alt: PSRL
:width: 420px
:align: center
:class: psrl-hero-brand
```


**PSRL** is an efficient asynchronous RL framework for LLM post-training. Developed by a joint team from **PKU & SJTU** and built on top of [veRL](https://github.com/volcengine/verl), PSRL features efficient RDMA weight synchronization via parameter servers, fine-grained staleness control, and flexible rollout coordination to achieve up to **2.68x throughput improvement**.

## Env框架：现状、对照证据与下一步

[Env专题首页](env-worker-deep-dive/README.md)是固定回顾入口。主基线为同次冻结实现的 episode 静态环境池、容量不足排队，已完成相同两步补测，具体耗时优势及跨次比较限制由实验总览维护。

| 想了解什么 | 直接阅读 |
|---|---|
| 当前结构与能力边界 | [Env架构现状](env-worker-deep-dive/knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md) |
| 相比基线的优势与限制 | [实验结论总览](env-worker-deep-dive/experiments/01-实验总览_调度收益与证据范围.md) |
| 已做实验、条件和原始证据 | [九轮资源分区实验](env-worker-deep-dive/experiments/09-资源分区_静态卡池与推理余量共享.md) |
| 用于研究交流的设计与实验 | [PSRL 环境 GPU 调度研究报告](env-worker-deep-dive/experiments/12-研究报告_PSRL环境GPU调度的设计与评估.md) |
| 扩展卡数、模型和环境数据 | [八卡单批次与多负载回放](env-worker-deep-dive/experiments/13-扩展对照_八卡单批次与多负载回放.md) |
| 下一步优先级及验证边界 | [研究路线](env-worker-deep-dive/milestones/05-调度研究_研究全景与证据详解.md#接下来可以做什么优先级与验收边界) |

---

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`versions;1.5em` Overview
:link: overview/index
:link-type: doc

Project introduction, key features, and performance highlights.
:::

:::{grid-item-card} {octicon}`rocket;1.5em` Quick Start
:link: tutorial/quickstart
:link-type: doc

Prepare the environment and run the two-node, 16-GPU DAPO example.
:::

:::{grid-item-card} {octicon}`book;1.5em` Tutorial
:link: tutorial/index
:link-type: doc

Step-by-step guides for installation, configuration, and first training run.
:::

:::{grid-item-card} {octicon}`code;1.5em` Examples
:link: examples/index
:link-type: doc

Training recipes for RLVR and Agentic RL, plus clearly marked planned recipes.
:::

:::{grid-item-card} {octicon}`gear;1.5em` Architecture
:link: design/architecture
:link-type: doc

Understand the decoupled train/gen design and module interactions.
:::

:::{grid-item-card} {octicon}`graph;1.5em` Designs & Features
:link: design/index
:link-type: doc

Deep-dive into staleness control, flexible rollout, and parameter server.
:::

::::

---

<!-- figure:map-b4d68dc855 -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![PSRL Documentation：阅读地图（1/2）](_static/img/document-figures/map-b4d68dc855-screen.png)](_static/img/document-figures/map-b4d68dc855.png)

[![PSRL Documentation：阅读地图（2/2）](_static/img/document-figures/map-b4d68dc855-screen-02.png)](_static/img/document-figures/map-b4d68dc855.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](_static/img/document-figures/map-b4d68dc855.svg)。
<!-- /figure:map-b4d68dc855 -->

## Key Features

::::{grid} 2 2 3 3
:gutter: 3

:::{grid-item-card} Efficient RDMA-based Weight Transfer
:class-card: psrl-feature-card

Push/Pull weights to/from CPU-side Parameter Server and P2P RDMA transfers.
:::

:::{grid-item-card} Fine-grained Staleness Control
:class-card: psrl-feature-card

Trajectory-level version binding with Reserve/Occupy/Consume protocol bounds accepted sample age while generation and training overlap.
:::

:::{grid-item-card} Flexible Rollout Coordination
:class-card: psrl-feature-card

Partial rollout, redundant rollout, intelligent routing, and load-balanced migration work together to minimize idle time.
:::

:::{grid-item-card} Easy-to-use Agentic RL Support
:class-card: psrl-feature-card

Native environment loops and SessionRouter/TITO support both integrated and black-box agents.
:::

:::{grid-item-card} Hierarchical KV Cache Management
:class-card: psrl-feature-card

SMG cache-aware routing, vLLM GPU prefix cache, LMCache offload, and P2P transfer reduce re-prefill.
:::

:::{grid-item-card} Multiple Backends and Algorithms Support
:class-card: psrl-feature-card

FSDP2 and Megatron training integrate with PPO, GRPO, DAPO, and generative reward workflows.
:::

::::

---

```{toctree}
:maxdepth: 2
:hidden:

overview/index
tutorial/index
examples/index
design/index
```
