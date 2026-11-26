# Designs & Features

This section describes the current PSRL architecture: SMG-based online request
routing, metadata-only sample flow through TransferQueue, asynchronous weight
transfer, and hierarchical KV cache management.

Read [the beginner workflow](../overview/01-PSRL入门_从一道题到模型更新.md) first if the roles are new.
Then choose one subsystem below. The pages explain motivation, mechanisms, examples
and configuration rather than requiring you to understand every component at once.

---

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`workflow;1.5em` System Architecture
:link: architecture
:link-type: doc

Overall system design with decoupled training and generation, module responsibilities, and deployment topology.
:::

:::{grid-item-card} {octicon}`hubot;1.5em` Router, SessionRouter & TITO
:link: router_tito
:link-type: doc

SMG-based rollout routing, PSRL worker selection, session affinity, and exact
multi-turn training-data capture.
:::

:::{grid-item-card} {octicon}`stack;1.5em` TransferQueue
:link: transfer_queue
:link-type: doc

Asynchronous sample storage and metadata-only data exchange across rollout, reward,
and training.
:::

:::{grid-item-card} {octicon}`server;1.5em` Parameter Server
:link: parameter_server
:link-type: doc

CPU-based distributed weight storage with unified sharding, RDMA push/pull, and broadcast initialization.
:::

:::{grid-item-card} {octicon}`iterations;1.5em` Flexible Rollout
:link: flexible_rollout
:link-type: doc

Four complementary coordination techniques: partial rollout, redundant rollout, intelligent routing, and migration.
:::

:::{grid-item-card} {octicon}`clock;1.5em` Staleness Control
:link: staleness_control
:link-type: doc

Trajectory-level version binding with the Reserve/Occupy/Consume protocol for bounded off-policy training.
:::

:::{grid-item-card} {octicon}`database;1.5em` KV Cache Management
:link: kv_cache
:link-type: doc

LMCache integration for CPU offloading, prefix reuse, and cross-instance P2P KV transfer.
:::

:::{grid-item-card} {octicon}`pulse;1.5em` Resource Elasticity
:link: resource_elasticity
:link-type: doc

TMS-backed worker lifecycle and elastic inference resources, with current colocation
constraints documented explicitly.
:::

::::

---

```{toctree}
:maxdepth: 2
:hidden:

architecture
router_tito
transfer_queue
parameter_server
flexible_rollout
staleness_control
kv_cache
resource_elasticity
env_worker
```

<!-- figure:map-0df1b907f1 -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![Designs & Features：阅读地图（1/2）](../_static/img/document-figures/map-0df1b907f1-screen.png)](../_static/img/document-figures/map-0df1b907f1.png)

[![Designs & Features：阅读地图（2/2）](../_static/img/document-figures/map-0df1b907f1-screen-02.png)](../_static/img/document-figures/map-0df1b907f1.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](../_static/img/document-figures/map-0df1b907f1.svg)。
<!-- /figure:map-0df1b907f1 -->
