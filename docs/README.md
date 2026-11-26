# PSRL 文档总览

> **核心结论**：第一次了解项目，先读一条完整流程，再按问题进入专题。实现说明、文献调研、研究提案和实验记录各有独立目录。同一问题的机制、例子和直接相关结果集中阅读，较长时用目录定位。

中文知识正文统一为“序号-主题_具体内容.md”，文件名与正文标题对应。序号是所在目录的阅读建议，按需查阅可直接跳读。系统知识集中在[系统基础](tools/系统基础/README.md)，Env机制分为[架构与执行](env-worker-deep-dive/knowledge/01-架构与执行/README.md)、[资源与调度](env-worker-deep-dive/knowledge/02-资源与调度/README.md)、[任务接入](env-worker-deep-dive/knowledge/03-任务接入/README.md)。

## Env快速回顾：现状、优势、实验与下一步

想回顾 Env，固定从[Env专题首页](env-worker-deep-dive/README.md)进入。也可以直接按问题打开以下主文档，不必先找历史阶段报告。

| 想知道什么 | 直接阅读 |
|---|---|
| 现在怎样工作，哪些已实现、默认是否启用 | [Env架构：现状与完整执行流程](env-worker-deep-dive/knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md#先看结论现有能力与主基线) |
| 相比统一主基线有什么优势，证据到哪里 | [实验总览：调度收益与证据范围](env-worker-deep-dive/experiments/01-实验总览_调度收益与证据范围.md#统一主基线与当前结论) |
| 具体做过什么实验，每轮结果与原因 | [九轮资源分区实验](env-worker-deep-dive/experiments/09-资源分区_静态卡池与推理余量共享.md#实验总览每一轮验证了什么)与[完整实验目录](env-worker-deep-dive/experiments/README.md) |
| 下一步还能做什么，先验证什么 | [研究优先级与验收边界](env-worker-deep-dive/milestones/05-调度研究_研究全景与证据详解.md#接下来可以做什么优先级与验收边界) |
| 论文还缺什么，自动持卡有什么研究价值 | [证据缺口、自适应持卡与训练依赖](env-worker-deep-dive/milestones/05-调度研究_研究全景与证据详解.md#论文还缺什么自适应持卡的研究切口) |

主基线固定为同次冻结实现的 episode 静态环境池，满池排队。Episode 补测已完成相同两步，Command 静态完整时间观察到少 14.77%，重叠交接补跑少 38.39%。实测差值包含启动和运行配置的差异。

<details>
<summary>查看PSRL全部主题的阅读地图</summary>

<!-- figure:map-0b5ca119d2 -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![PSRL 文档总览：阅读地图（1/2）](_static/img/document-figures/map-0b5ca119d2-screen.png)](_static/img/document-figures/map-0b5ca119d2.png)

[![PSRL 文档总览：阅读地图（2/2）](_static/img/document-figures/map-0b5ca119d2-screen-02.png)](_static/img/document-figures/map-0b5ca119d2.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](_static/img/document-figures/map-0b5ca119d2.svg)。
<!-- /figure:map-0b5ca119d2 -->

</details>

## 目录

- [Env快速回顾：现状、优势、实验与下一步](#env快速回顾现状优势实验与下一步)
- [第一次阅读](#第一次阅读)
- [可直接分享的 Science GPU 介绍](#可直接分享的-science-gpu-介绍)
- [使用、算法与架构](#使用算法与架构)
- [环境执行与资源调度](#环境执行与资源调度)
- [系统基础与开发工具](#系统基础与开发工具)
- [文档维护](#文档维护)
- [全量文档直达目录](#全量文档直达目录)
- [源码包中的既有文档](#源码包中的既有文档)

## 第一次阅读

1. [PSRL入门：从一道题到模型更新](overview/01-PSRL入门_从一道题到模型更新.md)：沿“做题 → 评分 → 更新模型”的例子串起所有主要组件，附源码阅读顺序和 CPU 小例子。
2. [系统与实验术语速查](overview/02-PSRL入门_系统与实验术语速查.md)：按需查进程、并发、容器、显存、租约和实验口径，每项附作用与深入入口。
3. [PSRL 概览](overview/index.md)：了解项目目标、可配置的后端和主要能力。

如果只是了解工作成果，可以直接读下一节的简报。关键术语在正文就地解释，无需先通读底层实现。

维护中的每篇文档都配有解释图，README 也有阅读地图或操作示意。全部解释图默认显示不超过800×600像素的屏幕分图，按编号阅读，点击可看完整图。[全文档彩图集](_static/img/document-figures/README.md#所有解释图的屏幕显示)可按主题查看资源布局、内存页、性能对照与状态变化图，并提供绘图源文件和覆盖检查。想用图理解调度研究，还可从[文献阅读地图](env-worker-deep-dive/literature-reviews/README.md#阅读入口)进入，配套[文献图集](_static/img/literature-reviews/README.md)提供借卡时间线与 Roofline 曲线。

## 可直接分享的 Science GPU 介绍

| 文档 | 适合什么读者 | 主要内容 |
|---|---|---|
| [PSRL 环境 GPU 调度研究报告](env-worker-deep-dive/experiments/12-研究报告_PSRL环境GPU调度的设计与评估.md) | 研究交流与设计评审 | 命令级借还、常驻共享与整卡分配，附等工作量的完整训练和交接对照 |
| [调度研究 Big Map 精简版](env-worker-deep-dive/milestones/04-调度研究_研究全景交流版.md) | 与师兄讨论选题 | 框架能力、容量与 PSRL 依赖主线、状态放置支线和证据边界，附[详细版](env-worker-deep-dive/milestones/05-调度研究_研究全景与证据详解.md) |
| [简版介绍](env-worker-deep-dive/milestones/01-Science介绍_已接入任务与研究价值.md) | 第一次了解这项工作 | 解决什么问题、数字分类题的完整过程、已有任务与研究价值 |
| [各 bench 任务与目录导览](env-worker-deep-dive/milestones/README.md#各-bench-的任务与代码入口) | 想找某个 bench 的代码或题库 | 各 bench 做什么、代码目录、任务准备器、已准备题库与两条接入方式 |
| [任务现状与路线图](env-worker-deep-dive/milestones/02-Science现状_任务覆盖与接入路线.md) | 评估覆盖和后续工作 | 逐题库数量、验收层次、剩余缺口与研究顺序 |
| [接入与验收详情](env-worker-deep-dive/milestones/03-Science验收_接入流程与环境隔离.md) | 需要复现或继续开发 | 执行和训练流程、环境隔离实例、历史验收与证据入口 |

## 使用、算法与架构

| 入口 | 能回答的问题 |
|---|---|
| [安装与运行](tutorial/index.md) | 怎样准备软件、模型、数据和一次有界运行 |
| [架构设计](design/index.md) | 参数、样本、生成请求怎样在组件之间流动 |
| [算法与应用示例](examples/index.md) | 数学做题、工具调用、软件修复分别怎样配置 |
| [PPO、GRPO 与 DAPO 完整原理](psrl/02-强化学习_PPO与GRPO及DAPO.md) | 从奖励、优势到裁剪与 token 损失，在同篇读完完整算法主线与手算 |
| [训练系统与源码知识](psrl/README.md) | 训推流水线、推理缓存、并行机制与源码入口 |
| [集群运行前检查](psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md) | 共享环境里怎样核对依赖、地址、Docker、GPU 和清理范围 |

## 环境执行与资源调度

[环境专题导航](env-worker-deep-dive/README.md)按阅读目的串联以下目录：

| 分类 | 内容 |
|---|---|
| [当前设计](env-worker-deep-dive/knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md) | 环境执行的现状入口，区分默认配置、可选后端和研究原型 |
| [机制与任务接入](env-worker-deep-dive/knowledge/README.md) | 准入、显存、命令执行、回收、任务协议及完整算例 |
| [文献调研](env-worker-deep-dive/literature-reviews/README.md) | 论文、基准调研和对项目的启发 |
| [研究提案](env-worker-deep-dive/research-proposals/README.md) | 候选问题、方案、对照与否证条件 |
| [实验报告](env-worker-deep-dive/experiments/README.md) | 测量结果、原因解释、失败证据和复现入口 |
| [阶段介绍](env-worker-deep-dive/milestones/README.md) | Science GPU 工作说明和带版本条件的阶段材料 |
| [历史实现](env-worker-deep-dive/archive/README.md) | 固定版本的旧架构和设计取舍，供追溯 |

例如，想知道“模型已在 A 卡上时是否值得换 B 卡”，可直接读[热卡等待与冷卡加载实验](env-worker-deep-dive/experiments/gpu-benchmarks/04-动态放置_热卡等待与冷卡加载对照.md)。代价公式和协程执行见[代码指南](env-worker-deep-dive/knowledge/02-资源与调度/09-动态放置_热卡等待与冷卡重建代码.md)，何时允许预测、怎样显式接入固定步数服务见[命令可预测性](env-worker-deep-dive/knowledge/02-资源与调度/08-命令预测_适用条件与时长估计.md)，哪些题库已有收益证据、哪些仍是候选见[逐 benchmark 判断](env-worker-deep-dive/knowledge/02-资源与调度/08-命令预测_适用条件与时长估计.md#11-当前-benchmark-中哪些值得使用这套机制)。

跨题库的公共调度研究见[未知时长下的并发准入](env-worker-deep-dive/research-proposals/07-并发准入_未知命令时长与反馈调度.md)。[运行依赖准备与 GPU 就绪](env-worker-deep-dive/research-proposals/08-运行准备_重复准备成本与GPU就绪调度.md)记录六类代表任务均获益的固定旧 runner 对照，区分缓存命中收益、首次准备成本与尚未验证的 GPU 调度增量。

想理解“提交模型权重后，PSRL 究竟训练谁”，读 [MLGym 权重评分与 Agent DAPO](env-worker-deep-dive/knowledge/03-任务接入/04-MLGym与SWE及Serve_任务协议与评分训练.md#权重如何评分探索轨迹如何训练-agent)，沿隐藏评分、探索轨迹、组内优势和策略更新走一个完整例子。

想研究 infra Agent 训练中的 GPU 沙盒调度，读 [RSI 基础设施调研](env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md)。同篇串联 CUDA Agent、Firepool 等直接相关工作、[PSRL 三项能力与研究边界](env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md#firepool-场景与-psrl-三项能力对照)，以及自生成与模型接管扩展。

读论文机制时，可从[论文原图目录](_static/img/literature-reviews/README.md#论文与官方原图)直达作者的架构、时间线和状态图。八篇文献正文以及vLLM、动态放置、训推共享和运行准备专题已就地补充中文读图说明，来源、图号和本地实现边界随图标注。

## 系统基础与开发工具

[基础与工具索引](tools/README.md)收录 Shell/Python、Linux 存储与进程、Codex 和 Claude Code 等专题。历史故障记录标注原版本，日常操作说明优先描述当前流程。

## 文档维护

- [可读性整理与核验记录](tools/文档维护/01-文档组织_阅读结构与维护规范.md)：本次修改、拆分、保留或删除依据，以及检查范围。
- [时效审计](tools/文档维护/02-文档时效_2026年11月历史巡检.md)：实现、历史证据和待核实内容的维护边界。
- [站点构建与维护](site-maintenance.md)：本地 Markdown 导航与公开文档站点的区别，构建及检查命令。

## 全量文档直达目录

此处按主题列出全部维护中的正文与目录入口。中文正文采用“序号-主题_具体内容.md”，编号表示所在目录的推荐阅读顺序。README负责导航，index保留站点入口约定，公开英文教程沿用原名。原始结果、脚本与冻结baseline通过所属目录查找。

- [项目概览](#项目概览)
- [安装与配置](#安装与配置)
- [架构设计](#架构设计)
- [训练与任务示例](#训练与任务示例)
- [训练原理与运行](#训练原理与运行)
- [环境总览](#环境总览)
- [环境机制与任务接入](#环境机制与任务接入)
- [论文与基准调研](#论文与基准调研)
- [研究提案](#研究提案)
- [实验结果与历史证据](#实验结果与历史证据)
- [阶段介绍与分享](#阶段介绍与分享)
- [历史实现](#历史实现)
- [验证脚本与审查材料](#验证脚本与审查材料)
- [AIRS 与 GPU 作业审查](#airs-与-gpu-作业审查)
- [GPU 作业证据](#gpu-作业证据)
- [计算机系统基础](#计算机系统基础)
- [开发工具与文档维护](#开发工具与文档维护)
- [站点资源](#站点资源)
- [站点模板](#站点模板)
- [站点入口与维护](#站点入口与维护)

### 项目概览

**`overview`**

- [项目概览与阅读起点](overview/README.md)（目录导航）
- [01 · PSRL入门：从一道题到模型更新](overview/01-PSRL入门_从一道题到模型更新.md)
- [02 · PSRL入门：系统与实验术语速查](overview/02-PSRL入门_系统与实验术语速查.md)
- [Overview](overview/index.md)


### 安装与配置

**`tutorial`**

- [安装、运行与配置](tutorial/README.md)（目录导航）
- [Configuration](tutorial/configuration.md)
- [Tutorial](tutorial/index.md)
- [Installation](tutorial/installation.md)
- [Performance Tuning](tutorial/performance_tuning.md)
- [Quick Start](tutorial/quickstart.md)


### 架构设计

**`design`**

- [架构与组件设计](design/README.md)（目录导航）
- [Overall System Architecture](design/architecture.md)
- [EnvWorker](design/env_worker.md)
- [Flexible Rollout Coordination](design/flexible_rollout.md)
- [Designs & Features](design/index.md)
- [KV Cache Management](design/kv_cache.md)
- [Parameter Server](design/parameter_server.md)
- [Resource Elasticity and Worker Memory](design/resource_elasticity.md)
- [Router, SessionRouter, and TITO](design/router_tito.md)
- [Fine-Grained Staleness Control](design/staleness_control.md)
- [TransferQueue Integration](design/transfer_queue.md)


### 训练与任务示例

**`examples`**

- [算法与任务示例](examples/README.md)（目录导航）
- [Examples](examples/index.md)

**`examples/agentic_rl`**

- [工具与软件任务](examples/agentic_rl/README.md)（目录导航）
- [Agentic RL](examples/agentic_rl/index.md)

**`examples/agentic_rl/retool`**

- [ReTool 工具调用](examples/agentic_rl/retool/README.md)（目录导航）
- [ReTool: Code Interpreter RL](examples/agentic_rl/retool/index.md)

**`examples/agentic_rl/swe`**

- [SWE 软件修复](examples/agentic_rl/swe/README.md)（目录导航）
- [SWE Agentic RL](examples/agentic_rl/swe/index.md)

**`examples/generative_reward_model`**

- [生成式奖励模型规划](examples/generative_reward_model/README.md)（目录导航）
- [Generative Reward Model](examples/generative_reward_model/index.md)

**`examples/rlvr`**

- [可验证奖励训练配方](examples/rlvr/README.md)（目录导航）
- [Common Infrastructure](examples/rlvr/common.md)
- [DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization](examples/rlvr/dapo.md)
- [GRPO: Group Relative Policy Optimization](examples/rlvr/grpo.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](examples/rlvr/index.md)
- [PPO: Proximal Policy Optimization](examples/rlvr/ppo.md)


### 训练原理与运行

**`psrl`**

- [PSRL：从一次做题理解训练框架](psrl/README.md)（目录导航）
- [01 · PSRL原理：AgentLoop与工具调用](psrl/01-PSRL原理_AgentLoop与工具调用.md)
- [02 · 强化学习：PPO与GRPO及DAPO](psrl/02-强化学习_PPO与GRPO及DAPO.md)
- [03 · 模型配置：上下文长度与生成限制](psrl/03-模型配置_上下文长度与生成限制.md)
- [04 · 高速传输：NIXL内存搬运机制](psrl/04-高速传输_NIXL内存搬运机制.md)
- [05 · 训练排障：DAPO退化诊断](psrl/05-训练排障_DAPO退化诊断.md)
- [06 · 训练排障：SciAccel显存不足与修复](psrl/06-训练排障_SciAccel显存不足与修复.md)
- [07 · 源码阅读：从Science脚本到训练流程](psrl/07-源码阅读_从Science脚本到训练流程.md)

**`psrl/cluster-pitfall-checks`**

- [集群问题检查脚本与证据](psrl/cluster-pitfall-checks/README.md)（目录导航）

**`psrl/psrl-run-guide`**

- [集群预检与运行故障](psrl/psrl-run-guide/README.md)（目录导航）
- [01 · PSRL运行：准备启动与监控](psrl/psrl-run-guide/01-PSRL运行_准备启动与监控.md)，可直达 [Ray 在 PSRL 中的职责与通用概念](psrl/psrl-run-guide/01-PSRL运行_准备启动与监控.md#ray-角色与-rollout-流程)。
- [02 · 集群预检：依赖地址与GPU资源核验](psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md)
- [03 · 启动排障：单机与多机故障复盘](psrl/psrl-run-guide/03-启动排障_单机与多机故障复盘.md)：先看近期Env实验的Ray注册超时、批量预启动和依赖错误，再查旧IP、数据和上下文故障。

**`psrl/psrl-run-guide/incidents`**

- [独立故障记录](psrl/psrl-run-guide/incidents/README.md)（目录导航）
- [01 · Docker故障：2026年10月20日存储挂起](psrl/psrl-run-guide/incidents/01-Docker故障_2026年10月20日存储挂起.md)

**`psrl/推理与训练`**

- [推理与训练专题：阅读顺序与章节索引](psrl/推理与训练/README.md)（目录导航）
- [01 · vLLM推理：计算瓶颈与KV缓存](psrl/推理与训练/01-vLLM推理_计算瓶颈与KV缓存.md)
- [02 · 训练并行：模型切分与通信带宽](psrl/推理与训练/02-训练并行_模型切分与通信带宽.md)
- [03 · 训推协作：异步流水线与版本协调](psrl/推理与训练/03-训推协作_异步流水线与版本协调.md)
- [04 · 性能分析：公式速查与历史实验](psrl/推理与训练/04-性能分析_公式速查与历史实验.md)


### 环境总览

**`env-worker-deep-dive`**

- [环境执行与回收专题](env-worker-deep-dive/README.md)（目录导航）


### 环境机制与任务接入

**`env-worker-deep-dive/knowledge`**

- [Env 环境执行：机制与接入指南](env-worker-deep-dive/knowledge/README.md)（目录导航）

**`env-worker-deep-dive/knowledge/01-架构与执行`**

- [Env架构与执行：从任务申请到容器回收](env-worker-deep-dive/knowledge/01-架构与执行/README.md)（目录导航）
- [01 · Env架构：现状与完整执行流程](env-worker-deep-dive/knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)
- [02 · 统一执行：EnvRouter与容器及预算策略](env-worker-deep-dive/knowledge/01-架构与执行/02-统一执行_EnvRouter与容器及预算策略.md)
- [03 · 容器后端：EnvWorker与Harbor统一实现](env-worker-deep-dive/knowledge/01-架构与执行/03-容器后端_EnvWorker与Harbor统一实现.md)
- [04 · GPU作业：生命周期与任务遥测](env-worker-deep-dive/knowledge/01-架构与执行/04-GPU作业_生命周期与任务遥测.md)
- [05 · EnvRouter：配置恢复与Worker重试](env-worker-deep-dive/knowledge/01-架构与执行/05-EnvRouter_配置恢复与Worker重试.md)
- [06 · 工程审查：待修复问题与优先级](env-worker-deep-dive/knowledge/01-架构与执行/06-工程审查_待修复问题与优先级.md)

**`env-worker-deep-dive/knowledge/02-资源与调度`**

- [Env资源与调度：容量、借还、状态和动态放置](env-worker-deep-dive/knowledge/02-资源与调度/README.md)（目录导航）
- [01 · 显存准入：实测占用与预约预算](env-worker-deep-dive/knowledge/02-资源与调度/01-显存准入_实测占用与预约预算.md)
- [02 · 命令调度：GPU申请归还与状态保留](env-worker-deep-dive/knowledge/02-资源与调度/02-命令调度_GPU申请归还与状态保留.md)
- [03 · 进程归属：PID身份与显存记账](env-worker-deep-dive/knowledge/02-资源与调度/03-进程归属_PID身份与显存记账.md)
- [04 · GPU并行：MPS执行后端](env-worker-deep-dive/knowledge/02-资源与调度/04-GPU并行_MPS执行后端.md)
- [05 · 训推共享：vLLM余量与容量调度](env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md)，含[默认放满的动态准入设计](env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md#默认放满并用完成时间反馈收紧准入)。
- [06 · 训推借卡：Rollout睡眠与归还协议](env-worker-deep-dive/knowledge/02-资源与调度/06-训推借卡_Rollout睡眠与归还协议.md)
- [07 · 状态复用：基线状态寿命与版本正确性](env-worker-deep-dive/knowledge/02-资源与调度/07-状态复用_基线状态寿命与版本正确性.md)
- [08 · 命令预测：适用条件与时长估计](env-worker-deep-dive/knowledge/02-资源与调度/08-命令预测_适用条件与时长估计.md)
- [09 · 动态放置：热卡等待与冷卡重建代码](env-worker-deep-dive/knowledge/02-资源与调度/09-动态放置_热卡等待与冷卡重建代码.md)

**`env-worker-deep-dive/knowledge/03-任务接入`**

- [Env任务接入：公共协议与各题库配方](env-worker-deep-dive/knowledge/03-任务接入/README.md)（目录导航）
- [01 · 任务接入：公共GPU任务与PSRL配方](env-worker-deep-dive/knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)
- [02 · 任务接入：EnvWorker配置与CUDA验证](env-worker-deep-dive/knowledge/03-任务接入/02-任务接入_EnvWorker配置与CUDA验证.md)
- [03 · AIRSBench：接入流程与有界验收](env-worker-deep-dive/knowledge/03-任务接入/03-AIRSBench_接入流程与有界验收.md)
- [04 · MLGym与SWE及Serve：任务协议与评分训练](env-worker-deep-dive/knowledge/03-任务接入/04-MLGym与SWE及Serve_任务协议与评分训练.md)
- [05 · REBench：聊天模型训练接入](env-worker-deep-dive/knowledge/03-任务接入/05-REBench_聊天模型训练接入.md)
- [06 · InferenceBench：评分机制与开关验证](env-worker-deep-dive/knowledge/03-任务接入/06-InferenceBench_评分机制与开关验证.md)


### 论文与基准调研

**`env-worker-deep-dive/literature-reviews`**

- [文献调研与论文精读](env-worker-deep-dive/literature-reviews/README.md)（目录导航）
- [01 · 调度论文：五篇近期顶会工作综述](env-worker-deep-dive/literature-reviews/01-调度论文_五篇近期顶会工作综述.md)
- [02 · 论文精读：RollArt与DynaRL的Agent调度](env-worker-deep-dive/literature-reviews/02-论文精读_RollArt与DynaRL的Agent调度.md)
- [03 · 论文精读：KREX共享GPU测量保护](env-worker-deep-dive/literature-reviews/03-论文精读_KREX共享GPU测量保护.md)
- [04 · 系统借鉴：YARN资源分配与反馈调度](env-worker-deep-dive/literature-reviews/04-系统借鉴_YARN资源分配与反馈调度.md)
- [05 · 基准调研：GPU任务类型与接入顺序](env-worker-deep-dive/literature-reviews/05-基准调研_GPU任务类型与接入顺序.md)
- [06 · 基准调研：AI4Infra性能反馈与验证](env-worker-deep-dive/literature-reviews/06-基准调研_AI4Infra性能反馈与验证.md)
- [07 · 智能调度：模型决策的已有工作与研究选择](env-worker-deep-dive/literature-reviews/07-智能调度_模型决策的已有工作与研究选择.md)
- [08 · 基础设施：RSI与GPU沙盒及模型接管](env-worker-deep-dive/literature-reviews/08-基础设施_RSI与GPU沙盒及模型接管.md)


### 研究提案

**`env-worker-deep-dive/research-proposals`**

- [GPU 环境调度：研究问题与验证进展](env-worker-deep-dive/research-proposals/README.md)（目录导航）
- [01 · 调度研究：Rollout完成时间与反馈保真](env-worker-deep-dive/research-proposals/01-调度研究_Rollout完成时间与反馈保真.md)
- [02 · 研究场景：非计时评分的GPU任务](env-worker-deep-dive/research-proposals/02-研究场景_非计时评分的GPU任务.md)
- [03 · 动态放置：会话与GPU执行分离](env-worker-deep-dive/research-proposals/03-动态放置_会话与GPU执行分离.md)
- [04 · 状态复用：真实任务中的收益空间核查](env-worker-deep-dive/research-proposals/04-状态复用_真实任务中的收益空间核查.md)
- [05 · 状态调度：主存受限时的准备状态保留](env-worker-deep-dive/research-proposals/05-状态调度_主存受限时的准备状态保留.md)
- [06 · 训推借卡：可撤回执行与反馈依赖](env-worker-deep-dive/research-proposals/06-训推借卡_可撤回执行与反馈依赖.md)
- [07 · 并发准入：未知命令时长与反馈调度](env-worker-deep-dive/research-proposals/07-并发准入_未知命令时长与反馈调度.md)
- [08 · 运行准备：重复准备成本与GPU就绪调度](env-worker-deep-dive/research-proposals/08-运行准备_重复准备成本与GPU就绪调度.md)


### 实验结果与历史证据

**`env-worker-deep-dive/experiments`**

- [Env 实验与验证入口](env-worker-deep-dive/experiments/README.md)（目录导航）
- [01 · 实验总览：调度收益与证据范围](env-worker-deep-dive/experiments/01-实验总览_调度收益与证据范围.md)
- [02 · 基线对照：低负载EnvRouter修复验证](env-worker-deep-dive/experiments/02-基线对照_低负载EnvRouter修复验证.md)
- [03 · Science训练：三步对照与显存压力](env-worker-deep-dive/experiments/03-Science训练_三步对照与显存压力.md)
- [04 · MPS训练：并行执行与控制修复对照](env-worker-deep-dive/experiments/04-MPS训练_并行执行与控制修复对照.md)
- [05 · 容器后端：原生实现修复与Science验证](env-worker-deep-dive/experiments/05-容器后端_原生实现修复与Science验证.md)
- [06 · 训推共享：常驻vLLM与环境并行实验](env-worker-deep-dive/experiments/06-训推共享_常驻vLLM与环境并行实验.md)
- [07 · vLLM并发：吞吐与请求延迟曲线](env-worker-deep-dive/experiments/07-vLLM并发_吞吐与请求延迟曲线.md)
- [08 · 模式选择：Command与Episode及等待预测](env-worker-deep-dive/experiments/08-模式选择_Command与Episode及等待预测.md)
- [09 · 资源分区：静态卡池与推理余量共享](env-worker-deep-dive/experiments/09-资源分区_静态卡池与推理余量共享.md)：按九轮实验回顾借卡、突发、完整训练、并发、分卡和同步。统一主基线为静态环境池＋episode 持卡＋满池排队，1200 秒 episode 补测已完成，历史四组的训练与完整作业分别比较。此前启动修复复测未触发借卡。新配对观察到 GPU2 交接由 98.11 降至 5.19 秒。首次重叠组命令超时，补跑完成两步。
- [10 · 进程归属：Procfs挂载障碍与处理](env-worker-deep-dive/experiments/10-进程归属_Procfs挂载障碍与处理.md)
- [11 · 历史审阅：低负载配对与GateAB设计](env-worker-deep-dive/experiments/11-历史审阅_低负载配对与GateAB设计.md)
- [12 · 研究报告：PSRL 环境 GPU 调度的设计与评估](env-worker-deep-dive/experiments/12-研究报告_PSRL环境GPU调度的设计与评估.md)
- [13 · 扩展对照：八卡单批次与多负载回放](env-worker-deep-dive/experiments/13-扩展对照_八卡单批次与多负载回放.md)，含[初始化分解与控制连接重叠](env-worker-deep-dive/experiments/13-扩展对照_八卡单批次与多负载回放.md#初始化为什么更久与怎样缩短)

**`env-worker-deep-dive/experiments/gpu-benchmarks`**

- [GPU benchmark 实验导航](env-worker-deep-dive/experiments/gpu-benchmarks/README.md)（目录导航）
- [01 · 任务验收：结果与证据导航](env-worker-deep-dive/experiments/gpu-benchmarks/01-任务验收_结果与证据导航.md)
- [02 · 状态复用：等待与重新加载的前期实验](env-worker-deep-dive/experiments/gpu-benchmarks/02-状态复用_等待与重新加载的前期实验.md)
- [03 · 动态放置：真实模型驻留与跨卡探索](env-worker-deep-dive/experiments/gpu-benchmarks/03-动态放置_真实模型驻留与跨卡探索.md)
- [04 · 动态放置：热卡等待与冷卡加载对照](env-worker-deep-dive/experiments/gpu-benchmarks/04-动态放置_热卡等待与冷卡加载对照.md)
- [05 · 评分复用：PostTrain准备成本与整批耗时](env-worker-deep-dive/experiments/gpu-benchmarks/05-评分复用_PostTrain准备成本与整批耗时.md)
- [06 · 显存压力：提前排队与失败重试](env-worker-deep-dive/experiments/gpu-benchmarks/06-显存压力_提前排队与失败重试.md)
- [07 · FlashInfer：进程复用与任务合批](env-worker-deep-dive/experiments/gpu-benchmarks/07-FlashInfer_进程复用与任务合批.md)
- [08 · 性能反馈：高压力调度实验](env-worker-deep-dive/experiments/gpu-benchmarks/08-性能反馈_高压力调度实验.md)

**`env-worker-deep-dive/experiments/gpu-benchmarks/figures`**

- [实验图表](env-worker-deep-dive/experiments/gpu-benchmarks/figures/README.md)（目录导航）

**`env-worker-deep-dive/experiments/gpu-benchmarks/flashinfer-env`**

- [FlashInfer 实验依赖](env-worker-deep-dive/experiments/gpu-benchmarks/flashinfer-env/README.md)（目录导航）

**`env-worker-deep-dive/experiments/gpu-benchmarks/scripts`**

- [辅助脚本](env-worker-deep-dive/experiments/gpu-benchmarks/scripts/README.md)（目录导航）

**`env-worker-deep-dive/experiments/history`**

- [历史实验与修复过程](env-worker-deep-dive/experiments/history/README.md)（目录导航）
- [01 · 历史验证：2026年10月13日EnvService修复](env-worker-deep-dive/experiments/history/01-历史验证_2026年10月13日EnvService修复.md)
- [02 · 历史对照：Science资源账本与Unit余量](env-worker-deep-dive/experiments/history/02-历史对照_Science资源账本与Unit余量.md)
- [03 · 历史对照：Science跨任务池显存竞争](env-worker-deep-dive/experiments/history/03-历史对照_Science跨任务池显存竞争.md)
- [04 · 历史对照：LAPS训练与EnvRouter开关](env-worker-deep-dive/experiments/history/04-历史对照_LAPS训练与EnvRouter开关.md)
- [05 · 历史对照：同任务多协程GPU并发](env-worker-deep-dive/experiments/history/05-历史对照_同任务多协程GPU并发.md)
- [06 · 历史过程：Science显存压力与原生任务](env-worker-deep-dive/experiments/history/06-历史过程_Science显存压力与原生任务.md)
- [07 · 历史过程：固定基线修复与单卡验证](env-worker-deep-dive/experiments/history/07-历史过程_固定基线修复与单卡验证.md)
- [08 · 历史过程：MPS训练并发对照](env-worker-deep-dive/experiments/history/08-历史过程_MPS训练并发对照.md)
- [09 · 历史故障：原生后端修复前生命周期](env-worker-deep-dive/experiments/history/09-历史故障_原生后端修复前生命周期.md)
- [10 · 历史提案：阶段租约与回填调度](env-worker-deep-dive/experiments/history/10-历史提案_阶段租约与回填调度.md)
- [11 · 历史过程：GPU基准接入验证](env-worker-deep-dive/experiments/history/11-历史过程_GPU基准接入验证.md)
- [12 · 历史阶段：2026年10月28日设计与收益](env-worker-deep-dive/experiments/history/12-历史阶段_2026年10月28日设计与收益.md)


### 阶段介绍与分享

**`env-worker-deep-dive/milestones`**

- [Env 阶段成果与 benchmark 导览](env-worker-deep-dive/milestones/README.md)（目录导航）
- [01 · Science介绍：已接入任务与研究价值](env-worker-deep-dive/milestones/01-Science介绍_已接入任务与研究价值.md)
- [02 · Science现状：任务覆盖与接入路线](env-worker-deep-dive/milestones/02-Science现状_任务覆盖与接入路线.md)
- [03 · Science验收：接入流程与环境隔离](env-worker-deep-dive/milestones/03-Science验收_接入流程与环境隔离.md)
- [04 · 调度研究：研究全景交流版](env-worker-deep-dive/milestones/04-调度研究_研究全景交流版.md)
- [05 · 调度研究：研究全景与证据详解](env-worker-deep-dive/milestones/05-调度研究_研究全景与证据详解.md)
- [06 · 历史阶段：2026年10月28日实现与规划](env-worker-deep-dive/milestones/06-历史阶段_2026年10月28日实现与规划.md)


### 历史实现

**`env-worker-deep-dive/archive`**

- [历史实现与阶段分析](env-worker-deep-dive/archive/README.md)（目录导航）
- [01 · Env历史：2026年9月EnvWorker原始提案](env-worker-deep-dive/archive/01-Env历史_2026年9月EnvWorker原始提案.md)
- [02 · Env历史：2026年10月12日资源单元与租约](env-worker-deep-dive/archive/02-Env历史_2026年10月12日资源单元与租约.md)
- [03 · Env历史：早期Router执行流程](env-worker-deep-dive/archive/03-Env历史_早期Router执行流程.md)
- [04 · Env历史：Science整卡锁与设备账本](env-worker-deep-dive/archive/04-Env历史_Science整卡锁与设备账本.md)
- [05 · Env历史：修复前账本回收与缺口](env-worker-deep-dive/archive/05-Env历史_修复前账本回收与缺口.md)
- [06 · Env历史：72ca50c版本GateAB机制](env-worker-deep-dive/archive/06-Env历史_72ca50c版本GateAB机制.md)
- [07 · Env历史：GateB回复节流机制](env-worker-deep-dive/archive/07-Env历史_GateB回复节流机制.md)


### 验证脚本与审查材料

**`env-worker-deep-dive/env-service-verification`**

- [Env 实验脚本与证据目录](env-worker-deep-dive/env-service-verification/README.md)（目录导航）

**`env-worker-deep-dive/env-service-verification/scripts`**

- [辅助脚本](env-worker-deep-dive/env-service-verification/scripts/README.md)（目录导航）

**`env-worker-deep-dive/env-service-verification/scripts/admission_study`**

- [未知时长准入实验脚本](env-worker-deep-dive/env-service-verification/scripts/admission_study/README.md)（目录导航）

**`env-worker-deep-dive/env-service-verification/scripts/feedback_pressure`**

- [性能测量干扰实验脚本](env-worker-deep-dive/env-service-verification/scripts/feedback_pressure/README.md)（目录导航）

**`env-worker-deep-dive/env-service-verification/scripts/joint_gpu_study`**

- [联合 GPU 放置实验脚本](env-worker-deep-dive/env-service-verification/scripts/joint_gpu_study/README.md)（目录导航）

**`env-worker-deep-dive/env-service-verification/scripts/posttrain_reuse_study`**

- [PostTrain 裁判复用实验脚本](env-worker-deep-dive/env-service-verification/scripts/posttrain_reuse_study/README.md)（目录导航）

**`env-worker-deep-dive/env-service-verification/scripts/resident_sharing_study`**

- [常驻模型计算共享实验脚本](env-worker-deep-dive/env-service-verification/scripts/resident_sharing_study/README.md)（目录导航）

**`env-worker-deep-dive/env-service-verification/scripts/state_reuse_study`**

- [状态复用实验脚本](env-worker-deep-dive/env-service-verification/scripts/state_reuse_study/README.md)（目录导航）

**`env-worker-deep-dive/env-service-verification/scripts/warm_router_study`**

- [热状态放置实验脚本](env-worker-deep-dive/env-service-verification/scripts/warm_router_study/README.md)（目录导航）


### AIRS 与 GPU 作业审查

**`env-worker-deep-dive/airs-bench-audit`**

- [AIRS 接入审查与隔离证据](env-worker-deep-dive/airs-bench-audit/README.md)（目录导航）
- [01 · AIRS审计：修复状态与验证范围](env-worker-deep-dive/airs-bench-audit/01-AIRS审计_修复状态与验证范围.md)
- [02 · APPS判题：代码隔离与验证边界](env-worker-deep-dive/airs-bench-audit/02-APPS判题_代码隔离与验证边界.md)

**`env-worker-deep-dive/airs-bench-audit/apps-followup`**

- [APPS 后续核验](env-worker-deep-dive/airs-bench-audit/apps-followup/README.md)（目录导航）

**`env-worker-deep-dive/airs-bench-audit/apps-isolation`**

- [APPS 隔离复现](env-worker-deep-dive/airs-bench-audit/apps-isolation/README.md)（目录导航）

**`env-worker-deep-dive/airs-bench-audit/review-round-two`**

- [AIRS 第二轮审查](env-worker-deep-dive/airs-bench-audit/review-round-two/README.md)（目录导航）


### GPU 作业证据

**`env-worker-deep-dive/gpu-job-validation`**

- [GPU 作业部署与延迟证据](env-worker-deep-dive/gpu-job-validation/README.md)（目录导航）


### 计算机系统基础

**`tools/系统基础`**

- [计算机系统基础：环境、进程、文件、网络与GPU](tools/系统基础/README.md)（目录导航）
- [01 · 运行环境：Shell与Python及Conda](tools/系统基础/01-运行环境_Shell与Python及Conda.md)
- [02 · 操作系统：进程与文件映射及SQLite](tools/系统基础/02-操作系统_进程与文件映射及SQLite.md)
- [03 · 文件系统：inode与链接及Docker写回](tools/系统基础/03-文件系统_inode与链接及Docker写回.md)
- [04 · 网络通信：HTTP请求与GateB调用链](tools/系统基础/04-网络通信_HTTP请求与GateB调用链.md)
- [05 · GPU并行：CUDA上下文与计算竞争](tools/系统基础/05-GPU并行_CUDA上下文与计算竞争.md)
- [06 · 远程登录：SSH 连接与 IP、网卡、端口](tools/系统基础/06-远程登录_SSH连接与IP网卡端口.md)


### 开发工具与文档维护

**`tools`**

- [开发工具知识索引](tools/README.md)（目录导航）

**`tools/claude-code`**

- [Claude Code 使用](tools/claude-code/README.md)（目录导航）
- [01 · ClaudeCode：命令执行与权限设置](tools/claude-code/01-ClaudeCode_命令执行与权限设置.md)

**`tools/codex`**

- [Codex 使用、规则与故障](tools/codex/README.md)（目录导航）
- [01 · 智能体配置：Rule与Memory及Skill](tools/codex/01-智能体配置_Rule与Memory及Skill.md)
- [02 · Codex配置：备份与按需恢复](tools/codex/02-Codex配置_备份与按需恢复.md)

**`tools/codex/scripts`**

- [辅助脚本](tools/codex/scripts/README.md)（目录导航）

**`tools/codex/session-archives`**

- [Codex 会话归档](tools/codex/session-archives/README.md)（目录导航）

**`tools/codex/session-management`**

- [Codex 终端与历史恢复](tools/codex/session-management/README.md)（目录导航）
- [01 · Codex会话：恢复命名与历史回看](tools/codex/session-management/01-Codex会话_恢复命名与历史回看.md)，含 [Fork 与多会话上下文汇总](tools/codex/session-management/01-Codex会话_恢复命名与历史回看.md#fork-与多会话上下文汇总)
- [02 · Codex终端：滚动与Tmux排查](tools/codex/session-management/02-Codex终端_滚动与Tmux排查.md)
- [03 · Codex故障：会话数据库与SIGBUS排查](tools/codex/session-management/03-Codex故障_会话数据库与SIGBUS排查.md)，含[code 26、名称与老会话修复](tools/codex/session-management/03-Codex故障_会话数据库与SIGBUS排查.md#再次出现-code-26状态库正常时只重建历史)、[索引重置与历史页损坏](tools/codex/session-management/03-Codex故障_会话数据库与SIGBUS排查.md#状态索引重置与历史页损坏)

**`tools/linux`**

- [Linux教学：CPU演示脚本与原始证据](tools/linux/README.md)（目录导航）

**`tools/文档维护`**

- [文档维护：组织规范与历史巡检](tools/文档维护/README.md)（目录导航）
- [01 · 文档组织：阅读结构与维护规范](tools/文档维护/01-文档组织_阅读结构与维护规范.md)
- [02 · 文档时效：2026年11月历史巡检](tools/文档维护/02-文档时效_2026年11月历史巡检.md)


### 站点资源

**`_static`**

- [站点静态资源](_static/README.md)（目录导航）

**`_static/css`**

- [站点样式](_static/css/README.md)（目录导航）

**`_static/img`**

- [站点图片](_static/img/README.md)（目录导航）

**`_static/img/literature-reviews`**

- [调度文献与机制图集](_static/img/literature-reviews/README.md)（目录导航）

**`_static/img/document-figures`**

- [全文档配图：彩图、来源与复现](_static/img/document-figures/README.md)（图集与维护入口）

**`_static/js`**

- [站点交互脚本](_static/js/README.md)（目录导航）


### 站点模板

**`_templates`**

- [站点模板](_templates/README.md)（目录导航）


### 站点入口与维护

**`.`**

- [PSRL Documentation](index.md)
- [文档站点构建与维护](site-maintenance.md)

## 源码包中的既有文档

下面 9 篇均已存在于固定 baseline `0eeead861ffafed6bf31c962d259fd847775dcf9`，属于项目原有模块说明与设计记录，随源码保留。2026-11-18 检查新增提交来源及当前差异，未发现我们新增而散落在源码包内的知识总结。新知识文档统一放在 docs。

| 文档 | 最初引入提交 | 作者 |
|---|---|---|
| [psrl/environments/README.md](../psrl/environments/README.md) | `04365d7` | Somoku |
| [psrl/eval/README.md](../psrl/eval/README.md) | `adc207d` | lhy101 |
| [psrl/utils/elastic_rm/research.md](../psrl/utils/elastic_rm/research.md) | `fd42351` | Somoku |
| [psrl/utils/elastic_rm/REVIEW.md](../psrl/utils/elastic_rm/REVIEW.md) | `fd42351` | Somoku |
| [psrl/bench/chunked_prefill/step_latency_model.md](../psrl/bench/chunked_prefill/step_latency_model.md) | `a311d3e` | lhy101 |
| [psrl/bench/chunked_prefill/USAGE.md](../psrl/bench/chunked_prefill/USAGE.md) | `a311d3e` | lhy101 |
| [psrl/utils/nixl/README.md](../psrl/utils/nixl/README.md) | `9618940` | lhy101 |
| [psrl/utils/converter/README.md](../psrl/utils/converter/README.md) | `c9c4b82` | lhy101 |
| [psrl/utils/visualization/README.md](../psrl/utils/visualization/README.md) | `da91f0d` | lhy101 |
