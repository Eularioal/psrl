# PSRL 文档总览

> **核心结论**：第一次了解项目，先读一条完整流程，再按问题进入专题。实现说明、文献调研、研究提案和实验记录各有独立目录。同一问题的机制、例子和直接相关结果集中阅读，较长时用目录定位。

## 目录

- [第一次阅读](#第一次阅读)
- [可直接分享的 Science GPU 介绍](#可直接分享的-science-gpu-介绍)
- [使用、算法与架构](#使用算法与架构)
- [环境执行与资源调度](#环境执行与资源调度)
- [系统基础与开发工具](#系统基础与开发工具)
- [文档维护](#文档维护)

## 第一次阅读

1. [从一道题理解 PSRL](overview/reading-guide.md)：沿“做题 → 评分 → 更新模型”的例子串起所有主要组件，附源码阅读顺序和 CPU 小例子。
2. [系统与实验术语速查](overview/glossary.md)：按需查进程、并发、容器、显存、租约和实验口径，每项附作用与深入入口。
3. [PSRL 概览](overview/index.md)：了解项目目标、可配置的后端和主要能力。

如果只是了解工作成果，可以直接读下一节的简报。关键术语在正文就地解释，无需先通读底层实现。

## 可直接分享的 Science GPU 介绍

| 文档 | 适合什么读者 | 主要内容 |
|---|---|---|
| [调度研究 Big Map 精简版](env-worker-deep-dive/milestones/env-scheduling-big-map-brief.md) | 与师兄讨论选题 | 已探索方向、主要证据、负结果和需要帮助判断的问题，附[详细版](env-worker-deep-dive/milestones/env-scheduling-big-map.md) |
| [简版介绍](env-worker-deep-dive/milestones/science-gpu-bench-brief.md) | 第一次了解这项工作 | 解决什么问题、数字分类题的完整过程、已有任务与研究价值 |
| [各 bench 任务与目录导览](env-worker-deep-dive/milestones/README.md#各-bench-的任务与代码入口) | 想找某个 bench 的代码或题库 | 各 bench 做什么、代码目录、任务准备器、已准备题库与两条接入方式 |
| [任务现状与路线图](env-worker-deep-dive/milestones/science-gpu-bench-status-and-roadmap.md) | 评估覆盖和后续工作 | 逐题库数量、验收层次、剩余缺口与研究顺序 |
| [接入与验收详情](env-worker-deep-dive/milestones/science-gpu-full-onboarding.md) | 需要复现或继续开发 | 执行和训练流程、环境隔离实例、历史验收与证据入口 |

## 使用、算法与架构

| 入口 | 能回答的问题 |
|---|---|
| [安装与运行](tutorial/index.md) | 怎样准备软件、模型、数据和一次有界运行 |
| [架构设计](design/index.md) | 参数、样本、生成请求怎样在组件之间流动 |
| [算法与应用示例](examples/index.md) | 数学做题、工具调用、软件修复分别怎样配置 |
| [训练原理与源码知识](psrl/README.md) | PPO/GRPO/DAPO、优势估计、推理缓存与并行机制怎样工作 |
| [集群运行前检查](psrl/psrl-run-guide/cluster-preflight.md) | 共享环境里怎样核对依赖、地址、Docker、GPU 和清理范围 |

## 环境执行与资源调度

[环境专题导航](env-worker-deep-dive/README.md)按阅读目的串联以下目录：

| 分类 | 内容 |
|---|---|
| [当前设计](env-worker-deep-dive/knowledge/current-design.md) | 环境执行的现状入口，区分默认配置、可选后端和研究原型 |
| [机制与任务接入](env-worker-deep-dive/knowledge/README.md) | 准入、显存、命令执行、回收、任务协议及完整算例 |
| [文献调研](env-worker-deep-dive/literature-reviews/README.md) | 论文、基准调研和对项目的启发 |
| [研究提案](env-worker-deep-dive/research-proposals/README.md) | 候选问题、方案、对照与否证条件 |
| [实验报告](env-worker-deep-dive/experiments/README.md) | 测量结果、原因解释、失败证据和复现入口 |
| [阶段介绍](env-worker-deep-dive/milestones/README.md) | Science GPU 工作说明和带版本条件的阶段材料 |
| [历史实现](env-worker-deep-dive/archive/README.md) | 固定版本的旧架构和设计取舍，供追溯 |

例如，想知道“模型已在 A 卡上时是否值得换 B 卡”，可直接读[热卡等待与冷卡加载实验](env-worker-deep-dive/experiments/gpu-benchmarks/warm-placement-matrix.md)。代价公式和协程执行见[代码指南](env-worker-deep-dive/knowledge/warm-placement-code-guide.md)，何时允许预测、怎样显式接入固定步数服务见[命令可预测性](env-worker-deep-dive/knowledge/command-predictability.md)，哪些题库已有收益证据、哪些仍是候选见[逐 benchmark 判断](env-worker-deep-dive/knowledge/command-predictability.md#11-当前-benchmark-中哪些值得使用这套机制)。

想理解“提交模型权重后，PSRL 究竟训练谁”，读 [MLGym 权重评分与 Agent DAPO](env-worker-deep-dive/knowledge/mlgym-weight-scoring-and-agent-dapo.md)，沿隐藏评分、探索轨迹、组内优势和策略更新走一个完整例子。

## 系统基础与开发工具

[基础与工具索引](tools/README.md)收录 Shell/Python、Linux 存储与进程、Codex 和 Claude Code 等专题。历史故障记录标注原版本，日常操作说明优先描述当前流程。

## 文档维护

- [可读性整理与核验记录](tools/docs-organization-audit.md)：本次修改、拆分、保留或删除依据，以及检查范围。
- [时效审计](tools/docs-freshness-audit.md)：实现、历史证据和待核实内容的维护边界。
- [站点构建与维护](site-maintenance.md)：本地 Markdown 导航与公开文档站点的区别，构建及检查命令。
