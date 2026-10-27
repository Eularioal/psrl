# 环境执行与回收专题

> **核心结论**：理解实现、概念和接入方式，从「知识与实现」开始。查实测结果、对照条件和故障证据，从「实验与验证」开始。知识文章中的提案不代表已经实现，实验结论只适用于报告注明的版本与条件。

## 目录

- [按阅读目的选择](#按阅读目的选择)
- [推荐阅读顺序](#推荐阅读顺序)
- [目录与归档规则](#目录与归档规则)

## 按阅读目的选择

| 想了解什么 | 入口 | 内容边界 |
|---|---|---|
| 快速介绍当前设计、收益与重点规划 | [Env 设计与收益简版](experiments/env-design-and-results-brief.md) | 可单文件转发，保留结构图、baseline 对照和必要实验边界。 |
| 详细了解实现进度、收益与规划 | [Env 状态与路线图](experiments/env-status-and-roadmap.md) | 详细版，以结构图和表格对比固定 baseline、当前实现及未来策略，实验条件、数据和边界均在正文中。 |
| 现在怎样工作，接口怎样使用 | [知识与实现](knowledge/README.md) | 流程、源码、配置、接入与能力边界。索引另列早期版本讲解。 |
| 测过什么，结果如何，哪里失败 | [实验与验证](experiments/README.md) | 训练对照、组件验证、性能分析、故障与未验证项。 |
| 从概念出发理解为什么这样设计 | [EnvWorker 设计讲解](../psrl/psrl-env-service-envworker-design.md) | 问题背景与设计动机，阅读时区分提案和实现。 |

例如，想知道「GPU 命令结束后为什么还要确认归还」，读[命令阶段配额](knowledge/command-gpu-reservations.md)。想查「固定 OFF 与 ON 的训练耗时如何比较」，读[真实训练对照](experiments/science-single-gpu-training-control.md)，同时核对基线、压力条件和计时边界。

## 推荐阅读顺序

项目交流先读[设计与收益简版](experiments/env-design-and-results-brief.md)，需要展开时读[详细版](experiments/env-status-and-roadmap.md)，再按下面的路径查实现细节。

1. [当前流程总览](knowledge/env-router-current-flow.md)：先串联任务放置、命令执行、配额归还与清理。
2. [统一执行后端](knowledge/unified-gpu-execution.md)和[GPU 作业与遥测](knowledge/gpu-jobs-and-task-telemetry.md)：理解实际接口、状态与部署条件。
3. 按需查[命令阶段配额](knowledge/command-gpu-reservations.md)、[显存准入](knowledge/gpu-memory-admission.md)或[Docker 借用 rollout GPU](knowledge/rollout-gpu-lending.md)。
4. 到[实验索引](experiments/README.md)选择对应证据，区分 CPU 教学模拟、真实 GPU 组件实验与完整训练。

## 目录与归档规则

| 目录 | 用途 |
|---|---|
| [knowledge/](knowledge/README.md) | 知识、实现与接入说明。早期版本文档在索引中单列，不作为当前能力总览。 |
| [experiments/](experiments/README.md) | 实验报告、对照分析与现场障碍记录。 |
| [env-service-verification/](env-service-verification/README.md) | 共用复现脚本、教学脚本、原始结果及早期验证报告。 |
| [gpu-job-validation/](gpu-job-validation/) | GPU 作业验证配套文件。 |
| [airs-bench-audit/](airs-bench-audit/) | AIRS-Bench 审查、修复报告与对应脚本、证据。报告也在实验索引登记。 |

一篇文章兼有原理与实测时，按主要阅读目的归类，在分类索引注明附带内容。新增长篇实验报告放到 `experiments/`，从相关知识文章链接过去。复现脚本和原始结果使用上述配套目录，阅读命令时以文中注明的工作目录为准。

[返回知识文档总索引](../README.md)
