# 环境执行与回收专题

> **核心结论**：了解目前设计，先读[当前设计](knowledge/current-design.md)，再到「知识与实现」查专题。查实测结果、对照条件和故障证据，从「实验与验证」开始。知识文章中的提案不代表已经实现，实验结论只适用于报告注明的版本与条件。

## 目录

- [按阅读目的选择](#按阅读目的选择)
- [推荐阅读顺序](#推荐阅读顺序)
- [目录与归档规则](#目录与归档规则)

## 按阅读目的选择

| 想了解什么 | 入口 | 内容边界 |
|---|---|---|
| 给第一次了解工作的读者看什么 | [Science GPU 简版](milestones/science-gpu-bench-brief.md) | 用数字分类题解释任务执行、评分和策略训练，再介绍已有成果。 |
| 任务覆盖多少、接下来做什么 | [任务现状与路线图](milestones/science-gpu-bench-status-and-roadmap.md) | 题库数量、验收层次、剩余接入和研究顺序。 |
| 接入时环境与依赖怎么安排 | [接入与验收详情](milestones/science-gpu-full-onboarding.md) | 真实流程、环境隔离实例、固定验收记录与复现条件。 |
| 查看可引用的实验收益与边界 | [Env 有效实验结论](experiments/env-design-and-results-brief.md) | 优先列出修复后的固定基线对照，旧结果集中归档。 |
| 详细了解实现进度、收益与规划 | [Env 状态与路线图](milestones/env-status-and-roadmap.md) | 2026-10-28 阶段详细版，比较固定 baseline、当时实现及未来策略，不能替代当前设计。 |
| 现在怎样工作，接口怎样使用 | [当前设计](knowledge/current-design.md) | 当前结构、模式区别、源码与能力边界，专题查[分类索引](knowledge/README.md)。 |
| 测过什么，结果如何，哪里失败 | [实验与验证](experiments/README.md) | 训练对照、组件验证、性能分析、故障与未验证项。 |
| 哪些真实任务值得研究状态复用 | [benchmark 状态核查](research-proposals/benchmark-state-reuse-audit.md) | 参考修复、状态寿命与历史轨迹，区分正确性修复、研究候选和未证实收益。 |
| 从概念出发理解为什么这样设计 | [EnvWorker 设计讲解](../psrl/psrl-env-service-envworker-design.md) | 问题背景与设计动机，阅读时区分提案和实现。 |

若还不熟悉“策略、轨迹、租约、评分器”等词，可按需查[术语速查](../overview/glossary.md)，或用[完整做题流程](../overview/reading-guide.md)串起背景。

例如，想知道「GPU 命令结束后为什么还要确认归还」，读[命令阶段配额](knowledge/command-gpu-reservations.md)。想查「固定 OFF 与 ON 的训练耗时如何比较」，读[真实训练对照](experiments/science-single-gpu-training-control.md)，同时核对基线、压力条件和计时边界。

## 推荐阅读顺序

当前实现以[当前设计](knowledge/current-design.md)为首读入口。查性能先读[有效实验结论](experiments/env-design-and-results-brief.md)。[10 月阶段报告](milestones/env-status-and-roadmap.md)只用于追溯，不作为当前能力或当前待办入口。

1. [当前设计](knowledge/current-design.md)：串联准入、容器提供者、direct / MPS 分支、预算持有与清理。
2. [统一执行后端](knowledge/unified-gpu-execution.md)和[GPU 作业与遥测](knowledge/gpu-jobs-and-task-telemetry.md)：理解实际接口、状态与部署条件。
3. 按需查[命令阶段配额](knowledge/command-gpu-reservations.md)、[显存准入](knowledge/gpu-memory-admission.md)或[Docker 借用 rollout GPU](knowledge/rollout-gpu-lending.md)。
4. 到[实验索引](experiments/README.md)选择对应证据，区分 CPU 教学模拟、真实 GPU 组件实验与完整训练。

## 目录与归档规则

| 目录 | 用途 |
|---|---|
| [knowledge/](knowledge/README.md) | 知识、实现与接入说明。早期版本文档在索引中单列，不作为当前能力总览。 |
| [experiments/](experiments/README.md) | 实验报告、对照分析与现场障碍记录。 |
| [env-service-verification/](env-service-verification/README.md) | 共用复现脚本、教学脚本、原始结果及早期验证报告。 |
| [gpu-job-validation/](gpu-job-validation) | GPU 作业验证配套文件。 |
| [AIRS 接入审计](airs-bench-audit/fixes-and-results.md) | 当前修复与剩余边界，配套目录保留 CPU 回归、GPU 证据核对、历史容器实验及脚本。 |

一篇文章兼有原理与实测时，按主要阅读目的归类，在分类索引注明附带内容。新增长篇实验报告放到 `experiments/`，从相关知识文章链接过去。复现脚本和原始结果使用上述配套目录，阅读命令时以文中注明的工作目录为准。

[返回知识文档总索引](../README.md)
