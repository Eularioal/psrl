# Env 环境执行文档导航

> **核心结论**：想了解“目前设计是什么”，直接读[当前设计](current-design.md)。本文是分类导航，专题文档用于查细节，提案与历史文章用于理解演进，不再把它们当作并列的首读入口。

## 目录

- [先读哪一篇](#先读哪一篇)
- [当前机制详解](#当前机制详解)
- [任务接入与验证](#任务接入与验证)
- [基础知识](#基础知识)
- [设计提案与阶段讨论](#设计提案与阶段讨论)
- [历史实现](#历史实现)
- [文档维护规则](#文档维护规则)

## 先读哪一篇

**默认阅读路径：[当前设计](current-design.md) → [统一执行层的当前结构](unified-gpu-execution.md#10-当前结构与前两阶段的对照) → 按问题选择下表专题。** 如果只想建立全局认识，第一篇即可。

| 你的问题 | 直接阅读 | 读完应能回答 |
|---|---|---|
| 现在整体是什么设计 | [当前设计](current-design.md) | 谁管理任务、谁准入、谁创建容器、谁执行和回收 |
| 为什么我的任务排队 | [显存准入](gpu-memory-admission.md) | unit 配额、实测占用和未兑现预约如何计算 |
| 同一个容器能否跑多个 GPU 作业 | [MPS 后端](mps-parallel-execution.md) | direct 与 MPS 的差异、并行条件和取消边界 |
| 我要接一个新任务 | [benchmark 接入 recipe](gpu-benchmark-recipes.md) | 如何准备资产、接入环境、选择持有范围并验证 |
| 现在到底测出了什么效果 | [实验索引](../experiments/README.md) | 哪些是功能、性能、训练验证，各自在什么条件下成立 |

例如，想确认“同容器并发是不是还没做”，先看[当前设计的能力边界](current-design.md#4-已实现与尚未完成的边界)，再读 MPS 专题。不要从旧总览中一句“暂不实施”判断当前全部后端。

## 当前机制详解

这些文章按机制查阅。部分正文包含阶段实验和提案，表中指出应优先读的位置与范围，当前总体结论以[首读文档](current-design.md)为入口。

| 专题 | 负责解释什么 | 阅读边界 |
|---|---|---|
| [统一 GPU 执行](unified-gpu-execution.md#10-当前结构与前两阶段的对照) | 容器提供者与公共作业执行层 | 先读第 10 节，前面阶段记录不代表所有后续能力 |
| [Harbor 与 EnvWorker 后端](native-harbor-backend.md) | 容器生命周期、命令与文件接口 | GPU 当前结构见统一执行层，正文 CPU 原型限制有版本范围 |
| [命令阶段配额](command-gpu-reservations.md) | 容器保留、逐命令借还、异步归还 | 第 9 节跨节点调度是提案 |
| [显存准入](gpu-memory-admission.md) | 实测、可信归属与未兑现预约 | 区分 unit 总容量和动态余量 |
| [MPS 并行执行](mps-parallel-execution.md) | 同容器及跨容器作业准入、并行、独立取消与控制通道串行化 | 专用环境 GPU，性能另查 [autoresearch 对照](../experiments/mps-training-comparison.md)，不等于 vLLM 同卡 MPS 已接入 |
| [GPU 作业与遥测](gpu-jobs-and-task-telemetry.md) | 作业身份、状态、归属与观测 | 含早期同容器提案，已落地后端查 MPS 专题 |
| [GPU 准入、归属与恢复](gpu-admission-attribution-ebpf-identity.md) | 内核身份、资源预算、节点健康及协程清理 | 深入机制阅读，不是全局入门 |
| [vLLM 与环境共享](vllm-env-opportunistic-sharing.md#10-常驻共享kv-上限与已睡眠窗口) | 独立 GPU 优先、显式共享与睡眠窗口 | 第 10 节查共享协议，MPS 探索须与环境专用后端区分 |
| [Docker 借用 rollout GPU](rollout-gpu-lending.md#16-当前交接能力与-procfs-报错的关系) | 睡眠借卡、交接和确认归还 | 旧互斥借卡与显式 command 共享是不同分支 |
| [重试与重派发](worker-retry-and-redispatch.md) | 重试、整组补采、监督和清理 | 已有恢复不等于原请求自动跨 worker 迁移 |

## 任务接入与验证

| 文档 | 阅读用途 |
|---|---|
| [五篇近期顶会论文](recent-top-conference-papers.md) | RollArt、DynaRL、Murakkab、Prism、Weave 与环境执行、GPU 调度课题的关系及阅读顺序 |
| [GPU benchmark 接入 recipe](gpu-benchmark-recipes.md) | 实际任务资产、准备、环境调用和 PSRL 接口，接入首选 |
| [command 调度研究](command-scheduling-research.md) | 整组完成时间、已有研究边界、反馈保真调度候选、实验与否证条件 |
| [RE-Bench 原生接入](re-bench-recipe.md) | 聊天模型训练、裁判凭据、command/episode 边界、卸载成本、本地裁判与 command 空窗权衡 |
| [GPU workload 调研](gpu-workload-benchmark-survey.md) | 比较任务候选，调研建议不等于全部已接入 |
| [InferenceBench 评分](inferencebench-scoring.md) | 官方性能奖励、质量门槛与本地功能检查分数的区别 |
| [EnvWorker 接入与 GPU 验证](onboarding-and-gpu-validation.md) | EnvWorker 环境类、AgentLoop 和分层验证，不替代当前 Router 接口 |
| [benchmark 阅读入口](../experiments/gpu-benchmarks/README.md) | 按任务查输入、目标、评分、证据与模式适用性 |

## 基础知识

| 文档 | 适合补什么知识 |
|---|---|
| [CUDA 上下文与计算竞争](cuda-context-and-compute-contention.md) | 上下文、流、计算资源与显存开销，硬件讨论须按文内条件理解 |
| [CPU 与 GPU 并行架构](cpu-gpu-parallelism-architecture-comparison.md) | 调度粒度和 MPS、MIG 的动机，不作为部署能力清单 |
| [不用 procfs 标识进程](process-identity-without-procfs.md) | 进程身份、PID 视图与 eBPF，先有归属问题再深入 |
| [从 Gate B 学 HTTP](http-basics-for-gate-b.md) | 端口、请求、连接和响应，使用的是旧 Gate B 教学场景 |
| [Gate B 新手讲解](gate-b-beginner-guide.md) | 旧版会话绑定和回复节流，不是当前资源准入总览 |

## 设计提案与阶段讨论

| 文档 | 状态与用途 |
|---|---|
| [GPU 命令动态放置](dynamic-command-placement.md) | 提案：会话和执行租约分离、同节点换卡及远程执行 |
| [EnvRouter 流程与通用接入设计讨论](env-router-current-flow.md) | 混合阶段记录：第 10 至 13 节展开通用接入、持有范围和恢复成本。文件名保留，但不再作为当前现状入口 |

其他提案保留在对应机制专题的明确章节中，例如命令配额第 9 节的跨节点调度。不要把一篇文章的“已实现”标签扩展到它包含的所有提案。

## 历史实现

| 文档 | 固定阅读范围 |
|---|---|
| [Env Service 初期实现](env-service-implementation.md) | 2026-10-12 的 unit、租约、节点策略和测试状态 |
| [EnvWorker 账本与回收缺口](implementation-and-gaps.md) | 早期实现分析和 AIRS-Bench 修复前审查，修复结果查对应报告 |
| [提交 72ca50c 的 Gate A/B](commit-72ca50c-gate-ab-walkthrough.md) | 指定提交的代码讲解，“当时没有”不等于当前没有 |
| [SciAccel GPU Gate](sciaccel-gpu-gate.md) | 早期整卡锁、回复节流与部署记录 |

## 文档维护规则

1. 当前结构、能力和默认行为变化，先更新[当前设计](current-design.md)，不要只在旧长文末尾追加进度。
2. 每个机制专题说明自己的适用路径。提案用“提案”标记，历史结果说明版本，不以日期较新自动覆盖不同配置分支。
3. 实验报告放在[实验目录](../experiments/README.md)，当前设计只链接证据及说明验证边界。
4. 新文档登记到对应分类。保持现有文件路径稳定，分类通过本索引表达，避免改名造成外部阅读链接失效。

[返回专题入口](../README.md) · [返回总索引](../../README.md)
