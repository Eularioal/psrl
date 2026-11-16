# Env 环境执行：机制与接入指南

> **核心结论**：初次了解项目，先读[当前设计](current-design.md)，建立“任务→容器→资源申请→执行→评分→清理”的整体认识，再按遇到的问题选择专题。需要看实验数字、研究方案或论文时，分别进入[实验](../experiments/README.md)、[研究提案](../research-proposals/README.md)和[文献调研](../literature-reviews/README.md)。


## 目录

- [先读哪一篇](#先读哪一篇)
- [当前机制详解](#当前机制详解)
- [任务接入与验证](#任务接入与验证)
- [基础知识](#基础知识)
- [设计提案与阶段讨论](#设计提案与阶段讨论)
- [历史实现](#历史实现)
- [文档维护规则](#文档维护规则)

## 先读哪一篇

| 你的问题 | 首读文档 | 能获得什么 |
|---|---|---|
| 整体是怎样运行的 | [当前设计](current-design.md) | 组件分工、完整例子、当前能力和深入入口 |
| 训练怎样产生环境任务 | [Router 配置与恢复](router-configuration-and-recovery.md#11-启动方题目rollout-与环境申请) | 题目、独立尝试、会话与命令之间的层次 |
| 想接入自己的 GPU 任务 | [接入 recipe](gpu-benchmark-recipes.md) | 资产准备、数据行、环境调用与验证步骤 |
| 任务为什么一直在等 | [显存准入](gpu-memory-admission.md) | 容量、实测使用、预约承诺和排队 |
| 当前有什么问题值得修 | [工程审查](engineering-review.md) | 安全恢复、MPS 故障、队列和长会话的优先级 |
| 需要给别人介绍任务与进展 | [任务简报](../milestones/science-gpu-bench-brief.md) | 任务做什么、已有覆盖和主要结论 |

例如，想理解“Agent 思考时，另一条任务能否用 GPU”，先看当前设计中的完整例子，再读命令阶段配额。这样可以先理解资源怎样转交，再深入设备权限和物理释放检查。

## 当前机制详解

| 专题 | 解释的问题 |
|---|---|
| [统一 GPU 执行层](unified-gpu-execution.md#10-当前结构与前两阶段的对照) | 容器创建方式与 GPU 作业执行怎样分工 |
| [原生 Harbor 后端](native-harbor-backend.md) | EnvWorker 怎样实现容器、文件、命令和生命周期 |
| [命令阶段配额](command-gpu-reservations.md) | 容器保留时，GPU 怎样申请、归还与再次申请，含相关实验、慢查询诊断和跨节点提案 |
| [GPU 作业与遥测](gpu-jobs-and-task-telemetry.md) | 提交、等待、取消与进程采样，含部署排障和早期并发设计的版本条件 |
| [显存准入](gpu-memory-admission.md) | 怎样保护已承诺但尚未分配的显存，避免重复记账 |
| [MPS 并行后端](mps-parallel-execution.md) | 环境专用 GPU 上，多作业怎样并行、独立取消和回收 |
| [GPU 身份、归属与恢复](gpu-admission-attribution-ebpf-identity.md) | 内核身份、容器预算、节点健康和取消安全怎样串联 |
| [vLLM 与环境共享](vllm-env-opportunistic-sharing.md#10-常驻共享kv-上限与已睡眠窗口) | 独立 GPU 优先、常驻余量、睡眠窗口和模型控制门禁 |
| [借用 rollout GPU](rollout-gpu-lending.md#16-当前交接能力与-procfs-报错的关系) | 推理引擎睡眠后怎样借卡、归还再唤醒 |
| [重试与重派发](worker-retry-and-redispatch.md) | 同 worker 重试、整组补采和原请求转移的区别 |
| [等待或重建代码指南](warm-placement-code-guide.md) | 研究控制器怎样比较模型亲和、复制和预测，怎样处理取消 |
| [命令可预测性](command-predictability.md) | 固定步数、完成校准与 off / observe / auto，默认关闭。含 CPU 演示、专用 GPU 服务接线与逐 benchmark 收益边界，新入口尚未实机复测 |

## 任务接入与验证

| 文档 | 阅读用途 |
|---|---|
| [公共 GPU recipe](gpu-benchmark-recipes.md) | 各类任务的共同接线和准备入口 |
| [AIRS GPU recipe](airs-gpu-recipe.md) | 独立 Coordinator/sandbox 路径与 19 题有界验证 |
| [RE-Bench recipe](re-bench-recipe.md) | 聊天训练、裁判反馈和 GPU 状态生命周期 |
| [MLGym / SWE-Serve recipe](mlgym-swe-serve-recipes.md) | 交付权重与修改推理服务两类本地协议 |
| [MLGym 权重评分与 Agent DAPO](mlgym-weight-scoring-and-agent-dapo.md) | 区分内层模型与外层 Agent，串联隐藏评分、轨迹掩码、组内优势和策略更新，附 CPU 算例 |
| [InferenceBench 评分](inferencebench-scoring.md) | 质量门槛、性能分和本地功能分怎样解释 |
| [EnvWorker 接入实操](onboarding-and-gpu-validation.md) | 自定义 Environment/AgentLoop 与最小真实 GPU 探针 |
| [固定基线状态生命周期](baseline-state-lifetimes.md) | 文件、常驻进程、源码版本和结果交付怎样分别管理 |

具体题数、完成组合和剩余验收统一查[任务状态与路线图](../milestones/science-gpu-bench-status-and-roadmap.md)，实测数据查[实验入口](../experiments/README.md)。

## 基础知识

术语在各文第一次出现时就地解释。这些文章用于进一步追问某个机制：

| 文档 | 适合继续了解什么 |
|---|---|
| [CUDA 上下文与计算竞争](cuda-context-and-compute-contention.md) | 为什么显存申请量与实测量不同，stream、上下文与 MPS 如何影响共享 |
| [CPU 与 GPU 并行架构](cpu-gpu-parallelism-architecture-comparison.md) | CPU 核心、GPU SM、MPS 和 MIG 各解决哪一层问题 |
| [不用 procfs 标识进程](process-identity-without-procfs.md) | PID 命名空间、进程出生身份、pidfd 与 eBPF |
| [从 Gate B 学 HTTP](http-basics-for-gate-b.md) | 一个请求怎样经 IP、端口、连接和路径抵达处理函数 |
| [Gate B 新手讲解](gate-b-beginner-guide.md) | 以历史回复节流机制理解会话绑定和控制钩子 |

## 设计提案与阶段讨论

[研究索引](../research-proposals/README.md)按问题和研究状态组织。已经实现的同机命令池见[动态放置](../research-proposals/dynamic-command-placement.md#11-已实现的同机命令池与资源账本)；状态复用是否符合真实任务，先看[benchmark 核查](../research-proposals/benchmark-state-reuse-audit.md)。

## 历史实现

[历史归档](../archive/README.md)保存初期实现、Gate A/B、固定提交审计和旧设计。当前能力从本文与[当前设计](current-design.md)进入，历史文章用来追溯形成原因和原始条件。

## 文档维护规则

当前结构与默认行为改变时，先更新[当前设计](current-design.md)，再同步专题。机制正文解释因果和例子，实验保存固定版本数据，提案标明尚待验证的部分。长文独立主题拆到同名子目录，原文保留导航；迁移时同步相对链接和原章节入口。

[返回专题入口](../README.md) · [返回总索引](../../README.md)
