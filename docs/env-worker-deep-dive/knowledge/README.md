# EnvWorker 知识与实现

> **核心结论**：先读当前流程总览，再按问题查具体机制。早期实现讲解单列在后半部分，其中的「当前」「已实现」以文内版本为准。各篇保留的实验章节是配套证据，集中查结果请转到[实验索引](../experiments/README.md)。

## 目录

- [流程与实现专题](#流程与实现专题)
- [基础概念与接入](#基础概念与接入)
- [早期实现与版本讲解](#早期实现与版本讲解)

## 流程与实现专题

| 文档 | 阅读用途与边界 |
|---|---|
| [EnvRouter 当前流程与优化总览](env-router-current-flow.md) | 首读入口。串联任务放置、命令借还、预热和清理，区分现有流程与通用接入提案。 |
| [Worker 重试、整组补采与重派发](worker-retry-and-redispatch.md) | 区分已有恢复和同请求跨 worker 转移，解释执行监督、熔断修复与真实进程死亡实验。 |
| [EnvWorker 与 Harbor 共用容器后端](native-harbor-backend.md) | 容器生命周期复用、两种 shell 语义及真实 Harbor oracle 验证，明确 CPU 原型与 science GPU 路径的边界。 |
| [GPU 统一执行后端](unified-gpu-execution.md) | episode 与 command 的提交、等待、取消和收尾，附组件验证及原生 science 验收边界。 |
| [GPU 作业入口与任务遥测](gpu-jobs-and-task-telemetry.md) | 作业状态、资源归属和部署条件。同容器并发部分为方案与 CPU 模型，未作为生产能力验收。 |
| [不用 procfs 标识进程](process-identity-without-procfs.md) | procfs 与 eBPF 的通俗比较、可选统一归属后端及双容器实测，也解释 SSH 容器层次和 pidfd 生命周期。 |
| [命令阶段配额](command-gpu-reservations.md) | 保留容器状态、归还 GPU 预算、后台归还与规划重叠，附实测和跨节点调度提案。 |
| [GPU 显存准入与未兑现预约](gpu-memory-admission.md) | 实测占用与逐任务预约余量如何共同参与准入，附故障修复和验证边界。 |
| [Docker 借用 rollout GPU](rollout-gpu-lending.md) | vLLM 睡眠、Docker 借卡与确认归还，区分当前实现、早期设计和配套实验。 |
| [vLLM 与环境同卡共享策略](vllm-env-opportunistic-sharing.md) | 独立 GPU 优先、常驻共享与已睡眠窗口，解释 KV 余量、MPS/SM 与时间片调度，区分功能共存和性能收益。 |
| [CUDA 上下文开销与计算竞争](cuda-context-and-compute-contention.md) | 零基础展开 CUDA 上下文、SM、stream、MPS 四个概念；**§4.1-4.3** 把"同上下文并发 vs 跨上下文时间片"拆成层1/层2讲清楚，并纠正 MPS 是"从源头上只建一个 context 共用"而非"事后合并两个已有 context"的常见误读；`accounted_gpu_memory_mb`/`_effective_gpu_mb` 怎么用实测值兜底上下文开销；**§3.1** 纠正 `model_headroom_mb * len(unit.gpu_indices)` 容易被误读成"一张卡挂了几个 GPU"的问题，讲清楚 `EnvUnit` 可跨多张卡（张量并行副本）。 |
| [CPU 与 GPU 多任务并行架构设计对比](cpu-gpu-parallelism-architecture-comparison.md) | 跳出 PSRL 具体代码，讲清楚 CPU 核心"按进程独立分配"和 GPU SM"按 context 整体轮转"这组调度粒度差异分别从哪来——CPU 面对少量异构任务、GPU 面对海量同构任务的设计取向差异，上下文切换成本为什么天生不对等；延伸到 MPS（软件代理，没改硬件调度粒度）和 MIG（硬件真分区，向 CPU 核心模型靠拢但要牺牲弹性共享）分别补的是哪一层。 |
| [GPU 准入/归属/恢复新设计：eBPF 身份识别与两个新协程模式](gpu-admission-attribution-ebpf-identity.md) | 零基础展开 eBPF/BTF/raw_tracepoint/libbpf FFI/cgroup v1 devices 控制器桥接技巧；`accounted_gpu_memory_mb` 显存会计公式、`ContainerBudget` 容器预算、`NodeHealth` 熔断恢复状态机逐一手算；`resource_telemetry.py`/`routing_pause.py` 两个新协程清理模式（有限并发+截止时间收尾、多持有者引用计数+取消安全清理）；跟着图读 `router.py`/`bpf_identity.py`/`bpf_identity.bpf.c` 等源码行号；配套可运行脚本复刻全部手算结果。 |

## 基础概念与接入

| 文档 | 阅读用途与边界 |
|---|---|
| [GPU workload 与 benchmark 接入调研](gpu-workload-benchmark-survey.md) | GPU kernel、容器训练、自动研究与 AI4Infra 候选比较，区分已有 recipe、接入提案和真实 GPU 验证。 |
| [EnvWorker 接入与 GPU 验证方法](onboarding-and-gpu-validation.md) | EnvWorker 路径的配置、环境类与 AgentLoop 接入，以及分层验证方法。不能直接替代 EnvRouter 当前接口说明。 |
| [Gate B 新手讲解](gate-b-beginner-guide.md) | 学习会话绑定、遥测轮询和模型回复节流，包含教学模拟。讲解的是旧 Gate B 路径。 |
| [从 Gate B 学 HTTP](http-basics-for-gate-b.md) | 从一次请求理解端口、URL、POST、连接和响应，以旧 Gate B 接口为例。 |

## 早期实现与版本讲解

以下文档适合追溯设计和学习源码，不宜用文中的历史「现状」替代[当前流程总览](env-router-current-flow.md)。

| 文档 | 版本或主题边界 |
|---|---|
| [Env Service 实现讲解](env-service-implementation.md) | 以 2026-10-12 改动为背景解释 unit、租约、overlay 和节点策略。显存准入后续变化见[专题](gpu-memory-admission.md)。 |
| [EnvWorker 账本与回收缺口](implementation-and-gaps.md) | 早期实现分析及 AIRS-Bench 修复前审查。修复结果见[实验报告](../airs-bench-audit/fixes-and-results.md)。 |
| [提交 72ca50c 的 Gate A/B 讲解](commit-72ca50c-gate-ab-walkthrough.md) | 限定到该提交的实现，不代表后来 EnvRouter 的完整能力。 |
| [SciAccel GPU Gate 接入与部署](sciaccel-gpu-gate.md) | 早期整卡锁、回复节流、资源池关系与部署验证，也包含文件锁基础。 |

[返回专题入口](../README.md) · [查看实验与验证](../experiments/README.md)
