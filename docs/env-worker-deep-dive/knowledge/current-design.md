# Env 环境执行：当前设计与阅读入口

> **核心结论**：了解现状先读本文。当前受管理路径由 EnvRouter 决定资源准入，Harbor 组织任务，原生 EnvWorker 或 Compose 创建容器，两者共用 GPU 作业执行接口。`episode` / `command` 决定预算持有多久，`direct` / `mps` 决定作业如何执行。这是两组不同的选择，不能混为一谈。
>
> **核对范围**：2026-11-05 对照主目录 `0326d46` 复核第 1、2 节的 runner 接线、提供者、执行后端和默认开关，并修正收尾源码定位。MPS 控制通道修复已包含在该提交中。开始前已有文档与实验未提交改动，第 4 节的局部验证仍按各自日期和范围阅读，未重新运行其中实验。本文不代表远端已合并版本或集群已部署状态。默认配置仍关闭 Env Service。[MPS 真实训练性能验证](../experiments/mps-training-comparison.md)单独记录实验条件和结果，不能从架构图推断吞吐收益。

## 目录

- [1. 现在的完整结构](#1-现在的完整结构)
- [2. 四个容易混淆的选择](#2-四个容易混淆的选择)
- [3. 资源如何计算与归还](#3-资源如何计算与归还)
- [4. 已实现与尚未完成的边界](#4-已实现与尚未完成的边界)
- [5. 跟着图读源码](#5-跟着图读源码)
- [6. 可运行例子与验证边界](#6-可运行例子与验证边界)
- [7. 接下来按问题阅读](#7-接下来按问题阅读)

## 1. 现在的完整结构

下面是已启用 Env Service 时的源码结构图。箭头表示调用或数据传递，不表示所有环节都在同一个进程里顺序执行。

```text
[1] Science runner：读取任务资源需求
       │ acquire：申请一次环境会话
       ▼
[2] EnvRouter / Ledger：选择 unit，检查预算，发放 lease
       │ unit = 节点上的资源单元，lease = 获准资源及身份凭据
       ▼
[3] Harbor 编排任务 → 容器提供者创建环境
       ├─ native EnvWorker：单容器生命周期
       └─ Harbor Compose：Compose 创建方式
       │ 两者都提供容器身份和命令执行通道
       ▼
[4] ManagedGpuExecution / GpuJobManager：提交、状态、等待、取消
       ├─ direct → GpuExecutionBackend → 同环境串行
       │             ├─ episode：使用会话持有的 GPU 预算
       │             └─ command：逐命令申请，确认归还后释放
       └─ mps → Router → 节点 MpsDomain：按作业准入与并发
                     │ 实际执行、遥测、定向取消及物理清理
                     ▼
[5] 作业结果返回 Harbor，继续下一轮任务或评分
       │ command 释放 GPU 预算不等于销毁容器
       ▼
[6] 环境结束：汇合作业清理 → 销毁所属容器 → 释放会话租约
```

图例：图里每个部分对应的源码位置。

| 节点或箭头 | 职责与实际函数 | 源码 |
|---|---|---|
| 1 → 2 | `run_harbor_episode` 在创建环境前申请租约 | [runner.py](../../../examples/sciaccel_rl/runner.py#L369) |
| 2 | `EnvUnitLedger.try_admit` 返回获准、等待或拒绝，重复会话请求复用租约 | [ledger.py](../../../psrl/workers/env_service/ledger.py#L113) |
| 2 → 3、4 | runner 将租约身份、持有模式、执行后端注入环境 | [runner.py](../../../examples/sciaccel_rl/runner.py#L412) |
| 3 | 两种容器提供者都继承公共执行层 | [EnvWorkerEnvironment](../../../psrl/workers/env_worker/harbor.py#L19)、[ManagedDockerEnvironment](../../../psrl/workers/env_service/docker.py#L25) |
| 4 | `_init_gpu_execution` 建立执行器与作业管理器，`start_gpu_job` 提交作业 | [managed_execution.py](../../../psrl/workers/env_service/managed_execution.py#L17)、[作业入口](../../../psrl/workers/env_service/managed_execution.py#L137) |
| 4 direct | `GpuExecutionBackend.execute` 串联排队、执行、返回和异常清理 | [gpu_execution.py](../../../psrl/workers/env_service/gpu_execution.py#L109) |
| 4 mps | `_execute_mps_job` 经 Router 把作业送到租约所属节点 | [managed_execution.py](../../../psrl/workers/env_service/managed_execution.py#L194)、[router.py](../../../psrl/workers/env_service/router.py#L259) |
| 4 → 5 | direct 路径先设置输出可读，再完成持有策略收尾 | [gpu_execution.py](../../../psrl/workers/env_service/gpu_execution.py#L151) |
| 6 | 原生后端 `stop` 清理作业和容器，runner 的 `finally` 完成外层资源释放 | [harbor.py](../../../psrl/workers/env_worker/harbor.py#L182)、[runner.py](../../../examples/sciaccel_rl/runner.py#L610) |

Harbor 是任务编排框架，负责执行任务、调用评分器并收集产物。EnvWorker 是容器执行实现。EnvRouter 是资源调度入口，Ledger 是它维护的资源账本。统一 GPU 执行层让两个容器提供者使用同一套作业协议，没有把它们的构建、网络和全部生命周期实现合并成一个提供者。

这些代码如果有理解上的问题，可以随时追问。

## 2. 四个容易混淆的选择

| 选择 | 回答的问题 | 当前含义 |
|---|---|---|
| 是否启用 Env Service | 这次任务是否进入 Router 管理 | [默认 `enable: False`](../../../psrl/trainer/config/psrl/env_service.yaml#L6)，实现存在不等于所有任务自动启用 |
| native / Compose | 谁创建容器 | 原生路径复用 EnvWorker，Compose 路径保留 Harbor 创建方式，两者共用 GPU 作业执行层 |
| episode / command | GPU 预算持有多久 | 前者贯穿环境会话，后者在每条 GPU 命令获得准入后持有，确认物理归还再释放 |
| direct / mps | 作业怎样执行 | direct 在同环境串行，mps 支持受管的同容器和跨容器作业并发 |

MPS 是 NVIDIA 的多进程服务。这里的 MPS 后端还有逐作业预算、身份和清理管理，不能把安装 MPS 服务等同于完成调度接入。当前后端面向环境专用单卡、cgroup v1 和同一可信实验域，不能直接推导为 vLLM 同卡 MPS 共享已完成。具体条件见 [MPS 专题](mps-parallel-execution.md)。

`episode` 不等于固定 OFF baseline。它仍然可以经过 EnvRouter 准入，只是持有期更长。固定历史 baseline 是提交 `0eeead861ffafed6bf31c962d259fd847775dcf9`，也不能用当前代码关掉开关替代。

## 3. 资源如何计算与归还

容器占用的 CPU、主存和容器名额，与 GPU 活动预算有不同生命周期。command 命令结束后，文件和 CPU 状态可以继续保留，GPU 预算则必须在节点确认任务不再占用后才可复用。启用同机命令池时，CPU/主存按会话的主机 unit 计费，GPU 按本次命令实际选中的 unit 计费。两个位置由同一个 Router 的 Ledger 维护，均在同一节点，不会把主机资源搬到目标 GPU unit。若后台服务还持有 GPU 状态，只是发送请求的客户端结束，不能据此宣布 GPU 已归还。

显存准入还要同时考虑实际使用和尚未兑现的承诺。当前 [accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9) 的计算为：

```text
计入占用 = 设备实测占用 + 每个任务 max(0, 预约量 - 可信归属于该任务的实测量) 之和
```

以下是教学假设，不是实测：三个任务各预约 20 GiB，当前各实际使用 14 GiB，总实测 42 GiB。若能够可信地确认归属，计入量为 `42 + 3 × (20 - 14) = 60 GiB`。三个任务后续尚可增长的 18 GiB 仍被保护。若没有可信归属，不能猜测哪些预约已经兑现，保守计入量为 `42 + 3 × 20 = 102 GiB`。因此，“还有空闲显存”与“账本允许再进一个任务”不是同一问题。

GiB 是显存容量单位，1 GiB 等于 1024 MiB。unit 配额是调度容量，并非给容器划出的显存硬分区。当前工作区 episode 归属与记账细节见[显存准入](gpu-memory-admission.md)。

## 4. 已实现与尚未完成的边界

2026-11-11 对 `ab74120` 加未提交改动的扩题能力局部核对：KernelBench 270 道源题已准备，快速 GPU 检查原尺寸通过 22 题、本地小输入档通过 193 题；新增 `kernel_bench_local` 独立评分身份与验证标记，不修改上游题库或原尺寸任务协议。FlashInfer 公共 recipe 已解除单一 RMSNorm 限制，30 个定义各一条原始随机 workload 经官方 evaluator 通过；MLE 本地数据扩为 digits、Iris、Wine、乳腺癌四题，CUDA 训练与隐藏评分通过。249 个通过项已写入统一 PSRL 任务池。当前策略更新闭环仍为六类各一个历史代表题，不能将本轮快速验证写成新增题目的策略训练完成。其他 bench 的官方题数、本地题目身份、数据缺口与路径见[逐题覆盖统计](../milestones/science-gpu-bench-status-and-roadmap.md#题库数量位置与逐题覆盖)。其余架构核对范围不变。

2026-11-10 局部更新：本地主目录已增加 Router 启动契约核对，以及节点 `node_diagnostics` / `recover_node` 接口。恢复不清空租约、不解除 GPU 隔离，真实 Ray/GPU 故障部署验收未执行。见[初学者讲解与修复边界](router-configuration-and-recovery.md)。其余章节仍按各自核对范围阅读。

2026-11-10 本地 GPU 任务接入新增 MLE 真实数字分类、InferenceBench 固定输出服务协议、RE-Bench 有界本地裁判模式，以及 FlashInfer 公共 Harbor 准备/验收入口。代码接线与已经通过的本地策略闭环验收记录在[本地完整接入报告](../milestones/science-gpu-full-onboarding.md)，不能继续将 FlashInfer 一概描述为只有探针，完整通过的结论只适用于报告列出的本地代表题，不外推到官方全题库。其余架构核对日期保持各节原范围。

2026-11-10 对 `bc5665f` 加本轮未提交修改的 benchmark 接线局部核对：公共训练 recipe 包含 KernelBench、MLE-bench、InferenceBench、autoresearch、RE-Bench、FlashInfer。MLE、Inference 与 RE 可显式选择本地协议，独立 GPU verifier 均已有代表任务证据，六类代表题均已完成真实策略采样、一次优化器更新、版本 1 参数回传及后续独立 GPU 评分，资源已经回收。AIRS 使用独立接线，其 GPU 子集未包含在本轮保证范围。当前任务边界与后续工作见[路线图](../milestones/science-gpu-bench-status-and-roadmap.md)，历史实测条件保留在完整接入报告与原始证据中，不据此刷新本篇其余架构核验日期。

同日针对 `ab74120` 加未提交改动补充局部边界修复：RE 缺失提交明确零分，MLE 拒绝格式损坏与超额 CSV，混合准备器拒绝显式来源身份冲突，Inference verifier 清理所属服务进程组。30 项 CPU 回归通过；未重新启动 GPU 训练，旧任务资产与镜像需重新准备才携带修复。具体行为及保留的 agent 生命周期限制见[边界修复与回归](../milestones/science-gpu-full-onboarding.md#本轮边界修复与回归)。

RE-Bench 本地裁判于 2026-11-05 对照 `cd09114` 加本次未提交修改局部核对，实机运行发生于 11-04。command 的组级测量预算与反馈保真调度仍是[研究提案](../research-proposals/command-scheduling-research.md)，不属于已实现能力。其文献与接口边界于 2026-11-05 局部核对，目标为固定 rollout 组全部有效反馈的完成时间，不另设公平性优化目标。现有 KernelBench 默认奖励仍为正确性，组反馈精度候选仅适合先做成本与误差验证，未确认独立研究收益。跨指标的评估契约与误差模型也仅为研究设计，未提供通用可靠性保证。

KernelBench command 于 2026-11-04 对照 `cd09114` 加未提交代码局部核对，见[接入与性能测量](gpu-benchmark-recipes.md#kernelbench-command-与性能测量)。其余章节保留原核对范围。

FlashInfer-Bench 于 2026-11-05 在 `0326d466` 加未提交修改的固定快照上完成三个官方 workload 的原生 EnvWorker command 探针，正确性、命令归还及物理清理通过，见[验证边界](../literature-reviews/ai4infra-performance-benchmarks.md#results-与验证边界)。这是已有执行接口的任务验证，没有新增生产调度策略，也没有扩大公共 benchmark runner 的 command 白名单或完成 agent 训练接入。这一句仅描述 11-05 固定探针；11-10 的公共 episode recipe 已完成前述本地策略闭环，不能用旧探针边界否定该结果。

本节动态放置与 benchmark 接入条件于 2026-11-03 对照 `96acfcc` 主目录源码局部复核，包含工作区文档修改。其余章节保留原核对范围，不表示全文重新验收或集群已部署。具体任务排序见[必要性评估](../research-proposals/dynamic-command-placement.md#9-已接入-benchmark-的必要性评估)。

command 放置边界于 2026-11-05 对照 `0326d466` 加本次未提交实现复核：显式同机 `command_gpu_pool` 已支持下一条 direct 命令重新选卡，主机归属与 GPU 分配分别记账。真实 EnvWorker 容器与 EnvRouterCore 验证了 GPU0 被占用时转到 GPU1，见[实现与实测](../research-proposals/dynamic-command-placement.md#11-已实现的同机命令池与资源账本)。2026-11-06 已补真实 Ray Router/节点 actor、固定工作量及已有 CPU prepare 对照，六组 288 次正确性测试执行通过，见[实测边界](../research-proposals/dynamic-command-placement.md#14-results真实-ray-与相同工作量对照)。仍未验证完整 Harbor episode runner、跨节点或训练收益，也未部署共享服务。版本有效准备与联合调度仍是研究提案。 2026-11-09 的[准备窗口增量试验](../research-proposals/non-timing-gpu-scheduling-scenarios.md#results准备并发窗口的增量收益验证)未提供足够收益证据，尚未新增生产控制策略。

| 问题 | 当前结论 | 深入入口 |
|---|---|---|
| 两种容器创建方式是否统一 GPU 作业接口 | 已共用提交、等待、取消和执行实现，创建层仍不同 | [统一执行层第 10 节](unified-gpu-execution.md#10-当前结构与前两阶段的对照) |
| 同容器能否同时运行多条受管 GPU 作业 | direct 串行，显式 MPS 后端支持并行 | [MPS 后端](mps-parallel-execution.md) |
| 命令归还可否与模型规划重叠 | direct command 支持异步归还，下一条 GPU 命令仍须通过准入 | [命令配额第 8 节](command-gpu-reservations.md#8-异步归还与模型规划并行) |
| 下一条命令能否自动换卡或跨节点迁移 | 显式同机池支持单卡 direct command 自动重新选卡。会话仍保留主机 unit；未配置池仍固定绑定。跨节点、MPS、模型共享卡与活 CUDA 状态迁移未支持 | [动态放置实现](../research-proposals/dynamic-command-placement.md#11-已实现的同机命令池与资源账本) |
| 等待模型时能否自动卸载任意 GPU 状态 | 未完成通用自动卸载与恢复，常驻模型需显式管理生命周期 | [接入 recipe](gpu-benchmark-recipes.md) |
| 是否能使用模型 GPU | 有独立 GPU 优先与显式共享路径，需遵守共享配置和模型余量边界 | [vLLM 共享第 10 节](vllm-env-opportunistic-sharing.md#10-常驻共享kv-上限与已睡眠窗口) |
| RE-Bench 训练任务是否接入原生 Env | 已有任务、独立 verifier、裁判请求桥接和 PSRL 接线。已支持独立卡本地 Qwen 裁判，真实小样本反馈验证通过，完整 512 条本地评分未验证。训练命令内部等待不会触发 command 归还，也不表示任务只能使用 episode，自动卸载仍未实现 | [RE-Bench recipe](re-bench-recipe.md) |
| 失败是否等于没有恢复能力 | 已有重试、补采与监督，不等于原请求能自动迁移到另一 worker | [重试与重派发](worker-retry-and-redispatch.md) |
| MPS 控制通道收尾错误是否已修复 | 已按服务串行管理连接，原失败训练组定向复测通过，长期稳定性未验收 | [故障定位与修复](../experiments/mps-training-comparison.md#6-控制通道故障定位与修复) |
| 功能通过是否代表训练更快 | 不能，性能结论需匹配版本、工作量和计时边界 | [实验索引](../experiments/README.md) |

## 5. 跟着图读源码

按下列顺序可以从入口读到清理，函数链接与图例对应。

1. **1 → 2**：读 `run_harbor_episode` 的资源申请，输入任务需求，输出租约或准入失败。再读 `EnvUnitLedger.try_admit`，看 unit 和预算怎样影响获准结果。
2. **2 → 3**：读 runner 的 `gpu_execution` 绑定及两种提供者。租约的模式和身份传入环境，容器创建后才能提供实际容器标识。
3. **3 → 4**：读 `ManagedGpuExecution._init_gpu_execution`、`start_gpu_job`，每次提交生成独立作业记录，调用方可以继续交互或等待。
4. **4 的分支**：direct 读 `GpuExecutionBackend.execute` 与 `EpisodeGpuReservation` / `CommandGpuReservation`。MPS 读 `_execute_mps_job`、`EnvRouter.submit_mps_job`，再到 [MpsDomain.admit](../../../psrl/workers/env_service/parallel_gpu.py#L208)。MPS 是远程调用与轮询状态链，不经过 direct 的串行执行锁。
5. **4 → 5 的并发点**：direct 输出就绪与物理归还不是一个瞬间。后台归还可以与调用方规划重叠，下一次执行仍等待旧归还完成。MPS 每个作业独立跟踪和回收，详情见 [MpsDomain.cleanup](../../../psrl/workers/env_service/parallel_gpu.py#L342)。
6. **6**：读原生 `stop` 及 runner 的 `finally`，会话结束要处理全部所属作业、容器和租约。单条命令完成不能代替这一步，清理失败也不能伪装成物理资源已经空闲。

## 6. 可运行例子与验证边界

复用已有的 [explain_unit_capacity.py](../env-service-verification/scripts/explain_unit_capacity.py)，从仓库根目录运行：

```bash
PYTHONPATH=. python docs/env-worker-deep-dive/env-service-verification/scripts/explain_unit_capacity.py
```

这是图中 **2 的准入与 6 的账本释放**的 CPU 教学例子，调用真实账本与显存计算函数，以模拟遥测代替 GPU，不创建容器。它没有模拟 Harbor、远程调用、CUDA 执行或物理回收协议，脚本中的释放前提由教学代码直接提供。

预期观察：24 GiB 配额只接纳第一项任务，96 GiB 配额在没有可信归属时接纳前三项，提供可信归属后四项均获准。三项任务的保守占用为 102 GiB，可信归属下为 60 GiB。释放后新的申请可以获准，重复声明同一设备的两个 unit 会被拒绝。这些观察用于理解准入，不能作为本次 GPU 性能结果。

2026-11-02 建立本入口时已实际运行上述 CPU 脚本，断言全部通过，输出与上述数字及准入顺序一致。2026-11-05 文档审计未重跑此实验。该教学脚本不运行 GPU 任务。

## 7. 接下来按问题阅读

- **还分不清训练、rollout 和环境申请**：读 [Router 入门第 1.1 节](router-configuration-and-recovery.md#11-启动方题目rollout-与环境申请)，从训练初始化追到 worker 协程、租约和清理，并运行四条 rollout 的 CPU 教学模拟。
- **要决定下一步修复**：读 [工程 review](engineering-review.md)，R1 至 R6 与同机命令池边界于 2026-11-10 局部复核，区分本地已实现的配置核对与人工重新探测、队列限制和后续研究。已实现不等于已部署验收。
- **要看 R1/R2 之后还剩什么**：读[剩余工程工作](engineering-review.md#9-从训练执行链理解剩余工作)，2026-11-11 对四项边界局部复核，按受控实验与长期共享部署分别排序，附 CPU 算例。
- **只想知道现状**：读到这里即可，不要求先读各阶段长文。
- **要改执行接口**：读[统一执行后端第 10 节](unified-gpu-execution.md#10-当前结构与前两阶段的对照)，再按图查源码。
- **要查排队和显存**：读[显存准入](gpu-memory-admission.md)，再查[命令配额](command-gpu-reservations.md)或 [MPS](mps-parallel-execution.md)。
- **要接入真实任务**：读[benchmark recipe](gpu-benchmark-recipes.md)，用对应[实验报告](../experiments/gpu-benchmarks/README.md)核对输入、结果及适用条件。
- **要追溯设计理由或旧阶段**：从[分类索引](README.md)进入提案、背景知识或历史实现。旧总览里的阶段性限制不能覆盖本文明确区分的当前分支。

维护约定：已实现能力变化时先更新本文的结构和边界，再更新专题细节。实验数据留在对应报告，提案保留明确标签，避免在本文连续追加互相覆盖的进度通知。


R5 的[显存、时长与后续依赖联合调度](../research-proposals/command-scheduling-research.md#显存时间与后续依赖的联合调度)于 2026-11-11 补充研究分析与教学模拟。生产策略没有加入预测器或跨节点命令迁移，不能保证多 GPU 空泡最少。


R1/R2 于 2026-11-11 补充[隔离实机验证](router-configuration-and-recovery.md#8-运行教学实验与回归)：真实 Ray 复用契约和真实 Docker 探测链路通过，节点故障为专属 actor 注入。仍不覆盖双 driver、活动资源故障清理或 MPS 服务失联，未更新共享服务部署。
