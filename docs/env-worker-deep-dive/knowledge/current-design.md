# Env 环境执行：当前设计与阅读入口

> **核心结论**：Env Service 把“运行一个任务”和“分配 GPU”分开管理。Harbor 负责把题目做完，EnvRouter 负责判断资源是否够用，EnvWorker 或 Compose 负责创建容器，公共执行层负责运行、取消和清理 GPU 命令。这样，容器里的文件可以保留，而 GPU 可以在命令之间交给其他任务。

例如，一个任务先用 GPU 训练 20 秒，再等 Agent 思考 10 秒，最后继续训练。采用逐命令借还时，20 秒训练结束并确认资源释放后，其他任务可以使用这张 GPU；原任务的文件仍留在容器里。Agent 发出下一条 GPU 命令时，再重新申请资源。这里的数字是教学假设，具体任务能否归还取决于后台进程是否仍持有 GPU。

本文先串起调用流程，再解释配置、资源记账与当前能力。需要看某个函数的细节时，可沿源码表和最后的专题入口继续阅读。版本和实验核对范围集中放在第 6 节。


## 目录

- [1. 现在的完整结构](#1-现在的完整结构)
- [2. 四个容易混淆的选择](#2-四个容易混淆的选择)
- [3. 资源如何计算与归还](#3-资源如何计算与归还)
- [4. 已实现与尚未完成的边界](#4-已实现与尚未完成的边界)
  - [任务接入：哪些可以开始使用](#任务接入哪些可以开始使用)
  - [固定工作量服务：等待热卡，还是在冷卡重建模型](#固定工作量服务等待热卡还是在冷卡重建模型)
- [5. 跟着图读源码](#5-跟着图读源码)
- [6. 可运行例子与验证边界](#6-可运行例子与验证边界)
  - [核对记录](#核对记录)
- [7. 接下来按问题阅读](#7-接下来按问题阅读)

## 1. 现在的完整结构

下面是已启用 Env Service 时的源码结构图。容器提供任务的文件和进程视图，命令进程在其中按需启动、退出。容器可以跨命令保留；后台模型进程仍存活时，命令返回后显存也可能继续被占用。箭头表示调用或数据传递，不表示所有环节都在同一个进程里顺序执行。

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

Harbor 是任务编排框架，负责执行任务、调用评分器并收集产物。EnvWorker 是容器执行实现。EnvRouter 是资源调度入口，Ledger 是它维护的资源账本。两个容器提供者保留各自的构建和网络管理，共用提交、查询、等待、取消 GPU 作业的协议。

AIRS 目前走另一条路径：`MLGymAgentLoop → EnvWorkerCoordinator → GPU sandbox → 独立 verifier`。其中 sandbox 是任务执行容器，verifier 是独立评分程序。GPU recipe（接入配置与运行脚本）申请一个由 Ray 分配的独立 GPU 槽位，CPU recipe 申请零 GPU；逐命令借还尚待接入。调用图与 19 题验证情况见 [AIRS GPU recipe](airs-gpu-recipe.md#任务怎样进入-psrl)和[接入审计](../airs-bench-audit/fixes-and-results.md)。

## 2. 四个容易混淆的选择

| 选择 | 回答的问题 | 当前含义 |
|---|---|---|
| 是否启用 Env Service | 这次任务是否进入 Router 管理 | [默认 `enable: False`](../../../psrl/trainer/config/psrl/env_service.yaml#L6)，实现存在不等于所有任务自动启用 |
| native / Compose | 谁创建容器 | 原生路径复用 EnvWorker，Compose 路径保留 Harbor 创建方式，两者共用 GPU 作业执行层 |
| episode / command | GPU 预算持有多久 | 前者贯穿环境会话，后者在每条 GPU 命令获得准入后持有，确认物理归还再释放 |
| direct / mps | 作业怎样执行 | direct 在同环境串行，mps 支持受管的同容器和跨容器作业并发 |

MPS 是 NVIDIA 的多进程服务，让多个程序共享 GPU 执行。这里在 MPS 外加了逐作业预算、身份和清理管理。当前接入要求环境专用单卡、cgroup v1（Linux 的进程资源管理机制）和同一可信实验域；vLLM 与环境共卡走单独的共享协议。具体条件见 [MPS 专题](mps-parallel-execution.md)。

`episode` 指一次完整环境会话：例如 Agent 从接题、修改代码到最终评分的全过程。它可以经过 EnvRouter 准入，并在整个会话持有预算。实验所说的固定历史 baseline（对照版本）则是提交 `0eeead861ffafed6bf31c962d259fd847775dcf9`，应使用该版本运行。

## 3. 资源如何计算与归还

容器占用的 CPU、主存和容器名额，与 GPU 活动预算有不同生命周期。command 命令结束后，文件和 CPU 状态可以继续保留，GPU 预算则必须在节点确认任务不再占用后才可复用。启用同机命令池时，CPU/主存按会话的主机 unit 计费，GPU 按本次命令实际选中的 unit 计费。两个位置由同一个 Router 的 Ledger 维护，均在同一节点，不会把主机资源搬到目标 GPU unit。若后台服务还持有 GPU 状态，只是发送请求的客户端结束，不能据此宣布 GPU 已归还。

显存准入还要同时考虑实际使用和尚未兑现的承诺。当前 [accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9) 的计算为：

```text
计入占用 = 设备实测占用 + 每个任务 max(0, 预约量 - 可信归属于该任务的实测量) 之和
```

以下是教学假设，不是实测：三个任务各预约 20 GiB，当前各实际使用 14 GiB，总实测 42 GiB。若能够可信地确认归属，计入量为 `42 + 3 × (20 - 14) = 60 GiB`。三个任务后续尚可增长的 18 GiB 仍被保护。若没有可信归属，不能猜测哪些预约已经兑现，保守计入量为 `42 + 3 × 20 = 102 GiB`。因此，“还有空闲显存”与“账本允许再进一个任务”不是同一问题。

GiB 是显存容量单位，1 GiB 等于 1024 MiB。unit 配额是调度容量，并非给容器划出的显存硬分区。当前工作区 episode 归属与记账细节见[显存准入](gpu-memory-admission.md)。

## 4. 已实现与尚未完成的边界

先按使用场景看能力。表中的“已实现”说明本地代码具备相应机制；是否适合某次实验，还要满足链接里的配置和验证条件。

| 你想做什么 | 当前能做到哪里 | 深入阅读 |
|---|---|---|
| 在两种容器后端运行 GPU 命令 | native EnvWorker 与 Compose 共用作业提交、等待、取消及执行层 | [统一执行层](unified-gpu-execution.md#10-当前结构与前两阶段的对照) |
| 命令之间让出 GPU | direct command 支持借还，物理归还可与 Agent 思考重叠；常驻后台模型需要显式结束或管理 | [命令阶段配额](command-gpu-reservations.md#8-异步归还与模型规划并行) |
| 同一容器同时跑多条 GPU 作业 | direct 串行；显式 MPS 后端支持受管并发与单独取消 | [MPS 后端](mps-parallel-execution.md) |
| 下一条命令换到同机空闲卡 | 显式 `command_gpu_pool` 支持专用单卡 direct command 重新选卡，保留原容器、主机资源和文件 | [动态放置](../research-proposals/dynamic-command-placement.md#11-已实现的同机命令池与资源账本) |
| 使用训练或推理所在的 GPU | 优先选独立 GPU；vLLM 有显式共享与睡眠借卡分支。Megatron 峰值协调和按推理需求主动退卡仍待实现 | [vLLM 共享](vllm-env-opportunistic-sharing.md#10-常驻共享kv-上限与已睡眠窗口)、[退卡提案](../research-proposals/reclaim-aware-env-sharing.md) |
| 长命令中途让出计算后继续 | 图像研究适配器可在完整更新后暂停，保留显存与票据。真实对照通过正确性，但未胜过现有共享；通用自动策略未接入生产 | [常驻模型计算对照](../experiments/resident-compute-sharing.md) |
| 复用跨命令的模型和准备状态 | 文件、应用自己保留的进程可以延续；生产尚无通用的代码版本检查、模型卸载恢复与 GPU 缓存管理 | [状态生命周期](baseline-state-lifetimes.md)、[真实任务核查](../research-proposals/benchmark-state-reuse-audit.md) |
| 服务配置变更或节点故障后继续工作 | 已实现同名 Router 配置核对、节点诊断和受控重新探测；隔离 Ray/Docker 链路已有验证 | [配置与恢复](router-configuration-and-recovery.md#8-运行教学实验与回归) |
| 任务失败后继续采样 | 已有重试、补采与执行监督；原请求自动跨 worker 转移仍需单独协议 | [重试与重派发](worker-retry-and-redispatch.md) |

**同机换卡保留的是容器和文件。** 正在运行的 CUDA 进程所持有的显存、上下文和模型状态，仍留在原 GPU。跨节点执行、MPS 池改派和活 GPU 状态迁移属于后续能力。

### 任务接入：哪些可以开始使用

当前基础范围包含 Harbor 六类 476 题、AIRS 原生 sandbox 19 题，以及独立 MLGym 9 个 GPU 任务。基础库已有逐题快检和代表题策略闭环，MLGym 的 9 题均有策略闭环。任务执行与独立评分构成环境链路，策略闭环还要求优化器更新、版本发布与新版本后续评分。

扩展库按各自完成范围使用：PostTrain 的 15/24 个组合、FastKernels 的 74/458 个目标已有独立 GPU 快检；SWE 导入 41 题，原测试与本地快速协议去重后 21 题全过，其中 17 题观察到 GPU 工作。策略闭环的逐组合与代表题覆盖、隔离情况和剩余工作统一维护在[接入状态与路线图](../milestones/science-gpu-bench-status-and-roadmap.md)，避免在架构入口重复全套验收表。

RE-Bench 已有任务、独立 verifier 和本地 Qwen 裁判桥接，小样本真实反馈验证通过，完整 512 条本地评分待验证。训练命令内部等待裁判时，当前仍由应用负责 GPU 生命周期，见 [RE-Bench recipe](re-bench-recipe.md)。

### 固定工作量服务：等待热卡，还是在冷卡重建模型

已有[类型化放置控制器](warm-placement-code-guide.md)比较三种策略：等待有模型的热卡、始终在另一张卡重建、按预计完成时间选择。新增[命令可预测性模块](command-predictability.md)和受管理队列执行器，把“何时允许使用时间预测”变成显式条件。调用者登记固定工作量、模型与代码版本、输入和可重建性，服务报告进度。系统先检查分块稳定性、报告新鲜度和近期独立完成命令的预测误差，再核对加载与执行画像及排队成本。

默认 `off` 沿用普通容量放置。`observe` 额外计算影子预测并积累完成校准，实际仍走普通策略。`auto` 逐命令通过检查后，才允许预计完成时间改变设备选择。普通放置优先复用可行的热副本，没有副本时仍允许按容量首次加载。容量检查同时计入模型和本次计算的临时显存，时间未知保留为未知。

[系统对照矩阵](../experiments/gpu-benchmarks/warm-placement-matrix.md)完成 48 个核心区块和 5 个故障区块，总计 53/57。主要结果是：热卡长时间繁忙时，冷卡重建可以缩短等待；热卡很快空闲时，保留模型更合算；接近分界点时，预测策略尚无稳定优势。该 GPU 实验采用 2026-11-13 冻结的研究策略。本次新增门控、受管理队列及固定步数 GPU 服务适配器，提供默认 off 的[专用入口](command-predictability.md#92-固定步数-gpu-服务接入)，已做 CPU 协议验证，尚未实机重跑。生产 Router 的通用 shell 路径未默认启用预测。

受管理服务通过 episode 会话租约登记常驻显存，结束时依次汇合请求、停止容器、释放租约并核对物理资源。普通 `command_gpu` 逐命令归还协议仍要求 GPU 状态真实退出，不在后台留下未记账模型。当前模块覆盖愿意提供可核验进度的结构化程序，任意 shell 时长预测、活 CUDA 状态迁移和全局最优调度仍是后续问题。调用关系和取消路径见[代码指南](warm-placement-code-guide.md)，门控原理、完整数字例子和 CPU 演示见[命令可预测性](command-predictability.md)，适用任务与历史收益的区别见[逐 benchmark 判断](command-predictability.md#11-当前-benchmark-中哪些值得使用这套机制)。其他研究结果与工程缺口分别查[研究索引](../research-proposals/README.md)和[工程审查](engineering-review.md)，性能结果查[实验结论入口](../experiments/env-design-and-results-brief.md)。

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

这是图中 **2 的准入与 6 的账本释放**的 CPU 教学例子：它调用真实账本与显存计算函数，以模拟遥测代替 GPU。运行前激活项目环境。脚本直接提供“资源已释放”的教学前提，适合观察记账逻辑；真实容器和物理回收需查看对应实验。

预期观察：24 GiB 配额只接纳第一项任务，96 GiB 配额在没有可信归属时接纳前三项，提供可信归属后四项均获准。三项任务的保守占用为 102 GiB，可信归属下为 60 GiB。释放后新的申请可以获准，重复声明同一设备的两个 unit 会被拒绝。这些观察用于理解准入，不能作为本次 GPU 性能结果。

### 核对记录

- 结构与默认开关：2026-11-05 对照 `0326d46` 核对 runner、容器提供者及执行后端；本轮文档整理复读相关入口，默认仍为 `enable: False`。
- 任务接入和研究状态：分别使用 2026-11-12/13 对 `6aa328d` 加工作区未提交内容的局部核查及链接中的固定实验。RE 候选源码三个直接入口已修复旧字节码问题，10 项 CPU 回归通过，新生成任务资产才携带修复，详见[真实任务核查](../research-proposals/benchmark-state-reuse-audit.md#results真实入口的旧字节码问题与修复)。
- 容量教学脚本：2026-11-02 实际运行，断言通过，本轮未重跑该脚本，历史训练和性能结果保持各自条件。
- 合作式计算让出：2026-11-16 在 `6aa328d` 加未提交内容上完成真实 vLLM、command 租约与 Docker 图像任务的 36 个对照区块，3 项针对性 CPU 回归通过，资源已清理。只增加研究适配器和可选更新钩子，生产调用图与默认策略未改变，详见[实验报告](../experiments/resident-compute-sharing.md)。
- 命令可预测性：2026-11-16 在 `6aa328d` 加主目录未提交内容上核对固定工作量门控、匹配成本画像、受管理队列及专用 GPU 服务接线。两批 CPU 回归共 325 项通过，实际范围维护在[对应教程](command-predictability.md#10-已验证范围与后续研究)，新入口未实机运行，历史 GPU 矩阵未重跑。

以上记录描述本地主目录与各报告的证据范围。运行中服务的版本、远端合并和部署状态需在部署时另行核实。

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
