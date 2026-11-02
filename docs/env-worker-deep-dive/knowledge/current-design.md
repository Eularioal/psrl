# Env 环境执行：当前设计与阅读入口

> **核心结论**：了解现状先读本文。当前受管理路径由 EnvRouter 决定资源准入，Harbor 组织任务，原生 EnvWorker 或 Compose 创建容器，两者共用 GPU 作业执行接口。`episode` / `command` 决定预算持有多久，`direct` / `mps` 决定作业如何执行。这是两组不同的选择，不能混为一谈。
>
> **核对范围**：2026-11-02 的主开发目录，HEAD 为 `8d1fc65`，包含当时尚未提交的显存记账与遥测修改。本文描述工作区源码，不代表远端已合并版本或集群已部署状态。默认配置仍关闭 Env Service。本次只核对源码和运行 CPU 教学例子，没有重新执行 GPU 或训练实验。

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
| 2 | `EnvUnitLedger.try_admit` 返回获准、等待或拒绝，重复会话请求复用租约 | [ledger.py](../../../psrl/workers/env_service/ledger.py#L112) |
| 2 → 3、4 | runner 将租约身份、持有模式、执行后端注入环境 | [runner.py](../../../examples/sciaccel_rl/runner.py#L412) |
| 3 | 两种容器提供者都继承公共执行层 | [EnvWorkerEnvironment](../../../psrl/workers/env_worker/harbor.py#L19)、[ManagedDockerEnvironment](../../../psrl/workers/env_service/docker.py#L25) |
| 4 | `_init_gpu_execution` 建立执行器与作业管理器，`start_gpu_job` 提交作业 | [managed_execution.py](../../../psrl/workers/env_service/managed_execution.py#L17)、[作业入口](../../../psrl/workers/env_service/managed_execution.py#L130) |
| 4 direct | `GpuExecutionBackend.execute` 串联排队、执行、返回和异常清理 | [gpu_execution.py](../../../psrl/workers/env_service/gpu_execution.py#L102) |
| 4 mps | `_execute_mps_job` 经 Router 把作业送到租约所属节点 | [managed_execution.py](../../../psrl/workers/env_service/managed_execution.py#L187)、[router.py](../../../psrl/workers/env_service/router.py#L259) |
| 4 → 5 | direct 路径先设置输出可读，再完成持有策略收尾 | [gpu_execution.py](../../../psrl/workers/env_service/gpu_execution.py#L144) |
| 6 | 原生后端 `stop` 清理作业和容器，runner 的 `finally` 完成外层资源释放 | [harbor.py](../../../psrl/workers/env_worker/harbor.py#L182)、[runner.py](../../../examples/sciaccel_rl/runner.py#L605) |

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

容器占用的 CPU、主存和容器名额，与 GPU 活动预算有不同生命周期。command 命令结束后，文件和 CPU 状态可以继续保留，GPU 预算则必须在节点确认任务不再占用后才可复用。若后台服务还持有 GPU 状态，只是发送请求的客户端结束，不能据此宣布 GPU 已归还。

显存准入还要同时考虑实际使用和尚未兑现的承诺。当前 [accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9) 的计算为：

```text
计入占用 = 设备实测占用 + 每个任务 max(0, 预约量 - 可信归属于该任务的实测量) 之和
```

以下是教学假设，不是实测：三个任务各预约 20 GiB，当前各实际使用 14 GiB，总实测 42 GiB。若能够可信地确认归属，计入量为 `42 + 3 × (20 - 14) = 60 GiB`。三个任务后续尚可增长的 18 GiB 仍被保护。若没有可信归属，不能猜测哪些预约已经兑现，保守计入量为 `42 + 3 × 20 = 102 GiB`。因此，“还有空闲显存”与“账本允许再进一个任务”不是同一问题。

GiB 是显存容量单位，1 GiB 等于 1024 MiB。unit 配额是调度容量，并非给容器划出的显存硬分区。当前工作区 episode 归属与记账细节见[显存准入](gpu-memory-admission.md)。

## 4. 已实现与尚未完成的边界

| 问题 | 当前结论 | 深入入口 |
|---|---|---|
| 两种容器创建方式是否统一 GPU 作业接口 | 已共用提交、等待、取消和执行实现，创建层仍不同 | [统一执行层第 10 节](unified-gpu-execution.md#10-当前结构与前两阶段的对照) |
| 同容器能否同时运行多条受管 GPU 作业 | direct 串行，显式 MPS 后端支持并行 | [MPS 后端](mps-parallel-execution.md) |
| 命令归还可否与模型规划重叠 | direct command 支持异步归还，下一条 GPU 命令仍须通过准入 | [命令配额第 8 节](command-gpu-reservations.md#8-异步归还与模型规划并行) |
| 下一条命令能否自动换卡或跨节点迁移 | 租约仍绑定 unit，两层租约和动态放置是提案 | [动态放置提案](dynamic-command-placement.md) |
| 等待模型时能否自动卸载任意 GPU 状态 | 未完成通用自动卸载与恢复，常驻模型需显式管理生命周期 | [接入 recipe](gpu-benchmark-recipes.md) |
| 是否能使用模型 GPU | 有独立 GPU 优先与显式共享路径，需遵守共享配置和模型余量边界 | [vLLM 共享第 10 节](vllm-env-opportunistic-sharing.md#10-常驻共享kv-上限与已睡眠窗口) |
| 失败是否等于没有恢复能力 | 已有重试、补采与监督，不等于原请求能自动迁移到另一 worker | [重试与重派发](worker-retry-and-redispatch.md) |
| 功能通过是否代表训练更快 | 不能，性能结论需匹配版本、工作量和计时边界 | [实验索引](../experiments/README.md) |

## 5. 跟着图读源码

按下列顺序可以从入口读到清理，函数链接与图例对应。

1. **1 → 2**：读 `run_harbor_episode` 的资源申请，输入任务需求，输出租约或准入失败。再读 `EnvUnitLedger.try_admit`，看 unit 和预算怎样影响获准结果。
2. **2 → 3**：读 runner 的 `gpu_execution` 绑定及两种提供者。租约的模式和身份传入环境，容器创建后才能提供实际容器标识。
3. **3 → 4**：读 `ManagedGpuExecution._init_gpu_execution`、`start_gpu_job`，每次提交生成独立作业记录，调用方可以继续交互或等待。
4. **4 的分支**：direct 读 `GpuExecutionBackend.execute` 与 `EpisodeGpuReservation` / `CommandGpuReservation`。MPS 读 `_execute_mps_job`、`EnvRouter.submit_mps_job`，再到 [MpsDomain.admit](../../../psrl/workers/env_service/parallel_gpu.py#L219)。MPS 是远程调用与轮询状态链，不经过 direct 的串行执行锁。
5. **4 → 5 的并发点**：direct 输出就绪与物理归还不是一个瞬间。后台归还可以与调用方规划重叠，下一次执行仍等待旧归还完成。MPS 每个作业独立跟踪和回收，详情见 [MpsDomain.cleanup](../../../psrl/workers/env_service/parallel_gpu.py#L353)。
6. **6**：读原生 `stop` 及 runner 的 `finally`，会话结束要处理全部所属作业、容器和租约。单条命令完成不能代替这一步，清理失败也不能伪装成物理资源已经空闲。

## 6. 可运行例子与验证边界

复用已有的 [explain_unit_capacity.py](../env-service-verification/scripts/explain_unit_capacity.py)，从仓库根目录运行：

```bash
PYTHONPATH=. python docs/env-worker-deep-dive/env-service-verification/scripts/explain_unit_capacity.py
```

这是图中 **2 的准入与 6 的账本释放**的 CPU 教学例子，调用真实账本与显存计算函数，以模拟遥测代替 GPU，不创建容器。它没有模拟 Harbor、远程调用、CUDA 执行或物理回收协议，脚本中的释放前提由教学代码直接提供。

预期观察：24 GiB 配额只接纳第一项任务，96 GiB 配额在没有可信归属时接纳前三项，提供可信归属后四项均获准。三项任务的保守占用为 102 GiB，可信归属下为 60 GiB。释放后新的申请可以获准，重复声明同一设备的两个 unit 会被拒绝。这些观察用于理解准入，不能作为本次 GPU 性能结果。

本次已实际运行上述 CPU 脚本，断言全部通过，输出与上述数字及准入顺序一致。没有运行 GPU 任务。

## 7. 接下来按问题阅读

- **只想知道现状**：读到这里即可，不要求先读各阶段长文。
- **要改执行接口**：读[统一执行后端第 10 节](unified-gpu-execution.md#10-当前结构与前两阶段的对照)，再按图查源码。
- **要查排队和显存**：读[显存准入](gpu-memory-admission.md)，再查[命令配额](command-gpu-reservations.md)或 [MPS](mps-parallel-execution.md)。
- **要接入真实任务**：读[benchmark recipe](gpu-benchmark-recipes.md)，用对应[实验报告](../experiments/gpu-benchmarks/README.md)核对输入、结果及适用条件。
- **要追溯设计理由或旧阶段**：从[分类索引](README.md)进入提案、背景知识或历史实现。旧总览里的阶段性限制不能覆盖本文明确区分的当前分支。

维护约定：已实现能力变化时先更新本文的结构和边界，再更新专题细节。实验数据留在对应报告，提案保留明确标签，避免在本文连续追加互相覆盖的进度通知。
