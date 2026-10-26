# GPU 作业入口、任务遥测与执行后端统一

> **核心结论**：GPU 作业现在有独立的提交、查询、等待、取消和遥测入口。调用方可以在作业排队、计算及归还期间执行无依赖的工作。确认物理归还之前，GPU 配额仍保留。默认 science Terminus2 已接入该工具。
>
> 遥测改用节点采集的进程数据，并按容器 cgroup 和进程出生身份过滤。普通 Docker 的整卡 NVML 视图没有被虚拟化。嵌套 PID 命名空间必须提供宿主机 procfs，否则拒绝归因。当前实验节点缺少该部署条件，正向进程归因只有 CPU 契约验证，没有实机验收通过。

> **同容器并发设计状态**：现有接口仍串行执行同容器 GPU 作业。第 11 至 17 节给出命令级预算、作业进程子组、独立设备授权和确认后归还的改造方案，并提供已运行的 CPU 契约模型。方案尚未接入生产框架，不能把模型通过写成同容器 GPU 并发已经实现。

> **适用目标纠正**：用户已澄清，原需求是同一道题的多次独立 rollout，而非单次尝试内部的多命令并发。因此上述提案保留为可选能力设计，不是验证原需求的前提。原需求应使用独立环境、跨租约的 command GPU 对照，详见[多 rollout 意图纠正](env-router-current-flow.md#9-多-rollout-意图纠正与-off-的同容器能力)。

## 目录

- [1. 完整执行链](#1-完整执行链)
- [2. 作业入口与重叠边界](#2-作业入口与重叠边界)
- [3. 任务遥测与评分](#3-任务遥测与评分)
- [4. 为什么统一 EnvWorker 和 Harbor](#4-为什么统一-envworker-和-harbor)
- [5. Results 与复现](#5-results-与复现)
- [6. 跟着图读源码](#6-跟着图读源码)
- [7. 尚未完成的目标](#7-尚未完成的目标)
- [8. 宿主机、procfs 与 proxy](#8-宿主机procfs-与-proxy)
  - [8.1 先看三层执行环境](#81-先看三层执行环境)
  - [8.2 PID 为什么会对不上](#82-pid-为什么会对不上)
  - [8.3 proxy 到底代理什么](#83-proxy-到底代理什么)
  - [8.4 正确的部署边界](#84-正确的部署边界)
  - [8.5 只读实验与 Results](#85-只读实验与-results)
  - [8.6 跟着图读源码](#86-跟着图读源码)
  - [8.7 下一阶段的最小验收](#87-下一阶段的最小验收)
- [9. 部署补丁与生产预检验收](#9-部署补丁与生产预检验收)
- [10. 开发工作树与推送边界](#10-开发工作树与推送边界)
- [11. 同容器并发的目标架构](#11-同容器并发的目标架构)
- [12. 命令级预算与容量准入](#12-命令级预算与容量准入)
- [13. 进程子组与设备授权](#13-进程子组与设备授权)
- [14. 独立取消与物理归还](#14-独立取消与物理归还)
- [15. 源码改造清单与接口](#15-源码改造清单与接口)
- [16. 可运行模型与验收边界](#16-可运行模型与验收边界)
- [17. 跟着方案读源码](#17-跟着方案读源码)

## 1. 完整执行链

```text
[1] Science runner：申请会话租约，接入工具，检查原生 CUDA 评分条件
      ↓
[2] 模型输出 psrl-gpu JSON → Terminus2 工具适配器
      ↓
[3] GpuJobManager：返回 job_id，后台拥有作业生命周期
      ├──────── 调用方继续独立 CPU 工作或模型规划
      ↓
[4] exec_gpu → Router FIFO 与配额检查 → 节点授权设备
      ↓
[5] 容器进程组运行前台 GPU 程序
      ├──────── [6] 节点 NVML + 宿主机 procfs → 仅保留本任务进程样本
      ↓
    输出就绪：调用方可使用结果，设备归还仍在进行
      ↓
[7] 撤销新访问 → 检查 GPU 描述符、映射与计算进程 → 配额释放
      ↓
[8] 汇合剩余作业 → verifier 正确性评分 → 主机活动归因门槛
      ↓
[9] 回收容器集合，释放会话租约
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责、函数及源码 |
|---|---|
| 1 | [runner 的 command 接线](../../examples/sciaccel_rl/runner.py#L390)及原生 CUDA 遥测预检 |
| 2 | [TruncatingTerminus2._execute_commands](../../examples/sciaccel_rl/agent.py#L104)，只拦截明确的作业工具请求 |
| 3 | [GpuJobManager.start](../../psrl/workers/env_service/gpu_jobs.py#L58)，后台任务拥有执行与清理 |
| 4 | [ManagedDockerEnvironment.exec_gpu](../../psrl/workers/env_service/docker.py#L78)，沿用资源账本和设备授权协议 |
| 5 | [_run_gpu_process](../../psrl/workers/env_service/docker.py#L192)及 [进程组助手](../../psrl/workers/env_service/gpu_job_process.py) |
| 6 | [TaskGpuTelemetry.sample](../../psrl/workers/env_service/task_telemetry.py#L56)，核对 PID 视图、cgroup 和进程出生身份 |
| 7 | [revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153)，物理确认后才解除 GPU 预算 |
| 8 | [apply_gpu_activity_gate](../../examples/sciaccel_rl/gpu_grading.py)，结合每个检查的宿主机报告调整训练奖励 |
| 9 | [ManagedDockerEnvironment.stop](../../psrl/workers/env_service/docker.py)，作业汇合后进入 Harbor 容器清理，runner 再归还租约 |

## 2. 作业入口与重叠边界

普通终端工具的“返回”只说明当前观察结束。例如，一个 30 秒的仿真被送进 tmux，工具观察 1 秒后返回，GPU 还可能计算 29 秒。将这次终端调用直接当作 GPU 完成边界会过早归还资源。

新入口将“提交工作”和“工作结束”分开。调用者得到一个作业编号 `job_id`，随后可以查询或等待它。

| 操作 | 行为 |
|---|---|
| `start` | 验证参数，立即返回编号。后台等待配额和执行，不阻塞调用者 |
| `status` | 返回排队、运行、归还中、完成、取消或失败状态，以及已有输出 |
| `wait` | 等待输出就绪，或设置 `released=true` 等待资源归还。观察超时不会取消作业 |
| `cancel` | 取消队列项或终止所属进程组，等待设备归还确认。不能确认时按原协议隔离相关 unit |
| `telemetry` | 返回本作业的节点侧归因报告。不可测时报告缺失原因 |

Science 的模型仍使用 Harbor 的命令字段，但可以提交以下独立一行。它由代理适配器解析，不在宿主机 shell 中执行：

```text
psrl-gpu {"op":"start","command":"./solver","cwd":"/workspace","tag":"aw-2d-256","timeout_sec":300}
psrl-gpu {"op":"status","job_id":"替换为返回的编号"}
psrl-gpu {"op":"wait","job_id":"替换为返回的编号","timeout_sec":1,"released":false}
psrl-gpu {"op":"telemetry","job_id":"替换为返回的编号"}
psrl-gpu {"op":"cancel","job_id":"替换为返回的编号"}
```

第一条提交一个前台求解程序。`cwd` 是程序目录，`tag` 将它关联到具体评分检查。GPU 作业在独立 bash shell 中执行，共享容器文件，但不隐式继承 tmux 的目录和变量，需显式传入 `cwd`、`env`。普通 CPU 命令仍走原终端。

作业管理器在 `psrl/workers/env_service/gpu_jobs.py`，不导入 Harbor 或 Ray。它接受一个异步执行器。当前生产适配器是 ManagedDockerEnvironment，负责真实 GPU 授权、进程控制和资源归还。其他后端必须提供同样的物理完成契约，不能只把普通 shell 调用接上就宣称安全。

默认 science 代理在 command 租约下自动切到带工具的本地 Terminus2 子类。它会向模型说明工具协议。自定义代理需要主动使用这些入口，不能假定任意 shell 字符串中的 CUDA 工作都会被识别。

**可以重叠的工作有依赖约束。** 教学例子：A 的 GPU 求解需要 6 秒，同时有 6 秒不依赖其结果的规划工作。串行约为 `6 + 6` 秒，并行约为 `max(6, 6)` 秒，另外加调度和工具开销。如果下一步必须读取求解结果，就必须等待 `output_ready`，不能靠异步消除数据依赖。正在使用的输入文件也不能被并行 CPU 操作覆盖。

同一容器的 GPU 作业串行，因为设备权限以容器为边界。不同容器仍可以在账本允许时并发。作业排队不占 GPU 活动配额，但会话的 CPU、主存和容器配额继续保留。当前 GPU 申报仍来自会话需求，不是每个作业自动预测实际显存。

每个 episode 最多提交 128 个作业。执行与排队超时默认各 300 秒，单项最多 3600 秒。状态输出尾部各最多 16000 字符。取消使用进程出生时间检查防止 PID 复用误杀。后台 CUDA 子进程遗留仍会阻止归还，没有实现任意进程暂停与恢复。

## 3. 任务遥测与评分

NVML 是 NVIDIA 提供的设备管理接口。整卡利用率回答的是“这张卡是否忙”，不回答“是谁让它忙”。Docker 的 PID 命名空间隔离，也不会自动把整卡显存和利用率变成任务专属数据。

当前设计提供的是**受管理的任务资源视图**：

1. 节点从 NVML 读取进程显存及带时间戳的进程利用率样本。
2. 在宿主机 procfs 中核对进程所属的设备 cgroup。cgroup 是 Linux 归组进程、实施资源控制的机制，此处用于匹配容器及其后代。
3. 检查进程出生时间，拒绝作业开始前或 PID 复用前的旧样本。
4. 只将匹配本任务的显存和利用率写入报告，记录样本数量、峰值和缺失原因。
5. GPU 作业返回后保留报告，runner 清理租约前写入作业目录中的 `gpu-job-reports.json`。

显存是匹配进程的 NVML 报告值之和。利用率报告使用观测到的进程样本峰值，不声称多个进程的百分比可以线性相加成精确的整卡占比。没有利用率样本时保留 `null`，不替换成整卡数据或零。瞬时、短命进程可能采不到样本，不能据此断言没有运行 GPU。

**宿主机 PID 与节点 PID 必须一致或可以可靠映射。** 本次实机中，节点 cgroup 看到的进程编号与 NVML 返回的编号不同。它们分别来自容器命名空间与物理宿主机，不能按相同数字直接关联。[NVIDIA 的问题记录](https://github.com/NVIDIA/nvidia-docker/issues/1483)也说明了这种差异。

部署时选择以下方式之一：

- 将节点遥测服务运行在物理宿主机 PID 视图中。
- 给节点服务只读挂载真正的宿主机 `/proc`，将 `PSRL_GPU_HOST_PROC` 设为挂载路径。此变量须传入实际 EnvNodeService 进程，不能只设在训练 driver 中。

实现检查 procfs 中 PID 1 对应的 Linux 初始 PID 命名空间身份，避免把嵌套 Docker 的“host”误当作物理宿主机。该身份常量来自 [Linux 的命名空间定义](https://github.com/torvalds/linux/blob/master/include/linux/proc_ns.h)。当前实现依赖 Linux cgroup v1 的设备控制协议。

不能在当前容器中把它自己的 `/proc` 再挂一次就解决问题，也不应通过容器内替换 `nvidia-smi` 可执行文件来宣称实现了资源隔离。任务还可以通过 NVML 或 CUDA API 读取物理设备信息。

如果要在硬件层让任务拥有独立的计算和显存分区，应另行考虑 [MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/index.html)，即将支持的 GPU 切成硬件实例。它会引入分区配置和粒度限制，且部分进程利用率接口在 MIG 下不支持。本次没有启用 MIG，也没有改变物理 GPU 配置。[NVML 接口说明](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)列出了逐进程利用率的支持边界。

**原生 `laps-accel-cuda` 的评分适配。** 它的 verifier 做数值正确性检查，同时依据任务保存的整卡 CSV 给出 GPU 活动判断。command 路径保留数值正确性奖励，改用节点报告判断三个检查 `aw-2d-256`、`aw-2d-512`、`aw-128` 是否各自有已完成、带匹配 tag 的 GPU 作业。每个检查至少需要三个有效观测及不低于 10% 的峰值利用率。

原始 verifier 文件不会被改写，最终训练奖励与主机报告另存。缺少主机进程视图或设备不支持利用率时，在启动原生 CUDA episode 前拒绝执行。运行中遥测失败也拒绝按成功样本评分，不能将基础设施缺失伪装成任务零分。

这一适配解决邻居活动误归因，不是恶意解题策略的完整验证器。任务仍能在自己的作业里运行无关 GPU 程序，也能给作业命名。证明“指定求解算法本身使用了 GPU”还需要更严格的执行或检查协议。

## 4. 为什么统一 EnvWorker 和 Harbor

原设计的核心理由是重复维护。旧 EnvWorker 同时承担任务放置与容量管理、容器创建和持续 shell 执行。Science 则另走 Harbor，形成另一套容器生命周期和清理路径。原稿的方向是保留并通用化调度层，统一容器执行层，Harbor 仍作为执行器。

这不是 Harbor 无法管理 Docker GPU。当前 ManagedDockerEnvironment 已经证明可以在 Harbor 生命周期中接入设备映射、命令授权、进程控制和资源归还。真正的问题是资源协议应该有一个明确的拥有者，并覆盖所有调用者。

建议的目标结构如下，**这是下一阶段提案，尚未完成迁移**：

```text
Science / MLGym / AIRS-Bench 的任务适配器
                  ↓
统一的环境会话接口
open / exec_cpu / start_gpu_job / wait / cancel / close
                  ↓
执行 Worker：持有会话、工具状态、作业与异常清理
                  ↓
Harbor Docker 环境 + 持久终端
                  ↔
EnvRouter / EnvNodeService：准入、GPU 授权、遥测、归还
```

图例：目标中的执行 Worker 与统一会话接口尚无完整实现。当前可复用部分是图 1 的 [作业管理器](../../psrl/workers/env_service/gpu_jobs.py)、[Harbor 适配器](../../psrl/workers/env_service/docker.py)和 [节点服务](../../psrl/workers/env_service/node.py)。旧执行器仍为 [ContainerShell / EnvWorker](../../psrl/workers/env_worker/worker.py)，Science 仍在 [agent_loop](../../examples/sciaccel_rl/agent_loop.py) 中调用 Harbor。

实施顺序应为：先固定会话协议与等价测试，再将 MLGym 接到 Harbor 环境和持久终端，随后把 science 的 Harbor 调用移到专门执行 Worker，最后删除重复的 shell 与线程执行路径。MLGym 不必被强行改成 Harbor 内置代理循环，它可以通过同一会话接口逐轮调用；共享的是容器执行后端。

执行 Worker 与轻量控制服务应保持职责分离。不要把耗时的构建、模型调用和整条任务塞到负责 GPU 授权的关键控制路径里。统一不应重新把“整条 episode 持有 GPU”作为默认行为。

收益包括统一取消和回收、统一资源协议、统一观测，以及让调用者与容器初始放置解耦。它不自动提供活容器跨节点迁移。已经创建的 GPU 作业仍在会话所在节点运行。

## 5. Results 与复现

本节记录限定规模的小实验，不代表完整 science 训练或固定 OFF 提交对照。原始日志保存在运行输出目录，便于分享的结果摘要在 [summary.json](gpu-job-validation/summary.json)。

| 检查 | 实际观察 | 结论边界 |
|---|---|---|
| 6 秒真实 CUDA + 6 秒独立等待，串行、并行、并行、串行各两次 | 串行均值 12.89 秒，作业重叠均值 7.01 秒，总时长减少 45.64% | 独立工作用等待模拟，没有运行 vLLM，也没有训练。对照是当前同步 `exec_gpu`，不是固定 OFF baseline |
| 未获授权的普通 CUDA 调用 | 被拒绝 | 明确工具入口不能被普通终端 GPU 调用绕过 |
| CPU 任务与邻居真实 CUDA 同卡 | CPU 任务看到整卡利用率 99% | 复现了整卡遥测污染 |
| 正向进程归因 | NVML 宿主机 PID 与节点 `/proc` 不匹配 | 私有节点与一个历史节点都缺宿主机视图。已实现拒绝和预检，正向真实归因仍未验收 |
| 作业取消与再次申请 | 取消状态完成，配额归零，后续真实 CUDA 内核返回 42 | 覆盖前台进程组及物理归还，不覆盖任意脱离会话的 GPU 守护进程恢复 |
| Harbor 与旧 EnvWorker shell | 两次调用保留目录、环境变量、shell 函数 | 使用缓存 science 镜像的 CPU 实验，不等于完整 MLGym 等价验收 |
| 回归 | 61 项针对性检查通过 | 覆盖作业生命周期、归还协议、归因过滤、原生任务预检及 science 工具接线 |
| 物理收尾 | GPU 计算进程为 0，显存回到约 0.19 MiB 的启动基线 | 所属容器及 sidecar 已清理，未停止共享 Ray 或重启 Docker |

![受控作业重叠时长](gpu-job-validation/latency.svg)

时长来自首轮作业原型，当时逐进程归因未正确匹配，不能用这组数值估计完整、有效遥测部署后的开销。后续只补验故障修复和未完成项目，没有为追求更漂亮的时长重跑对照。

首轮实机还发现进程出生身份比较不能直接比较两次换算后的墙钟时间。换算会有微小漂移，稳定身份应使用内核出生 ticks，墙钟只用于筛除旧样本。对应回归已覆盖。

最初 GPU 探针镜像没有 tmux，不能完成 shell 检查。随后只补跑缓存 science 镜像的 CPU shell 对照，没有重复已完成的 GPU 延迟实验。

本次修改文件的 Ruff、格式及注释风格检查通过，新增文档的路径、源码行号和目录锚点检查通过。全仓 `pre-commit run --all-files` 在隔离副本执行，仍有其他文件的既有格式及注释问题，不能报告为全仓通过。自动修改没有写回工作区。

在仓库根目录使用项目 Python 环境运行 CPU 验证：

```bash
python -m pytest tests/env_service/test_gpu_jobs.py tests/env_service/test_task_telemetry.py tests/env_service/test_runner_env_service.py -q
python docs/env-worker-deep-dive/env-service-verification/scripts/plot_gpu_job_results.py
```

第一条验证排队与取消、输出与归还分离、拒绝邻居样本、进程身份变化及 science 接线。归因用例含教学输入，不会分配 GPU。第二条从已提交的摘要再生成图，不重跑实验。

真实 GPU 脚本需要先按集群运行指南检查设备空闲、镜像、Docker 和容器 CUDA 能力。设置 `GPU_JOB_IMAGE` 为已缓存的探针镜像，`GPU_JOB_INDEX` 为现场核验的设备序号，`GPU_JOB_OUTPUT` 为新建的独立结果目录：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/gpu_job_probe.py \
  --image "${GPU_JOB_IMAGE:?Set a cached image with Python and bash}" \
  --gpu "${GPU_JOB_INDEX:?Select a verified idle GPU}" \
  --output "${GPU_JOB_OUTPUT:?Set a new output directory}"
```

脚本最多运行 300 秒，使用两只小容器和一张 GPU。默认要求正向归因通过。只诊断缺少宿主机视图的失败路径时，显式加 `--allow-unavailable-attribution`，结果不能称为正向归因成功。`--skip-latency` 用于已有时长证据、只补验后续项目的情况。

CPU 后端契约脚本是 [harbor_shell_contract.py](env-service-verification/scripts/harbor_shell_contract.py)，用 `--image` 指定包含 tmux 的缓存 science 镜像，`--output` 指定独立目录。它比较真实 ContainerShell 和 Harbor TmuxSession，不加载模型。

## 6. 跟着图读源码

| 步骤 | 图中编号与函数 | 输入、输出及并发关系 |
|---|---|---|
| 接入任务 | 1，runner command 分支 | 租约转成环境配置，默认代理开启 `gpu_jobs`。原生 CUDA 先检查遥测条件 |
| 提交工具 | 2，`_execute_commands` | JSON 转成结构化参数，普通 CPU 命令仍交给 Harbor |
| 后台作业 | 3，`start` / `_run` | 立即返回编号，后台获得容器作业锁及 GPU 配额。这里与模型工作并行 |
| 设备授权 | 4，`exec_gpu` / `acquire` | unit 预算通过后授权设备，普通命令在授权前不能使用 GPU |
| 执行与采样 | 5、6，`_run_gpu_process` / `sample` | 前台进程组和节点采样并行，不把终端观察窗口当作程序结束 |
| 输出与归还 | 7，`revoke_gpu_command` | 输出可读之后仍可能处于 returning，归还确认才释放容量 |
| 评分与退出 | 8、9，`apply_gpu_activity_gate` / `stop` | 等待必要作业、核对检查 tag、保存报告、清理整个会话 |

具体源码路径见第 1 节图例。这些代码或并发关系有不清楚的地方，可以随时追问。

## 7. 尚未完成的目标

- [ ] **TODO：ENV-GPU-SAME-CONTAINER，同容器多个 GPU 作业独立并发。** 容量足够时允许重叠，不足时排队；每作业独立预算、进程归属、取消与物理归还，取消 A 不影响仍运行的 B。设计见第 11 至 17 节。用户于 2026-10-26 明确要求先记账，暂不实施，不作为同题多 rollout 或通用 GPU 调度接入的前置条件。验收须覆盖低压力并行、高压力等待、独立取消与最终物理清理，CPU 模型通过不算完成。

本次没有删除旧 EnvWorker，也没有把所有 Harbor episode 搬到新的执行 actor。下一步应完成 MLGym 的超时、中断、后台输出、截断与最终奖励等价测试，再迁移并删除重复实现。

正向任务遥测还需要在具备宿主机进程视图的节点运行真实 CUDA 对照，并验证三项原生 LAPS 检查的训练奖励。当前只完成接口、过滤规则、错误拒绝和评分接线，不能写成原生 CUDA 训练已经通过。

任务标签的可信性、短命 GPU 进程的采样覆盖、MIG 设备支持、cgroup v2、服务重启恢复和跨节点重派发仍需独立设计和验收。GPU 作业接口没有引入任意 CUDA 进程抢占、自动显存预测或跨节点容器迁移。

## 8. 宿主机、procfs 与 proxy

### 8.1 先看三层执行环境

本节补充第 3 节的部署前提。**Docker 可运行 CUDA，不等于节点服务已经能正确识别每个 GPU 进程所属的任务。** `procfs` 是进程信息视图，`proxy` 是请求转发代理，两者是不同机制。

```text
[A] 物理宿主机：Linux 内核、GPU 驱动、宿主机 PID 视图
  │ GPU 驱动提供数据
  │     ↓
  └─ [B] 平台分配的实验节点容器：SSH 进入的位置
       ├─ [1] EnvNodeService / NvmlProbe → 逐进程 GPU 数据
       ├─ [2] TaskGpuTelemetry → 宿主机 procfs → cgroup 与出生身份
       │                        ↑ 当前缺少这个宿主机视图
       ├─ [3] Docker Unix socket → dockerd
       │                              ↓ 创建、执行、清理
       └─ [C] Science 任务容器 → 前台 CUDA 进程
                                      ↓ 结束
          [4] 撤权、核实物理归还 → Router 解除 GPU 配额
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责 | 对应源码 |
|---|---|---|
| A、B、C | 部署环境层级，内核和平台容器不是 PSRL 创建 | [只读诊断脚本](env-service-verification/scripts/host_proc_diagnostic.py)观测 procfs 挂载及 Docker daemon 的进程视图 |
| 1 | NVML 是 NVIDIA 的设备管理接口，返回进程显存和利用率样本 | [NvmlProbe.read_process_snapshot](../../psrl/workers/env_service/probe.py#L103) |
| 2 | 按 procfs 中的 cgroup 归属和出生时间筛选样本 | [process_identities](../../psrl/workers/env_service/task_telemetry.py#L10)、[check_proc_root / sample](../../psrl/workers/env_service/task_telemetry.py#L51) |
| 3 | 通过本地 Unix socket 请求 Docker 管理接口 | [_local_get](../../psrl/workers/env_service/docker_api.py#L9) |
| C | 适配器授权后执行 GPU 程序，进程组助手保留生命周期身份 | [_run_gpu_process](../../psrl/workers/env_service/docker.py#L192)、[gpu_job_process.py](../../psrl/workers/env_service/gpu_job_process.py) |
| 4 | 撤销设备访问，核对已有句柄、映射和计算进程，随后确认归还 | [revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153) |

宿主机是相对概念。C 的 Docker 管理服务运行在 B，B 是 C 的直接宿主环境。B 自身又是 A 上的平台容器，所以不能把 B 的进程列表当成 A 的完整进程列表。这里的容器共享底层内核，并不是每层都启动了一个独立 Linux 内核。

`namespace` 称为命名空间，用于给进程提供不同的系统视图。PID 命名空间隔离进程编号，网络命名空间隔离网卡、路由和端口，挂载命名空间隔离文件系统挂载视图。共享其中一种，并不自动共享其他几种。因而 `--network=host` 不会解决 PID 问题，能访问 GPU 设备也不代表能读取宿主机的所有进程。

### 8.2 PID 为什么会对不上

PID 是进程编号。`/proc` 是 Linux 内核提供的特殊文件系统，里面的数字目录用来查看进程状态，不是程序自己保存的普通日志。`/proc/self` 表示读取者自身，`/proc/PID/stat` 包含进程出生时间，`/proc/PID/cgroup` 包含进程所属资源组。procfs 展示哪组进程编号，取决于其对应的 PID 命名空间。[Linux procfs 文档](https://docs.kernel.org/filesystems/proc.html)和 [PID 命名空间手册](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html)解释了这些机制。

以下为**教学假设**，不是实测 PID 映射：

| 观察位置 | 同一个 CUDA 进程的编号 | 可见信息 |
|---|---|---|
| A，物理宿主机 | 2000 | 宿主机 procfs 能读取 2000 的归属 |
| B，实验节点 | 300 | B 的 procfs 用 300 识别这个进程 |
| C，任务容器 | 17 | 任务自身把它看作 17 |

假设 NVML 返回“PID 2000 使用 512 MiB 显存”。正确流程是查宿主机 `/proc/2000/cgroup`，发现它属于任务 A 的容器，然后把 512 MiB 归到任务 A。

如果错误地用 B 的 procfs 查 2000，目录可能不存在，也可能对应另一个进程。把没有匹配到的记录填成 0，会把正在计算的任务说成没用 GPU。当前实现先验证 procfs 是否属于初始 PID 命名空间，条件不满足就报告缺失，避免这种错误。这个前置检查仍不等于完整归因验收，还要验证权限、cgroup 路径一致性和设备接口支持。

cgroup 可以理解为内核给进程登记的资源组。这里使用其归属路径识别任务容器，并接受这个容器的子组。代码使用路径层级匹配，不能把 `/docker/task-a-neighbor` 因为字符串前缀相同也算到 `/docker/task-a`。PID 会在进程结束后复用，所以还要核对出生 ticks，即从开机起累计的时钟单位数，防止把旧进程的样本算给新进程。

将 B 的 `/proc` 再挂到 C 的 `/host/proc`，目录名字虽然有 host，内容仍是 B 的视图。类似地，在 B 的 Docker 中设置 `--pid=host`，共享的通常是 B 所在的 PID 命名空间。它不会自动进入 A 的初始命名空间。

### 8.3 proxy 到底代理什么

Proxy 的含义是代理，即接收请求后代调用者访问目标。一个直观的网络例子是：构建程序要下载依赖，先把请求送给 HTTP 代理，代理再连接软件源并转回响应。代理能控制出口、认证或访问范围，但不改变 CUDA 进程的编号。

本项目附近出现的几种机制应分别理解：

| 名称 | 谁调用谁 | 解决的问题 | 与任务 GPU 归因的关系 |
|---|---|---|---|
| Shell HTTP 代理 | 当前 shell 启动的下载工具 → 代理 → 网站 | 当前工具访问外部网络 | 不提供宿主机进程视图 |
| Docker daemon 代理 | dockerd → 代理 → 镜像仓库 | 拉取镜像等 daemon 发出的网络请求 | 不改变 PID 或显存归属 |
| 构建和容器代理变量 | 构建步骤或容器程序 → 配置的代理 | 构建时安装依赖、任务下载文件 | 需要传给实际发请求的程序 |
| Harbor 出口控制 sidecar | 任务网络流量 → 配套容器中的网络规则和透明代理 → 目标 | 按阶段限制外部访问 | 不参与 GPU 配额和进程身份关联 |
| Docker Unix socket | 节点服务 → 本机 dockerd | 创建、检查、执行和删除容器 | 管理入口，本身不是出站 HTTP 代理 |
| 宿主机遥测代理 | 节点服务 → 宿主机采集服务 → 驱动与 procfs | 在宿主机生成可信归因结果 | 可作为独立部署方案，当前尚未实现此服务 |

Sidecar 是与主业务容器配套运行的辅助容器。当前安装的 Harbor 在启用出口控制时，会让相关服务共享这个辅助容器的网络命名空间，使用 GOST 透明代理和网络策略限制出口。共享网络不等于共享 PID。并非每个任务都必须走这个模式，要看任务的 network policy 和平台支持。

Harbor 属于运行环境依赖，源码没有复制进 PSRL。可在项目 Python 环境中定位实际安装的实现：

```bash
python -c 'import inspect; from harbor.environments.docker.docker import DockerEnvironment; print(inspect.getfile(DockerEnvironment)); print(inspect.getsourcelines(DockerEnvironment._write_egress_control_services_compose_file)[1])'
```

它输出实际文件与函数起始行。检查同目录的 `docker-compose-egress-control.yaml` 和 `harbor-docker-egress-control-sidecar/gost.yaml`，可以看到辅助容器与透明代理的接线。这是外部依赖源码定位，不代表 PSRL 实现了一套新的网络代理。

**为什么 shell 的 curl 成功，Docker 构建仍可能失败？** shell 中设置代理只影响继承这些变量的程序。已运行的 dockerd 不会因为当前 shell 改了变量而自动更新配置，构建里的下载命令也需要自己的配置。Docker 分别提供 [daemon 代理配置](https://docs.docker.com/engine/daemon/proxy/)和[构建、容器代理配置](https://docs.docker.com/engine/cli/proxy/)。

当前 `_local_get` 用 `aiohttp.UnixConnector` 连接本地 Docker socket，且设置 `trust_env=False`。URL 中的 `http://localhost` 是 HTTP 请求的地址形式，实际传输走 Unix socket，不经过环境变量指定的 HTTP 代理。此前直接访问 Docker 接口的优化省去了 Docker CLI 子进程及客户端初始化，仍请求同一执行节点的 daemon。它没有通过代理绕开物理宿主机权限。

### 8.4 正确的部署边界

**现有实现支持的路径**：平台在创建 B 时，把 A 的真正宿主机 procfs 以只读方式提供给可信节点服务，服务进程设置 `PSRL_GPU_HOST_PROC=/host/proc`。挂载必须来自 A，不能在 B 内把自己的 procfs 换个名字。若平台使用 Kubernetes，`hostPath` 的来源是运行 Pod 的节点文件系统，具体挂载应由外层部署配置提供。[Kubernetes hostPath 说明](https://kubernetes.io/docs/concepts/storage/volumes/#hostpath)

这里的路径是部署约定示例，当前节点尚未提供该挂载。只读挂载并不自动解决所有访问权限，仍要确认能读目标进程的 `stat`、`cgroup`，并核对 cgroup 命名空间没有把路径解释成另一套相对路径。环境变量必须到达实际 EnvNodeService，设置在 driver 中而未传到远程服务没有效果。

**另一条设计路线**：让一个轻量的宿主机遥测代理在 A 上采集，按容器身份、GPU UUID、作业 epoch 和采样时间返回报告。epoch 是本次授权的递增编号，用于区分旧授权和新授权。这样 B 无需直接读取全部宿主机 procfs，但需要明确身份验证、租约绑定和故障处理。当前尚未实现该服务，也没有在现场发现可直接使用的已配置接口，不能把它写成现成能力。

Docker/cgroup 控制与采集可以分离。不能把整个 EnvNodeService 随意移到 A，同时继续假定 A 的 Docker socket、进程 PID 和 cgroup 路径与 B 的内部 dockerd 相同。运行位置改变后，必须重新验证这些身份关系。

### 8.5 只读实验与 Results

运行以下脚本只需要 Linux 和 Python 标准库，不启动容器、Ray 或 GPU 程序：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/host_proc_diagnostic.py --docker
```

`--docker` 会探测本地 Unix socket，单次超时为 3 秒。去掉它可只检查进程视图。`--proc-root` 可指定平台已经提供的 procfs，`--output` 可保存 JSON。脚本只输出代理变量名称是否存在，不输出代理 URL、凭证或进程环境变量。

对应图中的 B、2、3，脚本读取当前挂载表、选中 procfs 的 PID 1 命名空间、可见 dockerd 的命名空间及其 procfs，再请求 Docker `/_ping`。同时输出第 8.2 节的教学数据：错误编号关联没有匹配，正确关联得到 512 MiB。教学输出不是 GPU 实测。

退出码 0 表示所选 procfs 满足初始 PID 命名空间和基本可读性条件，且显式请求的 Docker 探活成功。退出码 3 表示至少一个前提不满足。退出码 0 **仍不代表**真实 GPU 归因通过，脚本没有调用 NVML，也没有证明全部任务 cgroup 路径等价。

本轮在优先实验节点执行的 Results：

| 只读检查 | 结果 | 含义 |
|---|---|---|
| Docker 探活与版本 | `OK`，26.1.3，cgroup v1 | 容器服务可响应 |
| GPU 空闲快照 | 8 张卡均为 0 MiB、0% | 只是检查时的快照，不是独占保证 |
| 外层 PID 1 与 dockerd | 同一非初始 PID 命名空间 | 实验节点内运行的是内层 Docker 服务 |
| procfs 挂载表 | 只有外层节点的完整 procfs，以及 NVIDIA 驱动信息的局部挂载 | 驱动目录不能用于查宿主机进程身份 |
| 所选 procfs | 不满足初始命名空间条件 | 诊断退出码为 3，原生 CUDA 归因验收仍受部署条件阻塞 |

本轮没有启动 CUDA 工作、训练或新容器，不存在新增 GPU 占用需要回收。最初一次临时探针误用了远端系统 Python 3.6，因不支持 `subprocess.run(capture_output=...)` 提前退出。随后在远端也激活项目环境，并运行标准库诊断脚本，保存结果。此错误属于探针解释器选择，与 Docker 或 GPU 故障无关。

本轮另运行任务遥测和 science runner 的 26 项 CPU 回归，全部通过。诊断脚本的 Ruff、格式及注释风格检查通过，索引和本文的本地链接、源码行号与目录锚点已检查。全仓 `pre-commit run --all-files` 在隔离副本执行，仍因其他已有文件的格式与注释问题失败，自动修改未回写工作区。

### 8.6 跟着图读源码

1. [B、2] [runner.py](../../examples/sciaccel_rl/runner.py#L396) 在原生 CUDA 任务创建前请求遥测能力预检。
2. [1、2] [Router.require_gpu_attribution](../../psrl/workers/env_service/router.py#L218) 找到租约归属节点。[check_gpu_attribution_support](../../psrl/workers/env_service/command_gpu_node.py#L111) 先检查 procfs，再查询 NVML 支持情况。
3. [3、C] 正常授权时，[DeviceCgroupFence](../../psrl/workers/env_service/command_gpu_node.py#L17) 从 Docker 容器身份得到其 cgroup，核对设备映射并控制访问权限。此步骤使用本地 Docker 对应的 PID 视图。
4. [1、2] 授权后启动后台遥测，GPU 执行和采样并行。[sample](../../psrl/workers/env_service/task_telemetry.py#L56) 用宿主机 PID 读取身份，[consume](../../psrl/workers/env_service/task_telemetry.py#L72) 聚合本任务样本。这一步需要与 NVML 一致的宿主机视图。
5. [4] [revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153) 撤权并停止采样，核实归还后把报告与完成结果交回。它不会因为报告缺失而虚构成功的 GPU 活动评分。

这些源码或两个 PID 视图的区别有不清楚的地方，可以随时追问。

### 8.7 下一阶段的最小验收

先解决宿主机视图的部署输入，再运行已有 GPU 探针的 `--skip-latency` 模式。验收只要求：忙任务获得自己的有效样本、相邻 CPU 任务不获得忙任务样本、取消与归还后所属进程和容器消失。限定一张空闲卡、两个小容器和 300 秒总预算，不重做已有时长对照。

随后才推进原生 `laps-accel-cuda` 的三个检查。已核对它声明 1 张 GPU、使用 allowlist 网络策略，并提供独立 verifier。依次核对镜像和执行入口，验证数值结果与任务归因报告能对应。当前只是源码与配置核对，没有构建、运行或评分成功记录。完整训练、模型指标变化和固定 OFF 对照仍是另一个验收范围。

## 9. 部署补丁与生产预检验收

**当前停止点是外层平台权限。** 实验节点的 ServiceAccount，即平台给 Pod 挂载的服务身份，可以请求自己的权限检查，但对当前 Pod 的读取、创建和修改均没有授权。读取当前 Pod 返回 HTTP 403。已经证实身份权限不足，不能通过修改内层 Docker 的参数来补齐外层挂载。

部署与验收按以下顺序衔接。平台应用模板和创建新节点是待完成步骤，补丁生成器与预检是现有实现：

```text
[1] 生成只读 procfs 挂载补丁
     ↓ 平台将补丁合入外层实验节点模板
[2] 新节点运行环境：/host/proc + PSRL_GPU_HOST_PROC
     ↓ 实际 EnvNodeService 读取该路径
[3] 生产节点预检：Docker、设备控制、宿主机视图、NVML 接口
     ├─ 条件不满足 → 关闭采集连接，保存失败，不启动任务容器
     └─ 条件满足 → [4] 两容器真实 GPU 归因与取消、归还验证
```

图例及跟着图读源码：

| 步骤 | 输入和状态变化 | 源码 |
|---|---|---|
| 1 | 外层可信容器名称和模板类型，输出 JSON 补丁，不调用平台接口 | [mount_patch](../../examples/sciaccel_rl/prepare/host_proc_mount_patch.py#L8) |
| 2 | 节点进程读取配置路径，挂载动作属于平台 | [TaskGpuTelemetry.__init__](../../psrl/workers/env_service/task_telemetry.py#L33) |
| 3 | 在创建任务容器前执行真实节点服务的能力检查 | [probe_node](env-service-verification/scripts/gpu_job_probe.py#L50)、[check_gpu_attribution_support](../../psrl/workers/env_service/command_gpu_node.py#L111) |
| 3 的退出 | 成功、失败或取消都进入节点连接清理 | [probe](env-service-verification/scripts/gpu_job_probe.py#L41) |
| 4 | 验证本任务样本、邻居隔离、进程取消和配额归还 | [gpu_job_probe.py](env-service-verification/scripts/gpu_job_probe.py) |

这些源码或部署边界有不清楚的地方，可以随时追问。

在仓库根目录生成一个**以 `experiment-node` 为示例容器名**的补丁：

```bash
python examples/sciaccel_rl/prepare/host_proc_mount_patch.py \
  --container experiment-node --target pod
```

该命令可以直接运行，只打印 JSON。实际使用时把 `experiment-node` 改成外层模板中已有的可信运行容器名，不能填 science 任务容器名。输出的关键内容是：

```yaml
spec:
  volumes:
    - name: psrl-host-proc
      hostPath:
        path: /proc
        type: Directory
  containers:
    - name: experiment-node
      volumeMounts:
        - name: psrl-host-proc
          mountPath: /host/proc
          readOnly: true
      env:
        - name: PSRL_GPU_HOST_PROC
          value: /host/proc
```

这是按名称合并的部署片段，不是完整 Pod，也没有在本次现场应用。`--target controller` 会把同样的内容放到 `spec.template.spec`，适用于具有该模板结构的标准控制器。平台自定义训练资源的嵌套位置可能不同，不能直接假定适用。提交给平台前还需核对已有卷名和挂载路径没有冲突。

应修改创建外层实验节点的模板并通过平台发布流程生效。普通 Pod 更新接口不能任意给运行中的 Pod 增加卷，不能把这个文件当作保证可热更新现有节点的命令。修改控制器模板通常会创建替代 Pod，平台需要安排对当前会话的影响。[Kubernetes Pod 更新与替换](https://kubernetes.io/docs/concepts/workloads/pods/#pod-update-and-replacement)

节点具备挂载后，在实际执行节点的项目环境中先运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/gpu_job_probe.py \
  --gpu 7 --preflight-only --output /tmp/psrl-host-proc-preflight
```

这里的 7 是命令示例，应改成现场核对的 GPU 序号。预检只读取 Docker 和 GPU 信息，不分配 CUDA 张量，也不启动 Harbor 容器。它不需要 `--image`。环境不满足时非零退出，`result.json` 保留 `stage=preflight`、`workload_started=false` 和错误原因，`node_service_closed=true` 表示采集连接已经清理。预检成功也不等于任务归因验收通过，后续仍需真实 CUDA 样本。

正式 GPU 验证现在默认执行同一个预检。缺少宿主机视图时会在工作负载之前停止。`--allow-unavailable-attribution` 仅用于有意复现缺失条件的诊断，不能与 `--preflight-only` 组合，不能当作修复办法。

**Results：** 本轮真实生产节点方法在优先实验节点执行，Docker 与设备控制检查通过，宿主机 PID 视图检查拒绝。没有启动任务容器或 CUDA 工作，节点服务连接已关闭。平台权限检查的三个结果均为拒绝，实际 Pod 查询为 403。两种部署补丁结构已在 CPU 上生成并检查，非法容器名被拒绝。GPU 正向归因、原生 science 三项评分和训练均未执行。

可分享的结果见 [deployment-summary.json](gpu-job-validation/deployment-summary.json)。新增 [4 项 CPU 回归](../../tests/env_service/test_gpu_probe_preflight.py)通过，覆盖正式运行前的拒绝、只预检模式的拒绝与通过、准备阶段取消后的连接清理。这里的通过分支使用测试替身，不是宿主机视图已经部署成功。

修改文件的 Ruff、格式及注释检查通过，索引与本文链接已校验。全仓 hooks 在隔离副本仍因其他文件的已有问题失败，自动格式修改没有回写。

这次请求没有更改平台 Pod、重启 Docker 或启动 Ray 集群。后续接续需要平台把上述只读挂载和环境变量交付给新的实验节点运行环境。只授予本节点内部更多 Docker 权限，不能满足这个条件。

## 10. 开发工作树与推送边界

**日常开发约定：以 `psrl/` 为主要目录，在其中切换分支。** 不再默认创建功能工作树或反复复制补丁。现有功能工作树保留作版本参照，只有固定 baseline 并行对照、正在运行的实验需要隔离等明确情形才使用临时工作树。下面的双工作树图描述已经核对的目录关系，不是后续日常开发的推荐流程。

`psrl` 和 `psrl-gpu-jobs` 是同一个 Git 仓库的两个工作树。工作树是某个分支的文件展开目录，各自有独立的当前分支、暂存区与未提交文件，但共享提交对象和分支引用。这个关系由 `git worktree list` 和 `git rev-parse --git-common-dir` 核验。[Git worktree 官方说明](https://git-scm.com/docs/git-worktree)

```text
同一个本地 Git 仓库，公共元数据在 psrl/.git
  ├─ psrl/          → main，保留未提交的工作区改动
  └─ psrl-gpu-jobs/ → feat/env-gpu-jobs，GPU 作业与遥测功能提交
                          ↓ git push
                     Eularioal/psrl 的同名远端分支
```

图中表示 Git 目录和分支关系，不是 PSRL 运行时调用链。`main`、`feat/env-gpu-jobs` 是分支名，目录名本身不决定推送目标。远端目标由 Git remote 和 push 参数决定。

本专题的 GPU 作业、进程遥测和部署预检，在独立功能工作树开发、运行针对性验证并提交。主目录原有较多未提交改动，因此使用独立工作树维护可检查、可回退的功能提交。相关补丁也同步回主目录，方便从惯用路径查阅源码和文档，但文件同步不会自动提交主目录，也不会把 main 分支移到功能分支的提交。

例如，功能工作树中的 `gpu_jobs.py` 已经纳入提交，主工作树中的同内容文件仍可能显示 `??`，表示它尚未被该工作树的暂存区跟踪。两边共享 Git 提交记录，不代表所有未提交文件会自动同步。

**Results：** 以功能提交 `d0ad050` 为核对点，主工作树的 main 仍指向 `37a06ea`。对比功能分支相对该基点涉及的 29 个文件，27 个内容一致，`docs/README.md` 与 `psrl/workers/env_service/docker.py` 有差异。索引保留主目录的额外条目，主目录的 Docker 适配器另有 `exec_gpu_prepared`、准备中任务登记及取消清理逻辑。这些额外逻辑和主目录其他未提交文件没有因此自动进入已推送分支。本节新增说明不改变这些功能代码的差异。

可以在工作区父目录执行以下只读检查，不切换分支、不修改文件：

```bash
git -C psrl worktree list
git -C psrl branch --show-current
git -C psrl-gpu-jobs branch --show-current
git -C psrl-gpu-jobs rev-parse --git-common-dir
git -C psrl status --short
git -C psrl-gpu-jobs status --short
git -C psrl-gpu-jobs remote get-url origin
```

## 11. 同容器并发的目标架构

**本节及后续改造均为设计提案。** 目标是一个任务容器内四条小命令可以按容量并行，放不下时排队，取消其中一条只回收它的进程和预算。实现单位应统一成 GPU 作业，容器只保留 CPU、主机内存和文件系统等会话级资源。它承接[同容器实验的归因](science-natural-gpu-concurrency.md#14-同容器限制与问题归因)，不改变已有实验结论。

```text
[P1] Science GPU 工具 / 同步执行接口 / 分阶段准备接口
                         ↓ 全部提交同一种 JobSpec
[P2] 作业注册：job_id、显存预算、截止时间、所属租约
       ├─ 可信启动器创建作业子组，GPU 禁止，完成可选 CPU 准备
       ↓ ready
[P3] Router 按卡准入：新鲜遥测 + 未兑现预约 + 外部占用
       ├─ 不够 → 有界队列，CPU 规划和其他作业继续
       └─ 够用 → 原子记入 RESERVED，生成本次授权身份
                         ↓ 节点 RPC 不持有调度锁
[P4] EnvNodeService：核验容器及作业子组 → 单独授权 A / B / C / D
                         ↓ 可以同时运行，分别跟踪进程树
[P5] 完成 / 取消 / 超时 → 只关闭目标作业入口并清理它的子组
                         ↓
[P6] 核验目标子组为空、所属 CUDA 活动消失 → 独立归还该作业预算
       ├─ 其他作业仍活动 → 保留其授权，继续执行
       ├─ 归还不确定 → QUARANTINED，保留预算并停止新的准入
       └─ 最后作业归还 → 关闭容器父级 GPU 权限
[P7] episode 结束 → 汇合全部作业 → 清理容器集合 → 释放会话租约
```

图例：图里每个部分对应的源码位置。表中链接指当前实现及改造落点，不表示图示新协议已经存在。

| 编号 | 当前职责、函数与改造落点 |
|---|---|
| P1 | [dispatch_gpu_tool](../../examples/sciaccel_rl/gpu_tools.py#L26) 接受作业请求；[execute_prepared](../../psrl/workers/env_service/prepared_gpu.py#L24) 处理分阶段程序，需要统一到同一作业后端 |
| P2 | [GpuJob / GpuJobManager](../../psrl/workers/env_service/gpu_jobs.py#L11) 已有独立 ID 与状态，但缺少命令预算，_run 仍持有串行锁 |
| P3 | [GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L49) 当前按 lease 计费且禁止同租约多请求；[账本显存求和](../../psrl/workers/env_service/ledger.py#L325) 当前依赖 gpu_active 布尔值 |
| P4 | [DeviceCgroupFence](../../psrl/workers/env_service/command_gpu_node.py#L14) 当前授权整个容器；[进程组启动助手](../../psrl/workers/env_service/gpu_job_process.py#L35) 当前不是作业 cgroup 启动器 |
| P5、P6 | [revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153) 当前扫描整个容器；[TaskGpuTelemetry](../../psrl/workers/env_service/task_telemetry.py#L30) 当前按容器路径归因，需收窄到作业路径 |
| P7 | [GpuJobManager.close](../../psrl/workers/env_service/gpu_jobs.py#L127) 与 [Router.finish_gpu_command_lease](../../psrl/workers/env_service/router.py#L240) 负责汇合和租约回收，需要遍历全部作业并核验物理状态 |

这些代码如果有理解上的问题，可以随时追问。方案需要同时改调度、节点执行和接口状态，不能只把 asyncio.Lock 换成 Semaphore(4)。后者只控制固定数量，不知道每条命令的显存大小，也不能解决 A 完成时撤销 B 权限的问题。

本次源码核对还发现，显式作业路径在 GpuJobManager._run 内也有串行锁。它和 Docker 的同容器锁、服务端单租约请求检查是三处独立限制，必须在新协议形成闭环后统一替换。之前 CIFAR 固定代理直接调用 exec_gpu_prepared，没有经过这个作业管理器，因此不能把新增发现倒算成那次实验的直接原因。

## 12. 命令级预算与容量准入

**资源分两本账。** 容器租约继续保留 CPU、主机内存和容器槽位。GPU 显存改为每个作业独立预约，不能继续用 lease.gpu_active=True 表示整个容器的一份 GPU 占用。四条命令每条申报 1024 MiB，全部获准时就必须计入 4096 MiB；A 完成只能扣除 A 的 1024 MiB。

每个 JobSpec 至少包含：所属租约和容器、唯一 job_id、程序及 cwd/env、显存需求、排队截止时间、执行截止时间、标签。预算必须为正值，且受任务配置的上限约束。语言模型给出的数字不能被当作可信上限，标准 science 检查应由任务配置提供预算。未知程序应使用受控保守预算或拒绝缺失声明，不能把 0 当作不占显存。预算仍是准入估计，不是 CUDA 显存硬隔离。

**不能只计算 max(整卡已用, 已预约)。** 一部分已用显存可能属于外部进程，同时已预约的新作业还没把内存真正分配出来。应同时保留这两部分。对每张物理 GPU，在同一份有效采样中定义：

- C：该卡物理总量与配置上限中的较小值。
- U：整卡已用显存。
- m_j：本服务已预约作业 j 在这次采样中可归属的实际显存。
- b_j：作业 j 的预约量。RESERVED、RUNNING、DRAINING、QUARANTINED 都持有预算。
- H：明确配置的安全余量，覆盖已知的上下文和采样误差；不是无限外部争抢的保证。

新作业预算 b_new 只有在下式成立时才允许进入：

```text
U + Σ max(b_j - m_j, 0) + b_new + H <= C
```

其中 U 已包含运行作业实际用量，求和项补上它们尚未兑现的预约，避免重复计算实际用量。等价理解是“外部已用 + 每个自有作业 max(预约, 实测) + 新预约 + 余量”。每个 PID 只能归属于一个作业子树，不能把整个容器用量复制到每个作业上。

**教学数字，不是新增 GPU 实测：** 设卡容量 8192 MiB、安全余量 256 MiB，每作业预算 1024 MiB。

| 场景 | 初始外部占用 | 准入过程 | 结果 |
|---|---:|---|---|
| 低压力 | 512 MiB | 512 + 4×1024 + 256 = 4864，低于 8192 | 四条都能获准 |
| 高压力 | 6500 MiB | 一条需 7780；两条需 8804，超过 8192 | 一条获准，三条排队 |
| 高压力下 A 已退出但还没确认回收 | 6500 MiB | A 的 1024 仍保留 | B 不能抢先进入 |
| A 确认回收后 | 6500 MiB | 扣除 A，加入 B 的 1024 | B 获准，其他继续等待 |

再看现有 max 公式可能遗漏的组合：外部使用 6000 MiB，A 预约 1500 但只分配了 100。U=6100，max(6100,1500)+新作业1000+余量256=7356，看起来放得下；实际承诺为 6000+1500+1000+256=8756，已经超过 8192。此时 B 应等待。这个例子解释了为何“删锁、按原账本累加”仍不足以实现正确的并发准入。

需要守住四个边界：

1. 预约在调度锁内原子提交，先计费再发授权。节点执行远程调用时不长期占用这把锁，其他请求可登记、查询和取消。返回后重验作业身份与状态。
2. 连续准入必须把上次采样后新增预约也算进去。采样要带时间和预约版本，结束作业的旧显存样本不能误归到新作业。m_j 总和与 U 不一致到超出容差时，视为无效样本。
3. 关键采样缺失、过期或归属无法确认时暂停新准入，等待受排队截止时间约束；不要将缺失当 0，也不因为一次监控失败取消仍正常运行的作业。
4. 按 GPU UUID 分别做上述检查，不能把两张卡剩余显存相加，允许一个单卡程序使用总和。首轮只验收单卡场景；后续多卡需求应使用逐卡预算向量，并原子预约整组设备。

首版采用明确 FIFO，即先到先服务，并设置单租约的运行数和排队数上限，防止一个任务提交无限作业。排队数是拒绝过载的容量上限，运行数只是容量准入之外的保护上限，不把正常并发固定成 1。大作业在队首可能暂时挡住小作业，这是明确的首版策略，避免未验证的预测回填。明显大于空卡有效容量的请求立即拒绝，不能永远排队。

## 13. 进程子组与设备授权

**推荐每作业一个 cgroup 子树，不仅是容器权限引用计数。** cgroup 的中文含义是控制组，Linux 用它把进程及其派生进程组织成可管理的集合。在同一个文件系统和容器里，可以把 A、B、C、D 放进不同控制组，分别识别、授权和清理。

目标设备控制层级如下，这是尚未实现的部署结构：

```text
容器设备父组：只作授权上界，不执行任意用户程序
  ├─ cpu：终端、代理、普通 exec，始终禁止 GPU
  ├─ job-A：A 及所有派生进程，获准时开放 GPU
  ├─ job-B：B 及所有派生进程，获准时开放 GPU
  └─ job-C：尚在准备或排队，禁止 GPU
```

此结构的代码落点是 DeviceCgroupFence、节点作业注册表与可信进程启动器，目前只有整个容器级实现，没有可引用的作业子组实现。cgroup v1 的子组不能获得父组禁止的设备权限；父级 deny 会向下收紧，而父级新增 allow 不自动传给既有子组。因此必须先打开父级，再单独授权指定 job 子组，关闭 A 时只操作 A 子组；有 B 活动时不能关闭父级。[Linux 设备控制器说明](https://docs.kernel.org/admin-guide/cgroup-v1/devices.html)

父级授权状态应由可信节点注册表中的作业身份集合推导，不能仅维护一个可重复加减的整数。重试同一个 grant 不重复计数，旧 ack 不能移除新授权。最后一个作业完成物理回收后才关闭父级；关闭失败则保留隔离状态，不能宣称容器 GPU 权限已经撤回。

**启动必须先归组、后运行任意用户代码。** 若先启动训练，再把 PID 移进去，训练可能已经创建 CUDA 上下文或 fork 子进程。建议容器创建时安装可信启动入口：在全部 GPU 权限仍拒绝时，建立 cpu 子组并安置容器初始进程。之后普通 exec 和 GPU 作业都通过受控启动器，启动器在执行用户命令之前等待节点确认目标子组与出生身份。不能继续允许普通 docker exec 在已开放的父组直接执行用户程序。

可选 CPU 准备从一开始就在对应 job 子组里运行，此时 GPU 仍禁止。这样准备阶段创建的子进程天然归属同一作业，后续授权不需要临时搬迁整棵进程树。原有 prepare/execute 分段保留，但它必须使用这个统一启动器，不能作为绕开作业生命周期的第二条路径。对 CPU 准备的并发也设上限，避免显存尚未占用时已经用尽主机内存。

当前 gpu_job_process.py 使用进程组与 PID 出生时间避免误杀已复用的 PID，这是可保留的身份检查，但不是完整作业边界。子进程可以创建新会话脱离原进程组；cgroup 子树能继续覆盖这些后代。节点应保留可信的 job_id、容器身份、宿主机 PID 与出生时间、cgroup 路径及授权代次。不能直接相信任务可写 JSON 文件里的 PID 来执行宿主机杀进程。

这个实现需要可信节点服务具备设备控制子树的创建和修改权限，以及与 NVIDIA 进程监控接口 NVML 一致的宿主机进程视图。普通任务容器不能修改 cgroup 或迁移自身逃出子组。第 9 节记录的宿主机 procfs 缺失仍是已知部署前提，**本次设计没有重新检查节点，也没有证明平台已补齐权限**。有 Docker CUDA 能力不等于已有这些能力。

设计边界是一个任务内合作的多个作业。它不把共享容器变成互相不信任的安全沙箱：同用户的恶意进程互相发信号、传递设备句柄或利用容器特权，需要更强隔离，不能靠作业记账解决。违反边界导致无法确定归属时必须进入异常清理路径，不能继续报告独立回收成功。

## 14. 独立取消与物理归还

每条作业拥有自己的状态和确认记录，不能再用环境对象单个 last_gpu_command 或 _pending_gpu_return 存所有命令。

```text
PREPARING → QUEUED → RESERVED → RUNNING → DRAINING → RELEASED
                └─取消→ CANCELLED             └─无法确认→ QUARANTINED
                       取消 RESERVED / RUNNING 也必须经过 DRAINING
```

这是拟议状态机。输出就绪、程序成功与资源归还分开表示：程序可以失败但已安全回收，也可以输出成功但仍有残留子进程。上述 RELEASED 表示资源已归还，业务 exit_code 另存；不能只因观察等待超时就取消作业。

以 A、B 同时运行、用户取消 A 为例：节点先记录 A 不再允许启动新工作，只撤回 A 子组的新设备访问，B 的子组和父级授权保持。停止 A 的整个子树，包括后台或重新建立会话的子进程。确认 A 子树为空、A 的 GPU 句柄和计算进程消失后，才释放 A 预算，再决定是否启动 C。不能要求“整个容器没有 GPU 进程”，因为 B 本来就应该继续运行。

撤回设备权限不会立即杀死已有 CUDA 工作，因此停止与核验不能省略。当前节点使用 cgroup v1，不能直接假设支持 v2 的 cgroup.kill。建议把“有界终止整个作业子树”封装成节点后端能力：在具备 freezer 控制器的 v1 节点，冻结目标作业子树并确认 FROZEN，核验进程身份后发送终止信号，按内核要求解冻让退出完成，随后复查子树及设备活动。freezer 的 FREEZING 不等于已经冻结，达不到 FROZEN 或不能清空必须超时失败。该执行协议仍须真实节点验收。[Linux freezer 说明](https://docs.kernel.org/admin-guide/cgroup-v1/freezer-subsystem.html)

v1 的设备和 freezer 可能是两棵不同的层级，可信启动器必须在执行用户代码前核验同一作业在两棵树的成员关系。不要为个人实验重挂载控制器或改共享节点层级。没有所需后端能力时拒绝启动该并发模式，不能静默退化成对单个 PID 发信号并声称完整回收。

关键竞态和故障按下表定义：

| 情况 | 必须满足的处理规则 |
|---|---|
| QUEUED 时取消 | 从队列退休，禁止迟到请求重新创建同一个 job_id；没有已分配 GPU 预算 |
| RESERVED 后授权 RPC 尚未返回就取消 | 已计费不能提前释放；节点保存取消墓碑，即该授权身份已作废的记录，迟到 activate 被拒绝或进入清理 |
| RUNNING 时取消 | 节点服务拥有独立清理任务；客户端协程取消不能撤销清理责任 |
| DRAINING 时 B 仍在运行 | 只检查 A 子组，B 继续有权运行；A 的预算继续保留直到核验完成 |
| 重复 grant、cancel、ack | 按完整身份幂等，即重复请求不产生额外授权或重复扣账 |
| 旧 ack 晚到 | 以 lease_id、job_id、授权 epoch 和服务 incarnation 核验。epoch 是单次授权代次，incarnation 区分服务重启，旧身份不得修改新状态 |
| 单条进程无法终止、归属不确定 | A 进入 QUARANTINED 并保留预算，整卡停止新准入；已确认安全的 B 可继续。只有需要销毁整个容器才能恢复安全时，才将 B 一并失败并明确报告升级清理原因 |
| lease 到期或 episode 退出 | 先关闭新提交和启动，取消全部排队项，收敛所有活动作业，核验后清容器和释放会话资源 |
| Router 或节点重启 | 重建完成前拒绝新准入。按节点可信作业登记与物理状态对账，不能从空内存字典推断 GPU 空闲；首版允许拒绝自动恢复、只做所属容器的有界清理 |

因此“独立取消”是正常及可确认故障路径的目标，不是驱动故障或容器级污染下的绝对保证。异常范围和升级原因必须暴露给调用方，不能悄悄杀掉 B 后仍标注“A 独立取消成功”。

## 15. 源码改造清单与接口

下面是可落地的改动顺序，不是已经完成的代码变更。尽量复用已有模块，只有职责实质独立时才新增节点作业控制模块。

| 模块 | 必要改动 | 不能遗漏的验证 |
|---|---|---|
| types.py / gpu_jobs.py | 增加命令需求与独立授权记录；管理器移除包住整个执行过程的锁，保留注册/状态修改短锁、数量和时间上限 | 并发启动、重复 ID、排队取消、观察超时不取消 |
| ledger.py / command_gpu.py | 命令预算按 job 求和；替换同 lease 单命令限制和 gpu_active 布尔计费；预约先于 RPC；逐卡容量公式 | 外部占用加未兑现预约、多个 RESERVED、清理期间预算保留、超额等待 |
| command_gpu_node.py / 可信启动器 | 建立每 job 的设备与终止子组，默认 CPU 子组拒绝 GPU；按作业授权和取消，按活动身份集合管理父级权限 | 排队作业不可执行 CUDA，取消 A 不影响 B，迟到授权拒绝，后台后代回收 |
| task_telemetry.py / probe.py | 按精确作业子树、宿主机 PID 与出生身份归因；同次采样给出整卡和作业用量 | A/B 样本不混合、PID 复用、采样缺失、同容器相邻子树 |
| docker.py / prepared_gpu.py | 去掉单值 last_gpu_command / _pending_gpu_return，保存每 job 独立结果和回收 future；同步执行、异步作业和准备接口统一走同一后端 | 四个回收不串错任务，取消准备中任务，environment.stop 汇合全部作业 |
| router.py / client / Science 工具 | 透传预算与作业身份，新增必要状态查询；租约回收先关闭准入再处理所有 job | 真正从 Science 入口发出并发请求，训练收尾无残留 |

拟议调用形式如下，仅展示接口契约，当前代码尚不接受 gpu_memory_mb：

```python
# Proposed API, not an executable call against the current implementation.
job = environment.start_gpu_job(
    command="python train.py",
    gpu_memory_mb=1024,
    timeout_sec=180,
    wait_timeout_sec=300,
)
await environment.cancel_gpu_job(job["job_id"])
```

start 返回独立 job_id，四次 start 可以登记四条作业。gpu_memory_mb 是这条命令的需求，timeout_sec 从实际运行开始计时，wait_timeout_sec 约束排队。分阶段准备有自己的上限，不能让它无限占 CPU 和主机内存。普通同步 exec_gpu 应成为“提交一个 job 并等待其回收”的薄封装，不保留另一条依赖同容器锁的旧调度通道。

生产落地时先完成纯调度和生命周期契约，再完成可信节点启动与子组回收，最后统一全部入口后开放并发。不能先删除锁上线，再靠异常路径补回收。现有工作区还包含准备接口与独立 GPU 作业分支的差异，实施应以包含两者的明确源码快照为基线，不能误删一侧未提交功能。

## 16. 可运行模型与验收边界

配套 [model_concurrent_gpu_jobs.py](env-service-verification/scripts/model_concurrent_gpu_jobs.py) 是确定性的标准库契约模型，模拟图中的 P3、P4、P6。它实际运行在 CPU 上，不启动进程树、容器、Ray 或 GPU。示例中的授权集合和 empty 确认都是模拟输入，不能据此宣称 Linux 子组隔离、异步 RPC 竞态或真实 CUDA 取消已通过。

在仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/model_concurrent_gpu_jobs.py
```

**Results：本次 CPU 模型实际输出。** 低压力同时持有四个模拟授权，高压力只准入一个；A 清理确认前不准入下一条，确认后 B 进入。取消低压力组 B 后，A/C/D 的模拟授权仍保留。其余检查覆盖排队取消、取消后的迟到激活、旧授权确认、无法确认清理时保留预算、外部占用加未兑现预约，以及遥测过期停止新准入。断言全部通过，模型代码中含对应输入和预期结果。

正式实现建议按下面的有界验收推进，到阶段目标齐备即停，不为解释设计启动长训练：

| 阶段 | 验收内容 | 边界和停止条件 |
|---|---|---|
| 实际调度器 CPU 回归 | 将模型场景转成真实 Scheduler/Ledger 测试，增加取消与 grant/ack 交错、重启拒绝、每 job future 独立 | 不依赖 GPU，单次测试设超时；模型通过不能替代该阶段 |
| 节点启动与回收 | 预检 host procfs、devices/freezer 子树权限、PID 映射；先 CPU 子进程验证归组和 setsid 后代回收 | 一容器、四作业、总预算 120 秒；前提失败立即停止，不修改共享控制器 |
| 小型真实 CUDA | 一卡一容器四作业：容量足够重叠；不足排队；取消 A 时 B 继续产生输出；普通终端和排队 C 无 GPU 授权；最后物理归零 | 最多四个预定场景，共 300 秒，不自动重试。用准入预算制造排队与真实外部压力要分开标注 |
| 真实 CIFAR 对照 | 相同四命令、每命令十步，固定 OFF 与新 ON；同时测 GPU 进程重叠、成功数、排队/初始化/执行/清理耗时 | 低压力与明确授权的高压力各一对，每组上限 300 秒；失败保留日志，不加压或重跑挑结果 |
| 外层 PSRL 接入 | 默认 Science 代理通过正式工具提交作业，完成采样、评分、外层优化器步骤和清理 | 双方各一外层步骤，模型初始化和工作分别设限；证明接入，不声称长期收敛或稳定吞吐 |

所有 GPU 阶段都需要重新现场预检、确认所属资源和独占或共享约定，不能拿之前空闲状态当保证。正式 CPU 回归、Linux 子组、真实 GPU 并发及外层训练均为后续实施验收项，**本次只完成源码设计核对和教学模型运行**。

模型脚本的定向 Ruff 与格式检查通过；本次三篇文档的本地路径、源码行号范围及章节锚点检查通过，新增源码引用另做语义核对。全仓 pre-commit 在隔离副本运行，仍被既有 Ruff 和注释风格违规阻塞，自动修改未回写共享工作区，日志见 [concurrent-design-precommit.log](gpu-job-validation/concurrent-design-precommit.log)。

## 17. 跟着方案读源码

以下按实际模块顺序列出阅读入口，同时说明未来改造的输入和状态变化；所有“应”均属于提案。

1. **提交 → P1/P2 → gpu_tools.dispatch_gpu_tool、GpuJobManager.start。** 输入命令、预算与时限，返回 job_id；四次调用应建立四份独立状态。当前管理器内部串行锁需要替换，不能只更改前端 gather。
2. **准备分支 → P2 → prepared_gpu.execute_prepared。** 当前创建准备进程再调用 exec_gpu，未来应在禁止 GPU 的专属子组中完成准备，然后把同一身份送入队列，不重启或丢失进程归属。
3. **准入 → P3 → command_gpu.acquire、ledger._declared_gpu_mb / _effective_gpu_mb。** 输入逐卡需求与有效遥测，原子增加 RESERVED，输出授权身份或等待原因；节点 RPC 与其他请求并发，返回后重新核对取消和租约状态。
4. **授权和启动 → P4 → command_gpu_node.grant_gpu_command、gpu_job_process.run。** 当前分别按容器和进程组实现，未来应先建立可信子组身份，再开放该子组设备并运行程序。普通 exec 也必须留在 CPU 子组。
5. **运行与采样并发 → P4 → task_telemetry.sample。** 按每个 job 子树过滤宿主机样本，返回同一次采样中的整卡 U 与各作业 m_j。不能把整容器遥测发给每个 job。
6. **取消或退出分支 → P5/P6 → command_gpu_node.revoke_gpu_command、command_gpu._return。** 输入目标 job 和授权身份，停止它的子树并核验；只在成功确认后扣这条预算和完成其回收 future，兄弟作业不受影响。
7. **最终汇合 → P7 → GpuJobManager.close、router.finish_gpu_command_lease。** 禁止新启动，收敛全部作业再清理容器；任何不确定的 GPU 残留都保留隔离记录，不能仅根据任务协程已结束解除配额。

源码跳转见第 11 节图例；完整 API 与物理子组接口仍需按上述契约实现。本节提供可评审的实现方向和验收口径，没有将生产改造或平台部署写成已完成。

前四条确认工作树和分支关系，中间两条分别检查未提交改动，最后一条确认远端仓库。命令已在本次核对中执行。要复现已推送功能，应固定功能分支的具体提交和依赖路径，不能仅因为主目录已经同步了部分文件，就把它当成相同的完整实验快照。

单目录切分支可以完成日常开发。需要先处理的是未提交状态：与目标分支不冲突的修改可能被带过去，切换会覆盖修改时 Git 通常拒绝。因此不能把切换分支当作自动保存工作。优先按任务范围提交可独立检查的改动，需要暂存工作进度时使用 stash，并明确未跟踪文件是否包含、何时恢复，不强制丢弃。

具体到当前目录，主工作树的 `docker.py` 还有 prepared-GPU 修改，功能分支没有这部分。如果只关注分支名而忽略未提交状态，就可能把实验代码与已发布代码混淆。独立工作树解决了最初的隔离，但持续同步两份文件又增加了核对成本。后续在主目录整理修改、切换功能分支、验证并提交，可减少这类维护负担。此次只记录开发约定，保留现有分支与文件，没有执行切换或删除工作树。
