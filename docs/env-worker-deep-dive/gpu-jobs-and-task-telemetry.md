# GPU 作业入口、任务遥测与执行后端统一

> **核心结论**：GPU 作业现在有独立的提交、查询、等待、取消和遥测入口。调用方可以在作业排队、计算及归还期间执行无依赖的工作。确认物理归还之前，GPU 配额仍保留。默认 science Terminus2 已接入该工具。
>
> 遥测改用节点采集的进程数据，并按容器 cgroup 和进程出生身份过滤。普通 Docker 的整卡 NVML 视图没有被虚拟化。嵌套 PID 命名空间必须提供宿主机 procfs，否则拒绝归因。当前实验节点缺少该部署条件，正向进程归因只有 CPU 契约验证，没有实机验收通过。

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

前四条确认工作树和分支关系，中间两条分别检查未提交改动，最后一条确认远端仓库。命令已在本次核对中执行。要复现已推送功能，应固定功能分支的具体提交和依赖路径，不能仅因为主目录已经同步了部分文件，就把它当成相同的完整实验快照。
