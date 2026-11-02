# EnvRouter 的 MPS 并行执行后端

> **核心结论**：配置 `gpu_execution_backend: mps` 后，一个环境容器内和不同环境容器之间的 GPU 作业都能按显存预算并发，放不下的作业排队。每条作业分别记账、取消和回收。native EnvWorker 与 Harbor Compose 共用这套执行实现，command 与 episode 仍决定预算的持有范围。
>
> 此后端面向同一可信实验域内的任务、环境专用的单张 GPU 和 cgroup v1 节点。它不是不可信租户之间的安全隔离，也不是 vLLM 与环境同卡共享功能。本文主体结果是功能验收，不是 science 训练加速比。[真实 GPU 训练并发对照](../experiments/mps-training-comparison.md)单独报告固定 autoresearch 工作量的性能证据。该验证暴露的并发管理连接竞争已通过按服务串行控制请求修复，原失败训练组的定向复测通过。该结论不等于所有工作负载长期稳定。

## 目录

- [1. 全局执行流程](#1-全局执行流程)
- [2. 一次并行与排队的例子](#2-一次并行与排队的例子)
- [3. 配置与任务入口](#3-配置与任务入口)
- [4. 独立取消与故障处理](#4-独立取消与故障处理)
- [5. Results 与复现实验](#5-results-与复现实验)
- [6. 跟着图读源码](#6-跟着图读源码)
- [7. 使用边界与后续工作](#7-使用边界与后续工作)
- [8. 原理与当前效果应该怎样理解](#8-原理与当前效果应该怎样理解)

## 1. 全局执行流程

MPS 是 NVIDIA 的多进程服务，负责让不同 CUDA 进程的 GPU 工作具备并发执行的条件。Router 解决的是另外一层问题：允许哪些任务进来、何时开始，以及何时才能把预算交给下一条任务。两者结合后，才形成可管理的并行执行流程。

```text
[1] Science GPU 工具 / 环境 start_gpu_job
                  │ 命令、job_id、显存需求、执行与排队期限
                  ▼
[2] ManagedGpuExecution → EnvRouter 检查活跃租约
                  ▼
[3] 节点 MpsDomain：unit 级 FIFO 队列与显存准入
        │ 容量足够                       │ 容量不足
        ▼                                └─ 留在队列
[4] Docker exec 的阻塞启动器 → 作业独立 devices/freezer cgroup
        │ 核对容器归属和 PID，迁移进程，授权 GPU，再发送 go
        ▼
[5] CUDA 作业 A、B… → 该 unit 的 MPS 服务 → 同一 GPU
        │                      ↑
        └─ eBPF 身份 + NVML ────┘ 按作业观测显存
                  ▼
[6] 正常结束 / 超时 / 取消 → MPS 定向释放 CUDA → 回收作业进程
                  │ 物理回收确认成功
                  ▼
[7] 归还作业预算，唤醒等待者。最后一份租约结束时停止自建 MPS 服务
```

第 6 步的查询和终止命令统一经过 [MpsControlChannel](../../../psrl/workers/env_service/mps.py#L11)，同一服务同时只建立一条受管管理连接。锁不覆盖第 5 步的 CUDA 计算，不改变多作业并行。

图例：图里每个部分对应的源码位置。cgroup 是 Linux 对进程的资源分组机制，devices 控制设备访问，freezer 用于短暂冻结进程。eBPF 在内核侧记录进程归属，NVML 是 NVIDIA 的 GPU 遥测接口。

| 图中部分 | 职责与对应实现 |
|---|---|
| 1 | [dispatch_gpu_tool](../../../examples/sciaccel_rl/gpu_tools.py#L32) 把 science 的 JSON 工具请求转换为受管作业调用 |
| 2 | [ManagedGpuExecution](../../../psrl/workers/env_service/managed_execution.py#L14) 传输作业与观察结果，[Router.submit_mps_job](../../../psrl/workers/env_service/router.py#L259) 核对租约 |
| 3 | [MpsDomain.admit](../../../psrl/workers/env_service/parallel_gpu.py#L208) 按新鲜遥测和未兑现预约判断准入 |
| 4 | [MpsDomain.launch](../../../psrl/workers/env_service/parallel_gpu.py#L267)、[JobFence](../../../psrl/workers/env_service/job_fence.py#L11) 建立作业归属与设备授权 |
| 5 | [MpsDomain.register](../../../psrl/workers/env_service/parallel_gpu.py#L45) 创建独立 Docker MPS 服务，[TaskGpuTelemetry](../../../psrl/workers/env_service/task_telemetry.py#L73) 采样 |
| 6 | [MpsDomain.cleanup](../../../psrl/workers/env_service/parallel_gpu.py#L342)、[MpsClientController.terminate](../../../psrl/workers/env_service/mps.py#L123) 执行定向回收 |
| 7 | [MpsDomain.release](../../../psrl/workers/env_service/parallel_gpu.py#L408) 回收租约和服务，[Node.cleanup](../../../psrl/workers/env_service/node.py#L182) 处理失联租约 |

GPU 内核究竟在多长时间内同时运行，由 GPU 和 MPS 调度决定。程序生命周期重叠、两条 CUDA 循环同时推进，不等于已经测得内核重叠百分比或吞吐提升。

## 2. 一次并行与排队的例子

以下是解释算法的教学数字，不是性能测量：unit 的 GPU 预算是 2304 MiB，MPS 服务及其他已用显存约 64 MiB。A、B、C 每条申报 1024 MiB。

1. A、B 共需 `64 + 1024 + 1024 = 2112 MiB`，可以一起开始。
2. 加上 C 需要 3136 MiB，超过预算，因此 C 排队。C 的容器可以已经存在，文件编辑等 CPU 工作仍可继续。
3. A、B 开始运行后，各实际使用约 570 MiB。不能因此立刻把剩余预约当成空闲，程序后面仍可能分配显存。
4. 取消 A 时，先回收 A 的 CUDA 上下文和进程。确认后才减去 A 的 1024 MiB 预约，C 随后开始，B 继续运行。

准入使用的基本式子是 `整卡实测占用 + Σ max(作业申报 - 已归属实际占用, 0)`。实际占用已经包含在整卡数值中，因此只额外计入尚未兑现的部分。无法可靠归属时，不给予该作业扣减抵免，宁可多等也不凭空增加容量。

| 比较项 | direct 后端 | MPS 后端 |
|---|---|---|
| 同容器受管 GPU 作业 | 串行 | 容量足够时并行 |
| 不同容器的 GPU 命令 | 由原 command/episode 准入路径管理 | 同一 unit 的作业进入共同队列 |
| GPU 工作执行 | 普通 CUDA 多进程 | 连接本 unit 的 MPS 服务 |
| 进程与遥测归属 | 原容器级命令路径 | 每条作业独立设备组和身份记录 |
| 普通终端直接执行 CUDA | command 禁止，episode 沿用原能力 | 两种模式均禁止，必须使用受管入口 |
| GPU 数量与训推共享 | 按原资源策略 | 一个 unit 一张环境专用 GPU |

command 和 episode 不是两个不同的 GPU 执行器。两者在 MPS 下使用同一个作业队列和回收器，但租约预算不同：

- **command**：环境存在时不一直占用整份 GPU 声明额度，每条作业执行期间申请预算，确认回收后归还。
- **episode**：会话先保留自己的 GPU 额度。会话内部作业可以并行，但活动作业的声明之和不能超过该会话额度。作业结束只释放内部可用额度，整份会话预算要等租约结束才归还。

例如 episode 申报 2048 MiB，两个 1024 MiB 作业能一起运行。两条都结束后，unit 仍为这个会话保留 2048 MiB。状态中的 `job_capacity_released` 表示作业已回收，`capacity_released` 在 episode 下仍为 false，避免把两层预算混为一谈。

## 3. 配置与任务入口

在实际 EnvRouter 配置的 `units` 项中设置以下字段。`node_ip`、GPU 索引和镜像要使用现场核验值，示例中的名字不代表一个可直接访问的节点。

```yaml
node_policy: any
gpu_identity_backend: ebpf
units:
  - unit_id: env-mps
    node_ip: YOUR_NODE_IP
    cpu_quota: 5
    memory_mb: 4096
    max_containers: 2
    gpu_indices: [0]
    gpu_memory_mb: 2304
    gpu_reservation: command
    gpu_execution_backend: mps
    mps_image: YOUR_CACHED_IMAGE
    mps_thread_percentage: 100
```

`cpu_quota` 和 `memory_mb` 包含 MPS 服务开销。服务保留 1 CPU、512 MiB 主存，因此此例给两个各 2 CPU、1024 MiB 的环境仍然留得下服务。Docker 服务不是临时显存占位，实际 GPU 开销也会进入 NVML 实测值。

任务通过已有 science GPU 工具指定单条作业预算：

```text
psrl-gpu {"op":"start","command":"python train.py","cwd":"/workspace","gpu_memory_mb":1024,"timeout_sec":120,"wait_timeout_sec":60,"tag":"train-a"}
psrl-gpu {"op":"start","command":"python eval.py","cwd":"/workspace","gpu_memory_mb":1024,"timeout_sec":120,"wait_timeout_sec":60,"tag":"eval-b"}
psrl-gpu {"op":"status","job_id":"返回的编号"}
psrl-gpu {"op":"cancel","job_id":"需要取消的编号"}
```

两个 start 都立即返回编号，是否真正同时开始由节点预算决定。两个任务必须独立，不应同时覆盖同一个输出文件。工具参数 `gpu_memory_mb` 是正整数，不能超过所属租约申报的 GPU 需求。不传时，每条作业使用租约的完整需求，不能期待两个满额作业总能同时执行。

`mps_thread_percentage` 控制客户端可使用的活跃线程比例，其可见 SM 数可能按硬件粒度向下取整。SM 是 GPU 上执行线程块的计算单元。这个设置不是固定划分某几块 SM，也不保证计算吞吐的最低份额。MPS 为客户端设置设备内存上限，但该上限是**每个客户端进程**的限制，不能把它写成一个派生多进程作业的显存硬隔离。作业仍应如实声明整个进程树的预算。

镜像需要 `sh`、`bash`、`python3` 和可工作的 CUDA 用户态环境。MPS 服务使用节点上与驱动匹配的二进制、驱动库和真实设备映射。客户端和服务使用相同 UID，目前受支持配置为 root，并移除所有 Linux capabilities。两者共享专用 IPC namespace 和 MPS 通信目录，IPC 指进程间通信，不是共享 PID 编号空间。

## 4. 独立取消与故障处理

每条命令单独创建 devices/freezer cgroup。该组是节点可见控制器根目录下的独立组，不是容器 devices 组的子组，否则父组的 GPU 禁止规则会让子组也无法开放 GPU。进程的 CPU、主存 cgroup 保持在原容器中，所以仍受原 Docker 的 CPU、内存限制。

启动器先阻塞，节点核对 Docker 租约标签、容器 PID namespace 和 `NSpid`，确认命令进程确实属于这个容器，才迁入作业组、授权 GPU 并放行。容器内看到的 PID 用来定位候选，MPS 控制使用节点/MPS 服务视图的 PID，GPU 遥测中的更外层 PID 由 eBPF 关联，不能直接互换。

取消过程如下：

1. 关闭该作业新打开 GPU 设备的权限，暂时冻结它以核对进程集合、出生身份、MPS 客户端 ID 和服务身份。
2. 解冻客户端，再调用 `terminate_client`。实测驱动需要客户端继续处理终止握手，持续冻结会让确认卡住。
3. 收到 CUDA 终止确认后，再次冻结和检查成员变化。新出现的客户端需要另行处理，循环有明确上限。
4. 回收这个作业的进程树，确认组内已无进程和可归属的 GPU 进程，然后释放预约。不会删除仍被其他作业使用的主容器。

NVIDIA 要求先完成 MPS 客户端 CUDA 回收，再结束进程，否则可能影响共享服务中的其他客户端。参见 [NVIDIA 的客户端提前终止说明](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html)。

失败路径也有明确行为：

- 排队取消不会先申请 GPU。观察超时只停止等待结果，执行超时才触发作业回收。
- 启动中取消先等待受控启动动作落定，不能在后台还可能创建进程时就归还预算。
- 无法确认 CUDA 或物理清理时，作业进入 `quarantined`，该 domain 停止接受新作业，预算保留。租约回收会有限重试清理。
- science 的最终清理、容器 provider 的 stop 和租约 TTL 回收，均须先经过 MPS 清理，不能用直接删除容器绕过终止握手。
- 节点服务进程丢失状态后，发现同 unit 的孤儿 MPS 服务会拒绝重新准入和盲目删除。需要维护人员处理，当前不承诺进程崩溃后的自动无损恢复。

## 5. Results 与复现实验

功能矩阵在同一张环境专用 H20 上运行，主容器使用已有 science 镜像，但负载是有界的 CUDA 功能程序，**不是原生 science 求解器或 RL 训练**。没有额外后台显存占位。实验将 unit 声明预算设小，以确定触发第三条作业排队。

| 真实组合 | 同容器两条 CUDA 同时推进 | 第三条排队 | 取消 A 后 B 继续、C 获准 | 最终容器与 GPU 进程 |
|---|---|---|---|---|
| native + command | 通过 | 通过 | 通过 | 全部回收 |
| native + episode | 通过 | 通过 | 通过 | 全部回收 |
| Compose + command | 通过 | 通过 | 通过 | 全部回收 |
| Compose + episode | 通过 | 通过 | 通过 | 全部回收 |

各组均要求两个 CUDA 循环已实际执行超过 50 次内核，再检查并行和取消，不以状态字段 running 代替 CUDA 执行证据。两个作业分别观测到约 570 MiB 显存，存活作业最终核对 GPU 写回值为 42。最终 NVML 无计算进程，剩余约 0.19 MiB 驱动常驻显存。

最后一组还通过 science 的 `dispatch_gpu_tool` 提交，并将 MPS 线程比例设为 50。该卡客户端可见 SM 从 78 变为 38。给客户端设置 256 MiB 上限后申请 512 MiB，CUDA 返回预期的内存不足错误 2，设备仍有充足物理显存。这验证的是客户端限额生效，不是整卡 OOM。结构化结果见 [验收汇总](../experiments/mps-parallel-results.json)。

额外故障用例覆盖排队取消、已执行 CUDA 的子进程超时回收、回收后的新作业执行，以及普通终端即使知道 MPS 管道地址也不能直接开启 GPU。CPU 回归还覆盖清理失败保留预算、冻结与终止握手顺序、启动中取消和服务开销记账。相关测试目录合计 374 项通过。首次广泛回归中，两项清理测试替身缺少新增 backend 字段，已补齐。另两项张量测试加载了未修复的共享 veRL，改用已有隔离依赖后通过，未修改共享环境。

本次文件的定向风格检查通过。全仓 `pre-commit run --all-files` 在隔离副本执行，仍有既存文件的超长行和注释风格问题，未把全仓检查标记为通过，也未带回其批量自动修改。

复现入口是 [mps_parallel_check.py](../env-service-verification/scripts/mps_parallel_check.py)。它需要真实 Docker GPU 环境，单次外层时限 180 秒，每条正常 CUDA 命令运行 12 秒，单次只使用选定的一张 GPU。先按[集群预检](../../psrl/psrl-run-guide/cluster-preflight.md#集群踩坑总表与最小复现2026-10-13)核验实际节点、Docker、设备映射和当前占用。

从仓库根目录执行，脚本交互读取现场参数，避免把历史节点身份当成当前配置：

```bash
read -r -p '执行节点 IP: ' NODE_IP
read -r -p '已核验空闲的 GPU 索引: ' GPU_INDEX
read -r -p '缓存镜像名: ' MPS_IMAGE
python docs/env-worker-deep-dive/env-service-verification/scripts/mps_parallel_check.py \
  --node-ip "$NODE_IP" --gpu "$GPU_INDEX" --image "$MPS_IMAGE" \
  --provider native --mode command --output /tmp/mps-parallel-check
```

输出目录必须尚不存在。切换 `--provider compose`、`--mode episode` 验证其余组合。`--faults` 增加超时与排队取消检查，`--limits` 检查客户端内存上限，`--thread-percent 50` 调整计算限制。结果保存在输出目录的 `evidence.json`，不能只检查启动日志。必要时使用 `nohup` 在后台运行并保存日志，且仍保留脚本自带时限。

CPU 契约测试无需启动 Docker 或 GPU，需项目 Python 依赖：

```bash
python -m pytest tests/env_service/test_parallel_gpu.py tests/env_service/test_mps.py -q
```

这些 CPU 测试使用受控替身检查状态顺序，不能代替上表的真实 Docker CUDA 验收。

## 6. 跟着图读源码

| 步骤 → 图中编号 | 入口与状态变化 |
|---|---|
| 提交 → 1、2 | `dispatch_gpu_tool → start_gpu_job → _execute_mps_job`。返回 job_id，客户端后台任务向 Router 提交，节点拥有执行生命周期 |
| 建域与核验 → 2、3 | Router 的租约准入注册 `MpsDomain`，准备管道与 IPC 配置。后续作业必须持有活跃的 MPS 租约 |
| 准入 → 3 | `admit` 使用 unit 锁串行检查容量。获准后状态变为 starting，GPU 执行本身不会一直持有这把锁 |
| 启动 → 4 | `launch → container_pid → JobFence.attach`。核验 PID 并迁移设备/冻结组，然后发送 go，状态变为 running |
| 并行执行 → 5 | 多个 Docker exec、输出收集和 `TaskGpuTelemetry` 采样并发运行。不是一条串行函数链 |
| 收尾 → 6 | 正常结束、取消或异常均进入 `cleanup`。MPS 回收确认和进程回收完成后，才记录 returned 时间 |
| 销毁 → 7 | `drain/release` 阻止新作业、处理剩余作业，最后一份租约结束后关闭该 domain 的服务。TTL 清理走同样的先后次序 |

各函数的可跳转位置见第 1 节图例。代码中的归属、状态或调用顺序如果有不清楚的地方，可以直接沿这些函数继续追问。

## 7. 使用边界与后续工作

- **控制通道约束**：[MpsControlChannel](../../../psrl/workers/env_service/mps.py#L11) 按 domain 串行管理连接，GPU 计算仍并行。排队和执行共用 10 秒期限，失败不盲目重试，无法确认物理释放仍保留预算。真实训练收尾复测见[故障定位与修复](../experiments/mps-training-comparison.md#6-控制通道故障定位与修复)。
- 默认仍是 direct，MPS 必须显式启用。模式选择仍是 unit 的静态配置，没有自动选择 command/episode、预测 offload 或动态迁移实现。
- 当前支持 cgroup v1 的 devices/freezer、具备相应管理权限的节点、支持所用 MPS 控制接口的 NVIDIA GPU/驱动。cgroup v2、非 root 客户端、多 GPU unit 明确不在本次支持范围。
- MPS 控制表查询与终止不是原子事务。当前实现用于可信实验域，配合作业组稳定身份，不宣称能防御恶意客户端重连、篡改工具或对同容器同 UID 进程发送信号。
- MPS 通信目录和 IPC namespace 在 domain 内共享，不是多租户安全边界。显存申报也不是恶意多进程任务的聚合硬上限。
- SM 利用率可能缺少可归属样本，记账支持不等于利用率评分一定可用。science 的活动奖励仍应遵守原归因门槛，不应补造利用率或把整卡忙碌当作本作业贡献。
- 固定 autoresearch 训练的串行、普通多进程和 MPS 对照见[性能报告](../experiments/mps-training-comparison.md)。它不覆盖原生 science 求解器或 vLLM 同卡性能，时间线采样与训练计时也分别报告。

## 8. 原理与当前效果应该怎样理解

本节区分 2026-10-30 的 MPS 功能验收与 2026-11-02 的固定 autoresearch 训练性能验证。性能结论只适用于测过的负载和条件，不能将“实现完成”理解为所有工作负载均已证明更快、无 OOM 或无故障。实现对应本地代码快照，实际实验版本和工作区补丁见性能报告，不代表远端合并或全局部署。

先沿第 1 节的完整图理解：工具提交 → Router/节点决定准入 → 作业组授权设备 → CUDA/MPS 执行 → 定向回收 → 归还预算。Router 不逐个调度 GPU 内核，MPS 也不替 Router 管理实验任务的生命周期。

**GPU 的时间复用是什么。** CUDA 上下文是进程使用 GPU 时建立的资源和执行状态集合。GPU 内核是提交到 GPU 执行的计算函数。在常规的独立 CUDA 上下文路径里，不同进程的计算工作通常按上下文轮流获得执行机会。A、B 都占着显存，两个进程也都在运行，不代表两者的计算内核此刻同时执行。同一进程内部的多个 CUDA stream 则是另一层并发，不能用“多进程轮流执行”否定一个上下文内部的并行。

**MPS 改变了什么。** MPS 为客户端协调共享的 GPU 执行资源，使不同进程的工作可以同时被 GPU 调度。对于 H20 这一代设备，客户端保留独立 GPU 地址空间，并直接向 GPU 提交工作，不是每次计算都由 MPS 服务进程代发。MPS 是驱动侧机制，应用通常不需要重写计算内核。[NVIDIA 架构说明](https://docs.nvidia.com/deploy/mps/595/architecture.html)介绍了上下文、时间复用以及 Volta 之后的客户端直接提交机制。这里的驱动实现属于外部机制，本仓库只有接入与调度代码，没有 NVIDIA 调度器源码。

**为什么可能更快，但不是必然更快。** 下面是教学假设，不是实测：假设 GPU 有 80 个 SM，SM 是执行线程块的计算单元。A、B 每次各只能提供足以使用约 20 个 SM 的计算。

- 按上下文轮流执行时，轮到 A 的阶段里，B 不能直接把自己的计算填入 A 未使用的资源。
- MPS 使 A、B 的计算有机会重叠，在这个假设下可以共同利用更多空闲资源。
- 如果 A 自己已经吃满 SM 或显存带宽，B 再进入会争用资源，A 的单任务延迟可能变长，总完成时间也不一定更短。

这些数字仅说明空闲资源如何被填补。真实执行还受线程块、寄存器、共享内存和带宽等限制，不能把“20+20”机械换算成两倍速度。`mps_thread_percentage=50` 也只是限制可用活跃线程，既不保证半张卡的吞吐，也不是把固定的某一半 SM 永久独占分配给某个进程。硬件会取整，本次历史实测可见 SM 从 78 变为 38。[NVIDIA 配置说明](https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html)给出了线程比例生效层级。

当前 `mps_thread_percentage` 默认是 100，各客户端都可以争取空闲执行资源，没有自动按作业数量平均分卡。两个客户端各设置 100 并不意味着预留了两张卡。50 是限额实验的显式配置。

MPS 也不自动释放等待模型回复期间的常驻显存。episode 作业等待 vLLM 时，如果显存仍占着，其他任务能否进入仍取决于剩余预算。offload/reload 是另外一层策略，不能用开启 MPS 代替。

**防 OOM 的主要贡献是谁。** 第 2 节的 2304 MiB 例子对应的是 Router/节点准入：两个各申报 1024 MiB 的作业先开始，第三条等回收。MPS 负责让已经获准的计算具备重叠条件，同时提供客户端资源限制。关闭 MPS 但保留正确的预算与排队，也能降低超额准入造成的 OOM 风险。只开 MPS、不做任务预算，仍然可能 OOM。

显存申报必须覆盖整个作业的实际需要。MPS 客户端内存上限按进程生效，多进程作业的总量不能靠单个进程的上限自动保证。预算低于真实需求时，即使没有其他任务竞争，本任务也可能因限额而分配失败。

**目前的证据分为三个层次。**

| 层次 | 已有证据 | 尚不能推出的结论 |
|---|---|---|
| 功能 | native/Compose × command/episode 四种组合通过。并发提交、队列、取消、子进程超时回收和最终清理经过实机验收 | 任意任务、任意驱动或长时间压力均可靠 |
| 资源控制 | eBPF 能区分两个作业的显存，线程比例生效，客户端超限返回预期 CUDA 错误 | 所有进程树都获得显存硬隔离，SM 利用率评分总能归属 |
| 性能 | 已增加真实 autoresearch 固定训练工作量的三组对照，具体数据与时间线边界见[性能报告](../experiments/mps-training-comparison.md) | 原生 science 训练加速、高负载尾延迟改善，或任意任务均因 MPS 而更快 |

374 项是相关目录的回归测试总数，不是 374 次真实 GPU 实验。真实 GPU 验收记录是第 5 节的功能矩阵及额外限额组，负载为小型 CUDA 程序，使用 science 镜像和工具入口也不等于运行了原生科学求解器。

正常取消已验证不影响邻居，但设备致命错误、驱动重置、服务意外死亡、客户端异常退出等还不能据此视为全面覆盖。特别是 MPS 客户端在无法连接服务时存在退回普通 CUDA 的驱动行为，当前 `launch` 设置连接环境并检查容器配置，没有实现覆盖所有服务失联时刻的持续连接证明。此项应列为后续故障注入验收，不能把配置了 MPS 环境变量当作任何时刻都一定通过 MPS 执行的证据。服务故障可能影响同 GPU 的其他 MPS 客户端，参见 [NVIDIA 故障与提前终止说明](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html)。

autoresearch 对照补充了真实训练计算的证据。若要进一步声称原生 science 收益，仍须使用真正的 GPU 科学求解器、固定输入和相同成功工作量，记录总完成时间、单任务延迟及失败率。三组机制对照不能冒充项目指定的固定历史 OFF baseline，也不应为了追求加速数字反复更换负载。

**可运行的 CPU 教学模型。** 从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/model_concurrent_gpu_jobs.py
```

脚本用模拟预算和清理确认演示第 1 节图中 3、4、6、7 的状态变化，源码入口是 [Model.admit](../env-service-verification/scripts/model_concurrent_gpu_jobs.py#L47) 和 [Model.acknowledge_empty](../env-service-verification/scripts/model_concurrent_gpu_jobs.py#L85)。它模拟设备授权和物理释放，不执行 CUDA、不实现 MPS，模型中的额外 headroom 参数也不是当前 MPS 后端同名配置。应观察低压力放行四条、高压力只放行一条、正在回收仍占预算、取消 B 后 A/C/D 保留授权。这解释准入与回收机制，不验证 GPU 性能。实际执行链仍应沿第 6 节源码顺序阅读，相关函数不清楚时可以继续追问。

本次实际运行该 CPU 模型通过：`low_pressure_grants=4`、`high_pressure_grants=1`、`draining_retains_budget=true`，取消后保留 A/C/D。它是本次的新观察，与 2026-10-30 的 GPU 验收记录分别计述。
