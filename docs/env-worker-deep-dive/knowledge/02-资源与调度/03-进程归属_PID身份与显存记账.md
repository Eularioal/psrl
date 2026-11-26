# 进程归属：PID身份与显存记账

> **核心结论**：先把驱动报告的宿主 PID 关联到可信任务，再结合实测显存与未兑现预约判断准入，最后确认物理释放才归还预算。pidfd 保护本地进程引用，procfs/eBPF 建立 GPU 进程归属，二者承担不同环节。本篇保留完整编号例子、账本手算与各阶段验证条件。

## 目录

- [进程编号、可信归属与后端验证](#进程编号可信归属与后端验证)
  - [用一个任务理解两种方案](#用一个任务理解两种方案)
  - [SSH 私有节点为什么也是容器](#ssh-私有节点为什么也是容器)
  - [先看当前链路缺什么](#先看当前链路缺什么)
  - [唯一编号究竟指什么](#唯一编号究竟指什么)
  - [pidfd 怎样避免认错进程](#pidfd-怎样避免认错进程)
  - [不挂载宿主 procfs 的替代方案](#不挂载宿主-procfs-的替代方案)
  - [真实 CPU 验证与复现](#真实-cpu-验证与复现)
  - [内核追踪与真实 Docker CUDA 对账](#内核追踪与真实-docker-cuda-对账)
  - [统一后端怎样接入框架](#统一后端怎样接入框架)
  - [统一后端的双容器实测](#统一后端的双容器实测)
  - [跟着图读源码](#跟着图读源码)
  - [后续实现顺序与边界](#后续实现顺序与边界)
  - [接口资料](#接口资料)
  - [PID 为什么分层及 MPS 应该用哪个编号](#pid-为什么分层及-mps-应该用哪个编号)
- [GPU 身份、归属与显存会计](#gpu-身份归属与显存会计)
  - [1. 先还原一个具体问题：同一个容器里，这笔显存该记给谁](#1-先还原一个具体问题同一个容器里这笔显存该记给谁)
  - [2. 全局图：一次 GPU 命令执行，身份/显存/预算/健康怎么串起来](#2-全局图一次-gpu-命令执行身份显存预算健康怎么串起来)
  - [3. 跟着图读源码](#3-跟着图读源码)
  - [4. 新知识点逐个展开](#4-新知识点逐个展开)
  - [5. GPU 显存会计公式：accounted_gpu_memory_mb 手算](#5-gpu-显存会计公式accounted_gpu_memory_mb-手算)
  - [6. 资源画像：harbor_resources + container_budget 手算一个任务的预算](#6-资源画像harbor_resources--container_budget-手算一个任务的预算)
  - [7. 节点健康状态机：NodeHealth 的熔断与恢复](#7-节点健康状态机nodehealth-的熔断与恢复)
  - [8. 协程知识回顾 + 两个新模式](#8-协程知识回顾--两个新模式)
- [取消安全与可运行实验](#取消安全与可运行实验)
  - [8.3 新模式 B：多持有者引用计数 + 取消安全清理（routing_pause.py）](#83-新模式-b多持有者引用计数--取消安全清理routing_pausepy)
  - [9. 可运行实验脚本](#9-可运行实验脚本)
  - [10. 一句话总结 + 延伸阅读](#10-一句话总结--延伸阅读)

## 进程编号、可信归属与后端验证

<!-- figure:pid-namespace-identity -->
这张图沿同一个场景解释：两个容器里都有 PID 12，取消任务时怎么避免杀错进程？

[![进程归属：容器内编号与宿主编号可能不同（1/2）](../../../_static/img/document-figures/pid-namespace-identity-screen.png)](../../../_static/img/document-figures/pid-namespace-identity.png)

[![进程归属：容器内编号与宿主编号可能不同（2/2）](../../../_static/img/document-figures/pid-namespace-identity-screen-02.png)](../../../_static/img/document-figures/pid-namespace-identity.png)

教学编号：容器 A 的 PID 12 对应宿主 418；容器 B 的 PID 12 对应宿主 907。 进程身份包含所在空间与出生信息。裸 PID 既会跨空间重复，也会在退出后被复用。 编号均为教学示意。可信归属的当前实现范围按正文区分，不扩大能力。 [查看矢量图](../../../_static/img/document-figures/pid-namespace-identity.svg)。
<!-- /figure:pid-namespace-identity -->

> **核心结论：** SSH 登录的私有节点环境是平台容器，内部再运行 Docker 任务容器。PSRL 已新增可选的 `ebpf` 进程归属后端，与原有 `procfs` 后端共用任务遥测和显存预算接口。它按内核进程身份、容器资源组及本轮命令注册号记账，不依赖进程名，也不需要挂载宿主 `/proc`。两个真实 Docker CUDA 任务的独立归属、取消后的记录失效与物理归还已通过。完整 science 训练及跨内核长期运行尚未验收。


### 用一个任务理解两种方案

我们要回答的其实是一个记账问题：**显卡报告某个进程用了 380 MiB，这笔显存该记到哪个任务名下？** 显卡信息来自 NVIDIA 的监控接口 NVML，两种方案都继续使用它。区别在于怎样查清进程身份和所属任务。

先看完整的数据流，图中的“名单”表示节点服务确认过的容器资源组，任务不能仅靠自报名字加入名单。

```text
                         [P1] PSRL 确认任务所属容器及资源组
                                      │
           ┌──────────────────────────┴─────────────────────────┐
           ↓                                                    ↓
  [P2] procfs：按需查档案                           [P3] eBPF：在内核中维护记录
  查宿主 /proc/PID/stat 和 cgroup                   注册本轮任务的资源组
  得到出生时间和所属资源组                          记录宿主 PID、出生时间、归属
           │                                                    │
           └──────────────────────────┬─────────────────────────┘
                                      ↓
  [P4] NVML：PID、显存、GPU 活动 → 核对身份 → [P5] 任务遥测与预算核销
                                      │
                                      ↓
                       [P6] 命令结束，确认物理释放，再归还预算
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与函数 | 源码 |
|---|---|---|
| P1 | `_command_container` 校验租约标签，`DeviceCgroupFence` 校验设备和资源组 | [command_gpu_node.py:181](../../../../psrl/workers/env_service/command_gpu_node.py#L181) |
| P2 | `process_identities` 读取宿主进程档案 | [process_identity.py:8](../../../../psrl/workers/env_service/process_identity.py#L8) |
| P3 | `BpfIdentity.bind` 注册任务，内核 `capture` 更新记录 | [bpf_identity.py:145](../../../../psrl/workers/env_service/bpf_identity.py#L145)、[bpf_identity.bpf.c:57](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L57) |
| P4 | `NvmlProbe.read_process_snapshot` 取得 GPU 数据 | [probe.py:109](../../../../psrl/workers/env_service/probe.py#L109) |
| P5 | `TaskGpuTelemetry.sample` 过滤样本，`read_command_memory` 核销已兑现的显存预约 | [task_telemetry.py:101](../../../../psrl/workers/env_service/task_telemetry.py#L101)、[read_command_memory:10](../../../../psrl/workers/env_service/task_telemetry.py#L10) |
| P6 | `revoke_gpu_command` 禁止新的设备打开，检查残留句柄与 GPU 进程，确认后注销本轮身份 | [command_gpu_node.py:207](../../../../psrl/workers/env_service/command_gpu_node.py#L207) |

**procfs 可以理解成查档案。** Linux 把进程信息显示为 `/proc` 里的文件。假设 NVML 返回宿主 PID 1000，节点就查这个进程的出生时间和所属资源组，确认属于任务 A 后，将它的显存记给 A。问题是当前 SSH 环境看到的是平台容器的档案范围，里面的 1000 不一定就是宿主的 1000，所以旧方案需要正确的宿主 procfs 挂载。

**eBPF 可以理解成在内核里做登记。** 节点先确认任务 A 的容器，然后将它的资源组登记给采集器。相关进程创建、执行 GPU 驱动调用时，内核程序记录真实的宿主进程号和出生时间。NVML 返回 PID 1000 后，我们查这份记录，就能判断它属于 A。进程退出或发生资源组迁移时，旧记录被删除，后续合适的事件可以重新建立记录。

这里的“资源组”就是 cgroup，Linux 用它把一组进程组织起来并限制资源。它不是程序的名字。两段程序即使都叫 `python`，甚至在各自容器里 PID 都是 8，也可以属于完全不同的资源组。

下面是教学数字：A 本轮命令的注册号为 101，内核记录某进程属于 A 的资源组，并带上 101。A 结束后，注册号注销。下一轮即使还是同一个容器，也会取得新的注册号 202，残留的 101 记录不能给新任务抵扣显存。再配上出生时间，可以避免只认 PID 导致进程号被重用时认错人。

| 比较项 | procfs 方案 | eBPF 方案 |
|---|---|---|
| 身份信息从哪来 | 采样时读取宿主进程文件 | 内核事件更新记录，采样时查记录 |
| GPU 显存从哪来 | NVML | 仍是 NVML，eBPF 不直接读取显存 |
| 当前节点的部署条件 | 需要宿主 PID 对应的 procfs 挂载 | 需要内核追踪权限、BTF 类型信息、clang 和 libbpf |
| 如何识别任务 | 宿主进程所属资源组 | 已注册资源组及本轮命令注册号 |
| 如何防止同号进程混淆 | 检查进程出生时间 | 检查内核出生时间，并在退出时使记录失效 |
| 如何处理故障 | 身份读不到时不给显存核销额度 | 采集器异常或注册失效时不给显存核销额度 |
| 主要成本 | 定期读取进程文件 | 内核事件处理、记录表和定期查询 |

不能仅凭机制不同就说 eBPF 更快。它解决了本机的挂载障碍，让逐任务记账能够工作。本次没有测两种方式的性能差异，也没有证明高并发训练更快。

新方案绕开的是**宿主 procfs 挂载**。容器设备权限检查仍会读取节点当前可见的 `/proc` 和 cgroup 文件。取消仍由执行器终止进程，eBPF 负责更新身份记录，不能把“记录删除”当成“进程已经被杀死或显存已释放”。这些代码如有理解问题，可以随时追问。

### SSH 私有节点为什么也是容器

SSH 是远程登录协议。SSH 服务可以运行在裸机、虚拟机或者容器里，能 SSH 登录、能用 GPU、看到 root 提示符，都不能单独判断自己位于物理宿主系统。

下面是依据本节点现场信息得到的部署层次图。平台调度器和 SSH 服务本身不是 PSRL 仓库实现，图中这部分对应现场证据，没有可引用的本地平台源码。

```text
[A] 宿主 Linux 内核、NVIDIA 驱动、GPU
 │   NVML 使用这一层的进程编号
 │
 └─ [B] 平台分配的运行容器，也就是 SSH 登录后的环境
      │   PID 1 为 taijirun，控制组属于 Kubernetes Pod
      │   PSRL driver、节点服务、Docker daemon 在这里运行
      │
      └─ [C] Docker 创建的 science / 验证任务容器
              独立文件系统与进程编号视图，按配置访问 GPU
```

图例：图里每个部分对应的证据与源码位置。

| 编号 | 含义 | 对应证据或实现 |
|---|---|---|
| A | GPU 进程的宿主编号来自内核，与 NVML 对账 | [真实 GPU 对账结果](../../env-service-verification/results/bpf-process-identity-20261027/gpu-python3/result.json) |
| B | 平台容器的 PID、用户、挂载命名空间，PID 1 与控制组 | [只读预检结果](../../env-service-verification/results/bpf-process-identity-20261027/preflight.json)，读取视图的可复用函数 [namespace_view](../../env-service-verification/scripts/host_proc_diagnostic.py#L15) |
| C | 本次创建、限制和回收内部 Docker 容器 | [verify_bpf_gpu_identity.py:17 `main`](../../env-service-verification/scripts/verify_bpf_gpu_identity.py#L17) |

现场证据是相互印证的：SSH 环境的 PID 命名空间为 `4026549832`，不是初始宿主视图。PID 1 的程序名为 `taijirun`，自己的控制组路径含有 `kubepods/burstable/pod…/容器ID`。Kubernetes 是部署容器的平台系统，Pod 是它的一种工作负载组织单元。仅一个进程名不足以下结论，结合命名空间、控制组和下文真实进程的三层编号，才能判断这里确实有外层容器。

“私有”描述的是平台资源分配与使用权限，不等同于“裸机登录”或“完整宿主管理员权限”。这些结果没有证明平台是否把整台物理服务器独占给当前实例，也没有排除宿主操作系统底下还存在其他虚拟化层。多个实例是否共享 GPU、CPU 或网络，需要平台配额与部署信息确认，不能从一个 SSH 地址反推。

容器共享所在宿主的内核，通过命名空间隔离进程等可见范围，通过控制组限制资源。因此容器里可以运行真实 CUDA，同时仍看不到宿主的完整进程列表。内部 Docker 的 `--pid=host` 中的 host 是相对它的 daemon 而言。在这里，它不能据此自动越过外层平台容器，变成物理宿主视图。

### 先看当前链路缺什么

procfs 是 Linux 将进程信息暴露为文件的接口，常见挂载点是 `/proc`。我们读取它不是为了给任务随机取名，而是为了查清 GPU 上的进程属于谁、是否仍是原来的那个进程。

下面展示 procfs 后端遇到的部署障碍，问号表示它缺少的对应关系。可选 eBPF 后端见后文统一后端章节。右侧 pidfd 是独立验证的生命周期能力，尚未进入生产执行器。

```text
[1] PSRL 创建 job_id
          ↓
[2] 容器执行器启动命令 → 本地进程 → [5] pidfd 稳定引用与退出事件
          │                              （独立验证，未接入生产）
          │ 任务归属
          ? 需要可信映射
          │
[3] NVML → 宿主 PID + GPU UUID + 显存/活动采样
          ↓
[4] 归属检查：进程出生信息 + 所属资源组 → 任务统计
          ↑
    当前依赖宿主 PID 对应的 procfs
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与函数 | 源码 |
|---|---|---|
| 1 | `GpuJobManager.start` 分配作业 UUID，并异步提交执行 | [gpu_jobs.py:60](../../../../psrl/workers/env_service/gpu_jobs.py#L60) |
| 2 | `run` 启动容器命令，`identity` 和 `cancel` 管理本地进程身份与取消 | [gpu_job_process.py:35](../../../../psrl/workers/env_service/gpu_job_process.py#L35)、[identity:12](../../../../psrl/workers/env_service/gpu_job_process.py#L12) |
| 3 | `NvmlProbe.read_process_snapshot` 取得按 PID 索引的 GPU 样本 | [probe.py:109](../../../../psrl/workers/env_service/probe.py#L109) |
| 4 | `TaskGpuTelemetry.sample` 调用 `_owners`，由所选后端过滤归属与陈旧样本 | [task_telemetry.py:101](../../../../psrl/workers/env_service/task_telemetry.py#L101)、[procfs 查询](../../../../psrl/workers/env_service/process_identity.py#L8) |
| 5 | 演示 `probe` 创建子进程并观察 pidfd，不调用生产执行器 | [process_identity_without_proc.py:46](../../env-service-verification/scripts/process_identity_without_proc.py#L46) |

本地进程取消使用的 `/proc` 与 GPU 遥测需要的宿主 `/proc` 是不同用途。即使把前者改成 pidfd，也不会自动补齐后者的映射。这些代码如果有理解上的问题，可以随时追问。

### 唯一编号究竟指什么

PID 是进程号，但有两个限制：进程退出后号码可以被重用，同一进程在不同 PID 命名空间中可以有不同号码。命名空间就是容器用来隔离进程编号和可见范围的机制。

以下全部数字为教学假设，不是实验结果。

| 视角或标识 | Science 作业 A | Science 作业 B | 回答的问题 |
|---|---|---|---|
| PSRL 作业编号 | `job-A` | `job-B` | 哪一次逻辑作业 |
| 节点运行容器看到的 PID | 300 | 301 | 本地哪个进程 |
| 物理宿主 PID | 2000 | 2001 | NVML 记录里的哪个进程 |
| 同一监督器持有的 pidfd 数字 | 7 | 8 | 监督器引用的具体进程对象 |
| NVML 显存记录 | PID 2000 占 512 MiB | PID 2001 占 1024 MiB | 对应宿主进程的 GPU 占用 |

如果只知道 `job-A → 本地 PID 300 → pidfd 7`，仍不能从两条 NVML 记录中确定哪条属于 A。需要额外得到 `本地 300 ↔ 宿主 2000`，或者直接得到 `job-A ↔ 宿主 2000`，才能把 512 MiB 记到 A 名下。

UUID 是应用生成的高概率不重复标识。PSRL 已经为每个 GPU 作业生成 UUID，但一个作业可能启动多个进程，驱动也不会自动知道这个 UUID。多生成一个 UUID 无法填上对应关系。

GPU UUID 标识设备。NVML 的 GPU instance ID 和 compute instance ID 标识 MIG 分区，MIG 是将一张 GPU 划分成多个实例的能力。这些也都不是进程的通用唯一编号。

### pidfd 怎样避免认错进程

pidfd 意为“进程文件描述符”。文件描述符是当前进程持有资源引用的整数入口，资源也可以是套接字或进程，并不要求它对应磁盘文件。内核通过 pidfd 维持对某个具体进程身份的引用。

假设 A 的本地 PID 为 300，监督器在 A 活着时调用 `pidfd_open(300, 0)`，得到描述符 7。随后 A 退出，内核以后把 PID 300 分给 B。再次按数字 PID 查找时可能找到 B，但仍持有的旧 pidfd 不会改指向 B。通过它发送信号、等待退出，可以避免“认错同号的新进程”。该接口不需要挂载 procfs。

**数字 7 本身不是全局唯一编号。** 另一个监督器也可以有描述符 7，关闭后本进程也能重用 7。不能把 `7` 写到 JSON，让另一个进程拿它当同一个句柄。设计上应由监督器持续持有句柄，以 `job_id → 受控进程引用` 建立表，需要跨进程共享时传递实际文件描述符，或者调用监督器的控制接口。

还有三项边界：

- 获取句柄时必须防止竞态。先查一个陌生 PID，等它退出并被重用之后才打开，仍可能拿错对象。本次演示让自己的子进程阻塞等待，打开后才允许退出。生产启动器也需要受控握手或原子创建机制。
- pidfd 按调用者可见的 PID 查找进程。在外层平台容器中直接把 NVML 的宿主 PID 传进去，不会自动跨到物理宿主视图。
- 一个 pidfd 不代表整棵进程树，也不等同于进程组。父进程退出不证明所有子进程和 GPU 上下文都已释放，不能直接拿它替换现有整组取消及资源回收确认。

### 不挂载宿主 procfs 的替代方案

关键是引入一个能同时关联“可信任务身份”与“驱动使用的宿主进程号”的观察点。宿主注册服务和程序内采样仍是候选方式，eBPF 已实现本页后文的可选后端。

| 方式 | 怎样建立对应关系 | 前提与边界 |
|---|---|---|
| 宿主进程注册服务 | 真实 GPU 进程连接本机 Unix 套接字，宿主服务读取内核提供的对端身份，再绑定受信任的作业授权 | 服务须位于物理宿主 PID 视图，需平台部署及套接字访问。可以不向实验容器暴露宿主 `/proc` |
| eBPF 内核追踪 | 在相关内核事件上记录宿主进程号、出生时间与注册资源组 | 已接入节点接口并完成双容器 CUDA 验证，其他内核及长期运行仍需验收 |
| 程序内 GPU 采样 | CUDA 程序把自身活动关联到传入的 job_id，例如使用 CUPTI 分析接口 | 需要任务配合，必须覆盖子进程。它改变采样方式，不是现有 NVML 归属的直接替换 |

**宿主注册服务的具体过程**：可信启动器为 `job-A` 发放一次性注册凭据，真正运行 CUDA 的进程连接宿主服务。Unix 套接字是同一内核中进程之间的通信接口，`SO_PEERCRED` 能提供连接对端的进程身份。宿主服务取得宿主 PID 2000，并在有生命周期保护的握手中绑定 `job-A`，之后就能关联 NVML 的 PID 2000。进程结束后服务撤销该绑定。

该设计仍要处理凭据认证、注册与退出竞态、子进程注册、监督器重启和旧采样失效。仅让一个代理进程替 CUDA 进程连接，得到的是代理身份。仅把服务放在当前外层平台容器里，也仍然只得到该容器的 PID 视图。实现前必须把这些边界纳入协议，不能把一段 socket 示例当作完整可信记账。

**eBPF 的具体含义**：它允许受约束的小程序在内核事件发生时运行。内核提供的 `bpf_get_current_pid_tgid` 可取得当前线程与进程的内核编号，其中高 32 位为线程组号，通常对应用户态所说的进程 PID。采集器还需关联任务所属控制组及出生/退出事件，才能形成完整账本。控制组即 cgroup，是 Linux 用来组织和约束一组进程的机制。

本项目当前使用的设备控制组属于 cgroup v1，不能假定某个返回 cgroup v2 ID 的 helper 可以直接作为现有归属键。内核版本、回移功能、用户命名空间和系统调用过滤都要实际核验。下面的 pidfd 实验只读权限位，后续独立 eBPF 实验才验证了程序加载、raw tracepoint 挂接以及真实 GPU 对账，不能把这两批证据混为一谈。

程序内采样中，CUPTI 是 NVIDIA 的 CUDA 性能分析接口。它的 context ID 用来区分进程内 CUDA 上下文，不是跨进程通用身份证，也不自动覆盖 NVML 统计的全部显存。对任意任务的自报数据，不能不经验证就用作可信 GPU 奖励。

### 真实 CPU 验证与复现

本节是 2026-10-27 私有节点实测。执行器以 `nohup`、独立会话和 60 秒外层超时在后台运行，日志与退出码持久保存。没有启动 Docker、CUDA 或训练，也没有读取宿主 `/proc` 或加载内核追踪程序。

最初直接调用 `os.pidfd_open` 失败，当前 Python 构建没有暴露这个属性。脚本通过 Linux 系统调用完成验证后通过。这说明 Python 包装接口缺失不能直接判断为内核不支持。

| 检查 | 实测观察 |
|---|---|
| 内核 | `5.4.241-1-tlinux4-0017.7` |
| pidfd 获取 | Linux 系统调用成功 |
| 子进程活着时的退出事件 | 无 |
| 子进程正常退出后的事件 | 有 |
| 对已回收进程的句柄发送信号 0 | 返回进程不存在 |
| 清理 | 子进程已回收，pidfd 已关闭 |
| NVML 宿主 PID 映射 | 未验证 |

信号 0 只是探测目标和权限，不实际发送终止信号。本次验证了真实生命周期事件，没有强行制造 PID 重用，没有测 GPU 归属或性能收益。

`capget` 系统调用读到 `SYS_ADMIN`、`SYS_PTRACE` 及标为 `PERFMON`、`BPF` 的位均为真。但上游 `CAP_BPF` 和 `CAP_PERFMON` 是 Linux 5.8 才引入，不能从这些位判断定制 5.4 内核的全部追踪能力，也不能判断权限属于哪个用户命名空间。必须另做受控能力验收。

在已激活项目环境后，从仓库根目录运行以下自包含 CPU 实验：

```bash
timeout 15 python docs/env-worker-deep-dive/env-service-verification/scripts/process_identity_without_proc.py
```

它只创建一个等待读取输入的 Python 子进程。父进程打开句柄，确认没有退出事件，再向子进程写入一个字节让它正常退出，最后检查退出事件并关闭句柄。预期 JSON 包含 `status: passed`、`exit_event_while_alive: false`、`exit_event_after_exit: true`、`child_reaped: true`。不支持接口或权限不足时退出码为 3，并保留错误。系统调用适配限定原生 64 位 x86_64 或 aarch64，实际验证的是目标节点的环境。

证据：[通过结果](../../env-service-verification/results/process-identity-20261027/result.json)、[Python 接口缺失记录](../../env-service-verification/results/process-identity-20261027/python-wrapper-unavailable.json)、[后台日志](../../env-service-verification/results/process-identity-20261027/background.log)。

### 内核追踪与真实 Docker CUDA 对账

本节为 2026-10-27 单进程原型实测，采用随机进程名识别受控对象，不是后文已经接入框架的资源组后端。目标限定为“不挂载宿主 `/proc`，取得自己启动的进程的宿主 PID，并关联真实 GPU 样本”。CPU 阶段外层超时 50 秒，GPU 阶段每次 100 秒，内部 CUDA 命令最多 45 秒。不训练模型，不运行 science rollout，不测吞吐或调度收益。

实验链路如下。它与上面的现有生产链路相互独立。

```text
[6] Python 验证器 → 创建独立 Docker 容器 → [7] CUDA 内核计算，等待握手
        │                                                   │
        ├─ 启动 [8] C 采集器 → 加载临时 eBPF 程序             │
        │                         ↓                         │
        │                  sys_enter 内核事件 ← 设置随机进程名后执行系统调用
        │                         ↓
        │                  只记录匹配名字的宿主 PID
        │                         ↓
        └─ [9] NVML 进程记录 ← 核对 PID、显存与内核结果
                ↓
          [10] 允许任务退出 → 删除容器 → 核实 GPU 释放
```

图例及跟着图读源码：

| 顺序与编号 | 函数、源码 | 输入、输出或状态变化 |
|---|---|---|
| 入口 6 | [验证器 `main`](../../env-service-verification/scripts/verify_bpf_gpu_identity.py#L17) | 检查 Docker 和 GPU，创建带唯一归属标签、1 CPU、512 MiB 内存上限的容器 |
| 容器侧并行执行 7 | [CUDA `main`](../../env-service-verification/scripts/bpf_cuda_identity.py#L11) | 分配 64 MiB 显存，执行内核并检查答案 42，写入就绪文件，等待 begin 信号 |
| 采集器加载和挂接 8 | [C `main`](../../env-service-verification/scripts/bpf_process_identity.c#L19) | 创建一项 BPF map，加载随机名字过滤程序，挂到 sys_enter，输出 READY |
| 内核回调 8 | [进程号 helper 调用](../../env-service-verification/scripts/bpf_process_identity.c#L57) | 命中随机名字后取得宿主线程组号与线程号，写入 map。其他事件不写入样本 |
| 回到验证器 9 | [NVML 对账](../../env-service-verification/scripts/verify_bpf_gpu_identity.py#L108) | 读取 map 捕获的宿主 PID，要求 NVML 有同 PID 且显存至少覆盖申请量的记录 |
| 收尾 10 | [验证器 finally](../../env-service-verification/scripts/verify_bpf_gpu_identity.py#L124)、[C 句柄关闭](../../env-service-verification/scripts/bpf_process_identity.c#L109) | 回收采集进程，删除本次容器，轮询进程消失及显存回落。临时 BPF 未固定到文件系统，关闭附着句柄后解除追踪 |

raw tracepoint 是直接挂接内核事件的接口。本实验挂接 `sys_enter`，也就是进程进入系统调用的事件。这样无需挂载 tracefs 或宿主 procfs。BPF map 是内核程序与用户态采集器交换小量数据的表，本次只有一个槽位，保存 64 位的线程组号和线程号。线程组号通常就是用户态所说的进程 PID。

采集程序在系统调用事件上执行，先检查完整的 15 字符随机进程名。只有命中本次标记才保存编号，捕获后立即退出并解除挂接。这不是按名字授权的生产方案，也不是高频全节点长期追踪的性能验收。名字可被任务修改或模仿，正式实现必须使用可信启动关系及控制组归属。

CPU 阶段已经捕获到同一个子进程的两种编号：外层平台容器 PID 为 `2521524`，内核宿主 PID 为 `158617`。随后 Docker CUDA 阶段得到以下实测：

| 观察视角 | 同一个 CUDA 进程的 PID |
|---|---|
| 内部 Docker 容器自行调用 getpid | `7` |
| 外层平台容器中 Docker top 返回 | `2522228` |
| eBPF 内核 helper 返回 | `203842` |
| NVML GPU 进程列表返回 | `203842` |

内核执行得到 `42`，申请显存 64 MiB，NVML 的该进程占用为 380 MiB。后者包括 CUDA 上下文等开销，不应要求它恰好等于申请量。只根据“占用了 380 MiB”猜身份不可靠，本次依据是内核取得的宿主 PID 与 NVML 精确一致。

容器退出码为 0，删除核验通过，目标 GPU 的计算进程列表恢复为空。设备总占用回到 586.25 MiB，低于启动前 589.5625 MiB。这一剩余设备级占用不是本实验进程显存，不写成“清理后显存为零”。采集器已回收，临时目录和附着句柄已释放。

第一次 GPU 启动因镜像没有 `python` 命令而退出，未进入 CUDA。改用 `python3` 后只重试 GPU 组一次并通过，CPU 组没有重跑。失败组容器已删除、显存释放已核验。镜像命令名需要现场核对，不能沿用外层 conda 的 `python` 假设。

证据：[CPU 映射](../../env-service-verification/results/bpf-process-identity-20261027/cpu-result.json)、[GPU 预检](../../env-service-verification/results/bpf-process-identity-20261027/gpu-preflight.json)、[首次启动失败](../../env-service-verification/results/bpf-process-identity-20261027/gpu/result.json)、[GPU 成功与清理](../../env-service-verification/results/bpf-process-identity-20261027/gpu-python3/result.json)。映射取号不读取宿主 procfs。只读部署诊断使用了当前容器的 procfs，Docker top 仅作为外层 PID 的辅助展示，不参与宿主 PID 的计算。

复现前激活项目环境，准备已有 CUDA 验证镜像，设置 `PROBE_IMAGE` 为该镜像名，`PROBE_OUTPUT` 为一个尚不存在的本地输出目录。命令从仓库根目录运行，要求 gcc、Linux BPF 头文件、NVML Python 包和已经验收的追踪权限。镜像需要 `python3` 和 `timeout`，驱动库按节点版本只读挂载。

```bash
probe_binary_dir=$(mktemp -d /tmp/psrl-bpf-repro-XXXXXX)
gcc -Wall -Wextra -Werror -O2 \
  docs/env-worker-deep-dive/env-service-verification/scripts/bpf_process_identity.c \
  -o "$probe_binary_dir/collector"
python docs/env-worker-deep-dive/env-service-verification/scripts/verify_bpf_gpu_identity.py \
  --binary "$probe_binary_dir/collector" \
  --image "$PROBE_IMAGE" --output "$PROBE_OUTPUT"
rm -rf "$probe_binary_dir"
```

这是会创建一个 GPU 容器的复现命令，不是纯只读诊断。本次实际运行采用后台执行器、独立会话、外层超时和只清理所属容器的退出钩子，执行文件及日志保存在上述结果目录。仅需学习稳定进程句柄时，运行前一节 CPU 脚本即可。这些代码如果有理解上的问题，可以随时追问。

### 统一后端怎样接入框架

新增配置在 `psrl.env_service` 下，默认保留 `procfs`。在已经验收追踪能力的节点上，可以显式选择：

```yaml
psrl:
  env_service:
    gpu_identity_backend: ebpf
```

这个配置决定“去哪里查进程身份”，不决定任务采用 command 还是 episode 调度模式，也不决定 GPU 显存是否常驻。当前接入的是已有 command 任务遥测和显存核销路径，不能把它理解成所有 episode 任务都新增了命令级遥测。

Router 将配置传给对应节点服务，节点按需创建一个共享采集器。选择 `ebpf` 后，加载或挂接失败会直接报告错误，不自动切回 procfs，也不自动进入 observe。`harbor.gpu_attribution_mode` 是另一个独立配置，控制 science 在可信归因能力不足时是否允许继续实验。

跟着图读源码，按实际调用路径展开：

| 步骤 | 图中编号 | 文件与函数 | 输入、输出或状态变化 |
|---|---|---|---|
| 配置传递 | P1 | [Router 创建节点](../../../../psrl/workers/env_service/router.py#L304) | 将 procfs 或 ebpf 选择传给执行节点 |
| 初始化 | P3 | [`_gpu_identity`](../../../../psrl/workers/env_service/command_gpu_node.py#L108)、[`BpfIdentity._load`](../../../../psrl/workers/env_service/bpf_identity.py#L66) | 读取 BTF 字段位置，编译 BPF，使用 libbpf 加载并挂接事件。初始化取消后也关闭迟到的采集器 |
| 命令注册 | P1、P3 | [`grant_gpu_command`](../../../../psrl/workers/env_service/command_gpu_node.py#L192)、[`bind`](../../../../psrl/workers/env_service/bpf_identity.py#L145) | 先确认容器与租约，登记资源组和随机本轮注册号，再开放 GPU 设备访问 |
| 并行的内核事件 | P3 | [`on_syscall`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L77)、[`on_fork`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L84)、[`on_exec`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L87) | 在驱动 ioctl、getpid、子进程创建及执行事件上更新受管进程记录。无需任务修改名字或自报宿主 PID |
| 退出与迁移分支 | P3 | [`on_exit`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L100)、[`on_move`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L103) | 删除该进程记录。资源组迁移后，后续事件按新的归属重新登记 |
| 采样与核销 | P4、P5 | [`BpfOwner.owners`](../../../../psrl/workers/env_service/bpf_identity.py#L187)、[`read_command_memory`](../../../../psrl/workers/env_service/task_telemetry.py#L10) | 核对资源组、注册号和出生时间，只用稳定身份的当前显存抵扣预约余量 |
| 归还与关闭 | P6 | [`revoke_gpu_command`](../../../../psrl/workers/env_service/command_gpu_node.py#L207)、[`EnvNodeService.close`](../../../../psrl/workers/env_service/node.py#L58) | 确认 GPU 句柄及属于本任务的 GPU 进程消失，注销命令。节点关闭时停止遥测并解除追踪 |

BTF 可以理解成内核附带的“数据结构说明书”。我们据此查找进程出生时间、资源组指针等字段的位置，不把这台机器的内存偏移写死到代码中。libbpf 是负责加载 BPF 程序、关联记录表和检查加载错误的库。clang 则把 C 程序编译成内核可执行的 BPF 指令。

本节点的 BTF 存在布局相同的重复类型定义，解析器允许相同布局，拒绝互相冲突的布局。clang 20 的默认指令选择与目标内核不完全兼容，因此明确编译为 BPF v1 指令集。加载检查失败时必须保留错误，不能解释成 GPU 任务自身 OOM。

实现边界需要明确：当前针对 cgroup v1 devices 控制组，检查最多 16 层祖先，支持最多 256 个同时注册的资源组、8192 条进程记录。内核记录表写入失败会将采集器标为异常，预算保持保守。注册资源组不能相互包含，避免一条进程被两个任务同时认领。注册目录保持打开，采样时检查路径身份和本轮注册号，关闭后旧记录不能继续提供额度。

这些限制不会把两个不同容器的并发变成“同一容器多个 GPU 命令并发”。同容器并发仍是独立 TODO。本后端也没有实现跨 Router 的统一全节点资源所有权协议。

### 统一后端的双容器实测

验收目标是“实际节点服务能区分两个 GPU 任务，并在一个任务退出后正确失效它的记录”。测试通过 `EnvNodeService` 的真实授权、遥测、显存查询和归还接口执行，未启动 Ray 集群、完整 science rollout 或训练。

后台执行器总预算 120 秒，创建两个独立 Docker 容器，各限制 1 CPU、512 MiB 系统内存，每个 CUDA 子进程显式申请 64 MiB 显存，CUDA 循环最多 20 秒。两个容器的父进程保持存活，分别启动一个可取消的 CUDA 子进程。这样可以检查“孩子退出但容器还活着”的情况。

| 项目 | 实测结果 |
|---|---|
| 两个任务的名字 | 都故意设置为 `same-gpu-name` |
| 两个任务在各自容器内的 PID | 都是 8 |
| 对应的宿主 PID | A 为 678508，B 为 678507 |
| CUDA 内核结果 | 两组均为 42 |
| 当前显存核销额度 | A、B 各 380 MiB，没有归属错误 |
| A 取消后 | A 的子进程退出码为 -15，内核身份记录消失 |
| B 在 A 取消后 | 仍占用并核销自己的 380 MiB |
| GPU 活动遥测 | 两组各取得 5 条有效活动样本，峰值利用率 99%，无遥测错误 |
| 最终收尾 | B 正常退出，两个容器删除，计算进程列表为空，采集器关闭 |

380 MiB 是 NVML 的实际进程占用，包含 CUDA 上下文，不能与程序申请的 64 MiB 混为一谈。设备级显存从约 774.6 MiB 降到取消 A 后约 389.1 MiB，最终回到启动前的 0.1875 MiB。本节使用框架的 NVML memory v2 口径，排除了驱动保留部分，不要直接与前节旧原型的设备总占用比较。

本轮取消动作由受控父进程向 A 的子进程发送 SIGTERM，框架随后执行真实的命令归还检查。它验证了退出后的记账和隔离结果，不等于验证了所有 Agent 工具的取消入口。没有为了产生更好看的指标扩大训练或重复 GPU 实验。

证据：[完整结果及清理](../../env-service-verification/results/bpf-backend-20261027/gpu/result.json)、[内核加载检查](../../env-service-verification/results/bpf-backend-20261027/load-result.json)、[CPU 回归日志](../../env-service-verification/results/bpf-backend-20261027/pytest-final.log)。CPU 回归覆盖陈旧注册号、进程出生身份改变、邻居归属、关闭后失效、采集中故障、初始化取消和节点关闭等情况，它们与真实 GPU 实测分开记录。

可运行脚本是 [verify_bpf_backend.py](../../env-service-verification/scripts/verify_bpf_backend.py)，容器负载为 [bpf_backend_workload.py](../../env-service-verification/scripts/bpf_backend_workload.py)。在具备上述依赖、已激活项目环境的执行节点，从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/verify_bpf_backend.py \
  --image "$PROBE_IMAGE" --output "$PROBE_OUTPUT"
```

`PROBE_IMAGE` 是已经准备好的 GPU 验证镜像，需要包含 python3 和 timeout。`PROBE_OUTPUT` 是尚不存在的输出目录。实际运行采用结果目录里的后台执行脚本，另设外层超时和仅所属容器的退出清理钩子。只需检查身份判断逻辑时，可运行 CPU 回归，无需 GPU：

```bash
pytest -q tests/env_service/test_process_identity.py tests/env_service/test_task_telemetry.py
```

### 跟着图读源码

这里分开列现有生产链路与独立实验，不将异步采样伪装成命令启动后的单线程调用。

| 步骤 | 图中编号 | 文件与函数 | 输入、输出或状态变化 |
|---|---|---|---|
| 提交作业 | 1 | [gpu_jobs.py:60 `start`](../../../../psrl/workers/env_service/gpu_jobs.py#L60) | 命令输入，生成 UUID，状态变为 queued，创建异步执行任务 |
| 容器内执行 | 2 | [gpu_job_process.py:35 `run`](../../../../psrl/workers/env_service/gpu_job_process.py#L35) | 命令输入，启动进程组，记录 PID 和出生信息，等待退出 |
| 并行采样 | 3 | [probe.py:109 `read_process_snapshot`](../../../../psrl/workers/env_service/probe.py#L109) | 设备与时间窗口输入，返回按 PID 的显存及利用率样本 |
| 检查与过滤 | 4 | [task_telemetry.py:101 `sample`](../../../../psrl/workers/env_service/task_telemetry.py#L101) | 检查所选后端，再取进程归属，过滤采样期间身份不稳定的进程 |
| 取消分支与本地收尾 | 2 | [gpu_job_process.py:20 `cancel`](../../../../psrl/workers/env_service/gpu_job_process.py#L20) | 校验出生信息后向进程组发信号。`run` 的 finally 移除身份文件，后续 GPU 物理释放仍需执行器确认 |
| 独立演示入口与清理 | 5 | [process_identity_without_proc.py:46 `probe`](../../env-service-verification/scripts/process_identity_without_proc.py#L46) | 自建子进程，取得 pidfd，观察退出，finally 回收子进程并关闭描述符 |

### 后续实现顺序与边界

独立进程归属后端已经接入节点服务，支持原有 procfs 和显式启用的 eBPF。真实双容器测试已达到本轮验收目标，因此停止 GPU 验证。本节之后的实现重点应是按实际接入需求验收，而不是为了证明可取号启动完整训练。

尚未验收的范围包括：其他内核和 cgroup v2、采集器或 Router 重启时恢复所有正在运行的任务、大规模长期事件开销、真实强制 PID 重用、所有资源组迁移路径，以及完整 science 原生评分链路。本轮没有进行这些实验，也没有声称训练吞吐提升。

默认配置仍为 procfs。后续 science 实验若显式启用 ebpf，应先通过实际执行节点的严格归因预检，并记录所选后端。已有 observe 结果不能改标为可信 GPU 评分。完整 science 严格模式验收之前，仍保留[观察模式说明](../../experiments/10-进程归属_Procfs挂载障碍与处理.md)中的结果标注和预算保守规则。

### 接口资料

- [Linux pidfd_open 手册](https://man7.org/linux/man-pages/man2/pidfd_open.2.html)：稳定进程引用、退出事件及不依赖 procfs 的接口。
- [Linux PID 命名空间手册](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html)：同一进程在不同视图中的编号及可见范围。
- [Linux Unix 套接字手册](https://man7.org/linux/man-pages/man7/unix.7.html)：对端凭据和文件描述符传递。
- [Linux BPF helpers 手册](https://man7.org/linux/man-pages/man7/bpf-helpers.7.html)：内核事件程序可调用的进程标识接口。
- [Linux capabilities 手册](https://man7.org/linux/man-pages/man7/capabilities.7.html)：权限位的用途与版本边界。
- [NVIDIA NVML 进程信息结构](https://docs.nvidia.com/deploy/archive/R510/nvml-api/structnvmlProcessInfo__t.html)：当前讨论的 PID、显存与 MIG 字段。
- [NVIDIA CUPTI 接口](https://docs.nvidia.com/cupti-python/cupti.html)：进程内 CUDA 上下文与采样接口，不能直接替代宿主 PID 映射。

### PID 为什么分层及 MPS 应该用哪个编号

本节接着上面的“SSH 私有节点为什么也是容器”和“唯一编号究竟指什么”，也对应 [GPU 遥测文档第 8.2 节](../01-架构与执行/04-GPU作业_生命周期与任务遥测.md#82-pid-为什么会对不上)。这三处讲的是同一个障碍：**编号属于某个观察范围，不能拿到另一个范围里直接用。**

先看全局层次。下图为 Linux 概念图，数字是教学假设，不是新增实测。

```text
[A] 物理宿主机的 PID 空间
    同一个 Python 进程 P 在这里叫 2000
    │
    └── [B] SSH 登录的私有节点，实际是平台容器
         P 在这里叫 300，内层 Docker daemon 在这里运行
         │
         ├── [C] science 任务容器
         │    P 在这里叫 17，任务通过 getpid() 得到 17
         │
         └── [D] MPS 服务容器
              默认独立空间：没有给同级 C 中的 P 分配本地编号
              使用 --pid=host：共享 B 的视图，可以用 300 找到 P
```

图例：图里每个部分对应的源码位置。

| 图中部分 | 对应机制或实现 |
|---|---|
| A、B、C 的分层编号 | Linux 内核机制，不是 PSRL 自己重写 PID。定义见 [PID namespace 手册](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html) |
| A 视角的 GPU 观测 | [NvmlProbe.read_gpu_snapshot](../../../../psrl/workers/env_service/probe.py#L85)，读取驱动报告的进程编号和显存 |
| C 的任务进程及本地取消 | [gpu_job_process.identity](../../../../psrl/workers/env_service/gpu_job_process.py#L12)、[cancel](../../../../psrl/workers/env_service/gpu_job_process.py#L20) |
| D 的服务视图选择 | [MPS Docker 探针](../../env-service-verification/scripts/mps_docker_probe.py)，只让服务使用 Docker 所在节点的 PID 视图 |
| B 到 D 的身份核对 | [MpsClientController](../../../../psrl/workers/env_service/mps.py#L93)，核对本地出生信息、namespace、服务代次和客户端连接 |

**一个进程为什么有三个号码。** PID 是 Process ID，即进程编号。namespace 是命名空间，可以理解为一张独立的“名字对应对象”表。创建进程 P 时，内核在它所属的空间及各级祖先空间分别登记编号。CPU 实际执行的仍是同一个进程，显存也不是复制三份。任务内部调用 `getpid()` 时得到 17，B 管理这个进程用 300，A 管理它用 2000。不同层也可能碰巧分到相同数字，所以不能靠数字大小猜层级。

**为什么这样设计。** 第一，为每个容器提供独立的进程世界，容器可以有自己的 PID 1 和完整进程树，不必依赖宿主当前已经用了哪些号码。第二，容器内部的进程不能仅靠遍历 PID 就看到或操作外部进程，减少信息暴露和误操作范围。第三，迁移或恢复容器时，可以保留内部进程编号，而不要求新宿主的全局编号恰好空闲。这是分层命名的用途，不表示 PID namespace 单独就实现了全部安全隔离，还需要权限、用户空间隔离和资源限制。[Linux 手册](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html)

**可见方向不是对称的。** A 可以看见自己的进程和 B、C 的后代进程，B 可以看见 C，C 不能通过本地 PID 操作 B 的其他进程。同级的独立 D 也看不到 C。这里“看见”指能在自己的编号空间定位到对象，是否有权发送信号还要检查权限。MPS 显示 PID 0、namespace 为问号是本次部署的不可定位现象，不能把 0 当成一个可取消的正常任务编号。

**为什么 `--pid=host` 还不是物理宿主。** `host` 是相对启动 Docker 的 daemon 而言。我们的内层 daemon 在 B，服务共享到的是 B。它获得 300 这个编号后就能管理 C，但不会自动获得 A 的 2000。把 B 的 `/proc` 挂载成 `/host/proc` 也只改访问路径，不会将 B 的进程表升级成 A 的表。

**`/proc`、NSpid 与 pidfd 分别是什么。** `/proc` 是内核提供的进程信息文件视图，展示哪层编号与该 procfs 挂载的 PID 空间有关。若从 A 的 procfs 读 P 的 `status`，`NSpid` 可能显示 `2000 300 17`，从 B 的视图则只显示 `300 17`。它不能倒推自己看不到的祖先号码。pidfd 是一个进程引用句柄，适合避免进程退出后 PID 复用导致误操作，但不会自动告诉你另一个 namespace 里的号码。查看自己当前可见的层次可以运行 `grep '^NSpid:' /proc/self/status`，输出数字数量取决于环境，不能把最左边一项无条件称为物理宿主 PID。

**回到 MPS，三个问题要分别回答。**

| 操作 | 应当使用的编号或身份 | 错用的后果 |
|---|---|---|
| 任务在自身容器里管理子进程 | C 中的 PID，配出生身份或 pidfd | 把宿主 PID 当本地 PID，可能找不到或找错进程 |
| MPS 定向取消客户端 | D 的服务所见 PID，本例为 B 中的 300 | 把 NVML 的 2000 发过去，可能无效或误指别的客户端 |
| 按 NVML 数据给任务记账 | 驱动报告的 PID，经可信归属链连接到 job | 没找到本地 `/proc/2000` 不代表显存为零 |

**新增控制组件如何工作。** 探针通过 Docker inspect 取得本次自建容器的进程号，不再仅凭任务 stdout 自报 namespace 来选择取消对象。控制组件读取该本地 PID 的出生 ticks 和 namespace，与 MPS 客户端表匹配，再记录服务进程出生信息。取消时要求同一 job 和 epoch，并复查客户端及服务身份，收到 MPS 成功确认之后才允许后续进程回收。epoch 是某次授权的代号，防止旧请求作用于后续一轮。

跟着图读源码：B 的 Docker 确定进程 → [bind](../../../../psrl/workers/env_service/mps.py#L100) 连接本地身份和 D 的客户端记录 → [validate](../../../../psrl/workers/env_service/mps.py#L113) 检查进程及服务是否变更 → [terminate](../../../../psrl/workers/env_service/mps.py#L123) 向 D 请求定向终止 → [统一执行器清理顺序](../../../../psrl/workers/env_service/gpu_execution.py#L151) 依次执行 MPS 确认、进程回收、预约取消。A 与 B 的 GPU 工作是并行存续的，不是这条控制流程中的串行任务。

这些代码如果有理解上的问题，可以随时追问。这个组件只解决可信、受控客户端的控制身份校验，还没有补齐 NVML 最外层身份映射，也没有消除校验与发送指令间的所有竞态。MPS 取消接口接受裸 PID，正式投产还需要停止新接入、串行管理生命周期，并验证派生进程与重连，不能将这次校验称为任意不可信任务下的原子身份保证。

**可运行教学模拟。** 下例只模拟不同名字表的查找，不创建真实 namespace，也不会给任何进程发信号。

```bash
python - <<'PY'
views = {'physical': {2000: 'P'}, 'node': {300: 'P'}, 'task': {17: 'P'}}
for view, table in views.items():
    print(view, next(iter(table)), '->', next(iter(table.values())))
assert views['node'].get(2000) is None
assert views['node'][300] == views['physical'][2000] == views['task'][17]
print('MPS in node view: use 300, not 2000 or 17')
PY
```

预期前三行的号码分别为 2000、300、17，全部指向 P，最后说明 MPS 在节点视图必须使用 300。真实 Docker GPU 入口与条件见 [MPS 接入探索](05-训推共享_vLLM余量与容量调度.md#13-docker-mps-接入探索与框架落点)。


> **核心结论**：GPU 资源管理需要连续回答三个问题：进程属于哪个任务，给它承诺了多少资源，结束后资源是否真的释放。本文沿这条链依次解释身份、预算、取消和清理，并保留完整实验。

例如，任务容器中看到 PID 8，GPU 驱动却报告宿主 PID 42008。eBPF 是在 Linux 内核中运行受校验小程序的机制，可跟踪真实进程身份及其 cgroup（进程资源组），让调度器把驱动观测记到正确任务上。正文就地解释 eBPF、BTF（内核类型描述）和相关术语；初读可先沿任务例子读，再深入实现。


## GPU 身份、归属与显存会计

> **本节要点**：先确定 GPU 进程属于哪个任务，再把它的实测显存与未兑现预约合并，才能既保护承诺又避免重复记账。下面用同一任务在容器与宿主看到不同 PID 的例子，逐步解释内核身份、容器预算和节点恢复，随后说明[取消与清理](#取消安全与可运行实验)。


### 1. 先还原一个具体问题：同一个容器里，这笔显存该记给谁

psrl 的 `env_service` 要做一件听起来很朴素的事：一张物理 GPU 上同时跑着好几个任务的命令（比如 4 个强化学习 episode 同时在同一张卡上执行用户代码），系统需要知道"这张卡现在用掉的显存，哪部分是 A 任务的、哪部分是 B 任务的"，这样才能在显存快满的时候正确地排队、在任务结束后正确地释放。

最直接的办法是：查显存用量时顺手查一下"这块显存是哪个进程在用"（NVIDIA 的 NVML 库能告诉你"PID 1234 用了 2GB"），再去确认"PID 1234 是不是属于我管理的这个任务"。第二步"确认 PID 属于谁"听起来应该很简单——Linux 的 `/proc/<pid>/cgroup` 文件就能告诉你这个进程属于哪个 cgroup（后面会解释 cgroup 是什么）。但这里有一个前提条件很容易被忽略：**你读到的 `/proc`，必须是"宿主机视角"的 `/proc`，PID 的数字才有意义。**

问题就出在这个前提上：psrl 的 `env_service` 节点服务（负责管理 GPU、确认身份的那个进程）现在自己也跑在一个容器里（文档称之为"SSH 私有节点"——一种给单个用户独占、通过 SSH 登录使用的容器化开发环境）。容器有自己的 PID 命名空间（namespace，一种让一组进程"看到"一套独立编号体系的内核机制），**容器内部看到的 PID 8，和宿主机上真正的 PID 8，通常不是同一个进程**。如果 `env_service` 直接读自己容器内的 `/proc`，它看到的 PID 体系就是"自己的"，而 NVML 返回的"PID 1234 用了 2GB"里的 1234，是**宿主机内核**分配的真实 PID——两套编号完全不在一个坐标系里，字面匹配毫无意义。

procfs 后端通过 `PSRL_GPU_HOST_PROC` 选择宿主视角的挂载，再用 [`check_support()`](../../../../psrl/workers/env_service/process_identity.py#L36) 核对 PID 1 的命名空间 inode。检查不符合预期会明确报错。随后 [`process_identities()`](../../../../psrl/workers/env_service/process_identity.py#L8) 读取进程出生时间和 devices cgroup，将 NVML 的 PID 与任务归属关联。eBPF 后端解决的是本机无法取得正确宿主挂载的部署障碍：在已确认的内核权限与可信容器注册条件下，从内核事件维护身份记录。两种后端的完整对比已在[前面的任务例子](#用一个任务理解两种方案)展开。

cgroup 将一组进程归入同一资源控制范围，PID 命名空间则改变进程看到的编号与可见范围。本机的三层编号和 pidfd 例子已在[进程编号与归属](#进程编号可信归属与后端验证)连续讲解，下面进一步把这些身份接到账本和恢复流程。

### 2. 全局图：一次 GPU 命令执行，身份/显存/预算/健康怎么串起来

```
                         ┌─────────────────────────────────────────┐
                         │ ①  任务提交：harbor_resources 把          │
                         │     task.toml 翻译成资源需求 EnvDemand    │
                         │     （含 container_budget）               │
                         └───────────────────┬───────────────────────┘
                                             ▼
                         ┌─────────────────────────────────────────┐
                         │ ②  router.py 按 EnvDemand 准入，          │
                         │     启动/复用节点服务，选择身份后端        │
                         │     build_identity_backend("procfs"/"ebpf")│
                         └───────────────────┬───────────────────────┘
                                             ▼
          ┌──────────────────────┐   ┌──────────────────────────────┐
          │ ③a procfs 后端        │   │ ③b eBPF 后端                  │
          │   读容器内挂载的       │   │   BpfIdentity 把 .bpf.c 编译   │
          │   宿主 /proc          │   │   加载进内核，6 个钩子实时追踪  │
          └──────────┬───────────┘   └───────────────┬────────────────┘
                     └────────────┬───────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────┐
                 │ ④  NVML 读取"这张卡上有哪些 PID、各用多少显存"│
                 │     再用③的身份后端核实这些 PID 归谁           │
                 │     （task_telemetry.py / read_command_memory）│
                 └───────────────────┬───────────────────────┘
                                     ▼
                 ┌─────────────────────────────────────────┐
                 │ ⑤  accounted_gpu_memory_mb 把"实测值"和    │
                 │     "尚未兑现的预约"合并成准入判断的那个数字   │
                 └───────────────────┬───────────────────────┘
                                     ▼
                 ┌─────────────────────────────────────────┐
                 │ ⑥  router.py 周期性健康检查 _check_node，   │
                 │     失败计数触发熔断，RayActorError 且无     │
                 │     未清理资源时才真正重建节点（恢复）         │
                 └─────────────────────────────────────────┘
```

**图例：图里每个部分对应的源码位置**

| 图上编号 | 对应代码 |
|---|---|
| ① `demand_from_task` / `with_container_budget` | [`harbor_resources.py:22`](../../../../psrl/workers/env_service/harbor_resources.py#L22)、[`harbor_resources.py:71`](../../../../psrl/workers/env_service/harbor_resources.py#L71) |
| ② `build_identity_backend` 的调用点 | [`router.py:49`](../../../../psrl/workers/env_service/router.py#L49)（选择哪种后端）、[`command_gpu_node.py:108-128`](../../../../psrl/workers/env_service/command_gpu_node.py#L108)（节点服务里异步创建并缓存） |
| ③a `ProcfsIdentity` | [`process_identity.py:28-56`](../../../../psrl/workers/env_service/process_identity.py#L28) |
| ③b `BpfIdentity` | [`bpf_identity.py:46-174`](../../../../psrl/workers/env_service/bpf_identity.py#L46)，内核侧程序 [`bpf_identity.bpf.c`](../../../../psrl/workers/env_service/bpf_identity.bpf.c) |
| ④ `read_command_memory` / `TaskGpuTelemetry` | [`task_telemetry.py:10-67`](../../../../psrl/workers/env_service/task_telemetry.py#L10)、[`task_telemetry.py:70-183`](../../../../psrl/workers/env_service/task_telemetry.py#L70) |
| ⑤ `accounted_gpu_memory_mb` | [`gpu_memory.py:9-28`](../../../../psrl/workers/env_service/gpu_memory.py#L9) |
| ⑥ `_check_node` / `NodeHealth` | [`router.py:395-421`](../../../../psrl/workers/env_service/router.py#L395)、[`health.py:6-27`](../../../../psrl/workers/env_service/health.py#L6) |

如果这些代码读起来有疑问，随时可以继续追问。

### 3. 跟着图读源码

按真实调用顺序整理成一张可以逐行核对的表（这是索引表，不是因果解释，具体"为什么"见第 4 节）：

| 步骤 | 文件:行号 | 做什么 |
|---|---|---|
| 1 | [`router.py:49`](../../../../psrl/workers/env_service/router.py#L49) | `EnvRouterCore.__init__` 读配置里的 `gpu_identity_backend`（默认 `"procfs"`），校验后存成 `self._gpu_identity_backend` |
| 2 | [`router.py:356-365`](../../../../psrl/workers/env_service/router.py#L356) | `_create_node` 把这个后端名字作为参数传给新建的 `EnvNodeService` Ray Actor |
| 3 | [`command_gpu_node.py:108-128`](../../../../psrl/workers/env_service/command_gpu_node.py#L108) | 节点服务收到第一次 GPU 命令请求时，才用 `asyncio.to_thread` 在后台线程里真正构建身份后端（编译 eBPF 很慢，不能占用事件循环） |
| 4 | [`process_identity.py:64-70`](../../../../psrl/workers/env_service/process_identity.py#L64) | `build_identity_backend` 按名字分派：`"procfs"` 直接 new 一个 `ProcfsIdentity`；`"ebpf"` 才 `import` `BpfIdentity`（避免没装 `libbpf` 的机器连 import 都失败） |
| 5 | [`bpf_identity.py:66-126`](../../../../psrl/workers/env_service/bpf_identity.py#L66) | `BpfIdentity._load`：读内核类型信息生成偏移量头文件 → 用 `clang` 编译 `.bpf.c` → 用 `libbpf` 加载 → 拿到 3 个 map 的文件描述符 → 把 6 个 BPF 程序分别挂到对应内核钩子上 |
| 6 | [`command_gpu_node.py:15-50`](../../../../psrl/workers/env_service/command_gpu_node.py#L15) | `DeviceCgroupFence.__init__` 确认容器的设备白名单、算出它在 "devices" cgroup 控制器下的真实路径 `self.path` |
| 7 | [`bpf_identity.py:145-160`](../../../../psrl/workers/env_service/bpf_identity.py#L145) | `BpfIdentity.bind(fence)` 把这个 cgroup 路径的 inode 号注册进 `registrations` map，生成一个随机的 `cookie` 作为这次命令执行的唯一凭证 |
| 8 | [`bpf_identity.bpf.c:76-106`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L76) | 6 个挂钩的 BPF 程序持续运行：进程发生 `fork`/`exec`/系统调用/`exit`/被移动 cgroup 时，分别更新或清除 `identities` map 里这个 PID 的记录 |
| 9 | [`bpf_identity.py:187-202`](../../../../psrl/workers/env_service/bpf_identity.py#L187) | `BpfOwner.owners(pids)` 查询一批 PID，只返回那些记录里的 `cookie`/`cgroup` 还和自己匹配的 PID（意味着这个进程此刻确实还属于这个任务） |
| 10 | [`task_telemetry.py:10-67`](../../../../psrl/workers/env_service/task_telemetry.py#L10) | `read_command_memory` 把 NVML 读到的"每个 PID 用了多少显存"，按第 9 步确认的归属关系拆成每个任务各自的 `owned_mb` |
| 11 | [`gpu_memory.py:9-28`](../../../../psrl/workers/env_service/gpu_memory.py#L9) | `accounted_gpu_memory_mb` 把实测值和"已确认归属"及"尚未兑现的预约"合并成一个判断准入用的总量 |
| 12 | [`router.py:395-421`](../../../../psrl/workers/env_service/router.py#L395) | `_check_node` 周期性探测节点，更新 `NodeHealth`，必要时触发节点重建（恢复） |

### 4. 新知识点逐个展开

#### 4.1 为什么不能只读 /proc：procfs 方案的具体失效场景

procfs（`/proc` 文件系统）由内核提供当前进程信息，多次读取并不是一个原子快照。例如先读到 PID 1234 属于任务 A，随后 A 退出，号码被任务 B 重用，再读取出生信息时对象已经变了。因此归属记录需要带出生身份并处理退出和陈旧采样；本地稳定引用的作用见[pidfd 例子](#pidfd-怎样避免认错进程)。

旧的 procfs 方案在 [`process_identities()`](../../../../psrl/workers/env_service/process_identity.py#L8-L25) 里用了一个补救手段：不只记录"属于哪个 cgroup"，还读 `/proc/<pid>/stat` 里的进程启动时间（`ticks`，内核自系统启动以来的时钟滴答数），换算成绝对时间戳。调用方在"确认归属"之后，会**再查一次**启动时间，如果两次查到的启动时间不一样，就说明这个 PID 已经被复用给了别的进程，之前查到的归属信息要作废（这个"查两次对账"的逻辑在 [`task_telemetry.py:107-113`](../../../../psrl/workers/env_service/task_telemetry.py#L107) 里能看到：`before = self._owners(pids)` 和 `after = self._owners(pids)` 各查一次，只有两次的启动时间戳一致才采信）。

这个补救手段解决了"PID 复用"问题，但解决不了本文开头说的"PID 命名空间视角不一致"问题——如果 `env_service` 自己就跑在容器里，它**压根读不到**宿主机视角的真实 `/proc`，除非运维额外挂载。这是两个独立的问题：PID 复用是"同一视角下，时间窗口内状态变了"；命名空间视角不一致是"你读到的坐标系本身就对不上"。procfs 方案对第一个问题有办法，对第二个问题没有办法——这正是 eBPF 方案要解决的那一半。

#### 4.2 eBPF 到底是什么，名字从哪来

**BPF** 全称 Berkeley Packet Filter（伯克利包过滤器），1992 年诞生于加州大学伯克利分校，最初只是一个很小的虚拟机，专门用来在内核里快速判断"这个网络数据包要不要接收"（比如 `tcpdump` 抓包时写的过滤表达式，底层就是编译成 BPF 指令塞进内核跑的）。2014 年前后，Linux 内核把这个小虚拟机扩展成了一个通用的、可以安全运行用户提供的小程序的内核态执行环境，改名叫 **eBPF**（extended BPF，扩展的 BPF），字面含义就是"扩展版的包过滤器"——虽然现在的 eBPF 早就不只用来过滤网络包了，但名字沿用了下来。

"安全运行用户提供的程序"这句话听起来有点可怕——内核是最高权限的地方，运行"用户提供的代码"岂不是很危险？eBPF 解决这个担忧的办法是**验证器（verifier）**：你写的 eBPF 程序（通常用一个受限的 C 子集写，编译成 BPF 字节码）在被允许加载进内核之前，必须先过一个静态分析器的检查——它会确认这段代码不会死循环（必须在有限步数内结束）、不会访问越界内存、不会导致内核崩溃。通过验证后，内核还会把 BPF 字节码**即时编译（JIT，Just-In-Time Compilation，运行前现场翻译成当前 CPU 能直接执行的机器码）**成原生指令，所以运行速度接近手写的内核模块，不是解释执行那种慢速方案。

和"读 `/proc` 查快照"相比，eBPF 的本质区别是：**它不是去问"现在是什么状态"，而是在状态真正发生改变的那个瞬间（比如进程调用 `fork` 系统调用的那一条内核指令执行到的那一刻）被自动触发去记录。** 这叫"事件驱动"而不是"轮询查询"——没有"两次查询之间状态已经变了"这种时间窗口问题，因为记录动作和状态改变动作，就是同一条内核执行路径上紧挨着的两步，中间不可能有别的进程插进来篡改。这是它比 procfs 快照方案更"可信"的根本原因，也是本文开头场景里"不依赖外部挂载配置，直接拿内核真相"这句话的技术含义。

#### 4.3 tracepoint / raw_tracepoint：这次用的 6 个钩子分别对应什么事件

**钩子（hook）** 在这里指内核代码里预先埋好的一个"插入点"：内核开发者在某些关键位置（比如"一个新进程刚刚 fork 出来"这个时刻）主动调用一个通知函数，如果你在这个通知点上挂了一段 eBPF 程序，内核执行到这里就会先跑一遍你的程序，再继续往下走。**tracepoint** 就是内核里这样一类"预埋好、专门留给追踪工具用"的插入点的统称；**raw_tracepoint** 是它的一个变体——普通 tracepoint 在触发时，内核会先把原始参数包装成一份带类型信息、格式友好的结构体，这个包装动作本身有一点点开销；`raw_tracepoint` 跳过包装，直接把原始参数（一组裸的 64 位数值，对应 [`bpf_identity.bpf.c:12`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L12) 里 `struct raw_args { u64 args[4]; }`）喂给你的程序，省掉这层包装开销，换来的代价是你自己要知道这几个参数分别是什么（文档里写清楚了每个事件对应哪几个参数）。

这次用的 6 个钩子，对应 6 种"进程身份可能发生变化"的内核事件：

```c
SEC("raw_tracepoint/sys_enter")            // 任意系统调用入口
SEC("raw_tracepoint/sched_process_fork")   // 进程 fork 出一个新进程/线程
SEC("raw_tracepoint/sched_process_exec")   // 进程执行 execve，替换成一个新程序
SEC("raw_tracepoint/sched_process_exit")   // 进程退出
SEC("raw_tracepoint/cgroup_attach_task")   // 进程被移动进另一个 cgroup
SEC("raw_tracepoint/cgroup_transfer_tasks")// 一批进程被整体转移 cgroup
```
—— [`bpf_identity.bpf.c:76-106`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L76)

`SEC(...)` 是一个宏（[`bpf_identity.bpf.c:6`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L6)），把紧跟着的函数放进 ELF 文件（一种可执行文件的标准格式）里一个按名字约定的"段"（section）里——`libbpf` 加载这个文件时，正是靠读这个段名知道"这个函数要挂到哪个钩子上"，而不需要你额外写配置文件。

`sys_enter` 这个钩子比较特殊——它在**每一次**系统调用发生时都会触发（系统调用是程序向内核请求服务的唯一通道，比如读文件、分配内存都要经过它），如果什么都记录会非常昂贵。代码里先做一次过滤：

```c
SEC("raw_tracepoint/sys_enter")
int on_syscall(struct raw_args *context) {
    /* CUDA enters the driver through ioctl, while getpid supports the startup handshake. */
    if (context->args[1] != IOCTL_NR && context->args[1] != GETPID_NR) return 0;
    return capture(current_task());
}
```
—— [`bpf_identity.bpf.c:76-81`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L76)

`args[1]` 是这次系统调用的编号，只在编号等于 `ioctl`（`IOCTL_NR`）或 `getpid`（`GETPID_NR`）时才继续往下做记录，其余系统调用直接返回不做任何事。为什么偏偏挑这两个：NVIDIA 的 CUDA 驱动没有自己的系统调用号，所有和 GPU 的交互（分配显存、提交计算任务）全部是通过标准的 `ioctl`（设备控制）系统调用完成的，所以"这个进程调用了一次 `ioctl`"是"这个进程正在和 GPU 驱动打交道"的一个很强信号；`getpid` 则是代码注释里提到的"启动握手"场景——一个刚被任务容器启动、还没来得及做任何 GPU 操作的新进程，用一次 `getpid`（查询自己的进程号，几乎所有程序启动时都会间接触发）就能尽早被记录下来，不用非得等到它真的开始用 GPU。

#### 4.4 BPF map 是什么：三张表怎么配合完成一次身份确认

**BPF map** 是一块由内核维护、同时能被"内核里跑的 BPF 程序"和"用户态的普通程序"读写的键值存储（key-value store）——你可以把它想象成一个内核态和用户态共享的字典，内核侧用专门的几个函数（`lookup`/`update`/`remove_key`，对应 [`bpf_identity.bpf.c:17-19`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L17) 里那几个函数指针，它们实际是内核提供的"BPF helper 函数"，通过固定编号调用，不是真的 C 函数调用）操作，用户态侧用 `bpf_map_lookup_elem`/`bpf_map_update_elem` 这类系统调用操作（对应 [`bpf_identity.py:36-38`](../../../../psrl/workers/env_service/bpf_identity.py#L36)）。

这次代码定义了三张 map：

```c
struct { UINT(type, 1); UINT(max_entries, 256);  TYPE(key, u64); TYPE(value, u64);       } registrations SEC(".maps");
struct { UINT(type, 1); UINT(max_entries, 8192); TYPE(key, u32); TYPE(value, struct identity); } identities SEC(".maps");
struct { UINT(type, 2); UINT(max_entries, 1);    TYPE(key, u32); TYPE(value, u64);       } health SEC(".maps");
```
—— [`bpf_identity.bpf.c:13-15`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L13)

- `registrations`（最多 256 条）：键是某个 device cgroup 的身份标识（一个 64 位整数，具体是什么见 [§4.8](#48-为什么用-devices-这个-cgroup-v1-控制器以及-inode-桥接技巧)），值是一个随机生成的 `cookie`（凭证）。这张表由**用户态**在 `bind()` 时写入——"我要开始追踪这个 cgroup，给它发一个凭证"。
- `identities`（最多 8192 条）：键是进程 PID，值是一个 `struct identity { birth_ns, cgroup, cookie }`（进程的启动时间、所属 cgroup、对应的凭证）。这张表由**内核态**的 BPF 程序在 6 个钩子触发时自动维护。
- `health`（1 条）：一个错误计数器，内核态程序遇到异常情况（比如写表失败）就自增，用户态定期检查这个数字还是不是 0，来判断"追踪有没有悄悄丢失记录"。

三张表怎么配合完成一次完整判断，用 `capture()` 函数（进程发生 fork/exec/系统调用时触发）逐步手算一遍：

```c
INLINE int capture(void *task) {
    void *leader = 0;
    u32 pid = 0;
    struct identity value = {0};
    if (READ(pid, task, TASK_TGID) || !pid || READ(leader, task, TASK_LEADER) || !leader) return 0;
    value.cookie = owner(device_css(task), &value.cgroup);
    if (!value.cookie) { remove_key(&identities, &pid); return 0; }
    ...
    if (update(&identities, &pid, &value, 0)) fault();
    return 0;
}
```
—— [`bpf_identity.bpf.c:57-74`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L57)（省略了同进程组校验的几行，整段见源码）

1. 先拿到这个进程的 `TASK_TGID`（线程组 ID，多线程程序里所有线程共享同一个值，相当于"进程 ID"）。
2. 调 `owner()`（[`bpf_identity.bpf.c:38-55`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L38)）：从这个进程当前所属的 device cgroup 开始，沿着 cgroup 的父子链一层一层往上查 `registrations` 表，直到找到一个被注册过的祖先 cgroup（最多查 16 层）——找到了就返回它的 `cookie`，这就是"确认这个进程归属于哪次已注册的任务"的那一步。
3. 如果 `owner()` 没查到任何已注册的祖先（`cookie` 为 0），说明这个进程根本不归任何正在被追踪的任务管，直接把 `identities` 表里这个 PID 的旧记录删掉（`remove_key`）并结束——**不记录"不确定"的状态，宁可什么都不写**。
4. 查到了就把 `{birth_ns, cgroup, cookie}` 写进 `identities` 表（`update`）。

举一个具体的手算例子：假设任务 A 的 device cgroup 的身份标识是 `9001`，`bind()` 时往 `registrations` 表写入 `{9001: cookie_A}`（`cookie_A` 是一个随机数，比如 `0x7f3a...`）。任务 A 容器里跑起来一个新进程 PID=4201，它 `fork` 的时候触发 `on_fork` 钩子调用 `capture()`：读到它的 device cgroup 正是 `9001`，`owner()` 在 `registrations` 表里查到 `9001 → cookie_A`，于是往 `identities` 表写入 `{4201: {birth_ns: ..., cgroup: 9001, cookie: cookie_A}}`。之后用户态查询 `owners([4201])` 时，发现记录里的 `cgroup` 和 `cookie` 都和自己持有的凭证吻合，就确认"PID 4201 确实属于我"（对应 [`bpf_identity.py:197-200`](../../../../psrl/workers/env_service/bpf_identity.py#L197) 的判断逻辑：`record.cookie == self.cookie and record.cgroup == self.inode`）。

如果这个进程后来被移动到了别的 cgroup（比如管理员手动 `cgroup_attach_task` 把它挪到了 `9002`），`on_move` 钩子触发 `invalidate()`，直接把 `identities` 表里 PID=4201 这条记录删掉——下次查询就查不到了，归属关系自动失效，不需要用户态再做额外的"过期检测"。

#### 4.5 BTF 与"不写死内核偏移量"：bpf_layout.py 怎么工作

前面 `capture()` 函数里有一步"读这个进程的 device cgroup"，代码里写的是：

```c
INLINE void *device_css(void *task) {
    void *set = 0, *css = 0;
    if (READ(set, task, TASK_CGROUPS) || !set) return 0;
    if (READ(css, set, CSS_SUBSYS + DEVICES_INDEX * 8)) return 0;
    return css;
}
```
—— [`bpf_identity.bpf.c:31-36`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L31)

`task`（指向内核的 `task_struct` 结构体，代表一个进程/线程）往下走 `TASK_CGROUPS` 这个字节偏移量，就能拿到它的 `cgroups` 字段；再往下走 `CSS_SUBSYS + DEVICES_INDEX * 8` 字节，就能拿到"devices 控制器"这个子系统对应的具体数据。这些"偏移量"（某个字段相对结构体起始地址偏了多少字节）取决于内核源代码里这个结构体字段的排列顺序——**不同版本的 Linux 内核，这个顺序可能不一样**（内核开发者加字段、调整字段顺序是很常见的事）。如果把这些偏移量写成固定数字硬编码进代码，换一个内核版本运行就可能读错内存位置，轻则读到垃圾数据直接被 verifier 拒绝加载，重则（理论上）读到不该读的地方。

**BTF**（BPF Type Format，BPF 类型格式）就是为了解决这个问题而设计的：从某个版本开始，Linux 内核在编译时会把自己所有内部数据结构的"类型描述"（每个结构体有哪些字段、每个字段的类型和偏移量）打包进一份元数据，暴露在 `/sys/kernel/btf/vmlinux` 这个文件里。只要在目标机器上实际读一下这份元数据，就能**现场**算出"在这台机器正在跑的这个内核版本里，`task_struct.cgroups` 到底偏移了多少字节"，不用提前猜、不用为每个内核版本单独编译一份代码。这正是 [`bpf_layout.py`](../../../../psrl/workers/env_service/bpf_layout.py) 这个文件做的事（这种"针对运行时内核现场适配，而不是为每个版本预编译"的思路，业界的通用叫法是 CO-RE，Compile Once – Run Everywhere，"编译一次、到处能跑"）。

`KernelTypes.__init__`（[`bpf_layout.py:10-32`](../../../../psrl/workers/env_service/bpf_layout.py#L10)）先读这份 BTF 文件的二进制头部，确认格式版本（magic number `0xEB9F` 是 BTF 格式的固定标识，类似很多文件格式开头都有一个"签名"用来快速识别），然后把文件剩下的部分解析成一个类型列表，每一项记录"这是什么类型（结构体/枚举/指针等等）、叫什么名字、里面有哪些成员"。`member()`（[`bpf_layout.py:51-59`](../../../../psrl/workers/env_service/bpf_layout.py#L51)）就是在这个列表里按"结构体名+字段名"查出对应的字节偏移量：

```python
def identity_defines(self):
    fields = {
        "TASK_TGID": ("task_struct", "tgid"),
        "TASK_CGROUPS": ("task_struct", "cgroups"),
        "CSS_SUBSYS": ("css_set", "subsys"),
        ...
    }
    values = {macro: self.member(*field)[0] for macro, field in fields.items()}
    values["DEVICES_INDEX"] = self.enum("cgroup_subsys_id", "devices_cgrp_id")
    ...
    return values
```
—— [`bpf_layout.py:69-86`](../../../../psrl/workers/env_service/bpf_layout.py#L69)

算出来的这一串数字（比如 `TASK_CGROUPS=112` 表示 `cgroups` 字段在 `task_struct` 里偏移 112 字节），会被 [`bpf_identity.py:73-75`](../../../../psrl/workers/env_service/bpf_identity.py#L73) 写成一个临时的 C 头文件 `identity_offsets.h`（每行一句 `#define TASK_CGROUPS 112` 这样的宏定义），这份头文件会在**编译 `.bpf.c` 的时候**被 `#include` 进去（[`bpf_identity.bpf.c:2`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L2)）——这样 C 代码里写的永远是 `TASK_CGROUPS` 这个符号名，实际数值是运行时现场生成、现场编译进去的，换一台内核版本不同的机器跑，这份头文件的内容会自动不一样，但 `.bpf.c` 源码一个字都不用改。

#### 4.6 没有现成库时，Python 怎么自己调用 libbpf

正常情况下，Python 想用 eBPF 功能，通常会用现成的封装库（比如 `bcc` 或 `libbpf-python`）。这次的代码选择了更底层的路线：直接用 **ctypes** 调用系统上的 `libbpf`（Linux 官方维护的、用 C 写的 BPF 加载/操作库，几乎所有现代 BPF 工具底层都靠它）。

**ctypes** 是 Python 标准库自带的一个模块，用来做 **FFI**（Foreign Function Interface，"外部函数接口"——让一种语言写的程序能直接调用另一种语言编译出来的二进制代码，这里是让 Python 调用 C 语言写的动态链接库）。它的基本用法是：先用 `ctypes.CDLL(库文件名)` 把一个 `.so` 动态库文件加载进当前进程，拿到一个代表这个库的对象；再告诉 ctypes 库里每个函数的"签名"（参数类型和返回值类型，因为 C 函数本身的二进制里不带这些类型信息，ctypes 需要你显式声明才知道怎么传参数、怎么解析返回值）；之后就能像调用普通 Python 函数一样调用这个 C 函数了。

```python
def library():
    """Use libbpf for ELF relocation, verification and map operations."""
    name = ctypes.util.find_library("bpf")
    lib = c.CDLL(name, use_errno=True)
    signatures = {
        "bpf_object__open_file": (c.c_void_p, [c.c_char_p, c.c_void_p]),
        "bpf_raw_tracepoint_open": (c.c_int, [c.c_char_p, c.c_int]),
        "bpf_map_lookup_elem": (c.c_int, [c.c_int, c.c_void_p, c.c_void_p]),
        ...
    }
    for symbol, (result, arguments) in signatures.items():
        function = getattr(lib, symbol)
        function.restype, function.argtypes = result, arguments
    return lib
```
—— [`bpf_identity.py:21-43`](../../../../psrl/workers/env_service/bpf_identity.py#L21)（节选）

`ctypes.util.find_library("bpf")` 在系统标准路径里找名叫 `libbpf` 的动态库文件；`c.CDLL(name, use_errno=True)` 加载它，`use_errno=True` 让 ctypes 在每次调用后帮你把 C 层面的全局错误码 `errno`（一个约定：C 函数执行失败时，不直接返回错误信息，而是把一个数字错误码写进一个全局变量，调用者失败后再去读这个变量判断具体错在哪）同步保存下来，之后能用 `c.get_errno()` 读取。`function.restype`/`function.argtypes` 就是在显式声明"这个函数返回什么类型、接受哪些类型的参数"——比如 `bpf_object__open_file` 返回一个 `c_void_p`（通用指针类型，这里实际代表"BPF 对象"这个 C 结构体的地址，Python 侧不需要知道这个结构体内部长什么样，只需要把这个指针原样传回给下一个 libbpf 函数）。

拿到声明好签名的 `lib` 之后，整个加载流程就是照着 libbpf 的标准用法依次调用：`bpf_object__open_file` 打开编译好的 `.o` 文件 → `bpf_object__load` 把它真正加载进内核（这一步会触发内核的 verifier 检查）→ `bpf_object__find_map_fd_by_name` 按名字拿到每张 map 的文件描述符（**文件描述符** 是 Linux 里代表"一个打开的资源"的小整数句柄，不管是真的文件、网络连接还是这里的 BPF map，内核都用同一套"文件描述符"机制统一表示，后续读写这个资源都靠这个小整数定位）→ `bpf_raw_tracepoint_open` 把某个 BPF 程序挂到某个钩子上。整段代码本质上就是用 Python 把 libbpf 这个 C 库原本要用 C 代码调用的那套流程，原样复刻了一遍，只是语法换成了 Python。

#### 4.7 为什么任务容器自己不能有 CAP_BPF/CAP_PERFMON

加载 eBPF 程序、挂追踪钩子这类操作，涉及读取内核内部数据结构（比如前面讲的 `task_struct` 字段），能力很强，所以 Linux 要求调用者必须拥有相应的**权能（capability）**——这是一种比"root/非 root"更细粒度的权限划分机制：传统的 Unix 权限只有"root 能做任何事，非 root 什么都不能做"两档，权能机制把 root 原本拥有的"任何事"拆成了几十个具体的小权限（比如"能不能修改系统时间"、"能不能绑定 1024 以下的端口"），一个进程可以只被单独授予其中几项，而不是全有或全无。和这次改动相关的几个：`CAP_SYS_ADMIN`（历史上的"万能权能"，覆盖一大堆系统管理操作）、`CAP_BPF`（较新内核版本里专门拆出来的、专门管"加载 BPF 程序"的权能）、`CAP_PERFMON`（专门管"用于性能监控/追踪目的的内核数据访问"的权能）。

这组改动里有一处检查：

```python
capabilities = {name.removeprefix("CAP_") for name in config.get("CapAdd") or ()}
if capabilities & {"ALL", "SYS_ADMIN", "BPF", "PERFMON"}:
    raise RuntimeError("GPU command containers cannot administer device cgroups or kernel tracing.")
```
—— [`command_gpu_node.py:22-24`](../../../../psrl/workers/env_service/command_gpu_node.py#L22)

这段检查在**创建用户任务容器**的时候生效，确认的是"这个任务容器自己的 Docker 配置，有没有被额外授予这几项权能"，如果有就直接拒绝创建。原因不难理解：eBPF 身份后端本身是运行在 `env_service` 节点服务进程里的（它拥有必需的权能，这是运维在部署节点服务时单独配置好的），**不是**跑在用户任务容器里；如果反过来让任务容器自己也拿到 `CAP_BPF`/`CAP_PERFMON`，那么一个恶意或者有 bug 的任务代码，理论上就能在自己的容器里加载 BPF 程序，直接读写内核维护的那几张 map——包括前面讲的 `identities`/`registrations` 表，也就是**自己伪造"这个显存该归我"的身份记录**，让系统误以为别的任务的显存归自己名下。这条检查就是从源头堵死这个篡改路径：追踪的权力只能握在管理者（节点服务）手里，被管理者（任务容器）绝对不能拿到。

#### 4.8 为什么用 "devices" 这个 cgroup v1 控制器，以及 inode 桥接技巧

这组代码的注释里反复出现"cgroup v1 devices controller"的字样（比如 [`bpf_identity.py:1`](../../../../psrl/workers/env_service/bpf_identity.py#L1) 的 docstring："Maintain trusted process identities from registered **cgroup v1** kernel events"）。这需要补一点背景：cgroup v1 与 v2 是 Linux 的两代资源管理接口。这里需要关注设备访问控制的差异：**devices 控制器**（专门管"这个 cgroup 里的进程能不能访问哪些设备文件，比如 `/dev/nvidia0`"的那个控制器）在 v2 里被整个移除了接口，改成了用 BPF 程序挂载在 cgroup 上做设备过滤决策。但很多现实世界的容器运行时（包括这里涉及的 Docker）在管理 GPU 设备白名单时，仍然沿用 v1 风格的 "devices" 控制器挂载点（`/sys/fs/cgroup/devices`，常见于"混合模式"——同一台机器同时挂载了 v2 统一层级和个别 v1 遗留控制器）。*这一段关于 Docker 具体实现细节属于通用背景知识，本工作区没有重新核实 Docker/runc 源码，仅供理解上下文参考。*

这次代码读写的正是这个 v1 风格的 "devices" 挂载点：[`command_gpu_node.py:18`](../../../../psrl/workers/env_service/command_gpu_node.py#L18) 里 `DeviceCgroupFence` 的默认根路径就是 `/sys/fs/cgroup/devices`，`set_access()`（[`command_gpu_node.py:63-78`](../../../../psrl/workers/env_service/command_gpu_node.py#L63)）通过写 `devices.allow`/`devices.deny` 这两个 v1 专属的控制文件来临时开启/关闭容器对某块 GPU 设备的访问权限（这就是"栅栏/Fence"这个类名的含义——像一道可以随时拉起/放下的闸门）。

这里还有一个精巧的"桥接"技巧值得展开：BPF 程序在内核里判断"这个进程属于哪个已注册 cgroup"，用的键是 `kernfs_node.id`（**kernfs** 是内核里专门用来实现"看起来像文件系统、但内容是内核数据实时生成"这类虚拟文件系统的通用框架，cgroup 文件系统正是用它实现的；`kernfs_node` 就是这个框架里代表"一个目录/文件节点"的内部结构体，`id` 字段是这个节点的唯一编号）。而用户态 Python 代码在 `bind()` 时，是直接对这个 cgroup 目录做标准的文件系统调用拿到它的 inode 号：

```python
self.directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
self.inode = os.fstat(self.directory_fd).st_ino
```
—— [`bpf_identity.py:182-183`](../../../../psrl/workers/env_service/bpf_identity.py#L182)

**inode**（index node，索引节点）是 Linux 文件系统里"一个文件/目录的唯一编号加元数据"，`os.stat()` 返回的 `st_ino` 就是这个编号——这是一个通用的 VFS（虚拟文件系统，内核对各种具体文件系统的统一抽象层）概念，不是 cgroup 专属的东西。关键在于：kernfs 框架在实现"让 cgroup 目录看起来像普通文件系统目录"的时候，**直接把自己的 `kernfs_node.id` 当成这个目录对外暴露的 inode 号**，这不是巧合，是 kernfs 的设计决定——所以用户态用标准的 `os.fstat().st_ino` 读到的数字，和内核态 BPF 程序用 BTF 偏移量读到的 `kernfs_node.id` 字段，**读的是完全同一个数值**，两边不需要任何额外的"翻译表"就能对上号。这就是为什么 `bind()` 里能直接把 `os.fstat` 读到的 inode 号当作 `registrations` 表的键写进去（[`bpf_identity.py:155-156`](../../../../psrl/workers/env_service/bpf_identity.py#L155)），内核里的 `owner()` 函数后续用同一个数值去查表（[`bpf_identity.bpf.c:46-50`](../../../../psrl/workers/env_service/bpf_identity.bpf.c#L46)）就一定能对上——这正是用户态注册和内核态追踪能够"认同一个身份证号"的根本原因。

### 5. GPU 显存会计公式：accounted_gpu_memory_mb 手算

<!-- figure:memory-promises -->
先用这张图区分实占、已兑现预约与尚未兑现的预约，再看下方的逐项手算。

[![显存准入：只补尚未兑现的预约（1/4）](../../../_static/img/document-figures/memory-promises-screen.png)](../../../_static/img/document-figures/memory-promises.png)

[![显存准入：只补尚未兑现的预约（2/4）](../../../_static/img/document-figures/memory-promises-screen-02.png)](../../../_static/img/document-figures/memory-promises.png)

[![显存准入：只补尚未兑现的预约（3/4）](../../../_static/img/document-figures/memory-promises-screen-03.png)](../../../_static/img/document-figures/memory-promises.png)

[![显存准入：只补尚未兑现的预约（4/4）](../../../_static/img/document-figures/memory-promises-screen-04.png)](../../../_static/img/document-figures/memory-promises.png)

沿①初始条件、②旧算法反例、③逐任务记账、④放行条件阅读。90 GiB容量、外部70 GiB、A和B各预约15 GiB是教学算例。A实占改变时剩余预约随之改变，总承诺仍为85 GiB；确认物理归还后才清除预约。 [查看矢量图](../../../_static/img/document-figures/memory-promises.svg)。
<!-- /figure:memory-promises -->

确认了"哪些进程属于哪个任务"之后，下一步是把这个归属关系换算成"准入判断该用的显存总量"。这件事比看起来复杂，因为显存的分配过程本身有时间差：一个任务被**允许**使用某块 GPU（拿到"预约"）和它的代码真正跑起来、CUDA 驱动真正**完成分配**（变成 NVML 能测到的"实测占用"）之间，可能隔着几秒甚至更久的初始化时间。如果只看"实测占用"，会漏算那些"已经批准但还没来得及真正占用"的显存，造成超卖；如果简单地"预约量"全部累加，又会在任务已经真正开始使用之后重复计费。

```python
def accounted_gpu_memory_mb(measured_mb, reservations, owned_mb=None):
    owned_mb = owned_mb or {}
    ...
    return measured_mb + sum(
        max(0.0, amount - owned_mb.get(lease_id, 0.0)) for lease_id, amount in reservations.items()
    )
```
—— [`gpu_memory.py:9-28`](../../../../psrl/workers/env_service/gpu_memory.py#L9)（节选核心逻辑）

公式的含义："总账 = 实测用量 + 每个任务（预约量 − 这个任务已被核实确实占用的那部分）"，`max(0.0, ...)` 保证"已核实占用超过预约量"的情况不会倒扣。两个来自真实测试用例（[`tests/env_service/test_gpu_memory.py:25-28`](../../../../tests/env_service/test_gpu_memory.py#L25)）的手算例子：

**例 1：没有归属信息时，预约量不能被打折。** `accounted_gpu_memory_mb(70, {"a": 15})`：实测用了 70MB，任务 a 预约了 15MB，没有 `owned_mb` 参数（意味着"还不知道这 70MB 里有没有 a 的份"）。代入公式：`70 + max(0, 15 - 0) = 85`。注意即使实测值后来涨到了 84（`accounted_gpu_memory_mb(84, {"a": 15})`），只要仍然没有归属确认，结果是 `84 + max(0, 15 - 0) = 99`——**哪怕实测用量的增长恰好等于 a 的预约量，系统也绝不会"猜"这增长就是 a 用的**，这正是测试断言里那句话的意思："Unknown ownership cannot be assumed to be ours"（未知归属不能被假定为己方占用）。

**例 2：一旦核实归属，同一笔钱不能收两次。** 紧接着的第三个断言：`accounted_gpu_memory_mb(84, {"a": 15}, {"a": 14})`。这次多了 `owned_mb={"a": 14}`——意味着通过第 4 节讲的身份确认流程，查到任务 a 确实有进程在用，而且核实出来的占用是 14MB（这 14MB 已经包含在实测值 84 里了，不是额外加的）。代入公式：`84 + max(0, 15 - 14) = 84 + 1 = 85`。对比例 1 的 99，少了 14——这 14 正是"已经被核实、已经包含在 84 这个实测值里的那部分，不应该在后面的预约量里再加一遍"。测试断言写的是："Verified memory must not be charged twice"（已核实的显存不能被收两次费）。

这条公式直接决定能不能再准入一个新任务——`EnvUnitLedger`（已有的账本模块）拿这个总量跟 GPU 的物理总容量比较，超过就拒绝或排队。

### 6. 资源画像：harbor_resources + container_budget 手算一个任务的预算

GPU 显存只是资源的一种，`harbor_resources.py` 负责把一个任务声明的 CPU/内存需求，连同它会启动多少个"配套容器"（sidecar，中文直译"配角容器"——除了跑主程序的那个容器之外，额外启动的辅助性容器，比如专门跑日志收集、专门跑某个外部依赖服务），翻译成一份具体的资源预算；`container_budget.py` 再把这份预算换算成 Docker 能直接执行的硬限制。

```python
@dataclass(frozen=True)
class ContainerBudget:
    main_cpu: float
    main_memory_mb: int
    max_sidecars: int
    environments: int = 1
    sidecar_cpu: float = 0.25
    sidecar_memory_mb: int = 128

    @property
    def cpu(self):
        return self.environments * (self.main_cpu + self.max_sidecars * self.sidecar_cpu)

    @property
    def memory_mb(self):
        return self.environments * (self.main_memory_mb + self.max_sidecars * self.sidecar_memory_mb)
```
—— [`container_budget.py:6-29`](../../../../psrl/workers/env_service/container_budget.py#L6)（节选）

`environments` 只能取 1 或 2——这对应 harbor 的一个设计：验证器（verifier，负责判断任务是否做对了的那部分逻辑）可以要求"独立环境"（`environment_mode == "separate"`），也就是验证器和主任务分别起一套完整的容器集合，互不干扰；不要求独立时，验证器复用主任务那一套容器，`environments=1`。

手算一个具体例子：假设一个任务的 `task.toml` 声明 `environment.cpus = 2.0`、`environment.memory_mb = 4096`，它的验证器要求 `environment_mode = "separate"`，它的 `docker-compose.yaml` 里除了 `main` 还声明了 2 个 sidecar 服务。走一遍 [`demand_from_task`](../../../../psrl/workers/env_service/harbor_resources.py#L22) 和 [`with_container_budget`](../../../../psrl/workers/env_service/harbor_resources.py#L71)：

1. `cpus=2.0`、`memory_mb=4096` 先被读进 `EnvDemand`。
2. `with_container_budget` 里 `separate=True`（因为验证器要求独立环境），`environments = 2 if separate else 1` → `2`。
3. `cpu = max([2.0, 1.0] + 各环境自己的 cpus) = 2.0`；`memory = max([4096] + 各环境自己的 memory_mb) = 4096`（代码见 [`harbor_resources.py:83-84`](../../../../psrl/workers/env_service/harbor_resources.py#L83)）。
4. 扫描 `docker-compose.yaml`，`sidecars = 2`（排除 `main`），`max_sidecars = sidecars + 1 = 3`（代码注释解释了这个 `+1`："Harbor 在准入之后可能还会给任意一边环境再加一个网络服务"，所以预算要多留一个名额，见 [`harbor_resources.py:89-94`](../../../../psrl/workers/env_service/harbor_resources.py#L89)）。
5. 代入公式：`budget.cpu = 2 × (2.0 + 3 × 0.25) = 2 × 2.75 = 5.5`（核）；`budget.memory_mb = 2 × (4096 + 3 × 128) = 2 × 4480 = 8960`（MB）。

这意味着：这个任务虽然自己只声明要 2 核 4GB，但因为验证器要独立环境、又带了 2 个 sidecar，它在节点上实际会被预留 **5.5 核、8960MB**——这个数字既是"调度时判断这台节点还有没有位置"用的那个数字，也是真正创建容器时写进 `cpus`/`mem_limit` 的那个数字（`service_limits()`，[`container_budget.py:31-46`](../../../../psrl/workers/env_service/container_budget.py#L31)），**两处用的是同一份计算结果**，不会出现"说要 5 份资源、却只真正限制了 2 份"的口径不一致。

### 7. 节点健康状态机：NodeHealth 的熔断与恢复

```python
@dataclass
class NodeHealth:
    failures: int = 0
    checked_at: float = float("-inf")
    error: str = ""
    pending_cleanup: bool = False

    @property
    def available(self):
        return not self.error and not self.pending_cleanup

    def success(self, now, pending_cleanup=False):
        self.failures = 0
        ...

    def fail(self, now, error):
        self.failures += 1
        ...
```
—— [`health.py:6-26`](../../../../psrl/workers/env_service/health.py#L6)

这是一个很朴素的计数器状态机：每次探测成功就把 `failures` 清零，探测失败就加一。配合 `router.py` 里的使用方式手算一遍一个失联场景：假设 `health_max_failures=3`（[`router.py:336`](../../../../psrl/workers/env_service/router.py#L336) 的默认值）。

- 第 1 次探测失败（比如节点进程暂时没响应）：`failures=1`，`_health_task`（[`router.py:423-436`](../../../../psrl/workers/env_service/router.py#L423)）发现 `1 < 3`，下一个探测周期还会继续排新的检查任务。
- 第 2、3 次都失败：`failures=3`。此时 `_health_task` 里 `state.failures >= self._health_max_failures` 成立（[`router.py:430`](../../../../psrl/workers/env_service/router.py#L430)），**直接 return，不再排新的健康检查任务**——这就是"熔断"：既然已经连续失败到阈值，继续高频探测一个大概率已经死掉的节点没有意义，干脆先停下来。
- 与此同时，`node_capacity()`（[`router.py:438-444`](../../../../psrl/workers/env_service/router.py#L438)）每次被上层调度逻辑调用时，都会把 `state.available`（此刻为 `False`，因为 `error` 非空）写进返回的 `NodeCapacity.available` 字段——**调度器天然就会因为这个标志不再往这个节点派新任务**，不需要专门写一条"如果节点坏了就别调度过去"的额外分支逻辑，坏节点自己把"不可用"这件事体现在它返回的容量信息里了。

那"恢复"具体发生在哪一步？只有探测失败的异常类型恰好是 `ray.exceptions.RayActorError`（代表 Ray 判断这个节点服务对应的 Actor 进程确实已经整个死掉，不是网络抖动之类的临时问题）时，`_check_node` 才会往下走一步，检查这个节点上是不是还有未了结的"身后事"——`active`（还有存活租约）、`bound`（还有 GPU 绑定）、`gpu_cleanup`（还有待回收的 GPU 资源）：

```python
if isinstance(exc, ray.exceptions.RayActorError):
    active = any(lease.node_ip == node_ip for lease in self._ledger._leases.values())
    bound = any(binding.unit.node_ip == node_ip for binding in self._gpu_bindings.values())
    gpu_cleanup = any(lease.gpu_indices for lease in self._ledger.pending_reaped(node_ip))
    if not active and not bound and not gpu_cleanup:
        self._node_services.pop(node_ip, None)
        self._node_handles.pop(node_ip, None)
```
—— [`router.py:413-419`](../../../../psrl/workers/env_service/router.py#L413)

只有这三项都为空，才会真正把这个节点的服务句柄清空——清空之后，下次有新任务需要这个节点，`_create_node`（[`router.py:356`](../../../../psrl/workers/env_service/router.py#L356)）会重新创建一个全新的 `EnvNodeService` Actor，这才是真正的"恢复"。如果还有未了结的资源，宁可先保留这个（已经死掉的）句柄不清空，也不贸然重建——因为一旦句柄被清空，这些"谁该负责清理"的信息就会跟着丢失，造成 GPU 资源永久卡死而没人知道该去回收。

### 8. 协程知识回顾 + 两个新模式

#### 8.1 快速回顾：三层叠加模型

先把执行单位分清：一个进程可以运行多个线程，同一线程中的事件循环又能让多个协程在等待点交替推进。`asyncio.create_task` 把协程登记为可调度任务，调用方随即继续；真正等待结果时再 `await`。例如遥测协程等待 GPU 查询时，控制协程可以处理取消，而不是等整条查询完成。不同 Ray Actor 通常位于不同进程，能否同时执行还受实际 CPU/GPU 资源限制。下面解释 `resource_telemetry.py` 与 `routing_pause.py` 在协程层使用的两个控制模式，项目完整角色流见[Router 运行流程](../01-架构与执行/05-EnvRouter_配置恢复与Worker重试.md#11-启动方题目rollout-与环境申请)。

#### 8.2 新模式 A：有限并发 + 截止时间收尾（resource_telemetry.py）

```python
async def read_node_resources(unit_ids, *, timeout_s=4):
    ...
    slots = asyncio.Semaphore(8)

    async def sample(row):
        async with slots:
            ...

    tasks = {row["Id"]: asyncio.create_task(sample(row)) for row in active}
    try:
        if tasks:
            await asyncio.wait(tasks.values(), timeout=timeout_s)
        readings = {
            container_id: task.result() if task.done()
            else {"memory_mb": None, "cpu_cores": None, "error": "Container stats deadline exceeded."}
            for container_id, task in tasks.items()
        }
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
```
—— [`resource_telemetry.py:24-59`](../../../../psrl/workers/env_service/resource_telemetry.py#L24)（节选）

这里要解决的问题是：一个节点上可能同时跑着几十个容器，要查每个容器的资源用量（调 Docker 的 `stats` 接口），但 Docker daemon 在压力大的时候，单次查询可能会很慢甚至挂住不返回。如果写成"挨个查、每个都等到返回"，一个慢查询就能拖垮整次采集。这段代码用了三个配合的手段：

1. **`asyncio.Semaphore(8)`**（信号量，一种"最多同时允许 N 个人进门"的并发限流器，在第 2 节的全局图文档里也用过类似的闸门概念）把同时在查的容器数量限制在 8 个以内，避免对 Docker daemon 造成瞬时压力。
2. **`asyncio.wait(tasks.values(), timeout=timeout_s)`**：给整批查询设一个统一的截止时间，到点之后不管还有多少个任务没完成，`wait` 都会立刻返回（而不是无限等下去）——用列表推导式里的 `if task.done() else {...}` 判断，没赶上截止时间的任务就填一个"超时"的占位结果，**不让一个慢容器拖累整批数据都拿不到**。
3. **`finally` 块里先 `cancel()` 所有还没完成的任务，再 `await asyncio.gather(..., return_exceptions=True)` 等它们真正结束**：这是容易被忽略的一步——`asyncio.wait` 超时返回之后，那些没完成的任务**并没有自动停止**，它们还在后台运行（比如还在等 Docker 的网络响应）。如果不管它们，这些"孤儿任务"会一直占着资源，甚至在没人等待的情况下，任务内部抛出的异常会变成一条容易被忽视的警告日志。`cancel()` 主动给这些任务发送取消信号，`gather(..., return_exceptions=True)` 等它们真正响应取消、彻底收尾（`return_exceptions=True` 保证即使某个任务取消时抛了异常，也不会打断对其他任务的等待）。


## 取消安全与可运行实验

> **本节要点**：多个操作同时要求暂停路由时，每个操作登记自己的持有记录，最后一个操作结束才恢复。即使调用方被取消，也要等待清理任务进入终态，并把清理失败或取消结果交给上层处理。下面先解释一个双操作例子，再提供可运行的标准库教学脚本。


### 8.3 新模式 B：多持有者引用计数 + 取消安全清理（routing_pause.py）

`RoutingPauseController` 要解决的问题：训练过程中，模型权重同步、副本迁移等好几个独立的操作，都可能需要"暂停往下游派发新的推理/生成请求"，而且这些操作可能**并发**发生（比如正在同步权重的同时，又触发了一次副本迁移）。如果简单地用一个布尔值记录"是否暂停"，第一个操作做完就把它改回 `False`，会错误地提前恢复路由——哪怕第二个操作还没做完。

```python
async def _acquire(self, owner):
    async with self._lock:
        ...
        self._holds.add(owner)
        await self._write_running(False)

async def _release(self, owner):
    async with asyncio.timeout(self._timeout_s):
        async with self._lock:
            if owner not in self._holds:
                return
            self._holds.remove(owner)
            await self._write_running(self._running and not self._holds)
```
—— [`routing_pause.py:24-38`](../../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L24)

这里的解法是用一个 **集合**（`self._holds`）记录"当前有哪些独立操作正在要求暂停"，而不是一个布尔值——这其实是**引用计数**思想的一种变体（引用计数：记录"有多少个使用者正在持有这份资源"，只有计数归零才真正释放；这里每个 `owner` 就是一个"使用者"）。`_write_running(False)` 只要集合非空就一直传 `False`；只有 `_release` 把最后一个 `owner` 也移出集合（`not self._holds` 变成 `True`）之后，才会真正把"恢复运行"的信号写回去。

更精细的地方在它的上层封装 `pause()`：

```python
@asynccontextmanager
async def pause(self):
    owner = object()
    try:
        async with asyncio.timeout(self._timeout_s):
            await self._acquire(owner)
        yield
    finally:
        cleanup = asyncio.create_task(self._release(owner))
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError
```
—— [`routing_pause.py:40-59`](../../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L40)

`@asynccontextmanager` 把一个写成"先做准备工作、`yield` 一下、再做收尾工作"的协程函数，包装成一个可以配合 `async with` 使用的"异步上下文管理器"——`async with controller.pause(): ...` 这种写法，进入代码块前自动跑 `yield` 之前那部分（这里是 `_acquire`，登记一个 hold），离开代码块（无论正常结束还是中途抛异常）自动跑 `finally` 里那部分（这里是 `_release`，释放这个 hold）。

真正精巧（也是本文想重点讲的新知识）的是 `finally` 块里处理"取消"（cancellation）的方式。协程的**取消**是 `asyncio` 里一种特殊的异常注入机制：外部代码可以调用 `task.cancel()`，这会在这个协程当前正挂起等待的那个 `await` 点上，强行注入一个 `asyncio.CancelledError` 异常，让它"看起来像是那个等待操作自己失败了"——协程可以选择捕获这个异常做清理，但默认行为是继续往外传播，最终让这个协程整体终止。

问题是：如果调用方正在执行 `pause()` 的 `finally` 块（也就是正在做"释放 hold、把暂停状态写回去"这件收尾工作）的过程中，**又**被取消了一次（比如上层调用方自己也被取消，取消信号层层传递下来），默认情况下这第二次取消会直接中断 `_release` 的执行——可能刚把 `self._holds.remove(owner)` 执行完、还没来得及调用 `_write_running` 把"恢复运行"的信号真正写出去就被打断了，造成"集合里明明已经没有任何 hold 了，但路由状态却一直卡在暂停"的不一致。

`asyncio.shield(cleanup)`（**shield**，字面意思"盾牌"，这里指"保护某个协程不被外部的取消信号直接打断"）就是为了防住这第二次取消：`cleanup = asyncio.create_task(self._release(owner))` 先把清理工作包装成一个独立的 Task（Task 有自己的生命周期，不会因为"等待它的那一方被取消"而自动跟着被取消）；`await asyncio.shield(cleanup)` 等待这个 Task，但即使等待方（当前协程）被取消，`shield` 也只会让"等待"这个动作本身抛出 `CancelledError`，`cleanup` 这个 Task 不受影响，继续在后台跑完。代码用一个 `while not cleanup.done()` 循环反复尝试等待它——哪怕中途再被取消几次（每次把 `cancelled` 标记设为 `True`），也会一直等到 `cleanup` 真正跑完为止，持续等待清理任务完成或报告异常；最后 `cleanup.result()` 取出结果（如果清理过程本身出了异常，这里会重新抛出），如果期间确实被取消过，再手动 `raise asyncio.CancelledError`——**把"我被取消了"这个事实如实告诉外部**，只是不让这次取消打断必须完成的清理工作。这是协程编程里一个经典但容易被忽略的教训：**取消信号可以、也应该被妥善处理，但绝不能让它打断必须完整执行的收尾逻辑。**

### 9. 可运行实验脚本

下面这份脚本把第 5/7/8 节讲到的四段真实逻辑（显存会计公式、容器预算、健康状态机、多持有者暂停控制器）原样搬过来跑一遍，再加一个纯 Python 手写的"迷你内核"模拟第 4 节的 eBPF 身份追踪状态机（真实的 eBPF 需要 Linux 内核 + root 权限，在这里只能用等价的数据结构和逻辑模拟，方便在任何机器上直接观察状态变化）。只用标准库，可以直接用 `python3` 运行：

```python
"""
自包含教学复刻脚本：对应 docs/env-worker-deep-dive/knowledge/02-资源与调度/03-进程归属_PID身份与显存记账.md
第 5/7/8 节的真实逻辑搬运 + 第 4 节 eBPF 身份追踪的纯 Python 模拟。

只用标准库（asyncio + dataclasses + contextlib），不依赖 Linux/BPF/Docker/PSRL 代码。

对照真实代码（仅供对照阅读，不在本脚本中导入）：
  - 显存会计公式 accounted_gpu_memory_mb -> psrl/psrl/workers/env_service/gpu_memory.py:9-28
  - 容器预算 ContainerBudget           -> psrl/psrl/workers/env_service/container_budget.py:6-46
  - 健康状态机 NodeHealth              -> psrl/psrl/workers/env_service/health.py:6-26
  - 暂停控制器 RoutingPauseController   -> psrl/psrl/workers/gen/rollout_coordination/routing_pause.py:7-59
  - eBPF 身份追踪（本脚本用 FakeKernel 模拟其状态机，不是真实 eBPF）
    -> psrl/psrl/workers/env_service/bpf_identity.bpf.c:57-106
"""

import asyncio
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass


# ---------- 第 5 节：GPU 显存会计公式（与 gpu_memory.py 逐行一致） ----------

def accounted_gpu_memory_mb(measured_mb, reservations, owned_mb=None):
    owned_mb = owned_mb or {}
    values = [measured_mb, *reservations.values(), *owned_mb.values()]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("GPU memory observations and reservations must be finite and nonnegative.")
    if owned_mb.keys() - reservations.keys() or sum(owned_mb.values()) > measured_mb:
        raise ValueError("GPU ownership credits must be disjoint and contained in measured usage.")
    return measured_mb + sum(
        max(0.0, amount - owned_mb.get(lease_id, 0.0)) for lease_id, amount in reservations.items()
    )


def demo_gpu_memory_accounting():
    print("=== 第 5 节：GPU 显存会计公式 ===")
    unknown = accounted_gpu_memory_mb(70, {"a": 15})
    print(f"未知归属: measured=70, reservations={{a:15}} -> {unknown}  (期望 85)")
    unknown_grew = accounted_gpu_memory_mb(84, {"a": 15})
    print(f"实测涨到84仍未知归属:            -> {unknown_grew}  (期望 99，不会被当成是a用的)")
    verified = accounted_gpu_memory_mb(84, {"a": 15}, {"a": 14})
    print(f"核实a占用14MB(已包含在84里):      -> {verified}  (期望 85，14MB不会被收两次)")
    print()


# ---------- 第 6 节：容器预算（与 container_budget.py 逐行一致） ----------

@dataclass(frozen=True)
class ContainerBudget:
    main_cpu: float
    main_memory_mb: int
    max_sidecars: int
    environments: int = 1
    sidecar_cpu: float = 0.25
    sidecar_memory_mb: int = 128

    @property
    def cpu(self):
        return self.environments * (self.main_cpu + self.max_sidecars * self.sidecar_cpu)

    @property
    def memory_mb(self):
        return self.environments * (self.main_memory_mb + self.max_sidecars * self.sidecar_memory_mb)


def demo_container_budget():
    print("=== 第 6 节：容器预算手算 ===")
    # 任务声明 2 核 4096MB，验证器要求独立环境(environments=2)，
    # docker-compose.yaml 里有 2 个 sidecar（+1 预留给 Harbor 可能追加的网络服务）
    budget = ContainerBudget(main_cpu=2.0, main_memory_mb=4096, max_sidecars=2 + 1, environments=2)
    print(f"任务自己声明: 2 核 / 4096MB")
    print(f"实际预留:     {budget.cpu} 核 / {budget.memory_mb}MB  (期望 5.5 核 / 8960MB)")
    print()


# ---------- 第 7 节：健康状态机（与 health.py 逐行一致） ----------

@dataclass
class NodeHealth:
    failures: int = 0
    checked_at: float = float("-inf")
    error: str = ""
    pending_cleanup: bool = False

    @property
    def available(self):
        return not self.error and not self.pending_cleanup

    def success(self, now, pending_cleanup=False):
        self.failures = 0
        self.checked_at = now
        self.error = ""
        self.pending_cleanup = pending_cleanup

    def fail(self, now, error):
        self.failures += 1
        self.checked_at = now
        self.error = f"{type(error).__name__}: {error}"


def demo_node_health():
    print("=== 第 7 节：节点健康熔断 ===")
    max_failures = 3
    state = NodeHealth()
    for t in range(1, 5):
        state.fail(float(t), RuntimeError("node unreachable"))
        tripped = state.failures >= max_failures
        print(f"第{t}次探测失败: failures={state.failures}, available={state.available}, "
              f"熔断(停止继续探测)={tripped}")
    print()


# ---------- 第 8.3 节：多持有者暂停控制器（与 routing_pause.py 逐行一致） ----------

class RoutingPauseController:
    def __init__(self, read_status, write_running, timeout_s=5.0):
        self._read_status = read_status
        self._write_running = write_running
        self._timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._holds = set()
        self._running = None

    async def _acquire(self, owner):
        async with self._lock:
            if self._running is None:
                status = await self._read_status()
                self._running = not status["paused"]
            self._holds.add(owner)
            await self._write_running(False)

    async def _release(self, owner):
        async with asyncio.timeout(self._timeout_s):
            async with self._lock:
                if owner not in self._holds:
                    return
                self._holds.remove(owner)
                await self._write_running(self._running and not self._holds)

    @asynccontextmanager
    async def pause(self):
        owner = object()
        try:
            async with asyncio.timeout(self._timeout_s):
                await self._acquire(owner)
            yield
        finally:
            cleanup = asyncio.create_task(self._release(owner))
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError


async def demo_routing_pause():
    print("=== 第 8.3 节：多持有者暂停，只有最后一个释放才真正恢复 ===")
    running_log = []

    async def read_status():
        return {"paused": False}

    async def write_running(value):
        running_log.append(value)

    controller = RoutingPauseController(read_status, write_running)

    async with controller.pause():  # 操作A 开始暂停
        print(f"操作A进入(第1次写入): running写入历史={running_log}  (期望末尾是False)")
        async with controller.pause():  # 操作B 并发也要求暂停
            print(f"操作B也进入(第2次写入): running写入历史={running_log}  (期望末尾仍是False)")
        print(f"操作B释放后(第3次写入): running写入历史={running_log}  (期望末尾仍是False，因为A还在持有)")
    print(f"操作A释放后(第4次写入): running写入历史={running_log}  (期望此时末尾才变成True)")
    print()


# ---------- 第 4 节：用纯 Python 模拟 eBPF 的三张 map 和 capture/invalidate ----------

class FakeKernel:
    """
    模拟 bpf_identity.bpf.c 的 registrations / identities 两张 map，
    以及 capture()（对应 fork/exec/sys_enter 钩子）、invalidate()（对应 exit/cgroup move 钩子）。
    真实版本在内核里用 BTF 算出的偏移量读 task_struct，这里直接用 Python 字典里的
    "cgroup" 字段代替，逻辑一一对应，只是省去了内核内存布局这一层。
    """

    def __init__(self):
        self.registrations = {}   # cgroup_id -> cookie
        self.identities = {}      # pid -> {"cgroup": ..., "cookie": ...}

    def bind(self, cgroup_id, cookie):
        """对应 bpf_identity.py:155-159 的 bind()：用户态注册一个要追踪的 cgroup。"""
        self.registrations[cgroup_id] = cookie

    def capture(self, pid, cgroup_id):
        """对应 bpf_identity.bpf.c:57-74 的 capture()：fork/exec/sys_enter 钩子触发时调用。"""
        cookie = self.registrations.get(cgroup_id)
        if cookie is None:
            self.identities.pop(pid, None)
            return
        self.identities[pid] = {"cgroup": cgroup_id, "cookie": cookie}

    def invalidate(self, pid):
        """对应 bpf_identity.bpf.c:93-96 的 invalidate()：exit/cgroup_attach_task 钩子触发时调用。"""
        self.identities.pop(pid, None)

    def owners(self, cookie, pids):
        """对应 bpf_identity.py:187-202 的 owners()：用户态按 cookie 查询一批 pid 还归不归自己。"""
        return {
            pid for pid in pids
            if pid in self.identities and self.identities[pid]["cookie"] == cookie
        }


def demo_ebpf_identity_state_machine():
    print("=== 第 4 节：eBPF 身份追踪状态机模拟 ===")
    kernel = FakeKernel()

    # 任务 A 的 device cgroup 是 9001，bind() 时拿到凭证 cookie_A
    cookie_a = "cookie_A_0x7f3a"
    kernel.bind(cgroup_id=9001, cookie=cookie_a)
    print(f"任务A注册: registrations={kernel.registrations}")

    # 任务A容器里 fork 出 PID=4201，属于 cgroup 9001 -> on_fork 钩子触发 capture()
    kernel.capture(pid=4201, cgroup_id=9001)
    print(f"PID 4201 fork（cgroup=9001）后: identities={kernel.identities}")

    owned = kernel.owners(cookie_a, pids=[4201, 9999])
    print(f"任务A查询 owners([4201, 9999]) -> {owned}  (期望只有4201，9999根本没记录)")

    # 管理员把 4201 挪到了别的 cgroup（9002）-> on_move 钩子触发 invalidate()
    kernel.invalidate(pid=4201)
    print(f"PID 4201 被移出 cgroup（on_move 触发 invalidate）后: identities={kernel.identities}")
    owned_after_move = kernel.owners(cookie_a, pids=[4201])
    print(f"任务A再查询 owners([4201]) -> {owned_after_move}  (期望空集，归属已自动失效)")
    print()


async def main():
    demo_gpu_memory_accounting()
    demo_container_budget()
    demo_node_health()
    demo_ebpf_identity_state_machine()
    await demo_routing_pause()


if __name__ == "__main__":
    asyncio.run(main())
```

**实测跑通的输出**（已在本地实际运行，不是推算）：

```
=== 第 5 节：GPU 显存会计公式 ===
未知归属: measured=70, reservations={a:15} -> 85.0  (期望 85)
实测涨到84仍未知归属:            -> 99.0  (期望 99，不会被当成是a用的)
核实a占用14MB(已包含在84里):      -> 85  (期望 85，14MB不会被收两次)

=== 第 6 节：容器预算手算 ===
任务自己声明: 2 核 / 4096MB
实际预留:     5.5 核 / 8960MB  (期望 5.5 核 / 8960MB)

=== 第 7 节：节点健康熔断 ===
第1次探测失败: failures=1, available=False, 熔断(停止继续探测)=False
第2次探测失败: failures=2, available=False, 熔断(停止继续探测)=False
第3次探测失败: failures=3, available=False, 熔断(停止继续探测)=True
第4次探测失败: failures=4, available=False, 熔断(停止继续探测)=True

=== 第 4 节：eBPF 身份追踪状态机模拟 ===
任务A注册: registrations={9001: 'cookie_A_0x7f3a'}
PID 4201 fork（cgroup=9001）后: identities={4201: {'cgroup': 9001, 'cookie': 'cookie_A_0x7f3a'}}
任务A查询 owners([4201, 9999]) -> {4201}  (期望只有4201，9999根本没记录)
PID 4201 被移出 cgroup（on_move 触发 invalidate）后: identities={}
任务A再查询 owners([4201]) -> set()  (期望空集，归属已自动失效)

=== 第 8.3 节：多持有者暂停，只有最后一个释放才真正恢复 ===
操作A进入(第1次写入): running写入历史=[False]  (期望末尾是False)
操作B也进入(第2次写入): running写入历史=[False, False]  (期望末尾仍是False)
操作B释放后(第3次写入): running写入历史=[False, False, False]  (期望末尾仍是False，因为A还在持有)
操作A释放后(第4次写入): running写入历史=[False, False, False, True]  (期望此时末尾才变成True)
```

对照着看几个现象：显存会计那组数字和第 5 节手算完全一致（第三个例子打印出的是整数 `85` 而不是 `85.0`——这是 Python `max(0.0, 1)` 的一个实现细节：`max`/`min` 返回的是"两个参数里较大/较小的那一个原始对象"，不会做类型统一转换，`1`（int）严格大于 `0.0`（float）时就直接原样返回这个 `int` 对象，不影响数值大小，只是打印形式不同）；容器预算 `5.5核/8960MB` 和第 6 节手算一致；健康熔断在第 3 次失败时 `熔断=True` 开始生效，之后继续失败也保持 `True`（对应 `_health_task` 不再排新检查）；eBPF 模拟部分，`invalidate` 之后归属立刻查不到了（不需要任何"过期时间"判断，状态变化即时生效）。

暂停控制器在每次 acquire/release 时都写出当前状态。顺序为：A 进入写 `False`；B 进入仍写 `False`；B 退出时 A 仍持有，继续写 `False`；最后 A 退出，集合清空，写 `True`。决定是否恢复的是完整持有者集合，四次动作都按当时集合计算。源码见 [`routing_pause.py`](../../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L24)。

脚本保存在 [`scripts/env_admission_attribution_demo.py`](../../env-service-verification/scripts/env_admission_attribution_demo.py)。

### 10. 一句话总结 + 延伸阅读

这条链路依次完成四件事：在正确 PID 视角中确认进程归属，把实测显存与未兑现预约合并，统一调度和容器创建的资源口径，再通过节点健康状态阻止未确认清理的资源重新准入。eBPF 在已验证权限和可信注册条件下提供宿主身份记录，pidfd 保护本地进程引用；最终资源释放仍须物理核查。

- [返回专题入口](../../README.md)
- psrl 仓库自带的权威设计文档（本文假设读者已经点开过，如果还没读建议先读这两篇再回来看本文的"零基础展开"部分）：
  - [本篇：进程编号、可信归属与后端验证](#进程编号可信归属与后端验证)
  - [显存准入：实测占用与预约预算](01-显存准入_实测占用与预约预算.md)
  - [统一执行：EnvRouter与容器及预算策略](../01-架构与执行/02-统一执行_EnvRouter与容器及预算策略.md)
- 协程、Worker 与资源申请：[Router 运行流程](../01-架构与执行/05-EnvRouter_配置恢复与Worker重试.md#11-启动方题目rollout-与环境申请)。
- cgroup、procfs、PID 视角与 pidfd：[本篇的进程编号章节](#进程编号可信归属与后端验证)。
