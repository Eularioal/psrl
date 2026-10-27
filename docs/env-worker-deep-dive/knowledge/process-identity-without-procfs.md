# 不用 procfs 标识进程：pidfd 与 GPU 记账的边界

> **核心结论：** SSH 登录的私有节点环境是平台容器，内部再运行 Docker 任务容器。PSRL 已新增可选的 `ebpf` 进程归属后端，与原有 `procfs` 后端共用任务遥测和显存预算接口。它按内核进程身份、容器资源组及本轮命令注册号记账，不依赖进程名，也不需要挂载宿主 `/proc`。两个真实 Docker CUDA 任务的独立归属、取消后的记录失效与物理归还已通过。完整 science 训练及跨内核长期运行尚未验收。

## 目录

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

## 用一个任务理解两种方案

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
| P1 | `_command_container` 校验租约标签，`DeviceCgroupFence` 校验设备和资源组 | [command_gpu_node.py:181](../../../psrl/workers/env_service/command_gpu_node.py#L181) |
| P2 | `process_identities` 读取宿主进程档案 | [process_identity.py:8](../../../psrl/workers/env_service/process_identity.py#L8) |
| P3 | `BpfIdentity.bind` 注册任务，内核 `capture` 更新记录 | [bpf_identity.py:145](../../../psrl/workers/env_service/bpf_identity.py#L145)、[bpf_identity.bpf.c:57](../../../psrl/workers/env_service/bpf_identity.bpf.c#L57) |
| P4 | `NvmlProbe.read_process_snapshot` 取得 GPU 数据 | [probe.py:109](../../../psrl/workers/env_service/probe.py#L109) |
| P5 | `TaskGpuTelemetry.sample` 过滤样本，`read_command_memory` 核销已兑现的显存预约 | [task_telemetry.py:101](../../../psrl/workers/env_service/task_telemetry.py#L101)、[read_command_memory:10](../../../psrl/workers/env_service/task_telemetry.py#L10) |
| P6 | `revoke_gpu_command` 禁止新的设备打开，检查残留句柄与 GPU 进程，确认后注销本轮身份 | [command_gpu_node.py:207](../../../psrl/workers/env_service/command_gpu_node.py#L207) |

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

## SSH 私有节点为什么也是容器

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
| A | GPU 进程的宿主编号来自内核，与 NVML 对账 | [真实 GPU 对账结果](../env-service-verification/results/bpf-process-identity-20261027/gpu-python3/result.json) |
| B | 平台容器的 PID、用户、挂载命名空间，PID 1 与控制组 | [只读预检结果](../env-service-verification/results/bpf-process-identity-20261027/preflight.json)，读取视图的可复用函数 [namespace_view](../env-service-verification/scripts/host_proc_diagnostic.py#L15) |
| C | 本次创建、限制和回收内部 Docker 容器 | [verify_bpf_gpu_identity.py:17 `main`](../env-service-verification/scripts/verify_bpf_gpu_identity.py#L17) |

现场证据是相互印证的：SSH 环境的 PID 命名空间为 `4026549832`，不是初始宿主视图。PID 1 的程序名为 `taijirun`，自己的控制组路径含有 `kubepods/burstable/pod…/容器ID`。Kubernetes 是部署容器的平台系统，Pod 是它的一种工作负载组织单元。仅一个进程名不足以下结论，结合命名空间、控制组和下文真实进程的三层编号，才能判断这里确实有外层容器。

“私有”描述的是平台资源分配与使用权限，不等同于“裸机登录”或“完整宿主管理员权限”。这些结果没有证明平台是否把整台物理服务器独占给当前实例，也没有排除宿主操作系统底下还存在其他虚拟化层。多个实例是否共享 GPU、CPU 或网络，需要平台配额与部署信息确认，不能从一个 SSH 地址反推。

容器共享所在宿主的内核，通过命名空间隔离进程等可见范围，通过控制组限制资源。因此容器里可以运行真实 CUDA，同时仍看不到宿主的完整进程列表。内部 Docker 的 `--pid=host` 中的 host 是相对它的 daemon 而言。在这里，它不能据此自动越过外层平台容器，变成物理宿主视图。

## 先看当前链路缺什么

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
| 1 | `GpuJobManager.start` 分配作业 UUID，并异步提交执行 | [gpu_jobs.py:60](../../../psrl/workers/env_service/gpu_jobs.py#L60) |
| 2 | `run` 启动容器命令，`identity` 和 `cancel` 管理本地进程身份与取消 | [gpu_job_process.py:35](../../../psrl/workers/env_service/gpu_job_process.py#L35)、[identity:12](../../../psrl/workers/env_service/gpu_job_process.py#L12) |
| 3 | `NvmlProbe.read_process_snapshot` 取得按 PID 索引的 GPU 样本 | [probe.py:109](../../../psrl/workers/env_service/probe.py#L109) |
| 4 | `TaskGpuTelemetry.sample` 调用 `_owners`，由所选后端过滤归属与陈旧样本 | [task_telemetry.py:101](../../../psrl/workers/env_service/task_telemetry.py#L101)、[procfs 查询](../../../psrl/workers/env_service/process_identity.py#L8) |
| 5 | 演示 `probe` 创建子进程并观察 pidfd，不调用生产执行器 | [process_identity_without_proc.py:46](../env-service-verification/scripts/process_identity_without_proc.py#L46) |

本地进程取消使用的 `/proc` 与 GPU 遥测需要的宿主 `/proc` 是不同用途。即使把前者改成 pidfd，也不会自动补齐后者的映射。这些代码如果有理解上的问题，可以随时追问。

## 唯一编号究竟指什么

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

## pidfd 怎样避免认错进程

pidfd 意为“进程文件描述符”。文件描述符是当前进程持有资源引用的整数入口，资源也可以是套接字或进程，并不要求它对应磁盘文件。内核通过 pidfd 维持对某个具体进程身份的引用。

假设 A 的本地 PID 为 300，监督器在 A 活着时调用 `pidfd_open(300, 0)`，得到描述符 7。随后 A 退出，内核以后把 PID 300 分给 B。再次按数字 PID 查找时可能找到 B，但仍持有的旧 pidfd 不会改指向 B。通过它发送信号、等待退出，可以避免“认错同号的新进程”。该接口不需要挂载 procfs。

**数字 7 本身不是全局唯一编号。** 另一个监督器也可以有描述符 7，关闭后本进程也能重用 7。不能把 `7` 写到 JSON，让另一个进程拿它当同一个句柄。设计上应由监督器持续持有句柄，以 `job_id → 受控进程引用` 建立表，需要跨进程共享时传递实际文件描述符，或者调用监督器的控制接口。

还有三项边界：

- 获取句柄时必须防止竞态。先查一个陌生 PID，等它退出并被重用之后才打开，仍可能拿错对象。本次演示让自己的子进程阻塞等待，打开后才允许退出。生产启动器也需要受控握手或原子创建机制。
- pidfd 按调用者可见的 PID 查找进程。在外层平台容器中直接把 NVML 的宿主 PID 传进去，不会自动跨到物理宿主视图。
- 一个 pidfd 不代表整棵进程树，也不等同于进程组。父进程退出不证明所有子进程和 GPU 上下文都已释放，不能直接拿它替换现有整组取消及资源回收确认。

## 不挂载宿主 procfs 的替代方案

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

## 真实 CPU 验证与复现

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

证据：[通过结果](../env-service-verification/results/process-identity-20261027/result.json)、[Python 接口缺失记录](../env-service-verification/results/process-identity-20261027/python-wrapper-unavailable.json)、[后台日志](../env-service-verification/results/process-identity-20261027/background.log)。

## 内核追踪与真实 Docker CUDA 对账

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
| 入口 6 | [验证器 `main`](../env-service-verification/scripts/verify_bpf_gpu_identity.py#L17) | 检查 Docker 和 GPU，创建带唯一归属标签、1 CPU、512 MiB 内存上限的容器 |
| 容器侧并行执行 7 | [CUDA `main`](../env-service-verification/scripts/bpf_cuda_identity.py#L11) | 分配 64 MiB 显存，执行内核并检查答案 42，写入就绪文件，等待 begin 信号 |
| 采集器加载和挂接 8 | [C `main`](../env-service-verification/scripts/bpf_process_identity.c#L19) | 创建一项 BPF map，加载随机名字过滤程序，挂到 sys_enter，输出 READY |
| 内核回调 8 | [进程号 helper 调用](../env-service-verification/scripts/bpf_process_identity.c#L57) | 命中随机名字后取得宿主线程组号与线程号，写入 map。其他事件不写入样本 |
| 回到验证器 9 | [NVML 对账](../env-service-verification/scripts/verify_bpf_gpu_identity.py#L108) | 读取 map 捕获的宿主 PID，要求 NVML 有同 PID 且显存至少覆盖申请量的记录 |
| 收尾 10 | [验证器 finally](../env-service-verification/scripts/verify_bpf_gpu_identity.py#L124)、[C 句柄关闭](../env-service-verification/scripts/bpf_process_identity.c#L109) | 回收采集进程，删除本次容器，轮询进程消失及显存回落。临时 BPF 未固定到文件系统，关闭附着句柄后解除追踪 |

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

证据：[CPU 映射](../env-service-verification/results/bpf-process-identity-20261027/cpu-result.json)、[GPU 预检](../env-service-verification/results/bpf-process-identity-20261027/gpu-preflight.json)、[首次启动失败](../env-service-verification/results/bpf-process-identity-20261027/gpu/result.json)、[GPU 成功与清理](../env-service-verification/results/bpf-process-identity-20261027/gpu-python3/result.json)。映射取号不读取宿主 procfs。只读部署诊断使用了当前容器的 procfs，Docker top 仅作为外层 PID 的辅助展示，不参与宿主 PID 的计算。

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

## 统一后端怎样接入框架

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
| 配置传递 | P1 | [Router 创建节点](../../../psrl/workers/env_service/router.py#L304) | 将 procfs 或 ebpf 选择传给执行节点 |
| 初始化 | P3 | [`_gpu_identity`](../../../psrl/workers/env_service/command_gpu_node.py#L108)、[`BpfIdentity._load`](../../../psrl/workers/env_service/bpf_identity.py#L66) | 读取 BTF 字段位置，编译 BPF，使用 libbpf 加载并挂接事件。初始化取消后也关闭迟到的采集器 |
| 命令注册 | P1、P3 | [`grant_gpu_command`](../../../psrl/workers/env_service/command_gpu_node.py#L192)、[`bind`](../../../psrl/workers/env_service/bpf_identity.py#L145) | 先确认容器与租约，登记资源组和随机本轮注册号，再开放 GPU 设备访问 |
| 并行的内核事件 | P3 | [`on_syscall`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L77)、[`on_fork`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L84)、[`on_exec`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L87) | 在驱动 ioctl、getpid、子进程创建及执行事件上更新受管进程记录。无需任务修改名字或自报宿主 PID |
| 退出与迁移分支 | P3 | [`on_exit`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L100)、[`on_move`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L103) | 删除该进程记录。资源组迁移后，后续事件按新的归属重新登记 |
| 采样与核销 | P4、P5 | [`BpfOwner.owners`](../../../psrl/workers/env_service/bpf_identity.py#L187)、[`read_command_memory`](../../../psrl/workers/env_service/task_telemetry.py#L10) | 核对资源组、注册号和出生时间，只用稳定身份的当前显存抵扣预约余量 |
| 归还与关闭 | P6 | [`revoke_gpu_command`](../../../psrl/workers/env_service/command_gpu_node.py#L207)、[`EnvNodeService.close`](../../../psrl/workers/env_service/node.py#L58) | 确认 GPU 句柄及属于本任务的 GPU 进程消失，注销命令。节点关闭时停止遥测并解除追踪 |

BTF 可以理解成内核附带的“数据结构说明书”。我们据此查找进程出生时间、资源组指针等字段的位置，不把这台机器的内存偏移写死到代码中。libbpf 是负责加载 BPF 程序、关联记录表和检查加载错误的库。clang 则把 C 程序编译成内核可执行的 BPF 指令。

本节点的 BTF 存在布局相同的重复类型定义，解析器允许相同布局，拒绝互相冲突的布局。clang 20 的默认指令选择与目标内核不完全兼容，因此明确编译为 BPF v1 指令集。加载检查失败时必须保留错误，不能解释成 GPU 任务自身 OOM。

实现边界需要明确：当前针对 cgroup v1 devices 控制组，检查最多 16 层祖先，支持最多 256 个同时注册的资源组、8192 条进程记录。内核记录表写入失败会将采集器标为异常，预算保持保守。注册资源组不能相互包含，避免一条进程被两个任务同时认领。注册目录保持打开，采样时检查路径身份和本轮注册号，关闭后旧记录不能继续提供额度。

这些限制不会把两个不同容器的并发变成“同一容器多个 GPU 命令并发”。同容器并发仍是独立 TODO。本后端也没有实现跨 Router 的统一全节点资源所有权协议。

## 统一后端的双容器实测

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

证据：[完整结果及清理](../env-service-verification/results/bpf-backend-20261027/gpu/result.json)、[内核加载检查](../env-service-verification/results/bpf-backend-20261027/load-result.json)、[CPU 回归日志](../env-service-verification/results/bpf-backend-20261027/pytest-final.log)。CPU 回归覆盖陈旧注册号、进程出生身份改变、邻居归属、关闭后失效、采集中故障、初始化取消和节点关闭等情况，它们与真实 GPU 实测分开记录。

可运行脚本是 [verify_bpf_backend.py](../env-service-verification/scripts/verify_bpf_backend.py)，容器负载为 [bpf_backend_workload.py](../env-service-verification/scripts/bpf_backend_workload.py)。在具备上述依赖、已激活项目环境的执行节点，从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/verify_bpf_backend.py \
  --image "$PROBE_IMAGE" --output "$PROBE_OUTPUT"
```

`PROBE_IMAGE` 是已经准备好的 GPU 验证镜像，需要包含 python3 和 timeout。`PROBE_OUTPUT` 是尚不存在的输出目录。实际运行采用结果目录里的后台执行脚本，另设外层超时和仅所属容器的退出清理钩子。只需检查身份判断逻辑时，可运行 CPU 回归，无需 GPU：

```bash
pytest -q tests/env_service/test_process_identity.py tests/env_service/test_task_telemetry.py
```

## 跟着图读源码

这里分开列现有生产链路与独立实验，不将异步采样伪装成命令启动后的单线程调用。

| 步骤 | 图中编号 | 文件与函数 | 输入、输出或状态变化 |
|---|---|---|---|
| 提交作业 | 1 | [gpu_jobs.py:60 `start`](../../../psrl/workers/env_service/gpu_jobs.py#L60) | 命令输入，生成 UUID，状态变为 queued，创建异步执行任务 |
| 容器内执行 | 2 | [gpu_job_process.py:35 `run`](../../../psrl/workers/env_service/gpu_job_process.py#L35) | 命令输入，启动进程组，记录 PID 和出生信息，等待退出 |
| 并行采样 | 3 | [probe.py:109 `read_process_snapshot`](../../../psrl/workers/env_service/probe.py#L109) | 设备与时间窗口输入，返回按 PID 的显存及利用率样本 |
| 检查与过滤 | 4 | [task_telemetry.py:101 `sample`](../../../psrl/workers/env_service/task_telemetry.py#L101) | 检查所选后端，再取进程归属，过滤采样期间身份不稳定的进程 |
| 取消分支与本地收尾 | 2 | [gpu_job_process.py:20 `cancel`](../../../psrl/workers/env_service/gpu_job_process.py#L20) | 校验出生信息后向进程组发信号。`run` 的 finally 移除身份文件，后续 GPU 物理释放仍需执行器确认 |
| 独立演示入口与清理 | 5 | [process_identity_without_proc.py:46 `probe`](../env-service-verification/scripts/process_identity_without_proc.py#L46) | 自建子进程，取得 pidfd，观察退出，finally 回收子进程并关闭描述符 |

## 后续实现顺序与边界

独立进程归属后端已经接入节点服务，支持原有 procfs 和显式启用的 eBPF。真实双容器测试已达到本轮验收目标，因此停止 GPU 验证。本节之后的实现重点应是按实际接入需求验收，而不是为了证明可取号启动完整训练。

尚未验收的范围包括：其他内核和 cgroup v2、采集器或 Router 重启时恢复所有正在运行的任务、大规模长期事件开销、真实强制 PID 重用、所有资源组迁移路径，以及完整 science 原生评分链路。本轮没有进行这些实验，也没有声称训练吞吐提升。

默认配置仍为 procfs。后续 science 实验若显式启用 ebpf，应先通过实际执行节点的严格归因预检，并记录所选后端。已有 observe 结果不能改标为可信 GPU 评分。完整 science 严格模式验收之前，仍保留[观察模式说明](../experiments/host-proc-blocker.md)中的结果标注和预算保守规则。

## 接口资料

- [Linux pidfd_open 手册](https://man7.org/linux/man-pages/man2/pidfd_open.2.html)：稳定进程引用、退出事件及不依赖 procfs 的接口。
- [Linux PID 命名空间手册](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html)：同一进程在不同视图中的编号及可见范围。
- [Linux Unix 套接字手册](https://man7.org/linux/man-pages/man7/unix.7.html)：对端凭据和文件描述符传递。
- [Linux BPF helpers 手册](https://man7.org/linux/man-pages/man7/bpf-helpers.7.html)：内核事件程序可调用的进程标识接口。
- [Linux capabilities 手册](https://man7.org/linux/man-pages/man7/capabilities.7.html)：权限位的用途与版本边界。
- [NVIDIA NVML 进程信息结构](https://docs.nvidia.com/deploy/archive/R510/nvml-api/structnvmlProcessInfo__t.html)：当前讨论的 PID、显存与 MIG 字段。
- [NVIDIA CUPTI 接口](https://docs.nvidia.com/cupti-python/cupti.html)：进程内 CUDA 上下文与采样接口，不能直接替代宿主 PID 映射。
