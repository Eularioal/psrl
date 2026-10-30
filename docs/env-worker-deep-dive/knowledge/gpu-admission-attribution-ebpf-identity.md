# GPU 准入 / 归属 / 恢复新设计：eBPF 身份识别与两个新协程模式

## TL;DR

- psrl 仓库最近 3 个提交（`feat: strengthen environment admission, attribution and recovery` 等）给 `env_service` 加了一组新模块：用 **eBPF**（而不是读 `/proc`）在内核态确认"这个 GPU 进程到底属于哪个任务"，用一条**显存会计公式**把"已用显存"和"还没兑现的显存预约"合并成一个不会重复计费的准入判断，用**容器预算**把一个任务的 CPU/内存需求转成 docker-compose 的硬限制，用一个**健康状态机**在节点失联时判断能不能安全重建。
- 核心动机一句话：**当"运行 env_service 的节点本身也是一个容器"时，容器看到的 `/proc` 只是它自己的命名空间视角，不能保证里面的 PID 8 就是宿主上真正该被记账的那个进程；eBPF 方案绕开这个视角问题，直接在内核里追踪进程的真实身份和它所属的 cgroup。**
- 本文只讲这组改动里**新出现**的知识点（eBPF/BTF/raw_tracepoint/libbpf FFI/cgroup v1 devices 控制器桥接技巧、两个新的 asyncio 清理模式），协程/线程/进程的基础三层模型已经在 [并发/并行全景](../../../../docs/psrl/psrl-concurrency-landscape.md) 讲过，这里只做必要的回顾和延伸，不重复推导。
- psrl 仓库自带的 [`process-identity-without-procfs.md`](process-identity-without-procfs.md)、[`gpu-memory-admission.md`](gpu-memory-admission.md) 已经把这组设计的背景和取舍写得很细，本文是以"零基础读者"为口径的补充讲解，侧重把那两篇文档里**假设读者已经懂**的术语（eBPF 本身是什么、BTF 是什么、为什么需要 FFI）展开讲清楚，并把两份文档串成一条可以顺着读的线。

## 目录

- [1. 先还原一个具体问题：同一个容器里，这笔显存该记给谁](#1-先还原一个具体问题同一个容器里这笔显存该记给谁)
- [2. 全局图：一次 GPU 命令执行，身份/显存/预算/健康怎么串起来](#2-全局图一次-gpu-命令执行身份显存预算健康怎么串起来)
- [3. 跟着图读源码](#3-跟着图读源码)
- [4. 新知识点逐个展开](#4-新知识点逐个展开)
  - [4.1 为什么不能只读 /proc：procfs 方案的具体失效场景](#41-为什么不能只读-proc procfs-方案的具体失效场景)
  - [4.2 eBPF 到底是什么，名字从哪来](#42-ebpf-到底是什么名字从哪来)
  - [4.3 tracepoint / raw_tracepoint：这次用的 6 个钩子分别对应什么事件](#43-tracepoint--raw_tracepoint这次用的-6-个钩子分别对应什么事件)
  - [4.4 BPF map 是什么：三张表怎么配合完成一次身份确认](#44-bpf-map-是什么三张表怎么配合完成一次身份确认)
  - [4.5 BTF 与"不写死内核偏移量"：bpf_layout.py 怎么工作](#45-btf-与不写死内核偏移量bpf_layoutpy-怎么工作)
  - [4.6 没有现成库时，Python 怎么自己调用 libbpf](#46-没有现成库时python-怎么自己调用-libbpf)
  - [4.7 为什么任务容器自己不能有 CAP_BPF/CAP_PERFMON](#47-为什么任务容器自己不能有-cap_bpfcap_perfmon)
  - [4.8 为什么用 "devices" 这个 cgroup v1 控制器，以及 inode 桥接技巧](#48-为什么用-devices-这个-cgroup-v1-控制器以及-inode-桥接技巧)
- [5. GPU 显存会计公式：accounted_gpu_memory_mb 手算](#5-gpu-显存会计公式accounted_gpu_memory_mb-手算)
- [6. 资源画像：harbor_resources + container_budget 手算一个任务的预算](#6-资源画像harbor_resources--container_budget-手算一个任务的预算)
- [7. 节点健康状态机：NodeHealth 的熔断与恢复](#7-节点健康状态机nodehealth-的熔断与恢复)
- [8. 协程知识回顾 + 两个新模式](#8-协程知识回顾--两个新模式)
  - [8.1 快速回顾：三层叠加模型](#81-快速回顾三层叠加模型)
  - [8.2 新模式 A：有限并发 + 截止时间收尾（resource_telemetry.py）](#82-新模式-a有限并发--截止时间收尾resource_telemetrypy)
  - [8.3 新模式 B：多持有者引用计数 + 取消安全清理（routing_pause.py）](#83-新模式-b多持有者引用计数--取消安全清理routing_pausepy)
- [9. 可运行实验脚本](#9-可运行实验脚本)
- [10. 一句话总结 + 延伸阅读](#10-一句话总结--延伸阅读)

## 1. 先还原一个具体问题：同一个容器里，这笔显存该记给谁

psrl 的 `env_service` 要做一件听起来很朴素的事：一张物理 GPU 上同时跑着好几个任务的命令（比如 4 个强化学习 episode 同时在同一张卡上执行用户代码），系统需要知道"这张卡现在用掉的显存，哪部分是 A 任务的、哪部分是 B 任务的"，这样才能在显存快满的时候正确地排队、在任务结束后正确地释放。

最直接的办法是：查显存用量时顺手查一下"这块显存是哪个进程在用"（NVIDIA 的 NVML 库能告诉你"PID 1234 用了 2GB"），再去确认"PID 1234 是不是属于我管理的这个任务"。第二步"确认 PID 属于谁"听起来应该很简单——Linux 的 `/proc/<pid>/cgroup` 文件就能告诉你这个进程属于哪个 cgroup（后面会解释 cgroup 是什么）。但这里有一个前提条件很容易被忽略：**你读到的 `/proc`，必须是"宿主机视角"的 `/proc`，PID 的数字才有意义。**

问题就出在这个前提上：psrl 的 `env_service` 节点服务（负责管理 GPU、确认身份的那个进程）现在自己也跑在一个容器里（文档称之为"SSH 私有节点"——一种给单个用户独占、通过 SSH 登录使用的容器化开发环境）。容器有自己的 PID 命名空间（namespace，一种让一组进程"看到"一套独立编号体系的内核机制），**容器内部看到的 PID 8，和宿主机上真正的 PID 8，通常不是同一个进程**。如果 `env_service` 直接读自己容器内的 `/proc`，它看到的 PID 体系就是"自己的"，而 NVML 返回的"PID 1234 用了 2GB"里的 1234，是**宿主机内核**分配的真实 PID——两套编号完全不在一个坐标系里，字面匹配毫无意义。

旧方案（`process_identities()`，[`process_identity.py:8-25`](../../../psrl/workers/env_service/process_identity.py#L8)）的权宜之计是要求运维额外挂载一份"宿主机视角"的 `/proc`（通过环境变量 `PSRL_GPU_HOST_PROC` 指定挂载点），并用 `check_support()`（[`process_identity.py:36-38`](../../../psrl/workers/env_service/process_identity.py#L36)）在用之前检查一下这确实是宿主的 PID 命名空间（判断依据是 inode 号等于固定值 `0xEFFFFFFC`，这是内核初始 PID 命名空间的一个已知标志）。这个方案能工作，但依赖"运维必须手动挂载并且挂载正确"这个外部前提，一旦挂载配置错了或者被篡改，整个归属判断就会悄悄给错结果而没有任何报错——这正是这次改动要解决的问题：**不依赖任何外部挂载配置，直接从内核本身拿到不可伪造的真相。** 这就是引入 eBPF 方案的直接动机，细节见 [§4.1](#41-为什么不能只读-proc procfs-方案的具体失效场景)。

关于 cgroup 和 PID 命名空间本身更系统的讲解，见已有的 [cgroup v2 资源控制知识笔记](../../../../docs/operating-systems/cgroup-v2-resource-control.md) 和 [procfs 知识笔记](../../../../docs/operating-systems/proc-filesystem.md)，本文不重复这两篇已经讲过的基础内容，只讲这次新代码引入的部分。

## 2. 全局图：一次 GPU 命令执行，身份/显存/预算/健康怎么串起来

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
| ① `demand_from_task` / `with_container_budget` | [`harbor_resources.py:22`](../../../psrl/workers/env_service/harbor_resources.py#L22)、[`harbor_resources.py:71`](../../../psrl/workers/env_service/harbor_resources.py#L71) |
| ② `build_identity_backend` 的调用点 | [`router.py:49`](../../../psrl/workers/env_service/router.py#L49)（选择哪种后端）、[`command_gpu_node.py:108-128`](../../../psrl/workers/env_service/command_gpu_node.py#L108)（节点服务里异步创建并缓存） |
| ③a `ProcfsIdentity` | [`process_identity.py:28-56`](../../../psrl/workers/env_service/process_identity.py#L28) |
| ③b `BpfIdentity` | [`bpf_identity.py:46-174`](../../../psrl/workers/env_service/bpf_identity.py#L46)，内核侧程序 [`bpf_identity.bpf.c`](../../../psrl/workers/env_service/bpf_identity.bpf.c) |
| ④ `read_command_memory` / `TaskGpuTelemetry` | [`task_telemetry.py:10-67`](../../../psrl/workers/env_service/task_telemetry.py#L10)、[`task_telemetry.py:70-183`](../../../psrl/workers/env_service/task_telemetry.py#L70) |
| ⑤ `accounted_gpu_memory_mb` | [`gpu_memory.py:9-28`](../../../psrl/workers/env_service/gpu_memory.py#L9) |
| ⑥ `_check_node` / `NodeHealth` | [`router.py:395-421`](../../../psrl/workers/env_service/router.py#L395)、[`health.py:6-27`](../../../psrl/workers/env_service/health.py#L6) |

如果这些代码读起来有疑问，随时可以继续追问。

## 3. 跟着图读源码

按真实调用顺序整理成一张可以逐行核对的表（这是索引表，不是因果解释，具体"为什么"见第 4 节）：

| 步骤 | 文件:行号 | 做什么 |
|---|---|---|
| 1 | [`router.py:49`](../../../psrl/workers/env_service/router.py#L49) | `EnvRouterCore.__init__` 读配置里的 `gpu_identity_backend`（默认 `"procfs"`），校验后存成 `self._gpu_identity_backend` |
| 2 | [`router.py:356-365`](../../../psrl/workers/env_service/router.py#L356) | `_create_node` 把这个后端名字作为参数传给新建的 `EnvNodeService` Ray Actor |
| 3 | [`command_gpu_node.py:108-128`](../../../psrl/workers/env_service/command_gpu_node.py#L108) | 节点服务收到第一次 GPU 命令请求时，才用 `asyncio.to_thread` 在后台线程里真正构建身份后端（编译 eBPF 很慢，不能占用事件循环） |
| 4 | [`process_identity.py:64-70`](../../../psrl/workers/env_service/process_identity.py#L64) | `build_identity_backend` 按名字分派：`"procfs"` 直接 new 一个 `ProcfsIdentity`；`"ebpf"` 才 `import` `BpfIdentity`（避免没装 `libbpf` 的机器连 import 都失败） |
| 5 | [`bpf_identity.py:66-126`](../../../psrl/workers/env_service/bpf_identity.py#L66) | `BpfIdentity._load`：读内核类型信息生成偏移量头文件 → 用 `clang` 编译 `.bpf.c` → 用 `libbpf` 加载 → 拿到 3 个 map 的文件描述符 → 把 6 个 BPF 程序分别挂到对应内核钩子上 |
| 6 | [`command_gpu_node.py:15-50`](../../../psrl/workers/env_service/command_gpu_node.py#L15) | `DeviceCgroupFence.__init__` 确认容器的设备白名单、算出它在 "devices" cgroup 控制器下的真实路径 `self.path` |
| 7 | [`bpf_identity.py:145-160`](../../../psrl/workers/env_service/bpf_identity.py#L145) | `BpfIdentity.bind(fence)` 把这个 cgroup 路径的 inode 号注册进 `registrations` map，生成一个随机的 `cookie` 作为这次命令执行的唯一凭证 |
| 8 | [`bpf_identity.bpf.c:76-106`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L76) | 6 个挂钩的 BPF 程序持续运行：进程发生 `fork`/`exec`/系统调用/`exit`/被移动 cgroup 时，分别更新或清除 `identities` map 里这个 PID 的记录 |
| 9 | [`bpf_identity.py:187-202`](../../../psrl/workers/env_service/bpf_identity.py#L187) | `BpfOwner.owners(pids)` 查询一批 PID，只返回那些记录里的 `cookie`/`cgroup` 还和自己匹配的 PID（意味着这个进程此刻确实还属于这个任务） |
| 10 | [`task_telemetry.py:10-67`](../../../psrl/workers/env_service/task_telemetry.py#L10) | `read_command_memory` 把 NVML 读到的"每个 PID 用了多少显存"，按第 9 步确认的归属关系拆成每个任务各自的 `owned_mb` |
| 11 | [`gpu_memory.py:9-28`](../../../psrl/workers/env_service/gpu_memory.py#L9) | `accounted_gpu_memory_mb` 把实测值和"已确认归属"及"尚未兑现的预约"合并成一个判断准入用的总量 |
| 12 | [`router.py:395-421`](../../../psrl/workers/env_service/router.py#L395) | `_check_node` 周期性探测节点，更新 `NodeHealth`，必要时触发节点重建（恢复） |

## 4. 新知识点逐个展开

### 4.1 为什么不能只读 /proc：procfs 方案的具体失效场景

procfs（`/proc` 文件系统）本质上是内核实时生成的一个"快照视图"：你每次读 `/proc/1234/stat`，内核都是现算现答，不是读一份提前写好的文件。这意味着两次连续读之间，进程状态完全可能已经变了——比如你先读到 PID 1234 属于任务 A 的 cgroup，还没来得及用这个信息做记账，1234 这个进程就已经退出、而**同一个 PID 号**被内核复用给了一个全新的、属于任务 B 的进程（这就是 procfs 笔记里讲过的"PID 复用"问题，见 [procfs 知识笔记 §5](../../../../docs/operating-systems/proc-filesystem/05-权限与PID复用.md)）。你如果不是在"读到状态"的同一瞬间做判断，中间隔了哪怕几毫秒的其他操作，结论就可能已经过期。

旧的 procfs 方案在 [`process_identities()`](../../../psrl/workers/env_service/process_identity.py#L8-L25) 里用了一个补救手段：不只记录"属于哪个 cgroup"，还读 `/proc/<pid>/stat` 里的进程启动时间（`ticks`，内核自系统启动以来的时钟滴答数），换算成绝对时间戳。调用方在"确认归属"之后，会**再查一次**启动时间，如果两次查到的启动时间不一样，就说明这个 PID 已经被复用给了别的进程，之前查到的归属信息要作废（这个"查两次对账"的逻辑在 [`task_telemetry.py:107-113`](../../../psrl/workers/env_service/task_telemetry.py#L107) 里能看到：`before = self._owners(pids)` 和 `after = self._owners(pids)` 各查一次，只有两次的启动时间戳一致才采信）。

这个补救手段解决了"PID 复用"问题，但解决不了本文开头说的"PID 命名空间视角不一致"问题——如果 `env_service` 自己就跑在容器里，它**压根读不到**宿主机视角的真实 `/proc`，除非运维额外挂载。这是两个独立的问题：PID 复用是"同一视角下，时间窗口内状态变了"；命名空间视角不一致是"你读到的坐标系本身就对不上"。procfs 方案对第一个问题有办法，对第二个问题没有办法——这正是 eBPF 方案要解决的那一半。

### 4.2 eBPF 到底是什么，名字从哪来

**BPF** 全称 Berkeley Packet Filter（伯克利包过滤器），1992 年诞生于加州大学伯克利分校，最初只是一个很小的虚拟机，专门用来在内核里快速判断"这个网络数据包要不要接收"（比如 `tcpdump` 抓包时写的过滤表达式，底层就是编译成 BPF 指令塞进内核跑的）。2014 年前后，Linux 内核把这个小虚拟机扩展成了一个通用的、可以安全运行用户提供的小程序的内核态执行环境，改名叫 **eBPF**（extended BPF，扩展的 BPF），字面含义就是"扩展版的包过滤器"——虽然现在的 eBPF 早就不只用来过滤网络包了，但名字沿用了下来。

"安全运行用户提供的程序"这句话听起来有点可怕——内核是最高权限的地方，运行"用户提供的代码"岂不是很危险？eBPF 解决这个担忧的办法是**验证器（verifier）**：你写的 eBPF 程序（通常用一个受限的 C 子集写，编译成 BPF 字节码）在被允许加载进内核之前，必须先过一个静态分析器的检查——它会确认这段代码不会死循环（必须在有限步数内结束）、不会访问越界内存、不会导致内核崩溃。通过验证后，内核还会把 BPF 字节码**即时编译（JIT，Just-In-Time Compilation，运行前现场翻译成当前 CPU 能直接执行的机器码）**成原生指令，所以运行速度接近手写的内核模块，不是解释执行那种慢速方案。

和"读 `/proc` 查快照"相比，eBPF 的本质区别是：**它不是去问"现在是什么状态"，而是在状态真正发生改变的那个瞬间（比如进程调用 `fork` 系统调用的那一条内核指令执行到的那一刻）被自动触发去记录。** 这叫"事件驱动"而不是"轮询查询"——没有"两次查询之间状态已经变了"这种时间窗口问题，因为记录动作和状态改变动作，就是同一条内核执行路径上紧挨着的两步，中间不可能有别的进程插进来篡改。这是它比 procfs 快照方案更"可信"的根本原因，也是本文开头场景里"不依赖外部挂载配置，直接拿内核真相"这句话的技术含义。

### 4.3 tracepoint / raw_tracepoint：这次用的 6 个钩子分别对应什么事件

**钩子（hook）** 在这里指内核代码里预先埋好的一个"插入点"：内核开发者在某些关键位置（比如"一个新进程刚刚 fork 出来"这个时刻）主动调用一个通知函数，如果你在这个通知点上挂了一段 eBPF 程序，内核执行到这里就会先跑一遍你的程序，再继续往下走。**tracepoint** 就是内核里这样一类"预埋好、专门留给追踪工具用"的插入点的统称；**raw_tracepoint** 是它的一个变体——普通 tracepoint 在触发时，内核会先把原始参数包装成一份带类型信息、格式友好的结构体，这个包装动作本身有一点点开销；`raw_tracepoint` 跳过包装，直接把原始参数（一组裸的 64 位数值，对应 [`bpf_identity.bpf.c:12`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L12) 里 `struct raw_args { u64 args[4]; }`）喂给你的程序，省掉这层包装开销，换来的代价是你自己要知道这几个参数分别是什么（文档里写清楚了每个事件对应哪几个参数）。

这次用的 6 个钩子，对应 6 种"进程身份可能发生变化"的内核事件：

```c
SEC("raw_tracepoint/sys_enter")            // 任意系统调用入口
SEC("raw_tracepoint/sched_process_fork")   // 进程 fork 出一个新进程/线程
SEC("raw_tracepoint/sched_process_exec")   // 进程执行 execve，替换成一个新程序
SEC("raw_tracepoint/sched_process_exit")   // 进程退出
SEC("raw_tracepoint/cgroup_attach_task")   // 进程被移动进另一个 cgroup
SEC("raw_tracepoint/cgroup_transfer_tasks")// 一批进程被整体转移 cgroup
```
—— [`bpf_identity.bpf.c:76-106`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L76)

`SEC(...)` 是一个宏（[`bpf_identity.bpf.c:6`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L6)），把紧跟着的函数放进 ELF 文件（一种可执行文件的标准格式）里一个按名字约定的"段"（section）里——`libbpf` 加载这个文件时，正是靠读这个段名知道"这个函数要挂到哪个钩子上"，而不需要你额外写配置文件。

`sys_enter` 这个钩子比较特殊——它在**每一次**系统调用发生时都会触发（系统调用是程序向内核请求服务的唯一通道，比如读文件、分配内存都要经过它），如果什么都记录会非常昂贵。代码里先做一次过滤：

```c
SEC("raw_tracepoint/sys_enter")
int on_syscall(struct raw_args *context) {
    /* CUDA enters the driver through ioctl, while getpid supports the startup handshake. */
    if (context->args[1] != IOCTL_NR && context->args[1] != GETPID_NR) return 0;
    return capture(current_task());
}
```
—— [`bpf_identity.bpf.c:76-81`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L76)

`args[1]` 是这次系统调用的编号，只在编号等于 `ioctl`（`IOCTL_NR`）或 `getpid`（`GETPID_NR`）时才继续往下做记录，其余系统调用直接返回不做任何事。为什么偏偏挑这两个：NVIDIA 的 CUDA 驱动没有自己的系统调用号，所有和 GPU 的交互（分配显存、提交计算任务）全部是通过标准的 `ioctl`（设备控制）系统调用完成的，所以"这个进程调用了一次 `ioctl`"是"这个进程正在和 GPU 驱动打交道"的一个很强信号；`getpid` 则是代码注释里提到的"启动握手"场景——一个刚被任务容器启动、还没来得及做任何 GPU 操作的新进程，用一次 `getpid`（查询自己的进程号，几乎所有程序启动时都会间接触发）就能尽早被记录下来，不用非得等到它真的开始用 GPU。

### 4.4 BPF map 是什么：三张表怎么配合完成一次身份确认

**BPF map** 是一块由内核维护、同时能被"内核里跑的 BPF 程序"和"用户态的普通程序"读写的键值存储（key-value store）——你可以把它想象成一个内核态和用户态共享的字典，内核侧用专门的几个函数（`lookup`/`update`/`remove_key`，对应 [`bpf_identity.bpf.c:17-19`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L17) 里那几个函数指针，它们实际是内核提供的"BPF helper 函数"，通过固定编号调用，不是真的 C 函数调用）操作，用户态侧用 `bpf_map_lookup_elem`/`bpf_map_update_elem` 这类系统调用操作（对应 [`bpf_identity.py:36-38`](../../../psrl/workers/env_service/bpf_identity.py#L36)）。

这次代码定义了三张 map：

```c
struct { UINT(type, 1); UINT(max_entries, 256);  TYPE(key, u64); TYPE(value, u64);       } registrations SEC(".maps");
struct { UINT(type, 1); UINT(max_entries, 8192); TYPE(key, u32); TYPE(value, struct identity); } identities SEC(".maps");
struct { UINT(type, 2); UINT(max_entries, 1);    TYPE(key, u32); TYPE(value, u64);       } health SEC(".maps");
```
—— [`bpf_identity.bpf.c:13-15`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L13)

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
—— [`bpf_identity.bpf.c:57-74`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L57)（省略了同进程组校验的几行，整段见源码）

1. 先拿到这个进程的 `TASK_TGID`（线程组 ID，多线程程序里所有线程共享同一个值，相当于"进程 ID"）。
2. 调 `owner()`（[`bpf_identity.bpf.c:38-55`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L38)）：从这个进程当前所属的 device cgroup 开始，沿着 cgroup 的父子链一层一层往上查 `registrations` 表，直到找到一个被注册过的祖先 cgroup（最多查 16 层）——找到了就返回它的 `cookie`，这就是"确认这个进程归属于哪次已注册的任务"的那一步。
3. 如果 `owner()` 没查到任何已注册的祖先（`cookie` 为 0），说明这个进程根本不归任何正在被追踪的任务管，直接把 `identities` 表里这个 PID 的旧记录删掉（`remove_key`）并结束——**不记录"不确定"的状态，宁可什么都不写**。
4. 查到了就把 `{birth_ns, cgroup, cookie}` 写进 `identities` 表（`update`）。

举一个具体的手算例子：假设任务 A 的 device cgroup 的身份标识是 `9001`，`bind()` 时往 `registrations` 表写入 `{9001: cookie_A}`（`cookie_A` 是一个随机数，比如 `0x7f3a...`）。任务 A 容器里跑起来一个新进程 PID=4201，它 `fork` 的时候触发 `on_fork` 钩子调用 `capture()`：读到它的 device cgroup 正是 `9001`，`owner()` 在 `registrations` 表里查到 `9001 → cookie_A`，于是往 `identities` 表写入 `{4201: {birth_ns: ..., cgroup: 9001, cookie: cookie_A}}`。之后用户态查询 `owners([4201])` 时，发现记录里的 `cgroup` 和 `cookie` 都和自己持有的凭证吻合，就确认"PID 4201 确实属于我"（对应 [`bpf_identity.py:197-200`](../../../psrl/workers/env_service/bpf_identity.py#L197) 的判断逻辑：`record.cookie == self.cookie and record.cgroup == self.inode`）。

如果这个进程后来被移动到了别的 cgroup（比如管理员手动 `cgroup_attach_task` 把它挪到了 `9002`），`on_move` 钩子触发 `invalidate()`，直接把 `identities` 表里 PID=4201 这条记录删掉——下次查询就查不到了，归属关系自动失效，不需要用户态再做额外的"过期检测"。

### 4.5 BTF 与"不写死内核偏移量"：bpf_layout.py 怎么工作

前面 `capture()` 函数里有一步"读这个进程的 device cgroup"，代码里写的是：

```c
INLINE void *device_css(void *task) {
    void *set = 0, *css = 0;
    if (READ(set, task, TASK_CGROUPS) || !set) return 0;
    if (READ(css, set, CSS_SUBSYS + DEVICES_INDEX * 8)) return 0;
    return css;
}
```
—— [`bpf_identity.bpf.c:31-36`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L31)

`task`（指向内核的 `task_struct` 结构体，代表一个进程/线程）往下走 `TASK_CGROUPS` 这个字节偏移量，就能拿到它的 `cgroups` 字段；再往下走 `CSS_SUBSYS + DEVICES_INDEX * 8` 字节，就能拿到"devices 控制器"这个子系统对应的具体数据。这些"偏移量"（某个字段相对结构体起始地址偏了多少字节）取决于内核源代码里这个结构体字段的排列顺序——**不同版本的 Linux 内核，这个顺序可能不一样**（内核开发者加字段、调整字段顺序是很常见的事）。如果把这些偏移量写成固定数字硬编码进代码，换一个内核版本运行就可能读错内存位置，轻则读到垃圾数据直接被 verifier 拒绝加载，重则（理论上）读到不该读的地方。

**BTF**（BPF Type Format，BPF 类型格式）就是为了解决这个问题而设计的：从某个版本开始，Linux 内核在编译时会把自己所有内部数据结构的"类型描述"（每个结构体有哪些字段、每个字段的类型和偏移量）打包进一份元数据，暴露在 `/sys/kernel/btf/vmlinux` 这个文件里。只要在目标机器上实际读一下这份元数据，就能**现场**算出"在这台机器正在跑的这个内核版本里，`task_struct.cgroups` 到底偏移了多少字节"，不用提前猜、不用为每个内核版本单独编译一份代码。这正是 [`bpf_layout.py`](../../../psrl/workers/env_service/bpf_layout.py) 这个文件做的事（这种"针对运行时内核现场适配，而不是为每个版本预编译"的思路，业界的通用叫法是 CO-RE，Compile Once – Run Everywhere，"编译一次、到处能跑"）。

`KernelTypes.__init__`（[`bpf_layout.py:10-32`](../../../psrl/workers/env_service/bpf_layout.py#L10)）先读这份 BTF 文件的二进制头部，确认格式版本（magic number `0xEB9F` 是 BTF 格式的固定标识，类似很多文件格式开头都有一个"签名"用来快速识别），然后把文件剩下的部分解析成一个类型列表，每一项记录"这是什么类型（结构体/枚举/指针等等）、叫什么名字、里面有哪些成员"。`member()`（[`bpf_layout.py:51-59`](../../../psrl/workers/env_service/bpf_layout.py#L51)）就是在这个列表里按"结构体名+字段名"查出对应的字节偏移量：

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
—— [`bpf_layout.py:69-86`](../../../psrl/workers/env_service/bpf_layout.py#L69)

算出来的这一串数字（比如 `TASK_CGROUPS=112` 表示 `cgroups` 字段在 `task_struct` 里偏移 112 字节），会被 [`bpf_identity.py:73-75`](../../../psrl/workers/env_service/bpf_identity.py#L73) 写成一个临时的 C 头文件 `identity_offsets.h`（每行一句 `#define TASK_CGROUPS 112` 这样的宏定义），这份头文件会在**编译 `.bpf.c` 的时候**被 `#include` 进去（[`bpf_identity.bpf.c:2`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L2)）——这样 C 代码里写的永远是 `TASK_CGROUPS` 这个符号名，实际数值是运行时现场生成、现场编译进去的，换一台内核版本不同的机器跑，这份头文件的内容会自动不一样，但 `.bpf.c` 源码一个字都不用改。

### 4.6 没有现成库时，Python 怎么自己调用 libbpf

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
—— [`bpf_identity.py:21-43`](../../../psrl/workers/env_service/bpf_identity.py#L21)（节选）

`ctypes.util.find_library("bpf")` 在系统标准路径里找名叫 `libbpf` 的动态库文件；`c.CDLL(name, use_errno=True)` 加载它，`use_errno=True` 让 ctypes 在每次调用后帮你把 C 层面的全局错误码 `errno`（一个约定：C 函数执行失败时，不直接返回错误信息，而是把一个数字错误码写进一个全局变量，调用者失败后再去读这个变量判断具体错在哪）同步保存下来，之后能用 `c.get_errno()` 读取。`function.restype`/`function.argtypes` 就是在显式声明"这个函数返回什么类型、接受哪些类型的参数"——比如 `bpf_object__open_file` 返回一个 `c_void_p`（通用指针类型，这里实际代表"BPF 对象"这个 C 结构体的地址，Python 侧不需要知道这个结构体内部长什么样，只需要把这个指针原样传回给下一个 libbpf 函数）。

拿到声明好签名的 `lib` 之后，整个加载流程就是照着 libbpf 的标准用法依次调用：`bpf_object__open_file` 打开编译好的 `.o` 文件 → `bpf_object__load` 把它真正加载进内核（这一步会触发内核的 verifier 检查）→ `bpf_object__find_map_fd_by_name` 按名字拿到每张 map 的文件描述符（**文件描述符** 是 Linux 里代表"一个打开的资源"的小整数句柄，不管是真的文件、网络连接还是这里的 BPF map，内核都用同一套"文件描述符"机制统一表示，后续读写这个资源都靠这个小整数定位）→ `bpf_raw_tracepoint_open` 把某个 BPF 程序挂到某个钩子上。整段代码本质上就是用 Python 把 libbpf 这个 C 库原本要用 C 代码调用的那套流程，原样复刻了一遍，只是语法换成了 Python。

### 4.7 为什么任务容器自己不能有 CAP_BPF/CAP_PERFMON

加载 eBPF 程序、挂追踪钩子这类操作，涉及读取内核内部数据结构（比如前面讲的 `task_struct` 字段），能力很强，所以 Linux 要求调用者必须拥有相应的**权能（capability）**——这是一种比"root/非 root"更细粒度的权限划分机制：传统的 Unix 权限只有"root 能做任何事，非 root 什么都不能做"两档，权能机制把 root 原本拥有的"任何事"拆成了几十个具体的小权限（比如"能不能修改系统时间"、"能不能绑定 1024 以下的端口"），一个进程可以只被单独授予其中几项，而不是全有或全无。和这次改动相关的几个：`CAP_SYS_ADMIN`（历史上的"万能权能"，覆盖一大堆系统管理操作）、`CAP_BPF`（较新内核版本里专门拆出来的、专门管"加载 BPF 程序"的权能）、`CAP_PERFMON`（专门管"用于性能监控/追踪目的的内核数据访问"的权能）。

这组改动里有一处检查：

```python
capabilities = {name.removeprefix("CAP_") for name in config.get("CapAdd") or ()}
if capabilities & {"ALL", "SYS_ADMIN", "BPF", "PERFMON"}:
    raise RuntimeError("GPU command containers cannot administer device cgroups or kernel tracing.")
```
—— [`command_gpu_node.py:22-24`](../../../psrl/workers/env_service/command_gpu_node.py#L22)

这段检查在**创建用户任务容器**的时候生效，确认的是"这个任务容器自己的 Docker 配置，有没有被额外授予这几项权能"，如果有就直接拒绝创建。原因不难理解：eBPF 身份后端本身是运行在 `env_service` 节点服务进程里的（它拥有必需的权能，这是运维在部署节点服务时单独配置好的），**不是**跑在用户任务容器里；如果反过来让任务容器自己也拿到 `CAP_BPF`/`CAP_PERFMON`，那么一个恶意或者有 bug 的任务代码，理论上就能在自己的容器里加载 BPF 程序，直接读写内核维护的那几张 map——包括前面讲的 `identities`/`registrations` 表，也就是**自己伪造"这个显存该归我"的身份记录**，让系统误以为别的任务的显存归自己名下。这条检查就是从源头堵死这个篡改路径：追踪的权力只能握在管理者（节点服务）手里，被管理者（任务容器）绝对不能拿到。

### 4.8 为什么用 "devices" 这个 cgroup v1 控制器，以及 inode 桥接技巧

这组代码的注释里反复出现"cgroup v1 devices controller"的字样（比如 [`bpf_identity.py:1`](../../../psrl/workers/env_service/bpf_identity.py#L1) 的 docstring："Maintain trusted process identities from registered **cgroup v1** kernel events"）。这需要补一点背景：[cgroup v2 资源控制笔记 §"为什么要从 v1 重做成 v2"](../../../../docs/operating-systems/cgroup-v2-resource-control.md#为什么要从-v1-重做成-v2) 讲过 v1 到 v2 的改造动机，但那篇笔记没有展开讲一个具体细节——**devices 控制器**（专门管"这个 cgroup 里的进程能不能访问哪些设备文件，比如 `/dev/nvidia0`"的那个控制器）在 v2 里被整个移除了接口，改成了用 BPF 程序挂载在 cgroup 上做设备过滤决策。但很多现实世界的容器运行时（包括这里涉及的 Docker）在管理 GPU 设备白名单时，仍然沿用 v1 风格的 "devices" 控制器挂载点（`/sys/fs/cgroup/devices`，常见于"混合模式"——同一台机器同时挂载了 v2 统一层级和个别 v1 遗留控制器）。*这一段关于 Docker 具体实现细节属于通用背景知识，本工作区没有重新核实 Docker/runc 源码，仅供理解上下文参考。*

这次代码读写的正是这个 v1 风格的 "devices" 挂载点：[`command_gpu_node.py:18`](../../../psrl/workers/env_service/command_gpu_node.py#L18) 里 `DeviceCgroupFence` 的默认根路径就是 `/sys/fs/cgroup/devices`，`set_access()`（[`command_gpu_node.py:63-78`](../../../psrl/workers/env_service/command_gpu_node.py#L63)）通过写 `devices.allow`/`devices.deny` 这两个 v1 专属的控制文件来临时开启/关闭容器对某块 GPU 设备的访问权限（这就是"栅栏/Fence"这个类名的含义——像一道可以随时拉起/放下的闸门）。

这里还有一个精巧的"桥接"技巧值得展开：BPF 程序在内核里判断"这个进程属于哪个已注册 cgroup"，用的键是 `kernfs_node.id`（**kernfs** 是内核里专门用来实现"看起来像文件系统、但内容是内核数据实时生成"这类虚拟文件系统的通用框架，cgroup 文件系统正是用它实现的；`kernfs_node` 就是这个框架里代表"一个目录/文件节点"的内部结构体，`id` 字段是这个节点的唯一编号）。而用户态 Python 代码在 `bind()` 时，是直接对这个 cgroup 目录做标准的文件系统调用拿到它的 inode 号：

```python
self.directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
self.inode = os.fstat(self.directory_fd).st_ino
```
—— [`bpf_identity.py:182-183`](../../../psrl/workers/env_service/bpf_identity.py#L182)

**inode**（index node，索引节点）是 Linux 文件系统里"一个文件/目录的唯一编号加元数据"，`os.stat()` 返回的 `st_ino` 就是这个编号——这是一个通用的 VFS（虚拟文件系统，内核对各种具体文件系统的统一抽象层）概念，不是 cgroup 专属的东西。关键在于：kernfs 框架在实现"让 cgroup 目录看起来像普通文件系统目录"的时候，**直接把自己的 `kernfs_node.id` 当成这个目录对外暴露的 inode 号**，这不是巧合，是 kernfs 的设计决定——所以用户态用标准的 `os.fstat().st_ino` 读到的数字，和内核态 BPF 程序用 BTF 偏移量读到的 `kernfs_node.id` 字段，**读的是完全同一个数值**，两边不需要任何额外的"翻译表"就能对上号。这就是为什么 `bind()` 里能直接把 `os.fstat` 读到的 inode 号当作 `registrations` 表的键写进去（[`bpf_identity.py:155-156`](../../../psrl/workers/env_service/bpf_identity.py#L155)），内核里的 `owner()` 函数后续用同一个数值去查表（[`bpf_identity.bpf.c:46-50`](../../../psrl/workers/env_service/bpf_identity.bpf.c#L46)）就一定能对上——这正是用户态注册和内核态追踪能够"认同一个身份证号"的根本原因。

## 5. GPU 显存会计公式：accounted_gpu_memory_mb 手算

确认了"哪些进程属于哪个任务"之后，下一步是把这个归属关系换算成"准入判断该用的显存总量"。这件事比看起来复杂，因为显存的分配过程本身有时间差：一个任务被**允许**使用某块 GPU（拿到"预约"）和它的代码真正跑起来、CUDA 驱动真正**完成分配**（变成 NVML 能测到的"实测占用"）之间，可能隔着几秒甚至更久的初始化时间。如果只看"实测占用"，会漏算那些"已经批准但还没来得及真正占用"的显存，造成超卖；如果简单地"预约量"全部累加，又会在任务已经真正开始使用之后重复计费。

```python
def accounted_gpu_memory_mb(measured_mb, reservations, owned_mb=None):
    owned_mb = owned_mb or {}
    ...
    return measured_mb + sum(
        max(0.0, amount - owned_mb.get(lease_id, 0.0)) for lease_id, amount in reservations.items()
    )
```
—— [`gpu_memory.py:9-28`](../../../psrl/workers/env_service/gpu_memory.py#L9)（节选核心逻辑）

公式的含义："总账 = 实测用量 + 每个任务（预约量 − 这个任务已被核实确实占用的那部分）"，`max(0.0, ...)` 保证"已核实占用超过预约量"的情况不会倒扣。两个来自真实测试用例（[`tests/env_service/test_gpu_memory.py:25-28`](../../../tests/env_service/test_gpu_memory.py#L25)）的手算例子：

**例 1：没有归属信息时，预约量不能被打折。** `accounted_gpu_memory_mb(70, {"a": 15})`：实测用了 70MB，任务 a 预约了 15MB，没有 `owned_mb` 参数（意味着"还不知道这 70MB 里有没有 a 的份"）。代入公式：`70 + max(0, 15 - 0) = 85`。注意即使实测值后来涨到了 84（`accounted_gpu_memory_mb(84, {"a": 15})`），只要仍然没有归属确认，结果是 `84 + max(0, 15 - 0) = 99`——**哪怕实测用量的增长恰好等于 a 的预约量，系统也绝不会"猜"这增长就是 a 用的**，这正是测试断言里那句话的意思："Unknown ownership cannot be assumed to be ours"（未知归属不能被假定为己方占用）。

**例 2：一旦核实归属，同一笔钱不能收两次。** 紧接着的第三个断言：`accounted_gpu_memory_mb(84, {"a": 15}, {"a": 14})`。这次多了 `owned_mb={"a": 14}`——意味着通过第 4 节讲的身份确认流程，查到任务 a 确实有进程在用，而且核实出来的占用是 14MB（这 14MB 已经包含在实测值 84 里了，不是额外加的）。代入公式：`84 + max(0, 15 - 14) = 84 + 1 = 85`。对比例 1 的 99，少了 14——这 14 正是"已经被核实、已经包含在 84 这个实测值里的那部分，不应该在后面的预约量里再加一遍"。测试断言写的是："Verified memory must not be charged twice"（已核实的显存不能被收两次费）。

这条公式直接决定能不能再准入一个新任务——`EnvUnitLedger`（已有的账本模块）拿这个总量跟 GPU 的物理总容量比较，超过就拒绝或排队。

## 6. 资源画像：harbor_resources + container_budget 手算一个任务的预算

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
—— [`container_budget.py:6-29`](../../../psrl/workers/env_service/container_budget.py#L6)（节选）

`environments` 只能取 1 或 2——这对应 harbor 的一个设计：验证器（verifier，负责判断任务是否做对了的那部分逻辑）可以要求"独立环境"（`environment_mode == "separate"`），也就是验证器和主任务分别起一套完整的容器集合，互不干扰；不要求独立时，验证器复用主任务那一套容器，`environments=1`。

手算一个具体例子：假设一个任务的 `task.toml` 声明 `environment.cpus = 2.0`、`environment.memory_mb = 4096`，它的验证器要求 `environment_mode = "separate"`，它的 `docker-compose.yaml` 里除了 `main` 还声明了 2 个 sidecar 服务。走一遍 [`demand_from_task`](../../../psrl/workers/env_service/harbor_resources.py#L22) 和 [`with_container_budget`](../../../psrl/workers/env_service/harbor_resources.py#L71)：

1. `cpus=2.0`、`memory_mb=4096` 先被读进 `EnvDemand`。
2. `with_container_budget` 里 `separate=True`（因为验证器要求独立环境），`environments = 2 if separate else 1` → `2`。
3. `cpu = max([2.0, 1.0] + 各环境自己的 cpus) = 2.0`；`memory = max([4096] + 各环境自己的 memory_mb) = 4096`（代码见 [`harbor_resources.py:83-84`](../../../psrl/workers/env_service/harbor_resources.py#L83)）。
4. 扫描 `docker-compose.yaml`，`sidecars = 2`（排除 `main`），`max_sidecars = sidecars + 1 = 3`（代码注释解释了这个 `+1`："Harbor 在准入之后可能还会给任意一边环境再加一个网络服务"，所以预算要多留一个名额，见 [`harbor_resources.py:89-94`](../../../psrl/workers/env_service/harbor_resources.py#L89)）。
5. 代入公式：`budget.cpu = 2 × (2.0 + 3 × 0.25) = 2 × 2.75 = 5.5`（核）；`budget.memory_mb = 2 × (4096 + 3 × 128) = 2 × 4480 = 8960`（MB）。

这意味着：这个任务虽然自己只声明要 2 核 4GB，但因为验证器要独立环境、又带了 2 个 sidecar，它在节点上实际会被预留 **5.5 核、8960MB**——这个数字既是"调度时判断这台节点还有没有位置"用的那个数字，也是真正创建容器时写进 `cpus`/`mem_limit` 的那个数字（`service_limits()`，[`container_budget.py:31-46`](../../../psrl/workers/env_service/container_budget.py#L31)），**两处用的是同一份计算结果**，不会出现"说要 5 份资源、却只真正限制了 2 份"的口径不一致。

## 7. 节点健康状态机：NodeHealth 的熔断与恢复

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
—— [`health.py:6-26`](../../../psrl/workers/env_service/health.py#L6)

这是一个很朴素的计数器状态机：每次探测成功就把 `failures` 清零，探测失败就加一。配合 `router.py` 里的使用方式手算一遍一个失联场景：假设 `health_max_failures=3`（[`router.py:336`](../../../psrl/workers/env_service/router.py#L336) 的默认值）。

- 第 1 次探测失败（比如节点进程暂时没响应）：`failures=1`，`_health_task`（[`router.py:423-436`](../../../psrl/workers/env_service/router.py#L423)）发现 `1 < 3`，下一个探测周期还会继续排新的检查任务。
- 第 2、3 次都失败：`failures=3`。此时 `_health_task` 里 `state.failures >= self._health_max_failures` 成立（[`router.py:430`](../../../psrl/workers/env_service/router.py#L430)），**直接 return，不再排新的健康检查任务**——这就是"熔断"：既然已经连续失败到阈值，继续高频探测一个大概率已经死掉的节点没有意义，干脆先停下来。
- 与此同时，`node_capacity()`（[`router.py:438-444`](../../../psrl/workers/env_service/router.py#L438)）每次被上层调度逻辑调用时，都会把 `state.available`（此刻为 `False`，因为 `error` 非空）写进返回的 `NodeCapacity.available` 字段——**调度器天然就会因为这个标志不再往这个节点派新任务**，不需要专门写一条"如果节点坏了就别调度过去"的额外分支逻辑，坏节点自己把"不可用"这件事体现在它返回的容量信息里了。

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
—— [`router.py:413-419`](../../../psrl/workers/env_service/router.py#L413)

只有这三项都为空，才会真正把这个节点的服务句柄清空——清空之后，下次有新任务需要这个节点，`_create_node`（[`router.py:356`](../../../psrl/workers/env_service/router.py#L356)）会重新创建一个全新的 `EnvNodeService` Actor，这才是真正的"恢复"。如果还有未了结的资源，宁可先保留这个（已经死掉的）句柄不清空，也不贸然重建——因为一旦句柄被清空，这些"谁该负责清理"的信息就会跟着丢失，造成 GPU 资源永久卡死而没人知道该去回收。

## 8. 协程知识回顾 + 两个新模式

### 8.1 快速回顾：三层叠加模型

[并发/并行全景](../../../../docs/psrl/psrl-concurrency-landscape.md) 已经讲过 psrl 里"协程（单线程内交替推进）→ 专用线程（跑自己的事件循环，比如避免阻塞调用拖慢主循环）→ Ray Actor 进程（真正的并行）"这三层是怎么叠在一起的，也讲过 `asyncio.create_task` 怎么"建好任务就返回、不等它跑完"（[文档 §5②](../../../../docs/psrl/psrl-concurrency-landscape.md#2-全局图一次-sciaccel-episode-要穿过几层并发原语)）。这次新代码里 `resource_telemetry.py` 和 `routing_pause.py` 用到的都还是"协程"这一层，但用了两个该文档还没出现过的、更精细的协程控制模式，下面展开讲。

### 8.2 新模式 A：有限并发 + 截止时间收尾（resource_telemetry.py）

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
—— [`resource_telemetry.py:24-59`](../../../psrl/workers/env_service/resource_telemetry.py#L24)（节选）

这里要解决的问题是：一个节点上可能同时跑着几十个容器，要查每个容器的资源用量（调 Docker 的 `stats` 接口），但 Docker daemon 在压力大的时候，单次查询可能会很慢甚至挂住不返回。如果写成"挨个查、每个都等到返回"，一个慢查询就能拖垮整次采集。这段代码用了三个配合的手段：

1. **`asyncio.Semaphore(8)`**（信号量，一种"最多同时允许 N 个人进门"的并发限流器，在第 2 节的全局图文档里也用过类似的闸门概念）把同时在查的容器数量限制在 8 个以内，避免对 Docker daemon 造成瞬时压力。
2. **`asyncio.wait(tasks.values(), timeout=timeout_s)`**：给整批查询设一个统一的截止时间，到点之后不管还有多少个任务没完成，`wait` 都会立刻返回（而不是无限等下去）——用列表推导式里的 `if task.done() else {...}` 判断，没赶上截止时间的任务就填一个"超时"的占位结果，**不让一个慢容器拖累整批数据都拿不到**。
3. **`finally` 块里先 `cancel()` 所有还没完成的任务，再 `await asyncio.gather(..., return_exceptions=True)` 等它们真正结束**：这是容易被忽略的一步——`asyncio.wait` 超时返回之后，那些没完成的任务**并没有自动停止**，它们还在后台运行（比如还在等 Docker 的网络响应）。如果不管它们，这些"孤儿任务"会一直占着资源，甚至在没人等待的情况下，任务内部抛出的异常会变成一条容易被忽视的警告日志。`cancel()` 主动给这些任务发送取消信号，`gather(..., return_exceptions=True)` 等它们真正响应取消、彻底收尾（`return_exceptions=True` 保证即使某个任务取消时抛了异常，也不会打断对其他任务的等待）。

### 8.3 新模式 B：多持有者引用计数 + 取消安全清理（routing_pause.py）

`RoutingPauseController` 要解决的问题：训练过程中，模型权重同步、副本迁移等好几个独立的操作，都可能需要"暂停往下游派发新的推理/生成请求"，而且这些操作可能**并发**发生（比如正在同步权重的同时，又触发了一次副本迁移）。如果简单地用一个布尔值记录"是否暂停"，第一个操作做完就把它改回 `False`，会错误地提前恢复路由——哪怔第二个操作还没做完。

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
—— [`routing_pause.py:24-38`](../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L24)

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
—— [`routing_pause.py:40-59`](../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L40)

`@asynccontextmanager` 把一个写成"先做准备工作、`yield` 一下、再做收尾工作"的协程函数，包装成一个可以配合 `async with` 使用的"异步上下文管理器"——`async with controller.pause(): ...` 这种写法，进入代码块前自动跑 `yield` 之前那部分（这里是 `_acquire`，登记一个 hold），离开代码块（无论正常结束还是中途抛异常）自动跑 `finally` 里那部分（这里是 `_release`，释放这个 hold）。

真正精巧（也是本文想重点讲的新知识）的是 `finally` 块里处理"取消"（cancellation）的方式。协程的**取消**是 `asyncio` 里一种特殊的异常注入机制：外部代码可以调用 `task.cancel()`，这会在这个协程当前正挂起等待的那个 `await` 点上，强行注入一个 `asyncio.CancelledError` 异常，让它"看起来像是那个等待操作自己失败了"——协程可以选择捕获这个异常做清理，但默认行为是继续往外传播，最终让这个协程整体终止。

问题是：如果调用方正在执行 `pause()` 的 `finally` 块（也就是正在做"释放 hold、把暂停状态写回去"这件收�020尾工作）的过程中，**又**被取消了一次（比如上层调用方自己也被取消，取消信号层层传递下来），默认情况下这第二次取消会直接中断 `_release` 的执行——可能刚把 `self._holds.remove(owner)` 执行完、还没来得及调用 `_write_running` 把"恢复运行"的信号真正写出去就被打断了，造成"集合里明明已经没有任何 hold 了，但路由状态却一直卡在暂停"的不一致。

`asyncio.shield(cleanup)`（**shield**，字面意思"盾牌"，这里指"保护某个协程不被外部的取消信号直接打断"）就是为了防住这第二次取消：`cleanup = asyncio.create_task(self._release(owner))` 先把清理工作包装成一个独立的 Task（Task 有自己的生命周期，不会因为"等待它的那一方被取消"而自动跟着被取消）；`await asyncio.shield(cleanup)` 等待这个 Task，但即使等待方（当前协程）被取消，`shield` 也只会让"等待"这个动作本身抛出 `CancelledError`，`cleanup` 这个 Task 不受影响，继续在后台跑完。代码用一个 `while not cleanup.done()` 循环反复尝试等待它——哪怕中途再被取消几次（每次把 `cancelled` 标记设为 `True`），也会一直等到 `cleanup` 真正跑完为止，保证清理工作**一定会被完整执行**；最后 `cleanup.result()` 取出结果（如果清理过程本身出了异常，这里会重新抛出），如果期间确实被取消过，再手动 `raise asyncio.CancelledError`——**把"我被取消了"这个事实如实告诉外部**，只是不让这次取消打断必须完成的清理工作。这是协程编程里一个经典但容易被忽略的教训：**取消信号可以、也应该被妥善处理，但绝不能让它打断必须完整执行的收尾逻辑。**

## 9. 可运行实验脚本

下面这份脚本把第 5/7/8 节讲到的四段真实逻辑（显存会计公式、容器预算、健康状态机、多持有者暂停控制器）原样搬过来跑一遍，再加一个纯 Python 手写的"迷你内核"模拟第 4 节的 eBPF 身份追踪状态机（真实的 eBPF 需要 Linux 内核 + root 权限，在这里只能用等价的数据结构和逻辑模拟，方便在任何机器上直接观察状态变化）。只用标准库，可以直接用 `python3` 运行：

```python
"""
自包含教学复刻脚本：对应 docs/psrl/env-worker-deep-dive/gpu-admission-attribution-ebpf-identity.md
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

暂停控制器部分的真实行为比我最初设想的更细致一点，值得纠正：**不是"只有首次加入和末次释放才触发写入"，而是每一次 `acquire`/`release` 都会触发写入，只是写入的那个布尔值是由"当前持有者集合是否为空"实时计算出来的**——B 自己进入时也会触发一次写入（第 2 次，`async with` 嵌套调用 `_acquire`，而 `_acquire` 内部无条件执行 `await self._write_running(False)`，见 [`routing_pause.py:24-29`](../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L24)），B 释放时也会触发一次写入（第 3 次，但因为 A 还在持有集合里，`self._running and not self._holds` 算出来仍是 `False`）；只有第 4 次——也就是最后一个持有者（A）释放、集合彻底清空的那一刻，这个表达式才第一次算出 `True`。这仍然是"引用计数"的核心思想：**决定对外生效的状态值，是整个持有者集合的大小（是否为空），而不是任意单次 acquire/release 动作本身**；只是这次代码选择了"每次都写、但写之前先用集合状态算出正确的值"这种更简单直接的实现方式，而不是额外判断"是否需要跳过这次写入"。

脚本保存在 [`scripts/env_admission_attribution_demo.py`](../env-service-verification/scripts/env_admission_attribution_demo.py)。

## 10. 一句话总结 + 延伸阅读

这组改动的动机：**当"运行 env_service 的节点本身也是一个容器"时，不能再相信容器自己视角里的 `/proc`，所以用 eBPF 直接在内核态拿到不依赖命名空间视角、事件驱动、不可伪造的进程归属真相，再配上一条不会重复计费的显存会计公式、一份调度和创建共用同一个口径的容器预算、一个让坏节点自动从调度候选中消失但不会贸然销毁未清理资源的健康状态机——四件事共同保证"显存和 CPU/内存记账记得准，节点挂了能安全恢复"。**

- [返回专题入口](../README.md)
- psrl 仓库自带的权威设计文档（本文假设读者已经点开过，如果还没读建议先读这两篇再回来看本文的"零基础展开"部分）：
  - [`process-identity-without-procfs.md`](process-identity-without-procfs.md)
  - [`gpu-memory-admission.md`](gpu-memory-admission.md)
  - [`unified-gpu-execution.md`](unified-gpu-execution.md)
- 协程/线程/进程三层叠加的基础模型：[并发/并行全景](../../../../docs/psrl/psrl-concurrency-landscape.md)
- cgroup / procfs 更系统的背景知识：[cgroup v2 资源控制知识笔记](../../../../docs/operating-systems/cgroup-v2-resource-control.md)、[procfs 知识笔记](../../../../docs/operating-systems/proc-filesystem.md)
