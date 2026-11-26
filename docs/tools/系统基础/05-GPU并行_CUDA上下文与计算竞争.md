# GPU并行：CUDA上下文与计算竞争

> **核心结论**：操作系统调度 CPU 线程，GPU 将 kernel（计算程序）的线程块交给 SM（流式多处理器）执行。多个进程共享 GPU 时，还要考虑 CUDA 上下文、计算竞争和 MPS/MIG 的执行边界。本篇先解释硬件怎样组织工作，再落到 PSRL 的显存与共享例子。

## 目录

- [从 CPU 到 GPU 的并行结构](#从-cpu-到-gpu-的并行结构)
  - [1. 先看结论：一张对照表](#1-先看结论一张对照表)
  - [2. CPU 的设计出发点：为什么核心天生能"独立分配给任意进程"](#2-cpu-的设计出发点为什么核心天生能独立分配给任意进程)
  - [3. GPU 的设计出发点：为什么 SM 天生不是按进程分配的](#3-gpu-的设计出发点为什么-sm-天生不是按进程分配的)
  - [4. 上下文切换成本：为什么 CPU 便宜、GPU 贵](#4-上下文切换成本为什么-cpu-便宜gpu-贵)
  - [5. 当 GPU 也要面对多租户：MPS 和 MIG 分别补的是哪一层](#5-当-gpu-也要面对多租户mps-和-mig-分别补的是哪一层)
  - [6. 回到 PSRL 的场景：这组差异意味着什么](#6-回到-psrl-的场景这组差异意味着什么)
  - [7. 延伸阅读](#7-延伸阅读)
- [CUDA 上下文、显存开销与计算竞争](#cuda-上下文显存开销与计算竞争)
  - [1. 从一组实测数字说起：64 MiB 申请为什么测出 380 MiB](#1-从一组实测数字说起64-mib-申请为什么测出-380-mib)
  - [2. CUDA 上下文是什么](#2-cuda-上下文是什么)
  - [3. 显存层面：这笔固定成本怎么被 PSRL 的账本处理](#3-显存层面这笔固定成本怎么被-psrl-的账本处理)
  - [4. 计算层面：没有 MPS 时，GPU 在两个上下文之间怎么调度](#4-计算层面没有-mps-时gpu-在两个上下文之间怎么调度)
  - [5. 把两层串回源码：跟着图读 vLLM 和环境命令的真实调用链](#5-把两层串回源码跟着图读-vllm-和环境命令的真实调用链)
  - [6. 一个可以手算、可以复现的任务依赖例子](#6-一个可以手算可以复现的任务依赖例子)
  - [7. 验证边界：已经证明了什么，还没证明什么](#7-验证边界已经证明了什么还没证明什么)

## 从 CPU 到 GPU 的并行结构

<!-- figure:gpu-capacity-and-compute -->
这张图沿同一个场景解释：用相同的10条命令，分别比较容量、计算竞争和整批完成时间。

[![GPU并发：装得下，不等于一起跑就划算（1/4）](../../_static/img/document-figures/gpu-capacity-and-compute-screen.png)](../../_static/img/document-figures/gpu-capacity-and-compute.png)

[![GPU并发：装得下，不等于一起跑就划算（2/4）](../../_static/img/document-figures/gpu-capacity-and-compute-screen-02.png)](../../_static/img/document-figures/gpu-capacity-and-compute.png)

[![GPU并发：装得下，不等于一起跑就划算（3/4）](../../_static/img/document-figures/gpu-capacity-and-compute-screen-03.png)](../../_static/img/document-figures/gpu-capacity-and-compute.png)

[![GPU并发：装得下，不等于一起跑就划算（4/4）](../../_static/img/document-figures/gpu-capacity-and-compute-screen-04.png)](../../_static/img/document-figures/gpu-capacity-and-compute.png)

图中90 GiB、每命令8 GiB，以及并发10秒或30秒、分两组共14秒都是教学假设，非新增实测。轻竞争与重竞争两种条件明确分开；两组各5条均按每组7秒计算。SM网格仅解释共享，不对应真实卡的SM数量，也不表示MPS提供固定硬分区。 [查看矢量图](../../_static/img/document-figures/gpu-capacity-and-compute.svg)。
<!-- /figure:gpu-capacity-and-compute -->

> **核心结论**：CPU 通常把独立线程交给不同核心运行；GPU 通常把一次计算拆成大量线程块，交给多个 SM 批量执行。SM（流式多处理器）是 GPU 内执行线程块的硬件单元。多个进程共享 GPU 时，还要考虑 CUDA 上下文、MPS 服务或 MIG 硬件分区，不能直接把“100 个 SM”换算成“100 个可独占的进程槽位”。

用一个例子贯穿：A、B 两个程序各有一项 GPU 计算。我们关心它们能否同时运行、怎样分配资源，以及其中一个失败会影响谁。以下讨论 NVIDIA CUDA 的常规多进程执行，具体支持能力取决于 GPU 和驱动配置。


### 1. 先看结论：一张对照表

| 问题 | 多核 CPU | CUDA GPU |
|---|---|---|
| 谁把工作交给硬件 | 操作系统把可运行线程分配到核心 | CUDA 程序启动 kernel，GPU 把线程块分配到 SM |
| 工作怎样拆分 | A、B 可以各有独立控制流程 | 一个 kernel 通常包含大量执行相近计算的线程块 |
| 多个进程怎样共用 | 不同核心可同时运行不同进程的线程 | 常规独立上下文的 kernel 按时间共享；MPS 可支持跨进程同时执行 |
| 要稳定划出资源怎么办 | CPU 亲和性、核心集合和资源组 | 支持的 GPU 可使用 MIG；MPS 的资源控制需按具体配置理解 |
| 是否只要空闲就更快 | 仍会竞争缓存、主存、带宽等 | 仍会竞争 SM、显存带宽、缓存等 |

kernel 是提交到 GPU 的一段计算程序，线程块是 GPU 安排其执行的基本组织单位。CUDA 上下文保存地址空间、已加载代码和执行相关状态。详细例子见[上下文教程](05-GPU并行_CUDA上下文与计算竞争.md)。

### 2. CPU 的设计出发点：为什么核心天生能"独立分配给任意进程"

CPU 擅长运行控制流程差异很大的程序，例如编辑器、数据库和网络服务。操作系统把每个进程中的线程登记为可调度对象，再挑选核心运行。每个核心能维护正在执行的寄存器和指令位置，因此 A 的线程可以在核心 0 运行，B 的线程同时在核心 1 运行。

这是简化的普通场景。实际调度还受 CPU 亲和性、资源组和处理器拓扑限制；两个线程也会争用共享缓存和主存带宽。因此“分到不同核心”说明可以并行，实际提速仍取决于工作负载。

### 3. GPU 的设计出发点：为什么 SM 天生不是按进程分配的

GPU 擅长把相近计算应用到大量数据。例如矩阵中的一百万个元素都需要相同类型的乘加，程序可以一次启动大量线程，由 GPU 把线程块分发到多个 SM。SIMT（单指令、多线程）是这种执行组织方式：一组线程执行共同的指令流，各自处理不同数据，分支会影响执行效率。

程序中的进程数量与 SM 数量属于不同层次。A 启动的一个大 kernel 就可能占用整张卡，也可能只占一部分；它能否与 B 的工作同时执行还取决于上下文、资源需求及执行模式。[CUDA 编程指南](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#concurrent-kernel-execution)说明了常规独立上下文与 MPS 的并发差异。

### 4. 上下文切换成本：为什么 CPU 便宜、GPU 贵

切换工作都需要保存和恢复执行状态。CPU 的影响包括寄存器切换、地址转换和缓存重新预热；GPU 的影响取决于架构、抢占粒度、正在执行的工作及驱动。CUDA 上下文本身还带来显存开销。

因此选择共享方式时，需要测“单独完成多久、一起执行多久、切换和初始化花多久”，再比较总完成时间。固定写成“CPU 一定几微秒、GPU 必然很重”，会掩盖硬件和工作负载的差异。NVIDIA 对上下文显存与时间成本的说明见 [CUDA 最佳实践](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-c-best-practices-guide/index.html#multiple-contexts)。

### 5. 当 GPU 也要面对多租户：MPS 和 MIG 分别补的是哪一层

#### 5.1 MPS：协调多个 CUDA 进程的同时执行

MPS（多进程服务）让多个 CUDA 客户端可以同时向 GPU 提交并执行工作。实现随架构演进：在 Volta 及后续架构上，每个客户端保留自己的 GPU 地址空间，并直接向 GPU 提交；服务端协调剩余共享资源。机制见 [NVIDIA MPS 架构](https://docs.nvidia.com/deploy/mps/595/architecture.html)。

例如 A、B 单独执行时都没用满 GPU，MPS 可能让它们的线程块同时运行，缩短整批时间。若两者都已占满相同的计算或带宽资源，则共享可能拉长各自耗时。资源上限和故障影响范围需要按架构与配置检查；PSRL 当前把它用于可信环境任务，见[MPS 后端](../../env-worker-deep-dive/knowledge/02-资源与调度/04-GPU并行_MPS执行后端.md)。

#### 5.2 MIG：把支持的 GPU 分成硬件实例

MIG（多实例 GPU）把物理卡划为拥有指定计算和显存资源的实例。A 使用实例 1，B 使用实例 2，可以得到比共同竞争资源池更明确的资源边界。代价是每个实例受分区容量限制：B 的空闲显存通常无法直接供 A 的实例使用。

例如两份任务各需 12 GiB，而某个分区配置给 A 的实例只有 10 GiB，整卡空闲很多也无法让 A 按原需求运行。需要选择合适的分区配置。支持型号、实例结构与隔离能力以[NVIDIA MIG 指南](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/introduction.html)为准，本文未配置或验证项目中的 MIG。

### 6. 回到 PSRL 的场景：这组差异意味着什么

当前有两条需要分别理解的路径：

- **环境专用 GPU 的 MPS 后端**：显式配置后按作业记账、并发、取消和回收，已经有真实 CUDA 功能证据。
- **vLLM 与环境共卡**：现有共享协议保护显存余量和模型控制操作，该分支尚未接入 MPS。计算竞争的大小仍需固定工作量测量。

资源准入判断“能否安全放下”，MPS/MIG 影响“进入之后怎样执行”，性能实验回答“整批是否更快”。三者需要配合。以 A 独占需 10 秒、B 独占需 10 秒为教学假设：若并发后各需 12 秒，整批由 20 秒降到 12 秒；若各需 22 秒，整批就变慢了。能并发本身无法决定哪一种结果出现。

### 7. 延伸阅读

- [CUDA 上下文开销与计算竞争](05-GPU并行_CUDA上下文与计算竞争.md)：从 64 MiB 申请与 380 MiB 观测的实际差异解释容量，再看任务依赖算例。
- [MPS 后端](../../env-worker-deep-dive/knowledge/02-资源与调度/04-GPU并行_MPS执行后端.md)：项目里的预算、进程身份和取消协议。
- [vLLM 与环境共享](../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md)：模型常驻、睡眠借卡与控制门禁。

[返回专题入口](../../env-worker-deep-dive/README.md)


## CUDA 上下文、显存开销与计算竞争

> **核心结论**：两个程序共享 GPU，先要分清“显存能否放下”和“计算是否互相拖慢”。CUDA 上下文是驱动为程序维护的 GPU 执行环境，会产生额外显存开销；计算竞争则发生在程序同时提交工作时。PSRL 用实测占用加未兑现预约做容量准入，计算收益需要另做同工作量对照。

本文从“只申请 64 MiB，为什么观测到 380 MiB”的真实例子开始，再解释上下文、stream（提交队列）、MPS（多进程服务），最后把机制放回 vLLM 与环境共卡的调用链。项目已有环境专用 MPS 后端；本文的 vLLM 共卡分支尚未接入 MPS。


### 1. 从一组实测数字说起：64 MiB 申请为什么测出 380 MiB

[不用 procfs 标识进程](../../env-worker-deep-dive/knowledge/02-资源与调度/03-进程归属_PID身份与显存记账.md#进程编号可信归属与后端验证)这篇文档里有一次真机实测：在容器里跑一个最小 CUDA 程序，代码里只显式申请（`cudaMalloc`）了 64 MiB 显存，执行一次内核算出结果 42。但用 NVML（NVIDIA 的显存/进程查询接口，`nvidia-smi` 底层调的就是它）去看这个进程实际占了多少显存，量出来的是 380 MiB：

> 内核执行得到 `42`，申请显存 64 MiB，NVML 的该进程占用为 380 MiB。后者包括 CUDA 上下文等开销，不应要求它恰好等于申请量。只根据"占用了 380 MiB"猜身份不可靠，本次依据是内核取得的宿主 PID 与 NVML 精确一致。

原始实测条件见[GPU 身份主文档的内核追踪与 CUDA 对账章节](../../env-worker-deep-dive/knowledge/02-资源与调度/03-进程归属_PID身份与显存记账.md#内核追踪与真实-docker-cuda-对账)。

`380 - 64 = 316 MiB`，这多出来的部分文档里管它叫"CUDA 上下文等开销"。这篇文档要讲清楚两件事：这 316 MiB 是什么（显存层面的额外成本），以及如果两个各自持有自己上下文的进程（vLLM 和环境容器）同时往 GPU 提交计算，会发生什么（计算层面的动态竞争）。

### 2. CUDA 上下文是什么

先回到一个你熟悉的类比，但要注意这个类比有边界，下面会具体说明哪里不一样。一个 CPU 进程被操作系统创建出来时，哪怕它一行业务代码都还没跑，操作系统已经替它准备好了一堆"执行环境"：虚拟地址空间、页表（记录虚拟地址到物理内存映射的表）、打开文件描述符表等等。这些东西占的内存，和进程自己 `malloc` 出来的业务数据是两笔完全不同的账——业务数据是进程"主动要的"，执行环境是"操作系统为了让这个进程能跑起来，必须先垄固好的"。

**CUDA 上下文就是 GPU 驱动层面的同一类东西。** 当一个 CPU 进程第一次调用 CUDA API（比如第一次 `cudaMalloc`，或者启动一个 kernel）时，NVIDIA 驱动会自动在 GPU 上（以及驱动自己在 CPU 内存里维护的一侧）给这个进程创建一个"上下文"。这个上下文里至少装着：

这个进程专属的 GPU 虚拟地址空间。和 CPU 进程的虚拟地址空间是同一个思路——不同进程的上下文之间默认互相隔离，A 进程手里的显存指针，B 进程拿去直接用是非法的，除非显式用进程间通信机制共享。这正是为什么"两个容器各自跑 CUDA"会变成"两个完全独立的上下文"，而不是共享同一份地址空间。

已经加载进这个上下文的 kernel 代码（cubin/PTX，也就是编译好的 GPU 机器码）。写的 CUDA 程序，以及它依赖的库（比如 cuDNN、cuBLAS，或者 vLLM/PyTorch 自带的各种算子）在第一次被调用时，会按 CUDA 版本、模块加载方式和调用情况加载代码；启用延迟加载时，一部分工作会推迟到实际使用。

默认 stream（CUDA 里的任务队列概念，决定提交的操作按什么顺序在 GPU 上执行）和一些事件、同步对象的簿记结构。

驱动为执行、同步及错误处理维护的辅助结构。

这些结构会产生额外占用，并且可能随模块加载和库初始化继续变化。64 MiB 业务分配与 380 MiB 观测之间的 316 MiB 差值包含这类开销，但该次实验没有逐项拆分。因此应把它读成一次特定程序、驱动和 GPU 下的实测差值；在新任务中仍需重新测量。NVIDIA 对上下文成本的说明见 [CUDA 最佳实践](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-c-best-practices-guide/index.html#multiple-contexts)。

### 3. 显存层面：这笔固定成本怎么被 PSRL 的账本处理

如果 PSRL 的显存准入账本只按"任务自己声明要多少显存"做加法，就会忽略上下文开销这笔成本，导致"账面上看起来还有空间，物理上放进去就爆了"。实际代码里是用两步兜底的：

第一步，在 [`gpu_memory.py:9`](../../../psrl/workers/env_service/gpu_memory.py#L9) 的 `accounted_gpu_memory_mb` 里，把"刚测到的物理占用"（`measured_mb`，这个数字天然已经包含了卡上所有进程各自的上下文开销，因为 NVML 测的就是真实物理占用）加上"还没来得及兑现的预约"，作为这张卡当前"被记账"的总量。换句话说，账本从来不单独计算"上下文开销应该扣多少"，而是直接信任 NVML 测到的真实数字，上下文开销自动被包含在里面，不需要额外建模。

第二步，在更上一层，[`ledger.py:466`](../../../psrl/workers/env_service/ledger.py#L466) 的 `_effective_gpu_mb` 把这个实测值、尚未兑现的预约，再加上一个 `model_headroom_mb`（给模型留的显存增长余量），加在一起，作为这个资源单元已经被占掉多少的最终判断依据：

```python
return accounted_gpu_memory_mb(
    measured or 0.0, reservations, owned_mb if measured is not None else None
) + unit.model_headroom_mb * len(unit.gpu_indices)
```

这个设计思路的要点是：**账本不假设"声明的显存等于实际会占的显存"，而是始终以实测值为准来兜底**。这正是为了不被上下文开销这类"程序自己说不清楚、但物理上确实存在"的隐藏成本坑到——如果单纯按每个任务自己报的显存数字相加，算出来的可用空间会比实际能放下的更宽松，一旦真的调度进去就会在物理层面爆显存，而不只是账本数字出错。更完整的公式细节和故障案例见[显存准入与未兑现预约](../../env-worker-deep-dive/knowledge/02-资源与调度/01-显存准入_实测占用与预约预算.md)。

#### 3.1 多卡 unit 为什么逐卡预留余量

unit 是 EnvRouter 的资源单元，可以包含一张或多张 GPU。[`gpu_indices`](../../../psrl/workers/env_service/types.py#L38) 保存它包含的设备编号。张量并行的推理副本需要多张卡一起工作，配置必须匹配该副本的完整设备集合，见[启动校验](../../../psrl/trainer/ppo/ray_trainer.py#L639)。

`model_headroom_mb` 是每张卡给模型保留的增长余量。假设一个 unit 含 4 张卡，每张预留 2048 MiB，则总余量为 `2048 × 4 = 8192 MiB`。这是教学数字。总账相加之外，[逐卡检查](../../../psrl/workers/env_service/ledger.py#L521)还会分别核查容量，因为卡 A 的剩余显存无法补上卡 B 的不足。

### 4. 计算层面：没有 MPS 时，GPU 在两个上下文之间怎么调度

显存层面讲的是"占多少空间"，是执行环境的额外成本，申请到就不会变。计算层面讲的是"谁先用 GPU 的计算单元跑、跑的时候会不会互相拖慢"，这是动态的、随时发生的竞争，需要先引入两个新概念才能讲清楚。

**SM（Streaming Multiprocessor，流式多处理器）**是 GPU 内部真正执行并行线程的硬件单元，可以粗略类比成 CPU 的"核心"——一张 GPU 卡上有几十到上百个 SM，每个 SM 同时能跑成百上千个线程。

#### 4.1 stream 是什么：一条"提交顺序队列"

**stream（流）是 CUDA 里的一个任务队列概念**：往里面按顺序扔进去的操作（启动一个 kernel、发起一次显存拷贝等等），会严格按照扔进去的顺序，一个接一个在 GPU 上执行——同一个 stream 内部，前一个操作没做完，后一个绝不会提前开始。

但关键的地方在于：**一个进程（一个 context）可以同时开好几条 stream，这几条 stream 之间没有顺序约束**——stream 1 里的操作和 stream 2 里的操作谁先谁后、是否重叠，CUDA 不做任何承诺，硬件可以按自己的判断去调度，只要两条 stream 之间没有显式加同步点。正因为没有顺序约束，硬件才有操作空间：如果 stream 1 这一刻提交的 kernel 和 stream 2 这一刻提交的 kernel，各自需要的 SM 数量都不多、GPU 上还有空闲的 SM，硬件完全可以把这两个 kernel 的线程块同时分发给不同的 SM，让它们在同一个时钟周期真的一起跑——这就是"并发 kernel 执行"。没有手动创建任何 stream 时，CUDA 会自动用第 2 节提到的那个**默认 stream** 兜底，所有没指定 stream 的操作都排在这一条队列里，顺序执行。

举个具体例子：vLLM 这类框架，内部经常会手动开好几条 stream——比如一条专门跑矩阵乘法这类计算，另一条专门做"把下一批数据从 CPU 搬到 GPU"这种显存拷贝，目的是让"算当前这批"和"搬下一批"这两件不冲突的事，靠不同 stream 同时进行，不用傻等着排成一条队串行做，这是同一个进程内部常见的性能优化手段。

#### 4.2 常规独立上下文怎样共享执行时间

在本文讨论的常规 CUDA 多进程路径中，A、B 各有独立上下文。CUDA 编程指南说明，这种情况下不同上下文的 kernel 按时间共享 GPU；同一上下文中的不同 stream 则在资源和依赖允许时具备并发机会。见[官方并发执行说明](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#concurrent-kernel-execution)。

例如 A 和 B 都启动了多个 stream，这只声明各自内部的操作关系。驱动仍需协调两个上下文。PSRL 共享票据没有给 A 永久划出 20% SM，也没有替驱动决定每次谁先执行。因此显存余量充足时，两边仍可能争用计算、带宽和缓存。

#### 4.3 MPS 怎样支持多个进程同时执行

MPS 协调多个 CUDA 客户端，使它们的工作可以同时在 GPU 上调度。需要按架构理解具体路径：早期实现与 Volta 及后续实现不同；Volta 及后续的客户端拥有各自的 GPU 地址空间，直接向 GPU 提交工作，服务端协调仍需共享的资源。见 [NVIDIA MPS 架构](https://docs.nvidia.com/deploy/mps/595/architecture.html)。

例如 A 的小 kernel 只使用了一部分资源，B 的 kernel 在资源允许时可以与它同时运行，从而利用余量。两者仍会竞争计算和带宽，实际提速要测量。地址空间分离也需要与具体的故障和管理边界一起理解，不能据此直接获得任意任务之间的隔离保证。

PSRL 当前的[MPS 后端](../../env-worker-deep-dive/knowledge/02-资源与调度/04-GPU并行_MPS执行后端.md)用于可信任务和环境专用 GPU；**vLLM 共卡分支**仍使用显式共享票据与模型控制门禁。这里的后续计算竞争分析针对该共卡分支。CPU、SM 与 MPS/MIG 的进一步比较见[并行架构教程](#从-cpu-到-gpu-的并行结构)。

### 5. 把两层串回源码：跟着图读 vLLM 和环境命令的真实调用链

[训推共享：vLLM余量与容量调度 第 11 节](../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md#11-计算竞争与控制等待已经验证到哪里)原本就在讲这件事，下面把图里每个节点在当前代码里的真实位置重新核对了一遍：

```text
[A] vLLM generate ───────────────→ 模型 CUDA 上下文 ─┐
                                                   ├→ [D] 驱动调度 GPU
[B] 环境 command 准入 → Docker → 环境 CUDA 上下文 ─┘      SM / 显存带宽 / 缓存
          │                                            (外部机制,没有 MPS = 时间片轮转)
          └→ 命令结束 → [C] 撤销设备权限、确认物理归还
                                     │
                                     ↓ 释放共享票据、通知等待者
[E] wake_up / pause_for_sync → GpuControlGate → 进入模型控制操作
                                     ↑
                     共享票据未归还时在这里等待
```

节点 A 是 vLLM 正常生成，不经过任何控制门禁，所以环境命令在跑的时候模型可以继续生成，入口是 [`vllm_async_server.py:945,generate`](../../../psrl/workers/gen/vllm_async_server.py#L945)。

节点 B 是环境命令拿"共享票据"的入口，[`gpu_handoff.py:76,acquire_env_command`](../../../psrl/workers/gen/gpu_handoff.py#L76)，实际的命令准入排队逻辑在 [`command_gpu.py:28,GpuCommandScheduler`](../../../psrl/workers/env_service/command_gpu.py#L28)。

节点 C 是命令结束、归还票据，对应 [`gpu_handoff.py:94,release_env_command`](../../../psrl/workers/gen/gpu_handoff.py#L94)，正常归还由 [`command_gpu.py:215,_return`](../../../psrl/workers/env_service/command_gpu.py#L215) 确认，超时则走 [`command_gpu.py:171,_expire_shared`](../../../psrl/workers/env_service/command_gpu.py#L171) 清理分支。

节点 D 是驱动真正调度 A 和 B 两个上下文谁先用 GPU——这是纯粹的 CUDA 驱动行为，PSRL 代码里没有、也不可能有一个伪造的本地调度函数去决定这件事，图里特意把它标成"外部机制"。

节点 E 是模型要唤醒或同步权重时，必须先等所有共享票据归还，入口是 [`vllm_async_server.py:735,wake_up`](../../../psrl/workers/gen/vllm_async_server.py#L735) 和 [`vllm_async_server.py:770,pause_for_sync`](../../../psrl/workers/gen/vllm_async_server.py#L770)，二者都经过 [`gpu_handoff.py:21,GpuControlGate.operation`](../../../psrl/workers/gen/gpu_handoff.py#L21) 等待独占控制条件。

**这里有一个容易混在一起的关键区分**：节点 E 这道控制门禁管的是**准入层面的互斥**——怕模型唤醒或换权重时显存状态被破坏，哪怕环境命令只是"睡着"地占用上下文（比如只持有显存不跑计算），门禁的等待逻辑一样会生效；而节点 D 的时间片轮转是只要两边**同时真的在提交 kernel**才会发生的、PSRL 完全管不到的底层调度行为。这两件事分属显存层面和计算层面，不要混为一谈。

### 6. 一个可以手算、可以复现的任务依赖例子

[训推共享：vLLM余量与容量调度 第 11 节](../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md#11-计算竞争与控制等待已经验证到哪里)给了一个教学例子：假设模型工作需要独占 GPU 2 秒，它的输出会触发一个 10 秒的 CPU 任务；同时有一个独立的环境 GPU 作业需要独占 10 秒。三种策略的总耗时不同：

模型先执行完再让环境进，模型占用 0 到 2 秒，之后 CPU 任务与环境 GPU 任务在 2 到 12 秒并行，总共 12 秒。环境先占满 GPU 再让模型进，环境占用 0 到 10 秒，模型 10 到 12 秒，CPU 任务 12 到 22 秒，总共 22 秒。两者共享 GPU，假设重叠期间各打对方五折速度，模型 0 到 4 秒（原本 2 秒的工作在抢占下花了 4 秒），环境剩余工作独占到 12 秒，CPU 任务 4 到 14 秒，总共 14 秒。

这个"五折"只是计算方便的教学假设，不是 CUDA 的固定时间片比例，也不是实测减速幅度——这一点在第 7 节会再强调一次。配套脚本 [`vllm_env_sharing_demo.py:19`](../../env-worker-deep-dive/env-service-verification/scripts/vllm_env_sharing_demo.py#L19) 把这套逻辑写成了可运行的 Python，不依赖 CUDA：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/vllm_env_sharing_demo.py
```

本次实际重新运行（在 psrl 仓库根目录下执行上述命令），输出 `model_first_makespan_s=12`、`docker_first_makespan_s=22`、`shared_makespan_s=14.0`，与文档记录的数字一致。这组数字说明的是"共享"比"环境先跑完再让模型进"（22 秒）好，但不如"让模型尽快跑完、尽早解锁依赖它的 CPU 任务"（12 秒）——这个结论只依赖任务之间的依赖关系和假设的减速比例，和真实硬件下的实际减速幅度没有关系，脚本本身也没有调用任何 CUDA API。

### 7. 验证边界：已经证明了什么，还没证明什么

| 问题 | 证据与当前结论 |
|---|---|
| 上下文是否带来额外显存 | 最小 CUDA 程序申请 64 MiB，进程观测为 380 MiB；差值未逐项拆解 |
| 模型与环境能否同时持有资源 | 固定共存探针中，环境常驻命令成功时模型仍可返回 token |
| vLLM 与环境持续计算时会慢多少 | 尚缺固定工作量的延迟对照，不能从容量或教学数字推出 |
| MPS 是否已进入项目 | 环境专用 GPU 后端已有实现；vLLM 共卡分支尚未接入 |

第 6 节的 12/22/14 秒来自任务依赖与假设减速比例，用于理解整体完成时间。如果要量化真实共卡影响，固定模型、输入输出长度、预热和设备，分别比较模型独占、环境只持有显存、环境持续计算，记录首 token 延迟、生成总时长和环境命令耗时。控制请求到达到放行的等待单独统计，方法见[共卡深入记录](../../env-worker-deep-dive/knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md#11-计算竞争与控制等待已经验证到哪里)。

本轮整理复读本地记账与共享门禁，核对 NVIDIA 官方机制，没有重跑 GPU 实验。历史 CPU 教学运行和 CUDA 探针均沿用原记录。

[返回专题入口](../../env-worker-deep-dive/README.md)
