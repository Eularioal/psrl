# CUDA 上下文开销与计算竞争：vLLM 和环境共享一张卡时到底在抢什么

> **核心结论**：同卡共享涉及两层完全不同的成本。显存层面是一次性的固定成本——任何进程只要用上 CUDA，哪怕不申请任何业务显存，也会先被扣掉一笔"上下文开销"，这笔钱[显存准入公式](gpu-memory-admission.md)用实测值兜底，不是靠声明值估算。计算层面是动态的、随时发生的竞争——没有部署 MPS 的情况下，vLLM 进程和环境容器进程各自持有独立的 CUDA 上下文，GPU 在两者之间只能"轮流"执行，不能真正同时跑在不同的流式多处理器上。前者已有真机实测数据，后者目前只有机制推理，没有量化实验，详见第 7 节的验证边界。

## 目录

- [1. 从一组实测数字说起：64 MiB 申请为什么测出 380 MiB](#1-从一组实测数字说起64-mib-申请为什么测出-380-mib)
- [2. CUDA 上下文是什么](#2-cuda-上下文是什么)
- [3. 显存层面：这笔固定成本怎么被 PSRL 的账本处理](#3-显存层面这笔固定成本怎么被-psrl-的账本处理)
  - [3.1 一个容易望文生义的地方：`乘以 len(unit.gpu_indices)` 不是在算"一张卡挂了几个 GPU"](#31-一个容易望文生义的地方乘以-lenunitgpu_indices-不是在算一张卡挂了几个-gpu)
- [4. 计算层面：没有 MPS 时，GPU 在两个上下文之间怎么调度](#4-计算层面没有-mps-时gpu-在两个上下文之间怎么调度)
  - [4.1 stream 是什么：一条"提交顺序队列"](#41-stream-是什么一条提交顺序队列)
  - [4.2 没有 MPS 时，不同上下文之间是"轮流"，不是真正同时跑](#42-没有-mps-时不同上下文之间是轮流不是真正同时跑)
  - [4.3 MPS 在修哪一层：不是"事后合并"，是"从源头上就只建一个"](#43-mps-在修哪一层不是事后合并是从源头上就只建一个)
- [5. 把两层串回源码：跟着图读 vLLM 和环境命令的真实调用链](#5-把两层串回源码跟着图读-vllm-和环境命令的真实调用链)
- [6. 一个可以手算、可以复现的任务依赖例子](#6-一个可以手算可以复现的任务依赖例子)
- [7. 验证边界：已经证明了什么，还没证明什么](#7-验证边界已经证明了什么还没证明什么)

## 1. 从一组实测数字说起：64 MiB 申请为什么测出 380 MiB

[不用 procfs 标识进程](process-identity-without-procfs.md)这篇文档里有一次真机实测：在容器里跑一个最小 CUDA 程序，代码里只显式申请（`cudaMalloc`）了 64 MiB 显存，执行一次内核算出结果 42。但用 NVML（NVIDIA 的显存/进程查询接口，`nvidia-smi` 底层调的就是它）去看这个进程实际占了多少显存，量出来的是 380 MiB：

> 内核执行得到 `42`，申请显存 64 MiB，NVML 的该进程占用为 380 MiB。后者包括 CUDA 上下文等开销，不应要求它恰好等于申请量。只根据"占用了 380 MiB"猜身份不可靠，本次依据是内核取得的宿主 PID 与 NVML 精确一致。

原文见 [`process-identity-without-procfs.md:270`](process-identity-without-procfs.md#L270)。

`380 - 64 = 316 MiB`，这多出来的部分文档里管它叫"CUDA 上下文等开销"。这篇文档要讲清楚两件事：这 316 MiB 是什么（显存层面的固定成本），以及如果两个各自持有自己上下文的进程（vLLM 和环境容器）同时往 GPU 提交计算，会发生什么（计算层面的动态竞争）。

## 2. CUDA 上下文是什么

先回到一个你熟悉的类比，但要注意这个类比有边界，下面会具体说明哪里不一样。一个 CPU 进程被操作系统创建出来时，哪怕它一行业务代码都还没跑，操作系统已经替它准备好了一堆"执行环境"：虚拟地址空间、页表（记录虚拟地址到物理内存映射的表）、打开文件描述符表等等。这些东西占的内存，和进程自己 `malloc` 出来的业务数据是两笔完全不同的账——业务数据是进程"主动要的"，执行环境是"操作系统为了让这个进程能跑起来，必须先垄固好的"。

**CUDA 上下文就是 GPU 驱动层面的同一类东西。** 当一个 CPU 进程第一次调用 CUDA API（比如第一次 `cudaMalloc`，或者启动一个 kernel）时，NVIDIA 驱动会自动在 GPU 上（以及驱动自己在 CPU 内存里维护的一侧）给这个进程创建一个"上下文"。这个上下文里至少装着：

这个进程专属的 GPU 虚拟地址空间。和 CPU 进程的虚拟地址空间是同一个思路——不同进程的上下文之间默认互相隔离，A 进程手里的显存指针，B 进程拿去直接用是非法的，除非显式用进程间通信机制共享。这正是为什么"两个容器各自跑 CUDA"会变成"两个完全独立的上下文"，而不是共享同一份地址空间。

已经加载进这个上下文的 kernel 代码（cubin/PTX，也就是编译好的 GPU 机器码）。写的 CUDA 程序，以及它依赖的库（比如 cuDNN、cuBLAS，或者 vLLM/PyTorch 自带的各种算子）在第一次被调用时，通常会把整份编译好的代码模块一次性加载进这个上下文，不是"用一个 kernel 就只加载一个 kernel"的按需方式。

默认 stream（CUDA 里的任务队列概念，决定提交的操作按什么顺序在 GPU 上执行）和一些事件、同步对象的簿记结构。

驱动自己为这个进程维护的固定大小的暂存区域和错误报告缓冲区。

这些加起来，哪怕一行 `cudaMalloc` 都没写、一个 kernel 都没跑，仅仅是"创建了一个 CUDA 上下文"这个动作本身，就会在显存里占掉一截——这就是第 1 节那 316 MiB 的来源。需要说明一下证据边界：第 1 节的实测只拿到了 380 MiB 这个黑盒总数，并没有进一步把 316 MiB 拆解成"地址空间簿记占多少、预加载的 cubin 占多少"，这个细分是 CUDA 生态的通用常识，不是本仓库源码或实验里验证出来的数字，这里特意说明清楚，避免把常识和实测混在一起讲。

## 3. 显存层面：这笔固定成本怎么被 PSRL 的账本处理

如果 PSRL 的显存准入账本只按"任务自己声明要多少显存"做加法，就会忽略上下文开销这笔成本，导致"账面上看起来还有空间，物理上放进去就爆了"。实际代码里是用两步兜底的：

第一步，在 [`gpu_memory.py:10`](../../../psrl/workers/env_service/gpu_memory.py#L10) 的 `accounted_gpu_memory_mb` 里，把"刚测到的物理占用"（`measured_mb`，这个数字天然已经包含了卡上所有进程各自的上下文开销，因为 NVML 测的就是真实物理占用）加上"还没来得及兑现的预约"，作为这张卡当前"被记账"的总量。换句话说，账本从来不单独计算"上下文开销应该扣多少"，而是直接信任 NVML 测到的真实数字，上下文开销自动被包含在里面，不需要额外建模。

第二步，在更上一层，[`ledger.py:409`](../../../psrl/workers/env_service/ledger.py#L409) 的 `_effective_gpu_mb` 把这个实测值、尚未兑现的预约，再加上一个 `model_headroom_mb`（给模型留的显存增长余量），加在一起，作为这个资源单元已经被占掉多少的最终判断依据：

```python
return accounted_gpu_memory_mb(
    measured or 0.0, reservations, owned_mb if measured is not None else None
) + unit.model_headroom_mb * len(unit.gpu_indices)
```

这个设计思路的要点是：**账本不假设"声明的显存等于实际会占的显存"，而是始终以实测值为准来兜底**。这正是为了不被上下文开销这类"程序自己说不清楚、但物理上确实存在"的隐藏成本坑到——如果单纯按每个任务自己报的显存数字相加，算出来的可用空间会比实际能放下的更宽松，一旦真的调度进去就会在物理层面爆显存，而不只是账本数字出错。更完整的公式细节和故障案例见[显存准入与未兑现预约](gpu-memory-admission.md)。

### 3.1 一个容易望文生义的地方："`乘以 len(unit.gpu_indices)`" 不是在算"一张卡挂了几个 GPU"

上面这行代码容易被读成"一张物理 GPU 卡上插了好几个 GPU 芯片，所以要乘这么多份"——这个理解是错的，物理上一张卡就是一个 GPU 设备，不会出现"一张卡对应多个 GPU"的情况。真正的原因是：**账本里统计的"这个单位"，不是字面意义上的一整块物理卡,而是 `EnvUnit`——PSRL 给 GPU 资源分配出去的最小单位。一个 `EnvUnit` 和一块物理 GPU 卡不是绑死的一一对应关系,它完全可以同时拥有好几块物理卡。**

证据在 `EnvUnit` 自己的字段文档里：

```37:37:/Users/liuwei/CodeBuddy/20260811161322/psrl/psrl/workers/env_service/types.py
gpu_indices (tuple[int, ...]): Whole devices owned by this unit. Empty means CPU only.
```

`gpu_indices` 是一个元组,可以装好几个设备编号(对应 `nvidia-smi` 里能看到的 GPU 0、GPU 1……),`len(unit.gpu_indices)` 就是这个 `EnvUnit` 一共拥有几块物理卡。

为什么一个资源单位会同时拥有好几块卡？典型场景是**张量并行的模型推理副本**：如果模型大到一张卡放不下,vLLM 会把权重切成几份摊到好几块卡上协同工作,对外是"一个模型副本",物理上占了 N 块卡。这几块卡要作为一个整体一起分配、一起释放,PSRL 就把它们当成一个 `EnvUnit` 来管理,而不是拆成 N 个互不相关的单位。训练代码里有一处校验直接体现了这个意图：

```637:646:/Users/liuwei/CodeBuddy/20260811161322/psrl/psrl/trainer/ppo/ray_trainer.py
        for unit in units:
            replica = unit.rollout_replica_rank
            if replica is None:
                replica = unit.shared_rollout_replica_rank
            name = f"rollout_{replica}"
            slots = placements.get(name, frozenset())
            actual = set(slots)
            expected = {(unit.node_ip, index) for index in unit.gpu_indices}
            if actual != expected:
                raise ValueError(f"GPU lending unit {unit.unit_id!r} does not match complete replica {name!r}.")
```

这段在 `_init_gpu_lending` 里校验"这个 `EnvUnit` 声称拥有的 `gpu_indices`，必须恰好等于某个 rollout 副本（`name = f"rollout_{replica}"`）实际用到的那一整组设备"——印证了 `EnvUnit` 的粒度确实是"一个模型副本的完整 GPU 占用"，不是"一张卡"。

搞清楚 `EnvUnit` 可以跨多张卡之后，再看 `model_headroom_mb` 本身的语义：它是**每一块卡单独要留的余量**，不是整个单位共享一份。证据是按卡逐一算账的 `_effective_gpu_by_device`：

```452:455:/Users/liuwei/CodeBuddy/20260811161322/psrl/psrl/workers/env_service/ledger.py
        return tuple(
            measured.get(index, 0.0) + sum(values[position] for values in reservations) + unit.model_headroom_mb
            for position, index in enumerate(unit.gpu_indices)
        )
```

这里对 `unit.gpu_indices` 里**每一块卡**都原封不动地加了一次同样的 `unit.model_headroom_mb`，不是先除以卡数再分摊——因为显存是物理隔离的，A 卡多出来的空闲显存不能拿去填补 B 卡的缺口；张量并行场景下，每块卡各自存着模型权重的一部分，推理时各自的 KV cache 也会各自增长，所以每块卡都要单独留一份增长余量。`_effective_gpu_mb` 要算的是"整个单位一共占了多少"，既然每块卡都要留一份，自然要 `× len(unit.gpu_indices)` 换算成这个单位名下所有卡加起来的总余量，这只是单位换算，不是什么特殊逻辑。

代入一个具体数字：假设一个 `EnvUnit` 做 4 卡张量并行，`gpu_indices = (0, 1, 2, 3)`，`model_headroom_mb = 2048`（2 GiB，仅为教学假设，非源码默认值）。按卡拆开看，0~3 号卡各自的有效占用都单独加一次 2048；按整个单位汇总看，总余量部分是 `2048 × 4 = 8192`，正好等于四块卡各自 2048 加起来的结果——两种算法在数值上是一致的，`_effective_gpu_mb` 只是把"按卡列清单"换算成了"按单位算总账"。

## 4. 计算层面：没有 MPS 时，GPU 在两个上下文之间怎么调度

显存层面讲的是"占多少空间"，是一次性的固定成本，申请到就不会变。计算层面讲的是"谁先用 GPU 的计算单元跑、跑的时候会不会互相拖慢"，这是动态的、随时发生的竞争，需要先引入两个新概念才能讲清楚。

**SM（Streaming Multiprocessor，流式多处理器）**是 GPU 内部真正执行并行线程的硬件单元，可以粗略类比成 CPU 的"核心"——一张 GPU 卡上有几十到上百个 SM，每个 SM 同时能跑成百上千个线程。

### 4.1 stream 是什么：一条"提交顺序队列"

**stream（流）是 CUDA 里的一个任务队列概念**：往里面按顺序扔进去的操作（启动一个 kernel、发起一次显存拷贝等等），会严格按照扔进去的顺序，一个接一个在 GPU 上执行——同一个 stream 内部，前一个操作没做完，后一个绝不会提前开始。

但关键的地方在于：**一个进程（一个 context）可以同时开好几条 stream，这几条 stream 之间没有顺序约束**——stream 1 里的操作和 stream 2 里的操作谁先谁后、是否重叠，CUDA 不做任何承诺，硬件可以按自己的判断去调度，只要两条 stream 之间没有显式加同步点。正因为没有顺序约束，硬件才有操作空间：如果 stream 1 这一刻提交的 kernel 和 stream 2 这一刻提交的 kernel，各自需要的 SM 数量都不多、GPU 上还有空闲的 SM，硬件完全可以把这两个 kernel 的线程块同时分发给不同的 SM，让它们在同一个时钟周期真的一起跑——这就是"并发 kernel 执行"。没有手动创建任何 stream 时，CUDA 会自动用第 2 节提到的那个**默认 stream** 兜底，所有没指定 stream 的操作都排在这一条队列里，顺序执行。

举个具体例子：vLLM 这类框架，内部经常会手动开好几条 stream——比如一条专门跑矩阵乘法这类计算，另一条专门做"把下一批数据从 CPU 搬到 GPU"这种显存拷贝，目的是让"算当前这批"和"搬下一批"这两件不冲突的事，靠不同 stream 同时进行，不用傻等着排成一条队串行做，这是同一个进程内部常见的性能优化手段。

### 4.2 没有 MPS 时，不同上下文之间是"轮流"，不是真正同时跑

这是[vLLM 与环境同卡共享策略](vllm-env-opportunistic-sharing.md#L459)专门强调、并引用 NVIDIA 官方说明的一点：

> 在常规、未启用 MPS 的多进程执行中，不同上下文通常在 GPU 上轮流获得执行时间。因而不应理解成"系统预先切出 20% 的 SM 永久给环境，剩下 80% 给 vLLM"。启用 MPS 后可以出现跨进程同时调度，但本项目本轮并未部署它。

为什么会"轮流"而不是"同时"？根源还是回到上下文本身：第 2 节讲过，不同进程各自持有**独立隔离的 GPU 虚拟地址空间**。GPU 硬件在某个时刻只能服务于一个上下文的指令流——要切换到另一个上下文执行，硬件需要先处理好当前上下文的执行状态，再切到另一个上下文的地址空间视角，这个"切换"本身就有开销，而且切换期间两边都没在真正干活。所以 vLLM 进程（模型上下文）和某个环境容器里的 CUDA 命令（环境上下文）**不会被静态地分成"各占一半 SM"**，而是硬件在两个上下文之间按时间片轮转。

**这里和多核 CPU 的直觉不一样**：容易把"GPU 有几十上百个 SM"直接类比成"CPU 有几十上百个核心，不同进程分到不同核心上真并行"，但这个类比在跨进程这一步是错的——CPU 核心是"按进程独立分配"的调度单位，GPU SM 是"当前被选中的那个 context 内部"的执行资源池，两者为什么会有这组差异、设计之初各自在优化什么问题，展开讲解见专题[CPU 与 GPU 多任务并行架构设计对比](cpu-gpu-parallelism-architecture-comparison.md)。这里只记住落到 PSRL 场景的结论：**GPU 没有把"一张卡的 SM"预先切成多份、分别常驻给不同进程**；"轮到谁"是以整个 context 为单位做二选一（或多选一），这正是 MPS 要补的那个差距。

有了 4.1 的 stream 概念，这句话可以拆成两层讲清楚，这也是"同上下文并发"和"跨上下文时间片"的本质区别，是理解这部分最容易搞混的地方：

- **层 1（context 内部，能并发）**：vLLM 进程自己这一个 context 里，哪怕它内部开了好几条 stream 各自塞了不同的 kernel，这些 kernel 之间有资格被硬件同时分发到不同 SM 上真并行——前提成立，是因为它们全都属于同一个 context，硬件不需要为了同时服务它们而切换任何地址空间，显存管理这边看到的始终是同一份映射。
- **层 2（跨 context，只能轮流）**：环境容器那边的 CUDA 命令，是另一个完全独立的 context——不管它内部开了几条 stream，这些 stream 和 vLLM 那边的任何一条 stream，都没有资格被同时分发到不同 SM，因为硬件在分发 block 给 SM 之前，首先要确定"当前轮到哪个 context"，而没有 MPS 时，这个"轮到谁"的判断是以**整个 context** 为单位做的，不会往下细看这个 context 内部具体有几条 stream——轮不到的那个 context，它名下所有 stream 的所有 kernel，在那一刻是彻底被挂起、一个都提交不进去的。

这就解释清楚了为什么"stream 内部能并发"这件事，救不了"跨进程（context）共享 GPU"这个场景——stream 带来的并发能力，是在"已经确定由哪个 context 使用 GPU"这个前提成立之后，才在 context 内部生效的，它本身跨不过 context 这道边界。

### 4.3 MPS 在修哪一层：不是"事后合并"，是"从源头上就只建一个"

**MPS（Multi-Process Service）**是 NVIDIA 驱动提供的一个守护进程，它的作用是把多个客户端进程的 CUDA 提交统一代理到**同一个共享上下文**里去真正提交给 GPU——这样原本属于不同进程、本该互相排队轮转的 kernel，现在可以像同一个上下文内的不同 stream 那样被调度，实现更细粒度的真正并发。但 PSRL 目前**没有部署 MPS**：

这里有两个容易望文生义、需要专门纠正的措辞。

**不是"事后合并"，而是"从源头上就只建一个"**：没有 MPS 时，进程 A 和进程 B 各自会自己创建一个属于自己的 CUDA context——这个 context 是"A 自己的地址空间 + A 自己已经装载到 GPU 上的代码 + A 自己分配的显存"，进程 B 同理，自己也有一份完全独立的。两份 context 各自诞生，互不相干，只是之后在 GPU 上排队轮流用而已。启用 MPS 之后，发生的事情不是"把 A 已经建好的 context 和 B 已经建好的 context 事后粘到一起"，而是：从一开始，只有 MPS server 这一个进程会去真正创建、拥有那个和 GPU 对话的 context。进程 A 和进程 B 不再各自建立自己独立通往 GPU 的那条路——它们该发的 CUDA 调用（分配显存、启动 kernel 之类）照常发，但这些调用会被转发给 MPS server，由 server 代为提交。所以严格说，这里没有发生"两个独立的东西合并成一个"的动作，而是"原本会生出两个独立东西的过程，被改成了只生出一个，大家共用"——没有"合并"这个中间步骤，从头到尾就只存在一个 context。

**不是物理搬运代码，而是提交路径被改**："统一代理到同一个共享上下文"这句话容易想象成"把 A 的代码和 B 的代码都塞进同一块显存里"这种物理搬运的画面。实际更准确的说法是：A 和 B 各自的 kernel 代码该怎么编译、该怎么提交，完全没变——变的只是"提交的落脚点"。原本 A 的提交会落到"A 自己的 context"里，现在统一落到"server 的 context"里，被当成这个 context 下的一条 stream 去处理；B 的提交也落到同一个 server context 里，被当成另一条 stream。所以准确的说法应该是"A 和 B 的提交路径，被改成共用同一个已经存在的 context"，而不是"把两份东西聚合/搬运到了一起"。

串到这里，**stream 和 MPS 的关系可以用一句话精确归位**：stream 是同一个 context 内部，用来表达"这些任务之间没有顺序依赖，你可以并发"的机制；MPS 是把原本分属不同 context 的任务，伪装成同一个 context 下的不同 stream，从而借到这条机制。MPS 自己没有新增任何"让不同 SM 同时服务不同 context"的硬件能力，它做的全部事情，就是让 A、B 不再各自持有 context，从而能蹭到"同 context 多 stream 并发"这条硬件早就支持的路。这也是为什么本节专门把"MPS 的作用"描述成"统一代理到同一个共享上下文里"——"共享上下文"这四个字，正是为了让双方的提交资格，从"跨 context（只能轮流）"升级成"同 context 的不同 stream（可以真并发）"。

> 当前共享协议没有给环境划分固定比例的计算单元，也没有部署 MPS。MPS 是 NVIDIA 的多进程服务，用于协调多个 CUDA 进程的执行，不是显存准入公式的一部分。

原文见 [`vllm-env-opportunistic-sharing.md:427`](vllm-env-opportunistic-sharing.md#L427)。所以现状是：vLLM 进程和环境容器进程各自持有独立上下文，谁的 kernel 先提交、驱动调度策略怎么在两者之间切换，PSRL 代码管不到，完全交给 CUDA 驱动这个外部机制黑盒决定。环境运行期间，模型可能少拿到执行时间，也可能受显存带宽（单位时间能搬多少数据，和显存容量是两件不同的事）、缓存及调度开销影响，结果可能是首个 token 更晚返回、后续 token 间隔变大。

## 5. 把两层串回源码：跟着图读 vLLM 和环境命令的真实调用链

[vllm-env-opportunistic-sharing.md 第 11 节](vllm-env-opportunistic-sharing.md#L425)原本就在讲这件事，下面把图里每个节点在当前代码里的真实位置重新核对了一遍：

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

## 6. 一个可以手算、可以复现的任务依赖例子

[vllm-env-opportunistic-sharing.md 第 11 节](vllm-env-opportunistic-sharing.md#L471)给了一个教学例子：假设模型工作需要独占 GPU 2 秒，它的输出会触发一个 10 秒的 CPU 任务；同时有一个独立的环境 GPU 作业需要独占 10 秒。三种策略的总耗时不同：

模型先执行完再让环境进，模型占用 0 到 2 秒，之后 CPU 任务与环境 GPU 任务在 2 到 12 秒并行，总共 12 秒。环境先占满 GPU 再让模型进，环境占用 0 到 10 秒，模型 10 到 12 秒，CPU 任务 12 到 22 秒，总共 22 秒。两者共享 GPU，假设重叠期间各打对方五折速度，模型 0 到 4 秒（原本 2 秒的工作在抢占下花了 4 秒），环境剩余工作独占到 12 秒，CPU 任务 4 到 14 秒，总共 14 秒。

这个"五折"只是计算方便的教学假设，不是 CUDA 的固定时间片比例，也不是实测减速幅度——这一点在第 7 节会再强调一次。配套脚本 [`vllm_env_sharing_demo.py:19`](../env-service-verification/scripts/vllm_env_sharing_demo.py#L19) 把这套逻辑写成了可运行的 Python，不依赖 CUDA：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/vllm_env_sharing_demo.py
```

本次实际重新运行（在 psrl 仓库根目录下执行上述命令），输出 `model_first_makespan_s=12`、`docker_first_makespan_s=22`、`shared_makespan_s=14.0`，与文档记录的数字一致。这组数字说明的是"共享"比"环境先跑完再让模型进"（22 秒）好，但不如"让模型尽快跑完、尽早解锁依赖它的 CPU 任务"（12 秒）——这个结论只依赖任务之间的依赖关系和假设的减速比例，和真实硬件下的实际减速幅度没有关系，脚本本身也没有调用任何 CUDA API。

## 7. 验证边界：已经证明了什么，还没证明什么

第 1 节的 380 MiB 实测证明的是"显存层面，两个各自持有 CUDA 上下文的进程确实能在物理上共存，占用会被准确测到"。第 4 节关于时间片轮转的说法，证据链是"没有部署 MPS"这个确定的部署事实，加上 CUDA 驱动的通用机制常识，但目前**没有一次真实实验去量化"两边真的同时跑计算时，vLLM 到底会慢多少"**。[vllm-env-opportunistic-sharing.md](vllm-env-opportunistic-sharing.md#L489) 自己列出的边界是：

两个进程能否持有 GPU 资源、模型能否继续生成，这一点已有真机证据，resident 命令成功执行时模型仍能返回 token。环境持续高负载计算时模型会慢多少，目前没有持续计算负载和等工作量下的 A/B 延迟对比数据，这一点**没有验证**。

也就是说，"显存能不能同时放下两个上下文"这件事已经真机验证过；但"两边真的同时跑计算时，vLLM 到底慢多少"，目前只有机制层面的推理——没有 MPS，所以理论上会互相拖慢，但拖慢多少、是否可接受，没有量化实验支撑。第 6 节那组 12/22/14 秒的数字也不能当作这个问题的答案，因为它只是教学假设，不涉及真实 CUDA 调用。如果后续要回答这个问题，需要固定模型、输入输出长度、预热和 GPU，分别测模型独占、环境仅持有显存、环境持续计算三组，记录首 token 延迟、生成总时长、命令耗时，并单独记录控制请求到达至门禁放行的时间，这一点同样记在 [`vllm-env-opportunistic-sharing.md:496`](vllm-env-opportunistic-sharing.md#L496)。

[返回专题入口](../README.md)
