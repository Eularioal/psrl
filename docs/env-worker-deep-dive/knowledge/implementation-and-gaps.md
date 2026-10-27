# EnvWorker 现状代码深读：账本结构、崩溃回收，以及提案里还没落地的部分

## TL;DR

- **修复后状态见 [2026-10-14 修复与对比实验](../airs-bench-audit/fixes-and-results.md)**。本文第 17 节保留修复前审查，相关链接改指向有 SHA-256 校验的 baseline 快照，不能再当成当前实现。
- 2026-10-13 的 AIRS-Bench 审查见本文第 17 节：标签暴露、shell 超时串线、资源回收缺口成立，但“永久死锁”“排队异常无人处理”“只改三个方法已完成接入”需要修正。
- 当前配置指向的 MLGym 源码可读，真实继承方法的最小重放确认初始化和超时中断都会因 `container_obj=None` 触发断言。
- 第 13、15 节是较早的实现分析，其涉及新 Env Service、GPU 支持及其他配方的“当前”判断不能作为本次代码版本的结论。第 17 节明确区分当前静态证据、CPU 复现与尚未验证的集群行为。

> 本文是 [`psrl-env-service-envworker-design.md`](../../psrl/psrl-env-service-envworker-design.md)（EnvWorker 设计详细版）原第 13、15 节拆分出来的独立文档。章节编号沿用原文档编号（13、15），没有从 1 重新编号，方便和速查表、其他文档里已经写好的"详细版 §13.x/§15.x"引用对应。
>
> 读本文前建议先读完详细版的第 0-10 节（问题背景、角色分工、unit、流程、GPU 难点）；本文只讲"现在跑着的代码具体长什么样、和提案原文逐节对照之后还差什么"，不重复讲设计动机。

## 目录

- [13. 现状聚焦：为什么不能"用一次就释放"，拆分 EnvRouter/EnvWorker 能不能合并](#13-现状聚焦为什么不能用一次就释放拆分-envrouterenvworker-能不能合并)
  - [13.1 现状的账本长什么样：三个类各管一件事](#131-现状的账本长什么样三个类各管一件事)
  - [13.2 一次 episode 里，资源到底被占用了多久](#132-一次-episode-里资源到底被占用了多久)
  - [13.3 资源管理为什么困难：三个不确定性叠在一起](#133-资源管理为什么困难三个不确定性叠在一起)
  - [13.4 EnvRouter 和 EnvWorker 分别在扛哪一种困难](#134-envrouter-和-envworker-分别在扛哪一种困难)
  - [13.5 能不能把两者合并：分场景回答](#135-能不能把两者合并分场景回答)
  - [13.6 多个 task 并发申请资源时，名额具体怎么分](#136-多个-task-并发申请资源时名额具体怎么分)
  - [13.7 agent 进程崩溃之后，容器和资源到底会不会被回收——要按"崩溃发生在哪一层"分情况](#137-agent-进程崩溃之后容器和资源到底会不会被回收要按崩溃发生在哪一层分情况)
- [15. 两份文档到底是什么关系：逐节对照，以及提案里没落地的部分现在该怎么看](#15-两份文档到底是什么关系逐节对照以及提案里没落地的部分现在该怎么看)
  - [15.1 逐节对照表：本文哪几节在讲 2026-09-21 提案的哪部分](#151-逐节对照表本文哪几节在讲-2026-09-21-提案的哪部分)
  - [15.2 哪几节不是在讲提案，而是在讲"现在真实跑着的代码"](#152-哪几节不是在讲提案而是在讲现在真实跑着的代码)
  - [15.3 原文里确实还没落地的部分：为什么值得做，初步怎么做](#153-原文里确实还没落地的部分为什么值得做初步怎么做)
    - [(1) Gate A 真正接到 Harbor 路径上，而不是只在旧 EnvWorker 里生效](#1-gate-a-真正接到-harbor-路径上而不是只在旧-envworker-里生效)
    - [(2) Gate B：SessionRouter 出口的 GPU 轮次节流](#2-gate-bsessionrouter-出口的-gpu-轮次节流)
    - [(3) 验证式的 borrow/return（RETURNING 状态）加真正接入调度的租约/心跳，取代现在"写了但没人调用"的空转超时](#3-验证式的-borrowreturnreturning-状态加真正接入调度的租约心跳取代现在写了但没人调用的空转超时)
    - [(4) 周期性负载快照（measured_gpu_memory_mb、measured_util_by_pid）](#4-周期性负载快照measured_gpu_memory_mbmeasured_util_by_pid)
    - [(5) 统一 Path A / Path B（Roadmap Phase 0）](#5-统一-path-a--path-broadmap-phase-0)
    - [(6) 两个已知的 GPU 驱动库 bug（B1/B2），核实结果是：现在的代码里确实还没修](#6-两个已知的-gpu-驱动库-bugb1b2核实结果是现在的代码里确实还没修)
    - [(7) 提升 retry_limit（B8）：成本最低，但也没做](#7-提升-retry_limitb8成本最低但也没做)
- [延伸阅读](#延伸阅读)
- [17. AIRS-Bench 接入批评核对与最小复现](#17-airs-bench-接入批评核对与最小复现)

---

## 13. 现状聚焦：为什么不能"用一次就释放"，拆分 EnvRouter/EnvWorker 能不能合并

前面两节讲的是"怎么用现在的代码"。这一节回到一个更根本的问题：现在的资源账本具体长什么样、多轮工具调用为什么决定了资源必须按 episode（一次完整尝试，定义见第 1 节）而不是按单条命令释放、以及"拆两个组件"这件事本身是不是可以省掉。全部结论都对照当前真实跑的代码，不是提案推演。

### 13.1 现状的账本长什么样：三个类各管一件事

当前实现里有三层对象，各自持有的状态完全不同：

```text
EnvWorkerManager（第 11 节讲过，只在训练启动时跑一次，不参与后续调度）
        │ 创建并注册
        ▼
EnvWorkerCoordinator（1 个 Ray actor，全局唯一）
    状态：{ worker_id → WorkerSlot(cpu_slots, used_cpu, gpu_slots, used_gpu) }
        │ 转发 create_sandbox
        ▼
EnvWorker × N（每个 unit 一个 Ray actor，绑定在具体某个节点上）
    状态：{ sandbox_id → ContainerShell }、_busy_gpu_indices、_last_used_at
        │ subprocess.Popen(["docker", "run", ...])
        ▼
真实容器（可能有很多个，同一个 EnvWorker 名下）
```

用一个具体数字例子过一遍："节点 B 的 cpu_slots=8"这句话在代码里到底是什么：`WorkerSlot`（[`coordinator.py:24-41`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L24)）只有 6 个字段，`used_cpu`/`used_gpu` 是两个整数计数器，不记录"是哪个任务占的"，也不记录任何超时时间。假设节点 B 已经跑了 5 个 CPU 任务、请求方要新开一个 GPU 任务：

```text
WorkerSlot(worker_id=1, cpu_slots=8, used_cpu=5, gpu_slots=2, used_gpu=1)
has_cpu_capacity  →  5 < 8            → True
has_gpu_capacity(1) →  1 + 1 <= 2     → True
```

两个布尔值都为真，这个 worker 才会进入 `select_by_method` 的候选集合（[`coordinator.py:67`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L67)）；`least_loaded` 策略再从候选集合里挑 `used_cpu` 最小的那个（[`coordinator.py:71-72`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L71)）。这一步"先检查、再在同一个 `async with self._lock` 块里原子地 `used_cpu += 1`"（[`coordinator.py:170-177`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L170)），正是第 4 节强调的"检查和预留必须是不可插入的一步"在代码里的样子——不是提案里才有，现在就是这么实现的。

### 13.2 一次 episode 里，资源到底被占用了多久

一个 episode（从 `create_sandbox` 到 `destroy`）在代码里的真实调用序列是：

```text
coordinator.create_sandbox(spec)                          ← 拿号，used_cpu/used_gpu 立刻 +1
  worker.create_sandbox(spec)                              ← docker run + "echo ready" 健康探测
  第 1 轮：handle.exec("cat solver.py", timeout_s=...)
  第 2 轮：handle.exec("sed -i ... solver.py", timeout_s=...)
  第 3 轮：handle.exec("pytest tests/test_solver.py", timeout_s=...)
  ……可能还有第 4、5、6 轮……
handle.destroy()                                           ← docker rm -f，然后 used_cpu/used_gpu -1
```

假设这道题模型平均要试 4 轮才能修对，每轮命令平均跑 6 秒，轮与轮之间还要等模型生成下一步操作（假设推理服务繁忙时要等 15 秒）：

```text
占用时长 ≈ 4 ×（6 秒命令执行 + 15 秒等模型）= 84 秒
```

这 84 秒里，容器绝大部分时间（60 秒）根本没有在执行任何命令，纯粹是在"等模型想下一步"，但 `used_cpu` 计数器从 `create_sandbox` 那一刻起就已经 +1，直到 84 秒后 `destroy()` 才会 -1。这不是实现疏漏，而是必须如此：

1. **容器状态要跨轮次保留。** 第 1 轮里 `sed -i` 改的文件，如果容器在第 1、2 轮之间被销毁重建，第 2 轮再 `cat solver.py` 就看不到刚才的修改了——这正是第 1 节"一个 episode 通常会有很多轮，但共用同一个任务容器"的原因。
2. **重建容器本身有真实开销。** `create_sandbox` 里除了 `docker run`，还要等 shell 探测通过（[`worker.py:248-254`](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L248)），这段延迟受 `startup_timeout_s` 预算限制但绝对不是 0；如果每轮都重开一次容器，这份开销会被乘以轮数，直接堆进模型等待时间里。
3. **GPU 任务还要加上显存/权重的重复加载成本。** 如果任务在第 1 轮往显存里加载了一个模型，第 3 轮的命令还要用它，中途销毁重建就要重新加载一遍——对大模型来说这可能比命令本身跑得还慢。

这里的释放粒度是整个 episode，不是单条命令。`handle.destroy()` 在 episode 结束时执行，容器状态才能跨轮保留。`EnvWorker._last_used_at` 虽然记录最后使用时间，[reap_idle_sandboxes](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L361) 却没有周期调用方，不能说它已经提供了自动兜底。并且该方法本身只销毁沙箱、不归还 coordinator 计数，具体证据见第 17.6 节。租约、心跳和清理确认需要形成完整协议，不能把一个空闲检查函数等同于它们。

### 13.3 资源管理为什么困难：三个不确定性叠在一起

把这件事和普通的批处理任务调度（例如提交一个跑 10 分钟的数据处理脚本）对比，能看清难点具体在哪：

| 维度 | 普通批处理任务 | agent loop 的一个 episode |
|---|---|---|
| 占用时长是否提前可知 | 基本可预估（脚本跑多久相对固定） | 不可知：13.2 的例子里 4 轮、84 秒只是假设；模型可能 1 轮就改对，也可能死循环试 20 轮 |
| 资源是否可压缩/可抢占 | CPU、内存通常能超卖或抢占 | GPU 显存基本不可压缩：第 7 节已经论证过，暂停容器不会释放已申请的显存，torch_memory_saver 管的是模型进程自己的内存，管不到 solver 这个独立进程 |
| "结束"信号是否可信 | 须确认任务及其子进程已退出 | 工具调用返回不保证后台进程结束。agent 崩溃可能留下容器，旧 EnvWorker 的空闲回收方法尚未接入，不能视为自动兜底 |

单独看某一项，问题都不算新——批处理调度器也要处理"时长未知"（比如用历史统计预测），云平台也要处理"资源不可压缩"（比如独占卡）。困难在于这里**三项同时成立**：不知道要占多久、占的东西不能强行拿走、也不能百分百确认它已经用完，这就迫使系统必须持续观察（第 7 节的"负载快照"）而不能只靠一次性的启动检查。

### 13.4 EnvRouter 和 EnvWorker 分别在扛哪一种困难

回头看现在这份代码，你会发现"拆分"这件事其实已经发生了，只是提案里把它进一步做重：`EnvWorkerCoordinator` 这个类的文档字符串第一行就写着"Route sandbox requests **without owning container or placement mechanics**"（[`coordinator.py:1`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L1)）——整份 `coordinator.py` 里没有一处 `subprocess`、`docker` 字样，它只做内存里的字典读写和一次 `worker_handle.create_sandbox.remote(spec)` 转发；真正跑 `docker run`/`docker exec`/`docker rm` 的代码全部在 `worker.py` 里。所以现在的 `EnvWorkerCoordinator` 就是一个"轻量版 EnvRouter"，`EnvWorker` 就是文档里说的"EnvWorker"，两者在职责边界上已经和提案是一致的，提案要加的只是 Gate A、跨训练任务的全局 unit 注册表、租约/TTL 这些还没做的能力（对应第 9 节的推进顺序）。

这两种活儿的"重量"差得很远，可以用数字感受：

- **决策这一侧（Coordinator）**：一次 `select_worker` 只是遍历内存里的 `WorkerSlot` 列表比大小（[`coordinator.py:44-76`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L44)），哪怕有几十个 worker，这个操作也是亚毫秒级的；配置里给它开的并发上限是 `coordinator_max_concurrency`，默认 **64**（[`manager.py:67`](../airs-bench-audit/baseline/psrl/workers/env_worker/manager.py#L67)）。
- **执行这一侧（Worker）**：一次 `exec` 是真的在等一个 shell 命令跑完，`ContainerShell.drain()` 的注释提到 MLGym 的命令"经常跑到 3600 秒"（[`worker.py:72`](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L72)）；配置给它开的并发上限是 `cpu_slots + 4`（[`manager.py:85`](../airs-bench-audit/baseline/psrl/workers/env_worker/manager.py#L85)），这个数字是按"这台机器能扛得住多少个容器"定的，不是按"每秒要处理多少次路由请求"定的。

一个亚毫秒级、要同时服务全局请求的决策服务，和一个单次调用可能挂起一个小时的执行服务，如果揉进同一个类、同一个 Ray actor，会发生什么？决策请求会被同一个并发池里正在阻塞的长 `exec` 调用挤占配额——本该几十微秒答完的"你有没有空位"，可能要排在一个还剩 3599 秒才会返回的 `exec` 后面。这是拆分的第一个理由：**决策的并发特性和执行的并发特性不是一回事，合用一个资源池谁都会吃亏**。

第二个理由是全局视野和本地执行天然冲突：选"哪个节点最空"需要同时看到所有节点的状态，但真正的 `docker run` 必须在容器要落地的那台机器上执行（`manager.py:86` 用 `NodeAffinitySchedulingStrategy` 把每个 `EnvWorker` 钉死在一个节点上）。这有点像 Kubernetes 里调度器（只做决策，不碰容器）和 kubelet（只在本机干活）的分工——这个类比只是帮助建立第一印象，PSRL 现在并没有等价的抢占式重调度能力，边界比 Kubernetes 窄得多，不要过度延伸。

> **配图：这个"决策/执行分离"不是 EnvWorker 独有的发明，是 PSRL 所依赖的 veRL 框架本身的设计**
>
> PSRL 是在 [veRL](https://github.com/volcengine/verl)（字节跳动开源的 RL 训练框架，HybridFlow 论文的开源实现；参见本目录上级 [`README.md`](../../README.md) 里"当前工作区没有...外部 verl/vLLM 源码"的说明）之上做的二次开发。veRL 官方文档 [`docs/hybrid_flow.rst`](https://github.com/volcengine/verl/blob/main/docs/hybrid_flow.rst)"Overall Execution Diagram"一节的原图（图片实际托管在社区仓库 [`eric-haibin-lin/verl-community`](https://github.com/eric-haibin-lin/verl-community/blob/main/docs/driver_worker.png)）：
>
> ![veRL execution diagram: single-process controller dispatching to multi-process actor/rollout/critic workers, credit to eric-haibin-lin/verl-community](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/main/docs/driver_worker.png)
>
> 原文原话（翻译）：**Controller 跑在单个进程里，generator/actor worker、critic worker 跑在多个进程里，按各自的资源组摆放**；rollout 时 controller 把数据传给 generator 生成样本，生成完再传回 controller 做下一步算法逻辑，其他 worker 类似。
>
> 把这张图和本节的结论对着看：
>
> | veRL 概念 | 本文对应 | 共同点 |
> |---|---|---|
> | Controller（单进程，只做数据流决策，不碰实际计算） | `EnvWorkerCoordinator`（[`coordinator.py:1`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L1)，docstring 就写着"without owning container or placement mechanics"） | 决策者天生该是单点、轻量、亚毫秒级 |
> | Generator/Actor/Critic Worker（多进程，真正跑计算，按资源组摆放） | `EnvWorker × N`（[`worker.py:133`](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L133)，真正 `docker run`/`exec`，钉死在具体节点上） | 执行者天生该是多点、可能长时间阻塞 |
>
> 也就是说：13.4 论证的"轻决策不能和重执行塞进同一个并发池"这个结论，不是本文凭空推导出来的设计原则，而是 PSRL 所依赖的底层框架本身就在用的架构模式——EnvWorkerCoordinator/EnvWorker 只是把这套模式从"调度模型计算"搬到了"调度做题容器"这个新场景。

### 13.5 能不能把两者合并：分场景回答

- **场景一：只有一个 unit（`workers_per_node=1` 且只有一个节点）。** 这时"选哪个节点"根本没有第二个候选，`select_by_method` 的候选集合永远只有一个元素，路由决策退化成"这一个就是答案"。这种规模下把 Coordinator 揉进 Worker 里、变成一个类，损失确实不大——现状代码没有这么做只是因为它要同时支持多节点场景，不是因为单节点场景下拆分有什么好处。
- **场景二：多个 unit/多个节点。** 合并会付两个具体代价：
  1. 如 13.4 所述，轻决策会被同一并发池里的重执行拖慢。
  2. 想要"选负载最低的节点"这个能力，就必须有某个地方掌握全局状态。合并之后，要么每个 `EnvWorker` 都要维护一份其它所有 `EnvWorker` 的负载副本（这是一个分布式一致性问题，比现在维护一个居中的字典复杂得多），要么每次决策都要现场跨节点问一圈"你现在多忙"（增加延迟，还会出现"两个节点同时都觉得自己最空，于是都接收了这个请求"的竞态）。这个竞态和第 3 节"两个各自守规矩的进程合计却超载"的例子是同一种问题，只是把"进程"换成了"节点"——当初要在两个进程之上加一份共享账本，就是为了消灭这种竞态；把 Coordinator 和 Worker 合并、让每个节点各自为政，等于把这个已经解决过的问题重新引入一遍。

结论：拆分不是提案为了显得"设计得多"而加的层，是"一旦要跨多台机器统一调度"这个目标本身逼出来的最小代价；只有在整个部署收缩到单节点、单 worker 的最简单形态时，合并才基本无害。现在代码里 `EnvWorkerCoordinator` 已经先把这道题回答了一次（只是功能比提案的 EnvRouter 简），提案要补的是 Gate A 准入检查、跨训练任务的 unit 账本、以及 13.2 提到的"心跳/租约"这类现在还没有的精确存活跟踪。

### 13.6 多个 task 并发申请资源时，名额具体怎么分

`select_worker`（[`coordinator.py:153-183`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L153)）是一个**轮询等待循环**，不是一个严格排队的等号系统。用 3 个并发请求过一遍具体过程：假设集群只有 2 个 worker，`cpu_slots` 都是 2，此刻都已经用满（`used_cpu=2`）；同一时刻来了 3 个 `create_sandbox` 请求 X、Y、Z：

```text
t=0.0s  X、Y、Z 几乎同时进入 select_worker 的 while 循环
        三者都拿不到锁内的名额（两个 worker 的 has_cpu_capacity 都是 False）
        三者各自 await asyncio.sleep(0.5)  ← _SELECT_POLL_INTERVAL_S
t=0.3s  某个已运行的沙箱调用 destroy()，coordinator.release() 把 worker A 的 used_cpu 减到 1
t=0.5s  X、Y、Z 的 sleep 几乎同时到期，重新尝试 async with self._lock
        asyncio 单线程事件循环里，谁先拿到这把锁，就先执行"检查+使用+worker_id 递增"这一整块
        假设调度顺序恰好是 Y → X → Z：
          Y 进锁：worker A 还有 1 个空位 → Y 拿到，used_cpu 变回 2
          X 进锁：两个 worker 都满了 → X 拿不到，继续 sleep 0.5s
          Z 进锁：同上，继续 sleep
```

几个关键点要分开说清楚：

1. **锁保护的是"检查 + 占用"这一步，不是整条等待队列。** `async with self._lock` 只在真正读写 `used_cpu`/`used_gpu` 的那几行生效，X/Y/Z 在 `sleep` 期间根本没有持锁，所以它们不是排在一条 FIFO 队列里等叫号，而是"每隔 0.5 秒各自醒来抢一次"。上面例子里 Y 抢到而 X 没抢到，纯粹取决于 asyncio 事件循环当次唤醒协程的顺序，这个顺序不保证按请求发起的先后来——也就是说，**当前实现不保证先申请的 task 先拿到资源**，存在（大概率很小但确实存在的）"后来的请求插队"的可能。
2. **选哪个 worker 由 `routing_method` 决定，默认 `least_loaded`。** 它只按 `slot.used_cpu`（外加 `worker_id` 破平局）排序（[`coordinator.py:71-72`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L71)），**不看 `used_gpu`**。也就是说，即使某个 GPU 任务的目标是"找 GPU 负载最低的节点"，`least_loaded` 实际比的是 CPU 占用数，这是当前实现的一个真实特点，不是笔误。
3. **等不到资源不会一直等下去。** `create_sandbox` 默认 `timeout_s=3600.0`（[`coordinator.py:185`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L185)），轮询到这个时限还抢不到名额就抛 `RuntimeError`；调用方（agent loop）需要自己处理这个失败，不是无限排队。
4. **拿到 worker 之后还可能失败一次，会换一个 worker 重试一次。** `create_sandbox` 最多试 2 次（`for attempt in range(2)`，[`coordinator.py:202`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L202)）：如果 coordinator 选中的 worker 在真正建容器时报错（比如该 worker 自己的物理 GPU 索引表 `_busy_gpu_indices` 因为某种原因已经不准，见 13.7），就释放这次预占的名额、换下一个 worker 再试一次；两次都失败才最终报错给调用方。

### 13.7 agent 进程崩溃之后，容器和资源到底会不会被回收——要按"崩溃发生在哪一层"分情况

先说会回收的情况：**agent 逻辑内部抛异常，但托管它的 Ray 进程本身没死。** `mlgym_agent_loop.py` 里整个 episode 执行是包在 `try/finally` 里的：

```text
try:
    await asyncio.wait_for(run_mlgym_agent(...), timeout=episode_timeout_s)
finally:
    await handle.destroy()      # 见 mlgym_agent_loop.py:130-136
```

不管中间是正常结束、抛业务异常，还是 `asyncio.wait_for` 因为超过 `episode_timeout_s` 抛 `TimeoutError`，只要执行这段 Python 代码的进程还活着，`finally` 就会跑，`handle.destroy()` 会依次调用 `worker.destroy_sandbox`（真正 `docker rm -f`）和 `coordinator.release`（把 `used_cpu`/`used_gpu` 减回去）。这是"正常"意义上的崩溃，容器和账本都会被清理。

再说**不会自动回收**的情况：**托管 agent loop 的 Ray 进程本身被杀掉**——比如被 OOM killer 杀掉、所在节点掉线、或者被 `ray.kill` 强制终止。这种情况下，进程连同它的调用栈一起消失，`finally` 根本没有机会执行，后果是两件事同时卡住：

```text
EnvWorker 侧：容器继续在物理机上跑，_busy_gpu_indices 里这几个 GPU 编号永远不会被清掉
Coordinator 侧：那个 worker 的 used_cpu/used_gpu 永远不会被减回去（release 从没被调用过）
```

注意这两份账本会**一致地卡死**，不会互相矛盾——它们本来就是同一次 `create_sandbox` 里前后脚一起加上去的，没人调用 `destroy()` 也就没人把两边一起减下去。表面看起来"资源统计没出错"，但实际是这台 worker 从此少了这几个名额，而且没有任何机制知道这是"真泄漏"还是"任务还在正常跑"。

那第 7 节提到的"idle 超时兜底"呢？`EnvWorker.reap_idle_sandboxes()`（[`worker.py:362-378`](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L362)）确实实现了"超过 `idle_sandbox_timeout_s`（默认 7200 秒）没人用就强制销毁"的逻辑，**但需要明确指出**：在当前代码里检索不到任何地方定时调用这个方法——`manager.py`、`ray_trainer.py` 里都没有起一个后台循环去周期性地对每个 worker 执行 `reap_idle_sandboxes.remote()`。也就是说，这个方法目前是一个**已经写好但没有被接入调度的能力**，不能当作"崩溃后 2 小时会自动清理"的既成事实——如果要依赖它兜底，需要额外自己起一个定时任务去调用它；现状下，一次 Ray 进程级崩溃造成的容器和资源泄漏，实际上要等到人工发现或整个训练任务重启（`EnvWorkerManager.shutdown()` 会 `ray.kill` 所有 worker 和 coordinator，[`manager.py:110-115`](../airs-bench-audit/baseline/psrl/workers/env_worker/manager.py#L110)）才会被清理。这正是提案第 7 节要引入租约（lease）、心跳（heartbeat）、TTL 的动机——用"调用方必须定期证明自己还活着"取代"被动等一个没人触发的超时检查"。

## 15. 两份文档到底是什么关系：逐节对照，以及提案里没落地的部分现在该怎么看

### 15.1 逐节对照表：本文哪几节在讲 2026-09-21 提案的哪部分

下表左列是详细版设计文档的章节，右列是它对应 `2026-09-21-env-service-design.md`（简称"原文"）的哪一节，以及对应关系的性质：

| 详细版章节 | 对应原文章节 | 对应关系 |
|---|---|---|
| 第 2 节 EnvWorker/Harbor/EnvRouter 分工 | 第 6.1 节（组件分工表）、第 2 节（两条现有路径为什么出现分叉） | 直接翻译，把"Harbor 是执行者，我们是调度者"这句话拆成具体例子 |
| 第 3 节 两个进程各开 8 个任务超卖的例子 | 第 3.3 节（"共享容量账本"的动机）、第 12.1 节（Ray 管不了 env 容器的 GPU 记账） | 用教学例子解释原文只用一句话带过的"为什么需要共享账本" |
| 第 4 节 unit 是什么 | 第 6.2 节 The unit（`EnvUnit` dataclass、两个不变量、状态图） | 直接翻译，数字例子（16 核/64 GiB/4 容器）是教学演算，不是原文数字 |
| 第 5 节 一次任务从申请到结束 | 第 6.3 节两个 Gate、第 6.4 节 Placement、第 6.5 节 Episode sequence | 直接翻译时间序列图，把 Gate A/B 的经济学差异（排队成本是否为零）具体化 |
| 第 6 节 GPU 任务为什么更难 | 第 4.5 节（时间尺度差三个数量级）、第 5.3 节（显存相加不可压缩）、第 5.4 节（telemetry 读的是整卡计数器）、第 4.6 节（后台进程导致"意图≠占用"） | 直接翻译并用 `nohup ./solver &` 的例子还原原文第 4.6 节 |
| 第 7 节 EnvWorker 为什么要持续上报、处理清理 | 第 6.2 节状态图里的 `LEASE_PENDING`/`RETURNING`、第 6.6 节 Telemetry、第 8.1 节"Unit slot TTL"、第 12.4 节配置示例里的 `slot_ttl_s` | 直接翻译，教学租约数字（60 秒/20 秒续期）是本文自己编的例子 |
| 第 8 节 SMG 和 SessionRouter 在哪 | 第 3.1 节映射表（`SMG worker selection` 对应 `select_unit`）、第 3.2 节（`SessionState.status` 是派生状态） | 背景补充，用于理解原文借用的 SMG/SessionRouter 术语，不是逐句翻译 |
| 第 9 节 当前代码状态、新方案接入顺序 | 第 2.1-2.3 节（两条路径怎么重叠）、第 7.1-7.2 节（L1/L2/L3）、第 9 节 Roadmap 各 Phase | 直接对照，把原文的 Phase 0-4 压缩成一张表 |
| 第 10 节 接入之后能改善什么 | 第 8.1 节 In scope 清单、散落在第 3/4/10 节（known bugs）的各个具体问题 | 汇总翻译，把原文分散在多节的"解决了什么问题"收进一张表 |
| 第 13.4 节（Coordinator 和 Worker 为什么要拆，本文件） | 第 6.1 节组件分工、第 9 节 Roadmap Phase 0 的现状对照 | 反向对照：拿现在真实跑的代码去验证原文的分工道理是否站得住 |
| 第 13.7 节（崩溃后能不能回收，本文件） | 第 6.2 节 `RETURNING` 状态、第 7 节"任务结束不代表资源已释放"背后的真实动机 | 反向对照：现状缺口恰好证明了原文为什么要引入 lease/heartbeat/TTL |

### 15.2 哪几节不是在讲提案，而是在讲"现在真实跑着的代码"

第 1、11、12、13 节（分布在详细版主文档和本文件）的主线不是翻译原文，原文里也找不到对应内容：

- **第 1 节**（修 `solver.py` 的代码题，详细版主文档）是本文自造的教学场景，用来在读者接触任何术语之前先建立"做题环境"的直觉，原文没有这个入门例子。
- **第 11、12 节**（怎么把现有 EnvWorker 接入 PSRL、怎么写 GPU 集成测试，见 [`onboarding-and-gpu-validation.md`](onboarding-and-gpu-validation.md)）讲的是**现在就能跑的旧 EnvWorker**（Path A），这是原文认为"应该保留、要在外面套一层 Harbor"的那部分执行体，但原文并未给出这两节里的接入步骤代码或测试文件，这是本文基于现状代码自己补充的落地路径，超出了原文的范围。
- **第 13 节**（本文件）主线是逐行核对现在跑着的 `coordinator.py`/`worker.py`，只有 13.4 和 13.7 两处明确拿现状去对照原文的分工道理和 lease 动机（见上表最后两行），其余小节（13.1-13.3、13.5-13.6）讲的是当前实现的真实行为，原文里没有对应段落。

### 15.3 原文里确实还没落地的部分：为什么值得做，初步怎么做

下面每一项先说"现状缺什么"，再说"不做会在什么场景吃亏"，最后说"原文给出的初步实现路径"。判断"有没有落地"的依据是直接读当前 `psrl/psrl/workers/env_worker/` 源码，不是猜测。

#### (1) Gate A 真正接到 Harbor 路径上，而不是只在旧 EnvWorker 里生效

**现状（截至本节最初写下时）：** `coordinator.py` 的 `has_cpu_capacity`/`has_gpu_capacity` 加原子锁（[`coordinator.py:170-177`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L170)）其实已经实现了 Gate A 的核心思路——"检查预算、原子预留、再创建容器"。但它只服务于走旧 EnvWorker 的 MLGym/AIRS-Bench 配方；走 Harbor 的 SciAccel 路径（Path B）完全绕开了这套账本，直接调用 Harbor 的 `Job.create`。而原文第 7.4 节已经核实过：**Harbor 自己的 `n_concurrent_trials` 只是一个不区分 cpu/mem/gpu 的计数信号量**，8 个并发配额在 2 卡主机上会把 8 个 `gpus=1` 的任务全部放行。

**更新（已部分落地，见 [`sciaccel-gpu-gate.md`](sciaccel-gpu-gate.md) §16.5）：** 上面说的是**数量预算**层面的问题（CPU/内存/GPU 总量核验），这部分目前仍未接入 `EnvWorkerCoordinator`，还是本节最初描述的状态，没有变化。但 §16 另外发现的**设备身份**层面问题（同时放行的多个 GPU episode 会拿到完全相同的 `/dev/nvidia0` 字符串，哪怕数量预算本身没超）已经用 §16.3 的"轻量方案"修复并部署：`GpuDeviceGate`（[`gpu_device_gate.py`](../../../psrl/utils/concurrency/gpu_device_gate.py)）接入了 [`runner.py`](../../../examples/sciaccel_rl/runner.py) 的 `run_harbor_episode`，两个问题不是同一件事，不要把"设备身份已修复"误读成"数量预算也已经接入"。

**不做会怎样：** 这不是假设性风险——按第 5.3 节的实测，显存是相加且不可压缩的；如果 8 个任务真被 Harbor 同时放行，且这台机器显存装不下，会直接触发 OOM，而不是排队等待。

**初步实现路径（原文第 9 节 Phase 1 加第 7.2 节）：** 在 SciAccel 的 agent loop 里、`Job.create` 之前，插入一次调用：用 `TaskConfig.model_validate_toml(task_toml.read_text())`（纯解析，不启动任何东西）读出这次任务声明的 `cpus/memory_mb/gpus`，拿去和 unit 的剩余预算做同一套"检查加原子预留"判断——这套判断逻辑不需要发明新机制，`coordinator.py` 里已经写好且能跑，缺的只是把 Harbor 路径接到同一个协调器上。原文特别注明这部分"规划阶段测试已经写过、12+12+9+2 个测试通过"，但这是原文自己的说法，本文没有独立验证过这些测试。

#### (2) Gate B：SessionRouter 出口的 GPU 轮次节流

**现状（最初写下本节时）：** 完全没有。当前 `SessionRouter` 的 `hang` 机制只挡"下一次模型请求"，不会在模型已经生成回复、agent 即将执行工具命令之前插入任何等待（第 8 节已说明）。

**更新（已实现，见 [`sciaccel-gpu-gate.md`](sciaccel-gpu-gate.md) §16.5）：** 已按下面"初步实现路径"描述的思路落地，挂在 `session_chat_completions` 返回模型回复之前（[`session_router.py`](../../../psrl/workers/gen/session_router.py) 的 `_wait_for_gpu_idle`），只对被 `POST /control/gpu_bind` 显式绑定过 GPU 设备的 session 生效——`examples/sciaccel_rl/runner.py` 的 Gate A 在拿到设备后会调用这个端点。空闲判断用的是 `nvidia-smi --query-gpu=index,utilization.gpu,memory.used`（真实整卡计数器，不是逐进程的 `pmon`——这是本次实现和下面"初步实现路径"建议的一处差异，会在整卡有其它无关进程时把"空闲"判断得偏保守，即误判为"忙"而多等，但不会误判为"空闲"而少等，方向上是安全的），默认最长等待 600 秒（`PSRL_GATE_B_MAX_WAIT_SEC`），超时直接放行。

**不做会怎样：** 直接对应第 5.4 节的实测结论——`nvidia-smi` 的 GPU 利用率读数是整卡计数器，不是按进程区分。如果同一张卡上有两个任务的 solver 真的同时在算，其中一个任务的评分会被另一个任务的 GPU 活动"污染"（把别人的计算误判成自己的）。这不是理论推演，是原文用两个各分配 8GB 的容器同时压 `/dev/nvidia0` 实测出来的现象（第 5.3 节）。

**初步实现路径（原文第 6.3 节加第 6.5 节）：** 挂在 SessionRouter 返回模型回复之前的出口 hook，只对 GPU unit 生效；最多等待 `admission_timeout_s`（默认 600 秒，必须小于 agent 自己的 LLM 请求超时——SciAccel 现在是 900 秒，见 [`runner.py:250-252`](../../../examples/sciaccel_rl/runner.py#L250)），超时就直接放行而不是判失败，因为放行只是"多花一点吞吐"，判失败是"丢掉最多四小时的 episode"。判断"GPU 是否真的空闲"要用 `nvidia-smi pmon` 的逐进程占用，不能只看 session 状态是不是 `env`（否则会被 `nohup ./solver &` 这种正常的后台用法骗过去，见第 6 节）。

#### (3) 验证式的 borrow/return（`RETURNING` 状态）加真正接入调度的租约/心跳，取代现在"写了但没人调用"的空转超时

**现状：** 第 13.7 节已经核实过：`reap_idle_sandboxes()` 这个方法本身存在（[`worker.py:362-378`](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L362)），但整个代码库里没有任何地方定时调用它。释放/归还只有两种状态——"占用中"和"已释放"，中间没有原文第 6.2 节设计的 `RETURNING`（正在归还，需要核验空闲后才能标记 `FREE`）过渡状态，也没有心跳/租约续期机制去区分"任务还在正常算"和"调用方已经崩溃"。

**不做会怎样：** 正是前面回答的那个缺口——一次 Ray 进程级崩溃会让容器和 coordinator 的账本一起卡死，没有任何自动机制能分辨这是真泄漏还是长 episode 正常运行，只能靠人工发现或整个训练任务重启。

**初步实现路径（原文第 6.2 节加第 7 节的教学租约例子）：** 引入显式状态机 `FREE → LEASE_PENDING → ENV → RETURNING → FREE`，每一次状态转移都包在 `try/finally` 里释放（原文明确点名要避免复刻 `check_and_migrate` 那个没有 `finally`、导致抽干超时后全局路由停摆的 bug，即 B7）；调用方按固定间隔上报心跳续约租约，租约到期（而不是"7200 秒没用过"这种一刀切的空闲判断）才触发回收，并且回收 GPU 时要用 `nvidia-smi --query-compute-apps` 核验设备确实空了之后才把 unit 标记回 `FREE`，而不是发完 `docker rm` 命令就直接认为完事。

#### (4) 周期性负载快照（`measured_gpu_memory_mb`、`measured_util_by_pid`）

**现状：** `WorkerSlot` 只有 `used_cpu`/`used_gpu` 两个自己维护的整数计数器（[`coordinator.py:24-41`](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L24)），没有任何一处真正去问 `nvidia-smi` 实际显存/利用率是多少，也没有定期上报的通道。

**不做会怎样：** 前面已经指出，默认 `least_loaded` 路由策略只看 `used_cpu`，连 `used_gpu` 都不看，更别提区分"申报 18 GiB"和"实测已经飙到 21 GiB"这种原文第 7 节强调的偏差。没有测量数据，调度只能盲选。

**初步实现路径（原文第 6.6 节，仿照 rollout 侧 `EngineStats` 的模式）：** 每个 EnvWorker 定期跑 `docker ps --filter label=...` 和 `nvidia-smi --query-compute-apps` 拼成一份快照，通过一条**独立于 `create_sandbox`/`exec` 控制路径**的通道发布——这一点很关键，第 13.4 节已经论证过"亚毫秒级决策"和"可能挂起一小时的执行"不能塞进同一个并发池，采集遥测数据同理不能占用处理 `create_sandbox` 请求的那条路径。

#### (5) 统一 Path A / Path B（Roadmap Phase 0）

**现状：** 第 9 节已经说明，SciAccel（Path B）现在仍然自己维护一个**进程级**（不是节点级）`asyncio.Semaphore`（[`agent_loop.py:60-84`](../../../examples/sciaccel_rl/agent_loop.py#L60)），没有接入 `coordinator.py` 的任何账本。

**不做会怎样：** 就是原文 B4 描述、且已经在启动脚本里写死变通方案的真实故障——同一节点上堆更多 worker 进程，会把这个节点的容器数按 worker 数成倍放大，原文引用的证据是 `examples/sciaccel_rl/fsdp_qwen35_4b.sh:105-114` 那段注释。

**初步实现路径（原文第 9 节 Phase 0 四步）：** 把 SciAccel 的 Harbor 调用挪到一个 EnvWorker actor 背后（行为不变）；把进程级 `Semaphore` 换成节点级的 `SlotManager`（[`psrl/utils/concurrency/slot.py`](../../../psrl/utils/concurrency/slot.py)，本来就是现成的、mini SWE loop 已经在用的节点级原语，只是 SciAccel 没用）；给容器打上自己的标签（这一步在旧 EnvWorker 里其实**已经做了**——`SANDBOX_LABEL_KEY`/`ACTOR_LABEL_KEY`，见 [`worker.py:21-22,189-196`](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L21)，只是 Path B 没有）；确认 Harbor 的 terminus-2 episode 能像旧 shell 一样在多轮之间保留容器内状态之后，再删掉旧的 exec-shell 后端。

#### (6) 两个已知的 GPU 驱动库 bug（B1/B2），核实结果是：现在的代码里确实还没修

这一条不是"提案设想的风险"，是刚刚直接读 [`sandbox.py:20-23`](../airs-bench-audit/baseline/psrl/workers/env_worker/sandbox.py#L20) 核实到的、现在就存在的问题：

```python
DEFAULT_DRIVER_LIB_GLOBS: tuple[str, ...] = (
    "/usr/lib64/libnvidia-ml.so.1",
    "/usr/lib64/libcuda.so.1",   # B1：原文明确说"永远不要挂载这个符号链接"
)
# 缺 libnvidia-ptxjitcompiler.so.1，对应 B2
```

**不做会怎样：** 对应原文第 5.2 节的实测——那台测试机上 `libcuda.so.1` 指向的是和内核驱动版本不匹配的用户态库，直接调用会返回 `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`；但因为 `libnvidia-ml.so.1` 指向的版本是对的，`nvidia-smi` 在容器里跑起来完全正常，只有真正的 CUDA 初始化会失败——这是一种"看起来很健康、实际不能用"的隐蔽故障，原文强调它"排查起来会被引导到错误的方向"。

**初步实现路径（原文第 5.2 节的 Fix 建议）：** 启动容器前先读宿主机的 `/proc/driver/nvidia/version`，用这个版本号去匹配挑选真正对应版本的库文件，挂载解析出来的那个具体路径，而不是无条件挂载 `.so.1` 符号链接；同时把 `libnvidia-ptxjitcompiler.so.1` 补进挂载列表，否则容器内 PTX JIT 编译会失败。

#### (7) 提升 `retry_limit`（B8）：成本最低，但也没做

**现状：** 已核实，`rollout.agent.retry_limit` 配置默认值是 `1`（[`psrl_rollout.yaml:154`](../../../psrl/trainer/config/rollout/psrl_rollout.yaml#L154)、[`rollout.py:84`](../../../psrl/workers/config/rollout.py#L84)），[`worker.py:389-427`](../../../psrl/workers/agent_loop/worker.py#L389) 里那段重试循环 `for retry_attempt in range(1, retry_limit + 1)` 在这个默认值下等价于只跑一次，重试分支实际是死代码。

**不做会怎样：** 一次偶发的基础设施错误（比如第 10 节提到的节点 Docker daemon 卡死）会直接判这个 episode 失败，白白扔掉已经花费的时间，不会有任何重试机会。

**初步实现路径：** 原文把这一条标为"应该在做任何重派发/迁移机制之前先评估"的最低成本改动——不需要写任何新代码，只是把配置默认值调高，观察 episode 成功率的变化。

以上七项之外，原文还提到"跨训练任务的全局 unit 注册表"（`EnvUnitRegistry` 需要在多个训练任务之间共享，而不是像现在 `EnvWorkerCoordinator` 一样只在单次 `EnvWorkerManager.__init__` 里创建、生命周期绑定单个训练任务）——这一条本文没有单独展开，因为它依赖前六项先落地之后才有意义讨论。

## 延伸阅读

- 设计动机和角色分工：[`psrl-env-service-envworker-design.md`](../../psrl/psrl-env-service-envworker-design.md) 第 0-10 节。
- 怎么把现有 EnvWorker 接进 PSRL、怎么写 GPU 集成测试：[`onboarding-and-gpu-validation.md`](onboarding-and-gpu-validation.md)。
- SciAccel 接 GPU Gate 的实测与部署（第 (1)(2) 条已落地的具体内容）：[`sciaccel-gpu-gate.md`](sciaccel-gpu-gate.md)。

## 17. AIRS-Bench 接入批评核对与最小复现

核对日期：2026-10-13。本次只做审查、CPU 最小复现和知识留档，没有修改训练实现或运行真实集群。PSRL HEAD 为 `90a4178f40b4de3e90ada1c9ebc4baf15815921a`，工作区已有未提交修改，因此判断以实际读取的文件为准，不以 HEAD 代表全部内容。复现文件的 SHA-256 摘要保存在 [results.json](../airs-bench-audit/results.json)。

### 17.1 全局串联图和源码位置

下图是代码声明的调用路径，不代表完整 episode 已经跑通。`[4]` 的 MLGym 初始化存在下文确认的阻断。

```text
[1] RayTrainer -> EnvWorkerManager
                  | 创建 coordinator 和 env 节点上的 EnvWorker
                  v
[2] MLGymAgentLoop.run -> env.reset（只准备任务信息）
                  -> session_scope（先创建 TITO 会话）
                  v
[3] coordinator.create_sandbox -> select_worker（预占计数/等待）
                  -> EnvWorker.create_sandbox -> docker run + shell 探测
                  v
[4] 线程池 run_mlgym_agent -> PSRLMLGymEnv -> MLGym BaseAgent
         |                                  |
         | 模型请求经 session URL           | 工具命令
         v                                  v
   TITO 记录 tokens                [5] communicate -> _communicate
                                         -> sync_exec -> handle.exec
                                         -> EnvWorker.exec -> 持久 Bash
                                            |
                                            +-> data/（整个任务目录，只读）
                                            |      +-> test_with_labels
                                            +-> evaluate.py -> 评分
                  v
[7a] _run_episode_logic 的 finally：handle.destroy
                  -> worker 删容器 -> coordinator.release
                  v
[6] runner 结果返回 run -> get_training_data -> compute_reward_score
                  v
[7b] session_scope 删除会话 -> MLGymEnvironment.close 本机兜底
```

图例：图里每个部分对应的源码位置。

| 图中部分 | 修复前快照与可跳转位置 |
|---|---|
| [1] 启动与 actor 池 | [ray_trainer.py:1867](../../../psrl/trainer/ppo/ray_trainer.py#L1867)，[EnvWorkerManager:63](../airs-bench-audit/baseline/psrl/workers/env_worker/manager.py#L63) |
| [2] 准备任务与提前开会话 | [MLGymEnvironment.reset:235](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L235)，[MLGymAgentLoop.run:171](../airs-bench-audit/baseline/psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L171)，[session_scope:85](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L85) |
| [3] 排队与启动 | [coordinator.select_worker:153](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L153)，[create_sandbox:185](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L185)，[worker.create_sandbox:198](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L198) |
| [4] 同步 runner 与外部 MLGym | [线程池等待:131](../airs-bench-audit/baseline/psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L131)，[run_mlgym_agent:91](../airs-bench-audit/baseline/examples/airs_bench/runner.py#L91)，[继承适配类:145](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L145) |
| [4] 模型请求与记录 | [host_url 配置:55](../airs-bench-audit/baseline/examples/airs_bench/runner.py#L55)，[session_api_url:93](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L93)，[get_training_data:183](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L183) |
| [5] 命令桥接 | [适配器 _communicate:175](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L175)，[sync_exec:103](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L103)，[handle.exec:94](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L94)，[worker.exec:256](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L256) |
| [5] 数据与评分边界 | [整目录挂载:94](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L94)，[prepare 标签要求:266](../../../examples/airs_bench/prepare/prepare_airs_data.py#L266)，[scripted 评分:184](../airs-bench-audit/baseline/examples/airs_bench/scripted_episode.py#L184) |
| [7a] 正常释放 | [handle.destroy:111](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L111)，[worker.destroy_sandbox:343](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L343)，[coordinator.release:219](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L219) |
| [6] 训练数据与奖励 | [runner 分数整理:151](../airs-bench-audit/baseline/examples/airs_bench/runner.py#L151)，[训练数据/奖励:190](../airs-bench-audit/baseline/psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L190)，[reward 分数消费:122](../../../examples/airs_bench/reward.py#L122) |
| [7b] 会话关闭及本机兜底 | [session_scope finally:90](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L90)，[env.close:269](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L269) |

这些代码如果有理解上的问题，可以随时追问。Actor 是 Ray 管理的有状态远程对象，这里分别承载调度计数和实际容器。Episode 是 agent 完成一次任务尝试的全过程。TITO 在本接入中负责记录模型输入输出的 token（模型处理文本的离散单位）和对应概率信息，供训练使用。

### 17.2 逐条裁定

| 原批评 | 裁定 | 必要修正 |
|---|---|---|
| AIRS 使用 env_worker，未使用 env_service 租约 | 成立 | 当前 sandbox.py 复用了 env_service 的 GPU 辅助函数，不等于接入它的租约、心跳和回收协议 |
| 只重写三个方法就能完整适配 MLGym | 不成立 | 实际方法名是 `_init_container`，不是 `_init_env`，且上游还有直接使用 Docker 对象的继承方法 |
| ① 标签对 agent 可读 | 成立，训练有效性 P0 | 证明的是暴露和可利用条件，没有运行 RL 来测量模型已经利用的频率 |
| ② shell 超时后输出/退出码串线 | 成立，CPU 实际复现 | 不保证总是恰好错位一格。当前 MLGym 上层还会因 interrupt 断言失败结束 episode，不能直接断言真实 AIRS 一定静默训练错误样本 |
| ③ 默认 56 请求争 8 槽，先开 session | 成立 | 56 是默认训练派发窗口的推算，48 是 56 个请求都已到达时的等待数，不是本次集群测量，也不是所有路径严格共享的硬上限 |
| ③ 排队不受 episode_timeout_s 管 | 成立 | 另有可选 trajectory_timeout 覆盖整个 run，默认 null，AIRS 脚本未设置 |
| ③ 排队 RuntimeError 会直接向外传播 | 局部成立 | 会离开 MLGymAgentLoop.run，但外层有错误分类、重试、组补采及熔断 |
| ③ 超过 64 等待者会永久死锁 | 风险成立，措辞过强 | 活跃等待 RPC 填满共享并发池时 release 被阻塞，但放置超时能打破等待，准确说是控制请求饥饿和长时间停顿 |
| ④ 崩溃后槽位泄漏，idle reaper 未接入 | 成立 | 仅接通 reaper 也不够，它只删容器，不归还 coordinator 计数 |
| 配置双轨、部分配置无消费者 | 成立 | 还发现 exec_default_timeout_s 只保存，没有用作 exec 的缺省超时 |
| 镜像启动秒退仍等 600 秒 | 条件性成立 | 已证实退出进程仍耗尽读取预算，但写管道也可能先报 BrokenPipeError，不是每次固定等 600 秒 |
| runner 超时不杀线程 | 成立，CPU 实际复现 | 删除容器只能促使后续命令失败，不能保证正在 HTTP 请求或重试中的线程立即结束 |
| host 网络、代理继承、sudo apt 权限宽 | 成立 | 容器 root 不等于已逃逸到宿主机，代理值包含凭据需要按实际环境判断，本次未读取/披露凭据 |
| README reaper 和 baseline 描述错误 | 成立 | cache_baseline_scores=False 控制写回 task YAML，不能推导为完全不算 baseline 或 agent 必然看不到 baseline |

### 17.3 P0：标签只读仍然暴露答案

证据链是完整的：准备阶段要求存在 train、test、test_with_labels，`build_sandbox_spec` 将它们共同的任务父目录挂在 agent 的 `data/` 下，挂载模式是 `ro`。`ro` 是 read-only，只禁止对挂载内容写入，不限制目录遍历或读取。沙箱评分程序能够以 agent 身份读标签，同一身份运行的 agent 命令也能读。

实际只读检查发现，配置的数据根目录下有 20 个任务的 `test_with_labels`。抽查 `TextualClassificationSickAccuracy` 和 `TimeSeriesForecastingSolarWeeklyMAE` 的标签 Arrow 文件为 `0644`，即文件所有者可读写、组和其他用户可读。Sick 任务的 evaluate.py:17（本地资料，未随仓库发布） 直接读取 `test_with_labels` 的 `label` 列。独立的 [scripted_episode.py:56](../airs-bench-audit/baseline/examples/airs_bench/scripted_episode.py#L56) 也直接在容器中读取标签目录。

手算教学例子：测试答案为 `[0, 1, 0, 1]`，正常模型预测 `[0, 0, 0, 1]`，准确率是 3/4。若直接读取答案并提交 `[0, 1, 0, 1]`，准确率是 4/4，却没有学会分类。这就是 reward hacking（利用评分漏洞获取奖励）：优化目标奖励了读取答案的行为，而非期望的学习过程。上述四条数据为教学示例，没有声称本次完成了实际攻击提交。

修复应建立独立评分边界：agent 只得到公开训练集和无标签测试输入，提交产物交给独立 verifier（可信评分进程/容器），只有 verifier 能读取隐藏标签和可信评分代码。仅在提交时把标签挂回同一容器仍有后台进程偷读的可能，改目录名、chmod 或只读挂载也不能提供所需隔离。

### 17.4 P0：哨兵协议损坏，与 MLGym 契约缺口分开判断

哨兵是命令执行完后额外打印的结束标记。当前 [wrap_command:23](../airs-bench-audit/baseline/psrl/workers/env_worker/shell.py#L23) 附加 `///PROCESS-DONE:<退出码>:PROCESS-DONE///`，但里面没有标识命令的唯一编号。[parse_sentinel:56](../airs-bench-audit/baseline/psrl/workers/env_worker/shell.py#L56) 取输出中第一个符合格式的标记，将它当成本次命令结果。

超时只是 Python 停止等待。Bash 的前台命令还在执行，所以新的 B 命令写入 stdin 后也要排在 A 后面。`drain()` 在 B 发出前只能清空已经到达的字节，无法清掉未来才出现的 A 尾部输出。结果就是 A 的结束标记触发 B 返回，再把 B 的结束标记交给后续命令。

本次直接执行当前 `EnvWorker.exec` 和 `ContainerShell`，将 Docker 进程替换为本地真实 Bash，得到：

| 调用 | 应有结果 | 实测返回 |
|---|---|---|
| A：打印 A-start，sleep 0.6，打印 A-end，成功。等待上限 0.1 秒 | 超时 | `A-start`，timed_out=true |
| B：sleep 0.35，打印 B-output，false | B-output，退出码 1 | A-end，退出码 0 |
| C：打印 C-output，true | C-output，退出码 0 | B-output 和 C-output，退出码 1 |

完整输出在 [results.json](../airs-bench-audit/results.json)。C 返回多少混合文本取决于读取分块时机，不能把“固定错一格”当成协议保证。退出码 0 表示 shell 命令成功，非零表示失败，本例说明成功/失败判断也会被串线。

当前配置指向的 MLGym checkout 实际存在，提交为 `9d40c1b5035202018cd7091fb4e83a9c68b377c0`，本次检查工作树无修改。其关键路径如下：

| 问题 | 实际源码 |
|---|---|
| 是 communicate 还是 step 捕获超时 | env.py:667（本地资料，未随仓库发布） 的 communicate 不捕获该超时，step:458（本地资料，未随仓库发布） 捕获并调用 interrupt |
| interrupt 是否绕开 _communicate | interrupt:803（本地资料，未随仓库发布） 断言两个容器对象非空，然后通过 Docker exec_run 杀进程、读取原 Popen 管道并做健康检查 |
| 子类提供了什么 | [mlgym_env.py:170](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L170) 将 container 与 container_obj 都设为 None，且没有覆盖 interrupt |
| 为什么更早就可能失败 | 父类初始化:249（本地资料，未随仓库发布） 调用 _reset_container:954（本地资料，未随仓库发布），之后的 _init_scripts:1045（本地资料，未随仓库发布） 立即断言 container_obj 非空 |
| 还有哪些绕过通信接口的调用 | _setup_workspace:1193（本地资料，未随仓库发布）、reset:334（本地资料，未随仓库发布）、add_commands:759（本地资料，未随仓库发布） 都直接依赖 Docker 对象 |

本次通过 AST（抽象语法树，即 Python 解析后的代码结构）提取真实方法，保留原文件和行号，用子类真实 `_init_container` 和最小对象状态重放。初始化在 `_init_scripts:1045` 抛 AssertionError，超时路径在 `interrupt:803` 抛 AssertionError。这不是完整 MLGym 运行，当前激活环境还缺 gymnasium 和 simple_parsing，但已能验证继承方法与子类状态冲突。

因此新增一个接入阻断级问题：不能认为“agent 没改、只换底层通信”已经足够。当前 MLGym 正常 action 超时更可能转成 runner 的 `exit_error`，再被 PSRL 丢弃/补采，不能据底层复现断言真实训练必定静默吃入错误轨迹。另一方面，底层串线仍必须修，否则其他直接调用者或以后补完上层适配后仍会受影响。TITO 可以忠实记录错误观测，错误观测不自动等价于 token 对齐算法本身出错。

修复要求：完整覆盖初始化、文件复制、工具安装、命令执行、中断、退出、清理等环境契约。命令层需要可靠取消运行中的进程、确认执行结束并同步输出边界，再允许下一条命令。唯一命令 ID 有助于匹配结束标记，但不能单独解决旧进程继续运行。`docker run -i` 没有分配终端，向管道写 Ctrl-C 字节不能当作可靠发出中断信号。

### 17.5 P1：56 对 8 的瓶颈、异常处理和控制请求饥饿

配方默认 N_RESP_PER_PROMPT=4、TRAIN_BATCH_SIZE=14、staleness=0，对应 [trainer 公式:395](../../../psrl/trainer/ppo/ray_trainer.py#L395) 的 4×14×1=56。一个 env 节点、每节点 1 worker、每 worker 8 个沙箱槽，对应同时 8 个沙箱。56 个请求均到达且没有完成时，48 个在等待。这里 cpu_slots 是沙箱数量槽，不是物理 CPU 核数，每个 SandboxSpec 实际另限 8 CPU 和 32g 内存。

这些是静态配置推算，不是实际排队测量。agent worker 的 `max_concurrency` 限制 RPC（远程方法调用），但 [generate_trajectory:304](../../../psrl/workers/agent_loop/worker.py#L304) 又创建独立 asyncio task 后返回。因此也不应把 trainer 的 56 当成所有训练/验证/补采路径严格统一的硬信号量。

会话创建早于沙箱等待，见图中 [2]、[3]。这会增加空闲会话的存续时间，但没有证据表明每个空会话都会立即占满 GPU 显存。`episode_timeout_s` 的 10800 秒仅包住 runner，未覆盖排队、启动及 finally 清理。外层 [base_agent_loop.py:690](../../../psrl/workers/agent_loop/loops/base_agent_loop.py#L690) 支持整条轨迹超时，但 [默认配置:136](../../../psrl/trainer/config/rollout/psrl_rollout.yaml#L136) 为 null，AIRS 启动脚本未覆盖。

排队到期确实抛 RuntimeError，但外层 [worker.py:397](../../../psrl/workers/agent_loop/worker.py#L397) 捕获为 ROLLOUT_ERROR。默认 retry_limit=1 是总尝试次数，当前 worker 不会再试第二遍，随后 [worker.py:430](../../../psrl/workers/agent_loop/worker.py#L430) 通知 manager，后者进行组补采，连续失败达到阈值则 [manager.py:558](../../../psrl/workers/agent_loop/manager.py#L558) 熔断。故问题是超时归类和失败成本，并非完全无人处理的裸异常。

Coordinator 的并发名额是另一套资源。`await asyncio.sleep` 让出事件循环的执行权，但方法尚未完成，仍占一个活跃 RPC 名额。[Ray 官方文档](https://docs.ray.io/en/latest/ray-core/actors/async_api.html) 将 max_concurrency 定义为同时运行的方法数量。当 64 个排队 create 调用占满名额后，release 也无法进入同一 actor，于是本来可以结束的沙箱不能及时归还计数。

不过这些 create 的放置等待默认 3600 秒后会抛异常，腾出 RPC 名额，所以不是由当前代码证明的永久死锁。最小脚本用 2 个 RPC 名额和 1 个已占满的沙箱槽模拟，两个等待者均超时后，原先受阻的 release 能完成。这是机制模型，不是实际 Ray 集群复现。

不要为了匹配 56 个请求直接把沙箱数改为 56。若每个都打满配额，56×8=448 CPU 配额、56×32 GiB=1792 GiB 内存上限总和；当前 8 个槽则分别是 64 和 256 GiB。这些是配置上限算术，不是资源预留量或实际用量。应按节点可用资源定准入容量，再在创建会话前施加有界排队和背压，让上游按下游能力派发。

### 17.6 P1：回收缺口不止是“缺少调用 reaper”

租约是附带有效期的占用凭证。TTL 是 time to live，即有效时长；心跳让仍存活的持有者定期续期。例如教学设定 TTL=90 秒、每 30 秒续约，持有者最后一次续约之后消失，90 秒到期才触发回收。这些数字是教学假设，当前 EnvWorkerCoordinator 没有这套协议。

[WorkerSlot:25](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L25) 只记录 used_cpu、used_gpu 等汇总计数，没有 sandbox_id 到持有者的租约关联。`finally: handle.destroy()` 只对协程正常展开有效，进程直接死亡时无法执行。

[reap_idle_sandboxes:361](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L361) 没有生产调用方，也不读取 spec.idle_timeout_s。即使手动调用，它只执行 destroy_sandbox，不调用 coordinator.release。本次用真实 reaper 方法、仅替换物理 Docker 删除动作的 CPU 实验，容器移除记录为 orphan，而 used_cpu 仍为 1。接上定时任务不能单独修复账本泄漏。

还不能把“多久没收到 exec”直接当作持有者死亡：`_last_used_at` 在命令结束时更新，一条长命令仍在执行时也可能被错误判空闲。租约续期与命令输出活跃度应分开。

确实存在通用 agent actor 的 [sidecar reaper:69](../../../psrl/workers/agent_loop/worker.py#L69)，不能笼统说整个项目没有 reaper。但该机制依赖本机 Docker 和 actor 标签。AIRS 请求没有显式把 agent actor_id 放进 spec，EnvWorker 的 [标签读取:193](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L193) 发生在另一个 actor 进程里，不能假定 agent 的 os.environ 修改会传播过去。它也没有归还旧 coordinator 计数的接口。

`MLGymEnvironment.close()` 直接调用 [本机 docker ps/rm:105](../../../psrl/utils/common/docker_utils.py#L105)。在通常连接本地 Docker daemon 的部署下，agent 不在 env 节点时清不到目标容器；即使同机删到了，也不自动归还计数。若 DOCKER_HOST 特意指向远程 daemon，位置结论会变，因此关键是清理请求是否发往容器实际归属端。

合理修复是把 AIRS 生命周期接入已有租约服务，并让实际 EnvWorker/Docker 执行端负责清理，验证容器确实消失后再归还容量。必须有独立清理通道、过期回收和幂等释放（同一租约多次释放只生效一次）。不能只给已有计数器减一，也不能只开 env_service.enable 就声称 AIRS 已接入。

### 17.7 P2 和其他需要修正的细节

| 项目 | 本次核对与证据 |
|---|---|
| CPU/内存双轨 | [run 脚本:188](../airs-bench-audit/baseline/examples/airs_bench/run_qwen3-4b.sh#L188) 写 env_worker.sandbox_cpus/memory，但实际 spec 来自 [runtime 配置:91](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L91)。同名默认值碰巧一致会掩盖改配置不生效的问题 |
| 输出截断双轨 | [runtime.max_observation_chars:48](../airs-bench-audit/baseline/examples/airs_bench/config.py#L48) 无执行消费者，真正传给 worker 的是 [manager.py:93](../airs-bench-audit/baseline/psrl/workers/env_worker/manager.py#L93) 读取的 env_worker 配置 |
| 默认命令超时 | [exec_default_timeout_s:153](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L153) 被存储，但 exec 要求调用者显式传 timeout_s，没有取该缺省值 |
| 启动进程退出 | [read_until:103](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L103) 不检查 poll/EOF。本次退出码已为 7 的 Bash 仍等待完整的 0.2 秒实验预算。放大到默认 600 秒是代码路径推论，不是实测等待了 10 分钟。若 write 先遇到断管，则会更早失败 |
| 重试节点 | [create_sandbox:202](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L202) 不排除上次失败 worker，least_loaded 释放计数后可再选同一台，与 docstring 不符 |
| 线程取消 | [wait_for + executor:131](../airs-bench-audit/baseline/psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L131) 超时后不能杀掉已执行线程。本次标准库实验确认 future 被取消时线程仍运行，脚本随后显式通知退出，没有遗留线程 |
| 网络与代理 | [build_sandbox_spec:84](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L84) 继承代理变量并使用 host 网络，agent 能读传入容器的变量，也能访问宿主网络命名空间内服务。是否含凭据、能访问哪些服务须按部署核实 |
| sudo apt | [Dockerfile:35](../../../examples/airs_bench/docker/Dockerfile#L35) 对 apt/apt-get 免密放行且不限制参数。APT 的 [Pre-Invoke 官方说明](https://manpages.debian.org/bookworm/apt/apt.conf.5.en.html#HOW_APT_CALLS_DPKG(1)) 明确支持执行 shell hook，因此不能把这看成仅有“安装包”而没有容器内 root 执行能力。没有据此声称发生了宿主逃逸 |
| README reaper | [故障表:147](../airs-bench-audit/baseline/examples/airs_bench/README.md#L147) 描述了并未接通的周期性空闲回收，容易错误引导排障 |
| baseline 开关 | 实际开关在 [runner.py:131](../airs-bench-audit/baseline/examples/airs_bench/runner.py#L131)，上游 env.py:365（本地资料，未随仓库发布） 仍调用 baseline 更新函数，376 行开关控制结果写回 task YAML。README 将它描述为缓存容器镜像并推断 agent 看不到 baseline，混淆了 cache_baseline_scores 和 cache_task_images |

### 17.8 跟着图读源码

| 顺序 | 对应图 | 入口/函数 | 观察重点 |
|---|---|---|---|
| 1 | [1] | [RayTrainer 初始化:1867](../../../psrl/trainer/ppo/ray_trainer.py#L1867) → [EnvWorkerManager.__init__:52](../airs-bench-audit/baseline/psrl/workers/env_worker/manager.py#L52) | 命名 actor、worker 放置、64 个 RPC 名额和 8 个沙箱槽是不同限制 |
| 2 | [2] | [MLGymAgentLoop.run:138](../airs-bench-audit/baseline/psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L138) → [env.reset:235](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L235) → [session_scope:85](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L85) | 任务路径如何进入 observation，会话何时创建 |
| 3 | [3] | [_run_episode_logic:89](../airs-bench-audit/baseline/psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L89) → [coordinator.create_sandbox:185](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L185) → [worker.create_sandbox:198](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L198) | 先预留后启动，失败如何归还，等待和启动分别计时 |
| 4 | [4] | [runner.run_mlgym_agent:125](../airs-bench-audit/baseline/examples/airs_bench/runner.py#L125) → [适配子类初始化:148](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L148) | 当前继承契约断在 _init_scripts，不应假定已经进入 agent.run |
| 5 | [5] | [sync_exec:127](../airs-bench-audit/baseline/psrl/environments/mlgym_env.py#L127) → [handle.exec:101](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L101) → [worker.exec:284](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L284) → [read_until:88](../airs-bench-audit/baseline/psrl/workers/env_worker/worker.py#L88) | 同步线程通过事件循环发远程调用，结束标记没有请求身份 |
| 6 | [7a] | [episode finally:135](../airs-bench-audit/baseline/psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L135) → [handle.destroy:111](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L111) → [release:219](../airs-bench-audit/baseline/psrl/workers/env_worker/coordinator.py#L219) | runner 已整理分数，返回 run 之前先删容器并归还计数 |
| 7 | [6] | [collect_scores_from_info:64](../airs-bench-audit/baseline/examples/airs_bench/runner.py#L64) 整理的结果 → [训练数据/评分:190](../airs-bench-audit/baseline/psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L190) | 分数与 token 记录是两条数据流，最终在输出中汇合 |
| 8 | [7b] | [session_scope finally:90](../../../psrl/workers/agent_loop/loops/session_agent_loop.py#L90) → [env.close:212](../airs-bench-audit/baseline/psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L212) | 退出会话后执行 env 本机兜底，容器删除、槽位归还、线程退出、会话关闭是不同状态 |
| 9 | 错误分支 | [base wrapper:690](../../../psrl/workers/agent_loop/loops/base_agent_loop.py#L690) → [worker retry:391](../../../psrl/workers/agent_loop/worker.py#L391) → [manager 补采:842](../../../psrl/workers/agent_loop/manager.py#L842) | 局部异常不等于直接打崩 trainer，也不等于训练已成功 |

该顺序是一次 episode 的调用关系。不同 actor、runner 线程和 asyncio 协程可以并发执行，不能把整个训练系统理解成一个顺序线程。

### 17.9 可运行复现和验证边界

脚本：[reproduce.py](../airs-bench-audit/reproduce.py)。从本工作区根目录执行：

```bash
python docs/env-worker-deep-dive/airs-bench-audit/reproduce.py
```

脚本仅依赖 Python 标准库和 Bash，不启动 Ray、Docker、GPU 或模型，不读数据集，不修改生产源码。默认从 AIRS 的 config.py 解析 MLGym 路径，也可以用 `--mlgym-repo` 指定真实 checkout。源文件不存在时明确报告未运行该项。`--output` 可指定保存 JSON 结果的文件路径。

脚本保留源代码文件名及行号，直接编译当前生产定义，仅替换环境依赖导入和物理删除动作。输出包括真实 shell 串线、已退出进程读取等待、空闲回收后账本计数、线程取消，以及 MLGym 两条继承路径断言。并发池部分明确是信号量模型，用于展示等待者占满名额后释放请求进不去，不是 Ray 实机证明。代码注释标注了图中步骤，可逐项对照。

没有执行完整 MLGym 初始化、LLM 调用、训练 step、真实容器取消、远程 actor 崩溃回收或评分攻击。本次也没有测量队列长度、吞吐、超时频率、reward hacking 发生率。现有 [MLGym adapter 测试](../../../tests/environments/test_mlgym_env.py) 主要验证配置、挂载、history processor 和 sync_exec，不能替代真实上游环境契约测试。[scripted_episode:153](../airs-bench-audit/baseline/examples/airs_bench/scripted_episode.py#L153) 手动上传文件并直接 exec，不经过 PSRLMLGymEnv/MLGymEnv 初始化，所以它通过也不能证明完整接入成立。

### 17.10 是否修，以及合理顺序

建议修复，但不直接按原清单打局部补丁。本次仅给方案，未实施。

1. **先确立可运行且可验收的 MLGym 适配契约。** 固定外部依赖版本，补足初始化、文件操作、命令中断、退出和清理，再用真实 MLGym 加小任务检查完整 episode。现有 scripted smoke 不覆盖这个边界。
2. **正式 RL 前必须同时修评分隔离和 shell 协议。** 可信 verifier 持有标签和评分代码；命令具有唯一身份，取消会停止实际执行并恢复明确输出边界，无法恢复时终止本 episode，不能继续使用污染 shell。
3. **接入已有租约生命周期，而不是再造第二套计数补丁。** 统一所有者、心跳、过期清理、远端容器归属、清理确认和幂等释放。把获取沙箱放在创建 TITO 会话前，采用覆盖排队与启动的 deadline，并处理“创建请求取消但远端已经创建成功”的回收竞态。
4. **按真实 CPU/内存约束控制准入和上游派发。** 等待与释放/心跳通道分离。消除配置双轨，启动失败快速返回并识别坏 worker，完成后再更新 README。

与已有笔记的关系：第 13 节解释旧计数器的结构，第 15 节列出当时的缺口；本节把 AIRS 的具体证据和修正结论补齐。[Env Service 实现笔记](env-service-implementation.md) 介绍另一套租约机制，其存在不意味着 AIRS 已自动受保护。
