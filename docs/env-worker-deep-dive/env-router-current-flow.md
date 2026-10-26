# EnvRouter 当前流程、固定基线差异与验证边界

> **核心结论**：当前 EnvRouter 将容器生命周期的 CPU／内存预算与 GPU 命令阶段预算分开。显存够用时，不同容器、不同租约的命令可以并行，不够时排队；同一容器当前仍只执行一条 GPU 命令。命令结束后经实际执行节点确认归还，预算可用于其他容器，而原容器继续 CPU／模型交互。直接 Docker 查询和 NVML 连接复用降低重复控制开销，节点异步预热减少首次服务启动的关键路径等待。
>
> 固定 OFF 基线为 `0eeead861ffafed6bf31c962d259fd847775dcf9`。命令阶段复用与 vLLM 同卡借还是不同配置分支，尚不能任意组合。最新实机数字维护在[预热与 OFF／ON 对照报告](router-baseline-control-validation.md#10-results私有节点预热与固定-off-复验)。本页解释当前实现，不把合成 CUDA 小实验写成真实训练收益。

> **与同容器并发提案的关系**：command GPU 已支持显式命令间隙归还，以及归还检查与模型规划重叠；新方案扩展的是同一租约内多个 GPU 作业的独立记账、同时执行和独立回收，不是重新实现此前的异步。详见第 7 节。

> **通用接入进展**：已完成[统一执行后端第一阶段](unified-gpu-execution.md)，episode 与 command 共用受管理作业接口和执行生命周期。完整通用接入仍按下文设计推进，保留完整会话与独立执行阶段两种 GPU 持有范围。常驻 GPU 状态不能因等待模型而自动交还。第 10 至 13 节给出当前限制、改造接口、释放恢复成本和 CPU 教学模拟。同容器多作业并发已单列 TODO，暂不实施，也不阻塞跨容器多 rollout 的通用接入。

## 目录

- [1. 完整流程与角色](#1-完整流程与角色)
- [2. 相比固定 baseline 改变了什么](#2-相比固定-baseline-改变了什么)
- [3. 用一张卡说明准入、并发和回填](#3-用一张卡说明准入并发和回填)
- [4. 预热的实现与生命周期](#4-预热的实现与生命周期)
- [5. 跟着图读源码](#5-跟着图读源码)
- [6. 当前限制与可复现实验](#6-当前限制与可复现实验)
- [7. command GPU 异步与同容器并发有什么区别](#7-command-gpu-异步与同容器并发有什么区别)
- [8. 容器与作业的申请关系](#8-容器与作业的申请关系)
- [9. 多 rollout 意图纠正与 OFF 的同容器能力](#9-多-rollout-意图纠正与-off-的同容器能力)
- [10. episode 与 command 的区别及通用性边界](#10-episode-与-command-的区别及通用性边界)
- [11. 通用 GPU 执行服务的具体设计](#11-通用-gpu-执行服务的具体设计)
- [12. Results：持有范围的 CPU 教学验证](#12-results持有范围的-cpu-教学验证)
- [13. 常驻 GPU 状态与释放恢复成本](#13-常驻-gpu-状态与释放恢复成本)

## 1. 完整流程与角色

```text
[1] Trainer 创建 EnvRouter
      ├─ 继续模型初始化
      └─ 异步预热本轮配置的节点服务 [6]

[2] AgentLoopManager 读取节点可用容量，选择执行任务的 worker
      ↓
[3] science runner / EnvServiceClient 请求会话租约
      → EnvRouter / Ledger：按 unit 检查 CPU、主存、容器数
      ← 允许 / 等待 / 拒绝；允许时返回租约、GPU 身份和节点接口
      ↓
    worker 内的 Harbor 调用该节点 Docker 创建任务容器
      ↓
[4] 容器需要 GPU → ManagedDockerEnvironment.exec_gpu
      → Router 的命令队列：检查申报预算和当前物理占用
      → [6] 实际执行节点检查容器归属、设备身份并授权
      ← 获准后，由 worker 执行容器中的 CUDA 命令
      ↓
[5] 命令结束
      ├─ 容器继续 CPU／下一次规划
      └─ 节点撤权、检查设备句柄及计算进程
           → Router 确认归还、释放 GPU 预算 → 放行其他请求 [4]
      ↓
[7] 评分及任务清理 → 归还会话的 CPU、主存、容器位置
      心跳丢失／异常：保留待清理状态，节点按归属清理后才解封

独立分支 [8]：unit 绑定整组 vLLM 推理副本
      模型 sleep 确认 → Docker 环境阶段
      → 容器归还／暂停确认 → 模型 wake up → 再次进入环境阶段
      当前此分支不能同时设置 gpu_reservation: command
```

图例：图里每个部分对应的源码位置：

| 编号 | 职责、数据与源码 |
|---|---|
| 1 | [训练器初始化](../../psrl/trainer/ppo/ray_trainer.py#L591) 调用 [create_env_router](../../psrl/workers/env_service/router.py#L408)；提交预热后继续模型工作，首请求按需等待节点 |
| 2 | [AgentLoopManager._refresh_env_capacity](../../psrl/workers/agent_loop/manager.py#L909) 更新调度容量，[get_dispatch_plan](../../psrl/workers/agent_loop/manager.py#L947) 分配任务组；这是正式训练入口，本轮合成实验直接调用 runner |
| 3 | [EnvServiceClient.acquire](../../psrl/workers/env_service/client.py#L159)、[EnvUnitLedger.try_admit](../../psrl/workers/env_service/ledger.py#L105)、[runner](../../examples/sciaccel_rl/runner.py#L259)：会话需求变为带归属标签的容器配置 |
| 4 | [ManagedDockerEnvironment.exec_gpu](../../psrl/workers/env_service/docker.py#L92)、[GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L49)：请求标识、队列、显存预算与授权确认 |
| 5 | [begin_return](../../psrl/workers/env_service/command_gpu.py#L112)、[_return](../../psrl/workers/env_service/command_gpu.py#L127)：服务持有后台归还任务，锁外检查，锁内提交配额释放 |
| 6 | [EnvNodeService](../../psrl/workers/env_service/node.py#L19)、[grant_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L139)、[revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153)：同节点 Docker 身份校验、GPU 权限和实时状态确认 |
| 7 | [runner 等待延迟归还](../../examples/sciaccel_rl/runner.py#L505)、[Node.cleanup](../../psrl/workers/env_service/node.py#L68)、[Ledger.block_gpu_cleanup](../../psrl/workers/env_service/ledger.py#L332)：错误不能变成成功，配额过期不等于物理资源已消失 |
| 8 | [GpuLendingBinding](../../psrl/workers/env_service/lending.py#L27)、[EnvUnit 配置约束](../../psrl/workers/env_service/types.py#L50)：确认借还的状态机与整副本绑定限制 |

这些代码如果有理解上的问题，可以随时追问。

这里的 unit 是预先配置的一组资源：节点、GPU 设备集合、CPU 配额、主存预算、显存预算和并发容器上限。租约是某条任务获得这些资源的登记，心跳用于证明持有者仍然存活。unit 不是“每个任务必须整份独占的显存块”，多个任务的申报可以累计装入同一份预算。

Router 是记账与授权协调者，Harbor 负责实际任务容器执行，节点服务负责在正确的机器上核实物理状态。Docker 命令不需要经 Router 转发全部输出；只有资源准入、授权与归还经过它。名称相近的通用 EnvWorker 沙箱池有自己的 coordinator，这次 science／Harbor 对照不经那条通用沙箱执行链，不能将两套角色混为一个服务。

## 2. 相比固定 baseline 改变了什么

| 环节 | 固定 baseline 的行为 | 当前 ON 的行为 | 实际作用 |
|---|---|---|---|
| 任务并发限制 | runner 进程内信号量，静态 GPU Compose 映射 | 共享 Router 依据 unit 管理预算，正式任务分发可读取节点容量 | 多个任务入口能看到同一份资源账本 |
| GPU 使用许可 | 映射到相同 GPU 的活动任务可以直接执行 | 根据队列、申报和实时占用决定是否允许命令 | 高压力时把失败竞争转为受控等待 |
| GPU 配额保留 | 没有这份动态显存账本 | `command` 模式在 GPU 阶段持有，确认归还后释放 | 其他容器可利用 CPU／模型交互间隔 |
| 容器状态 | 任务执行和清理由 Harbor 管理 | 仍保留容器与文件系统，GPU 配额单独借还 | 不必为了释放 GPU 每次销毁整个容器 |
| 归还与规划 | 没有 Router 归还确认 | 可让独立 CPU／规划与后台归还并行，错误后续必须收敛 | 避免安全确认阻塞所有后续工作 |
| 容器查询 | 不执行当前授权协议 | 执行节点直接访问本机 Docker Unix socket | 降低命令行子进程查询开销，保留身份检查 |
| GPU 查询 | 不执行当前授权协议 | NVML 精确查询，节点服务复用初始化连接，每次重读状态 | 降低重复驱动初始化开销 |
| 首次节点服务启动 | 没有 EnvRouter 节点服务 | 模型准备期间异步预热，启动与首请求共用任务 | 尽量覆盖首次启动成本 |
| 异常回收 | 主要依赖任务执行路径清理 | 标签、心跳、后台清理、未确认资源隔离 | 避免账本归零却仍有容器／GPU 使用 |

基线代码的静态映射见[本次运行快照](env-service-verification/results/router-prewarm-private-20261020/baseline-runner.py#L191)。保留了必要的运行修复，详细补丁见[baseline.patch](env-service-verification/results/router-prewarm-private-20261020/baseline.patch)。它不是当前代码关闭 `env_service` 的分支；当前关闭服务的路径已经包含其他 GPU gate，不能替代指定基线。

表中的直接 Docker 接口、NVML 连接复用和预热，都是为了降低 ON 自身安全协议的成本。不能说 baseline 原本也花了这些检查时间，然后把节省次数直接算作相对 OFF 的收益。相对 OFF 的主要收益需要竞争、失败补采或更好任务放置等实际条件来体现；低负载仍可能付出净管理成本。 具体耗时边界、长尾和优化优先级见[低负载关键路径分析](router-baseline-control-validation.md#11-results低负载慢在哪里与优化优先级)。

## 3. 用一张卡说明准入、并发和回填

本次实验条件是一张约 95.6 GiB 的 H20，unit 预算 80 GiB，最多四个存活容器。每条任务申报比 CUDA 分配多 1 GiB，为上下文等开销留余量。

低负载每条分配 1 GiB、申报 2 GiB，四条累计 8 GiB，预算允许四个独立容器、不同租约的 GPU 命令并行。这里不是同一个容器中的四条命令。unit 的短暂检查锁用于协调读取和承诺额度，不覆盖命令完整运行时间；另有覆盖同容器命令执行的锁，两者作用范围不同。

高压力每条分配 50 GiB、申报 51 GiB。两个声明需要 102 GiB，超过 80 GiB 预算，因此 ON 同时只放行一条 GPU 命令。A 完成后，节点确认它不再持有 GPU，释放 51 GiB 预算；A 的容器继续做 8 秒 CPU 工作，B 此时可以执行 GPU 命令。A 下一次需要 GPU 时重新排队，不必让 B 等 A 整条任务结束。

```text
A：GPU 阶段 → 确认归还 → CPU／规划────────→ 再申请 GPU
B：等待额度───────────→ GPU 阶段 → 归还 → CPU／规划
GPU 预算：51 GiB → 0 → 51 GiB → 0 → …
CPU／主存／容器位置：A、B 仍各自保留，直到自己的任务结束
```

此图是对应实验设置的机制示意，精确时间以原始日志为准。OFF 也可能在 A 物理释放后碰巧回填，但没有防止两个 50 GiB 分配同时重叠的准入机制，所以同样的空档复用可能伴随 OOM 和重试。ON 的目标是使放行有依据，并减少失败浪费。

显存申报仍是预算，不是硬件硬隔离。任务低报、外部进程突然进入、设备内部碎片和未计入的开销都可能造成风险，不能把一组零 OOM 写成任意任务永不 OOM。unit 的 GPU 集合仍静态配置，不能将多卡总预算自动解释为任意单卡都放得下。

## 4. 预热的实现与生命周期

环境服务启用时，默认配置为 `prewarm.enable: True`，启动预算 30 秒、最多四个节点并行准备。[create_env_router](../../psrl/workers/env_service/router.py#L408) 等待 [begin_prewarm](../../psrl/workers/env_service/router.py#L329) 确认提交，随后训练器继续初始化模型；它不等待全部节点启动完成。[wait_prewarm](../../psrl/workers/env_service/router.py#L336) 提供显式等待接口，实验用它测量串行与并行准备。

[_node_task](../../psrl/workers/env_service/router.py#L317) 保存每节点唯一启动任务，预热、准入、GPU 借用绑定均复用它。一个请求取消不会取消所有人共用的启动任务。[_start_node](../../psrl/workers/env_service/router.py#L297) 的超时覆盖并发名额等待和实际准备；失败会销毁本次刚创建的服务，失败结果留给后续消费者，不自动无限重试。

[Node.prepare](../../psrl/workers/env_service/node.py#L49) 在执行节点探测 Docker，首次读取 GPU 身份和实时占用，使该服务的 NVML 连接就绪。它不创建租约、不授权容器，也不把这次状态读数缓存成以后 GPU 空闲的证明。节点只读准备与模型加载可以重叠，真正的借用 GPU 仍要求模型 sleep 后再次确认。 节点服务连接复用与 runner 准备是两条不同路径。runner 生成租约 overlay 仍使用短生命周期的实时设备查询，但已移到后台线程，Docker 同步探测也通过异步准备移出主事件循环；实现、取消边界和实机对照见[同步准备优化](router-baseline-control-validation.md#13-results同步准备路径移出事件循环)。

[shutdown_nodes](../../psrl/workers/env_service/router.py#L342) 拒绝丢弃活动租约；租约结束后取消未完成启动、关闭服务接口与连接、销毁所属服务 actor。训练器正常停止以及模型初始化异常的退出路径都调用环境服务停止入口。实际 CUDA、容器、后台评分器的释放仍单独核验，不能用服务 actor 消失替代物理清理检查。

## 5. 跟着图读源码

| 步骤 | 图中编号 | 输入、调用与状态变化 |
|---|---|---|
| 初始化 | 1 → 6 | unit 配置 → `create_env_router` → `begin_prewarm`，后台 pending 启动任务与模型初始化并行 |
| 选择任务执行节点 | 2 | 正式训练的 `get_dispatch_plan` 读取容量快照；快照辅助放置，最终准入仍由 Router 判定 |
| 建立任务会话 | 3 | runner 通过 `EnvServiceClient.acquire` 提交需求；`try_admit` 等服务就绪，账本授予租约，Harbor 创建容器 |
| 申请 GPU 命令 | 4 → 6 | `exec_gpu` 带租约、容器和请求标识进入 `acquire`；先入先出队列决定次序，预算和实时查询决定能否放行 |
| 执行 | 6 → 4 | 节点核对容器租约与设备映射，开启设备访问；worker 调用 Docker 执行实际 CUDA 命令 |
| 归还与并行 CPU 工作 | 5 | `begin_return` 将归还交给服务；CPU 分支继续，`_return` 等节点撤权及确认后释放 GPU 配额 |
| 下一次 GPU／评分 | 4、7 | 下一条 GPU 命令、显式等待或容器停止会收敛延迟归还错误；评分不能把归还失败当作成功 |
| 会话结束与故障 | 7 → 6 | 正常结束释放剩余预算；心跳超时进入待清理，按归属物理回收后才恢复不确定资源 |
| 模型借卡分支 | 8 | `GpuLendingBinding.set_phase` 根据阶段和序号推进 sleep／归还／wake，失败进入隔离，不能混用命令模式 |

## 6. 当前限制与可复现实验

已经实现的并非跨机器运行中容器迁移，也没有预测下一次 GPU 调用时间的调度器。任务启动前可以选择节点，活着的 Harbor 容器仍留在原节点，命令阶段在其所在 unit 排队。CPU／主存／容器数预算仍按会话保留，可能成为吞吐瓶颈。先入先出队列也没有抢占正在执行的 CUDA 命令。

vLLM 借用模式要求完整副本绑定、每个借用 unit 一个容器位置，并且不能配置 `gpu_reservation: command`。本次单卡对照只验证独立环境 GPU 的命令配额、归还、预热与失败补采，不验证模型 GPU 同卡借还。实际训练的奖励、loss、entropy 和收敛效果未在本次测量。

无需 GPU 的原理实验：[node_prewarm_demo.py](env-service-verification/scripts/node_prewarm_demo.py) 模拟三种启动顺序，说明为什么只有真正的重叠才减少总时间。真实容器实验：[science_router_control_ab.py](env-service-verification/scripts/science_router_control_ab.py)；分析及画图：[plot_router_prewarm_ab.py](env-service-verification/scripts/plot_router_prewarm_ab.py)。

```bash
# CPU 教学模拟，不代表真实系统性能。
python docs/env-worker-deep-dive/env-service-verification/scripts/node_prewarm_demo.py

# 使用已保存的实机结果，重建对比图，不再次启动容器或 GPU。
python docs/env-worker-deep-dive/env-service-verification/scripts/plot_router_prewarm_ab.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-prewarm-private-20261020
```

教学模拟的预期是串行预热只移动等待位置，并行预热才缩短总时间；真实实验分析则核对四个任务均完成、八条有效 CUDA 命令、全部失败补采计时、GPU 身份一致和最终清理状态。完整条件、结果与停止预算见[实机报告](router-baseline-control-validation.md#10-results私有节点预热与固定-off-复验)。

## 7. command GPU 异步与同容器并发有什么区别

**用户记得的异步设计已经存在。** command GPU 的基本能力是把 GPU 使用期从整个任务会话中分离：A 的某条 GPU 命令结束并确认归还后，A 的容器仍然存活，可以等待模型或执行 CPU 工作，B 则使用腾出的容量。`overlap_return=True` 又允许 A 的模型规划与后台归还检查重叠。新提案继续保留这些能力，增加同容器内多作业同时使用 GPU、逐作业取消和回收。

先看完整关系。图中 A、B 是不同任务容器；A1、A2、A3 是同一容器内的不同作业。前半段是已有实现，后半段是尚未落地的提案。

```text
已有 command GPU，示例假设环境卡一次只能容纳一个命令：
[C1] A 的 GPU 命令得到稳定输出，服务接管归还
          ├─ [C2] A 请求模型，规划下一步，保留容器文件和 CPU 状态
          └─ [C3] 节点检查 A 的物理归还，确认前预算仍由 A 持有
                           ↓ 确认后
                    [C4] B 获准使用环境 GPU
                           ↓ B 归还后
          [C5] A 的模型结果已就绪 → 下一命令重新申请、排队和执行
                           ↓
                    [C6] 会话结束，收敛回收并清理容器

同容器并发提案，示例假设容量足够容纳两个作业：
[N1] 容器 A 提交 A1、A2、A3，每作业独立预算
          ├─ A1 运行 ───→ 仅回收 A1 ───→ A3 获准
          └─ A2 运行 ───────────────────→ 仅回收 A2
          GPU 工作重叠；结束 A1 不撤销 A2 的权限
          容量不足时仍按 C3/C4 的原则，确认回收后再放行
```

图例：图里每个部分对应的源码位置。

| 编号 | 当前实现与源码 |
|---|---|
| C1 | [exec_gpu](../../psrl/workers/env_service/docker.py#L92)，[overlap_return 分支](../../psrl/workers/env_service/gpu_execution.py#L55) 等待服务接管后返回输出 |
| C2 | [Science 模型请求 _query_llm](../../examples/sciaccel_rl/agent.py#L125)；显式异步作业也可由 [GpuJobManager.start](../../psrl/workers/env_service/gpu_jobs.py#L60) 返回 job_id，让调用方做独立工作 |
| C3 | [begin_return](../../psrl/workers/env_service/command_gpu.py#L112) 创建服务持有的后台任务，[_return](../../psrl/workers/env_service/command_gpu.py#L127) 等待 [revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153) 后才释放预算 |
| C4、C5 | [acquire](../../psrl/workers/env_service/command_gpu.py#L49) 判断队列与容量；[同租约限制](../../psrl/workers/env_service/command_gpu.py#L63) 禁止一份租约多个待执行或活动命令，说明不是同容器多 GPU 作业并行 |
| C6 | [wait_gpu_command_returns](../../psrl/workers/env_service/router.py#L234)、[finish_gpu_command_lease](../../psrl/workers/env_service/router.py#L240) 收敛归还与租约清理 |
| N1 | 新协议没有生产实现，落点与契约模型见 [GPU 作业并发提案](gpu-jobs-and-task-telemetry.md#11-同容器并发的目标架构)，不能把现有同容器锁说成新调度器 |

这些代码如果有理解上的问题，可以随时追问。

**三个经常被混用的“异步”。** 协程是可在等待点暂停、让其他调用继续推进的程序执行方式。它让出 CPU 执行机会，不会自动释放它持有的 GPU 资源。现有作业 start 接口立即返回 job_id，表示调用方不必等待作业完成；作业此时可能排队，也可能正在使用 GPU。`overlap_return=True` 则发生在输出稳定之后，让调用方不必等物理回收检查全部结束才能规划。它们都不等于“两条 GPU 作业已经同时执行”。

| 能力 | 当前 command GPU 及异步作业 | 同容器并发提案 |
|---|---|---|
| GPU 命令结束后，容器继续等待模型，其他容器利用空档 | 已有，物理归还后放开预算 | 保留 |
| 物理归还检查与独立 CPU／模型规划重叠 | 已有，overlap_return 显式开启；异步作业路径使用它 | 保留 |
| 提交 GPU 作业后先做其他独立工作 | 已有 start/status/wait/cancel 接口 | 保留 |
| 不同容器的 GPU 命令在容量足够时并行 | 当前准入机制允许，仍受配置和遥测约束 | 保留 |
| 同一容器的 A1、A2 在容量足够时并行 | 当前同容器锁、作业管理锁与同租约协议不允许 | 新增目标 |
| 同容器每条命令申报不同预算并单独记账 | 当前命令额度来自 lease.demand，不是各 job 的独立需求 | 新增目标 |
| A1 结束或取消时，A2 保持运行和授权 | 当前不允许这种同容器 GPU 重叠，因此未提供此种隔离 | 新增目标 |
| 自动暂停任意仍持有显存的 CUDA 程序，把它稍后无损恢复 | 没有 | 此提案也不提供 |
| 独立环境卡 command 与 vLLM 同卡借还直接组合 | 配置明确拒绝 | 此提案也不自动解决组合问题 |

不能把区别粗略概括成“旧方案只能串行，新方案才会并行”。旧方案已经有跨容器并行、跨任务空档复用和 CPU／模型规划重叠。新增范围是同一容器、同一会话租约里的多个 GPU 作业，要求进一步细化资源归属。

同一容器也不是必须等某条协程的模型请求结束才能继续用 GPU：只要上一条 GPU 命令已经确认归还、同容器锁可用，另一条独立协程已提交的 GPU 命令就可以顺序接力。旧协议限制的是同时活动的 GPU 命令数，不是所有 CPU／模型等待期间都锁住容器。是否能提前推进还取决于调用方的数据依赖，依赖模型新计划的命令当然必须等计划产生。

**数字例子一：你记得的空档复用，旧方案就可以做。** 以下是教学假设：环境卡可用预算 4 GiB，A、B 各需要 3 GiB，所以不能同时运行；模型推理用另一张卡。把 A 第一条命令输出就绪作为 t=0，物理归还需 3 秒，模型规划需 8 秒，B 的 GPU 命令需 2 秒，A 下一条命令需 4 秒。

- 开启异步归还：t=0 至 3，A 的模型规划与归还检查重叠，但 B 尚不能使用 A 的预算；t=3，确认归还后 B 开始，t=5 结束；t=8，A 得到计划，重新获准运行到 t=12。
- 同步等待归还：A 在 t=3 才开始模型规划，到 t=11 得到计划，下一命令到 t=15 才完成。B 仍可在 t=3 至 5 运行，因此空档复用和提前规划是两个可区分的收益。
- 这里节省的 3 秒来自隐藏回收等待，没有让两个 3 GiB 作业同时挤进 4 GiB。B 若更长、占用到 t=13，A 即使 t=8 已有计划也必须继续等；模型提前返回不等于 GPU 立即可用。

此例假设 B 归还无额外延迟，隔离出 A 的等待成本，不是生产时延预测。

**数字例子二：旧方案缺少的同容器能力。** 同一个容器里 A1、A2 各需要 1 GiB，卡上有足够容量。当前仍让 A1 执行并归还后再执行 A2。提案让 A1、A2 各自记 1 GiB，同时运行，A1 结束只释放自己的 1 GiB；A2 仍运行时可以准入 A3。若卡只剩足够一个作业的容量，新方案也必须排队，它不会改变资源不足这一事实。

**“发给 vLLM 就暂时不用 GPU”需要区分三种情况。** vLLM 是大模型推理服务，替代理生成下一步操作。代理等待回复时，可以不占环境 GPU，但推理服务本身仍需使用模型 GPU。

1. **独立模型卡和环境卡。** A 的环境命令结束且资源释放后，A 在模型卡上请求推理，B 在环境卡上执行。command GPU 解决的就是这类环境空档复用。发送模型请求本身不触发资源释放，释放由命令退出与归还协议触发。
2. **Docker 与同一个 vLLM 实例共用一张卡。** 需要另一条 GPU lending 状态机：模型释放／休眠后才让环境用卡，环境物理归还后才让模型恢复。当前 [GpuLendingBinding._transition](../../psrl/workers/env_service/lending.py#L51) 和 [agent 的阶段切换](../../examples/sciaccel_rl/agent.py#L136) 实现这一分支；[EnvUnit 校验](../../psrl/workers/env_service/types.py#L54) 明确禁止它与 command 模式直接叠加。异步提交请求不等于同卡 vLLM 能提前计算。
3. **同一个仍在运行的 GPU 程序中途 await 模型请求。** 如果该进程保留张量、上下文或后台子进程，它仍持有 GPU。command 模式不会因为协程等待或利用率下降而自动收回。必须拆出显式可归还边界，或者程序配合实现释放和恢复；本次提案也没有通用的任意 CUDA 程序暂停恢复能力。

如果以后要组合“vLLM 借卡 + 多个环境作业”，需要父级 env/model 阶段和子级 job 预约两层协议，切回 model 前必须收敛所有环境作业。这是额外的组合设计，不是同容器并发补丁自然附带的功能。

**跟着图读源码。** C1 的 exec_gpu 输出稳定后进入 overlap_return 分支，将请求交给 C3 的 begin_return。C2 是调用方可以推进的独立工作，与 C3 并发；它不是 _return 自动调用模型。C3 成功提交预算释放后，C4 的另一个 lease 才可能获准。C5 从自己的容器再次申请，仍受 FIFO 队列、实时容量及前次归还屏障约束。最终 C6 汇合错误并清理。若使用 start_gpu_job，调用方更早拿到 job_id，但 GpuJobManager._run 的执行锁仍存在。N1 的新协议要替换这个限制并独立回收每个作业，不能只提前返回 job_id。

**可运行实验与 Results。** 本节复用已有两个标准库脚本，无需再写一套相似模拟。在仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/async_gpu_planning_demo.py
python docs/env-worker-deep-dive/env-service-verification/scripts/model_concurrent_gpu_jobs.py
```

第一个脚本的 short_queue_sync / short_queue_async 对应数字例子一，教学逻辑时间分别为 15 和 12 秒；两组 B 都从 t=3 开始。脚本另有 remote 分支，是未实现跨节点方案的教学假设，不属于本次 command 已有能力。第二个脚本对应同容器新协议的预算与回收契约，低压力四授权、高压力一授权，取消 B 后 A/C/D 的模拟授权保留。

本次实际在 CPU 中调用上述模拟及契约检查，结果保存于 [models.json](env-service-verification/results/command-vs-concurrency-20261026/models.json)。它们没有调用生产 Router、Docker、vLLM 或 CUDA。生产同容器并发仍处于设计状态。本次修改在主开发目录完成，只更新说明和验证记录，没有新增功能分支、修改生产调度代码、提交、推送或合并。

本页及总索引的 94 个本地链接路径、源码行号范围与章节锚点检查通过，相关源码位置已核对并修正。全仓 pre-commit 在隔离副本运行，仍因仓库既有 Ruff 与注释风格违规失败，未将自动修改回写工作区；检查记录与日志保存在上述结果目录。

## 8. 容器与作业的申请关系

**同容器多作业不是必然条件，也不是 PSRL 默认把独立任务合并。** 容器是一次任务执行的环境，GPU 作业是这个环境里一次受管理的程序运行。一个环境可以运行零条、一条或多条 GPU 命令；多条通常可以顺序执行，是否同时提交取决于代理或实验脚本。上一轮同容器并发方案解决的是明确选择这种使用方式后的能力缺口，不是所有 science 训练都必须完成的升级。

先看当前 Science／Harbor 路径的真实层次。此图针对启用 Env Service 的 command 模式，不是所有 PSRL 后端的通用一对一关系。

```text
[R1] 一次 Science episode：代理尝试完成一道题的一次执行
        ↓ runner 先调用 acquire(session_id, EnvDemand)
[R2] Router 授予环境租约 L
     记录节点、设备、CPU、主机内存和一个会话名额
        ↓ Harbor 按租约生成的配置创建执行环境
[R3] 主容器 C：代码、数据、文件和终端在本次 episode 内持续保留
     另可能有归属同一任务的 sidecar / verifier 容器
        ↓ 代理选择在何时、运行什么程序
[R4] 普通 CPU 命令                    GPU 作业提交 J1、J2、J3
     直接在环境中执行                 均绑定已有 L 与主容器 C
                                        ↓ 不是再次创建 C
[R5] Router 逐次处理 GPU 命令申请：等待 → 授权 → 执行 → 确认归还
     当前同容器只允许一条活动 GPU 命令，其他已提交作业等待
        ↓ 可以继续模型交互，再提交下一条命令
[R6] episode 结束：汇合命令及回收结果 → 清理所属容器 → 释放 L
```

图例：图里每个部分对应的源码位置。

| 编号 | 实现、输入和状态 |
|---|---|
| R1、R2 | [run_harbor_episode 申请租约](../../examples/sciaccel_rl/runner.py#L346) → [EnvServiceClient.acquire](../../psrl/workers/env_service/client.py#L159) → [Ledger.try_admit](../../psrl/workers/env_service/ledger.py#L105)，输入 session_id 和环境需求，输出 EnvLeaseHandle |
| R3 | [_apply_env_lease](../../examples/sciaccel_rl/runner.py#L99) 配置设备与归属；[JobConfig](../../examples/sciaccel_rl/runner.py#L440) 绑定一个任务和代理；[Job.create / run](../../examples/sciaccel_rl/runner.py#L502) 进入 Harbor 执行 |
| R4 | [dispatch_gpu_tool](../../examples/sciaccel_rl/gpu_tools.py#L28) 调用已有环境的 [start_gpu_job](../../psrl/workers/env_service/docker.py#L156)，[GpuJobManager.start](../../psrl/workers/env_service/gpu_jobs.py#L60) 创建 job_id 并启动后台生命周期，没有申请新容器 |
| R5 | [exec_gpu](../../psrl/workers/env_service/docker.py#L92) 读取已有 lease_id 和 main container ID，将它们与 request_id 送入 [Scheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L49)；[物理归还](../../psrl/workers/env_service/command_gpu_node.py#L153) 成功后释放 GPU 预算 |
| R6 | [runner 的 finally](../../examples/sciaccel_rl/runner.py#L555) 先清理本次 Compose 容器，再调用 [lease_handle.release](../../examples/sciaccel_rl/runner.py#L577)，停止心跳并释放环境资源 |

这些代码如果有理解上的问题，可以随时追问。图中的 sidecar 是网络等辅助容器，verifier 是评分器。它们说明“一次 episode 的容器集合”可能不止一只 Docker 容器，不能把 lease 和物理容器总数机械视为一对一。当前 GPU 命令接口要求绑定一个运行中的 main 主容器。

**先消除 job 这个词的重名。** Science task 是题目定义，包含初始文件、说明和评分规则；episode 是代理对这道题的一次实际尝试。Harbor 的 Job 是外层运行组织对象，本 runner 配置一个 task、一个 agent、一次 attempt 和一个 concurrent trial。本文的 GpuJob 是容器内部一次 GPU 程序执行，两种 Job 不是同一个对象。Python 的 asyncio.Task 又只是调度协程的对象，不会因为创建了它就自动创建容器、申请 GPU 或生成新的 Science episode。

一个 GPU 作业也不必恰好只有一个进程：一条程序可以启动多个子进程或多个计算进程，它们仍属于这条作业。反过来，模型请求、文件编辑和编译也不一定产生 GPU 作业。不能用“有四个协程”直接推断“四个容器”或“四份 GPU 预约”。

**两层申请分别解决什么。** 第一层申请让这次 episode 有地方运行，保留 CPU、主机内存、环境名额，并确定 GPU 所在 unit。unit 是预先配置的节点及资源集合。command 模式在这一层不持有整段会话的活动 GPU 显存预算，[Ledger 创建 lease 时 gpu_active 为 false](../../psrl/workers/env_service/ledger.py#L182)，[环境准入](../../psrl/workers/env_service/ledger.py#L376) 主要检查会话资源与需求是否可能装入。

第二层申请是环境里某条命令真正准备使用 GPU 时的准入。它复用已有租约与主容器，不再扣一份完整 CPU／主机内存会话资源。当前每次 GPU 需求仍取自 lease.demand，确认归还后释放活动 GPU 预算，但租约和容器继续存在。拟议的每 job 独立显存需求是在第二层细化，不能写成当前已实现。

例如，**教学假设**一次 episode 申请 4 CPU、8 GiB 主机内存，里面先后执行三条 GPU 命令：环境账本在会话期间只保留这份 4 CPU、8 GiB，GPU 命令申请和归还发生三次。如果改成三个独立 episode 同时执行，每个同样申报 4 CPU、8 GiB，就需要三份环境资源，共 12 CPU、24 GiB。这里是账本数字，不是三个容器必然实际占用 24 GiB，也不保证 sidecar 和 verifier 的物理限制完全等同于主容器。

**同容器为什么可能有多条作业，但不一定需要并行。** 假设一道科学任务要求修改求解器并验证结果：

1. 代理先修改代码，运行 GPU 正确性检查 J1。
2. J1 的结果返回，代理请求模型，决定下一步修改。
3. 再运行 GPU 性能检查 J2，之后运行另一个参数组 J3。

这已经是“一个容器生命周期中有多条 GPU 作业”，但 J1、J2、J3 可以完全顺序执行，当前 command 协议即可处理。容器保留修改后的代码、输入和结果，不必每次重新搭建任务环境。

只有任务本身允许独立计算，且调用方选择提前提交时，才有同容器并发需求。例如代码版本固定后，分别对三份独立输入求解，结果写入三个不同目录，再汇总评分。此时 J1 尚未结束就提交 J2/J3，才需要讨论是否能同时运行。若三条命令都修改同一源码或覆盖同一输出目录，即使显存充足也不能据此判定可安全并行，必须先解决数据依赖。

共享容器主要共享环境和文件，不等于自动共享每个 Python 进程的内存、已导入库或 CUDA 上下文。因此“少建几个容器”也不意味着所有启动开销都会消失。

| 使用方式 | 环境关系 | 当前是否需要同容器 GPU 并发改造 |
|---|---|---|
| 一个 episode，只执行一次 GPU 检查 | 一份租约，一个主环境，一条 GPU 命令 | 不需要 |
| 一个 episode，多轮修改，每轮执行一条 GPU 检查 | 一份租约，一个主环境，多条命令顺序执行 | 不需要 |
| 多个独立 episode，各自执行一条 GPU 检查 | 多份租约、多个独立主环境，Router 跨租约调度 | 不以同容器改造为前提 |
| 一个 episode，同一代码版本上同时运行多组独立 GPU 检查 | 一份租约，一个主环境，多条同时提交的命令 | 若要求 GPU 执行重叠，需要新能力 |

**此前四命令实验为什么用了这种结构。** 当时我把用户的“一个真实 science 任务多开几个协程”理解成一次 episode 内部提交多条命令，因此实验脚本只调用一次 [run_harbor_episode](env-service-verification/scripts/science_natural_gpu_ab.py#L180)，固定代理再用 [asyncio.gather](env-service-verification/scripts/science_natural_gpu_agent.py#L47) 提交四条训练命令。用户已于 2026-10-26 澄清，实际指同一道题多次独立 rollout 尝试。前述实验结构没有准确覆盖该意图，不能再称为按澄清后的需求完成了验证。它也不是 PSRL 自动合并独立任务，纠正后的关系见第 9 节。

严格说，该实验直接调用 exec_gpu_prepared／exec_gpu，四条是四个 GPU 命令申请，没有通过 start_gpu_job 创建四个 GpuJob 管理对象。讨论调度时把它们统称为四个作业，是为了描述四份独立程序工作，不能因此混淆实际入口。单容器四命令是实验选择，足以暴露这个场景的限制，不能据此认定所有训练都受该限制影响。

配置也应按层读：实验中的 max_concurrent_episodes=1 限制 episode 并发，unit.max_containers=1 在 [账本中按活动 lease 数计名额](../../psrl/workers/env_service/ledger.py#L380)，都不是“容器内最多只能提交一个命令”。把 max_containers 改成 4 不会自动生成四个环境，也不会把四个命令拆成四份租约。必须真正启动四次独立 episode，才会沿 R1/R2 产生相应环境申请。

**跟着图读源码。** R1 调用 runner 后先进入 R2，成功返回 lease 后才进入 R3 的 Harbor 环境执行。R4 可以发生零次或多次，每次使用同一个 environment；GPU 分支进入 R5，CPU 分支不申请 GPU。多条 GPU 调用可以同时登记，但当前在同容器锁处等待，不能把提交当成执行。每条 R5 结束仅归还 GPU，后续仍可回到 R4；整个 episode 结束才进入 R6，关闭所属容器并释放会话资源。独立 episode 则各自运行这条链，可以在 Router 的容量约束下并发。

**可运行教学示例。** 复用 [explain_same_container_lock.py](env-service-verification/scripts/explain_same_container_lock.py)，在仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/explain_same_container_lock.py
```

脚本对应 R4/R5：四条协程都提交并完成准备，共享锁时执行区间峰值为 1，无锁模型为 4。它不创建容器或租约，不执行 GPU，不能作为新协议实现证据；作用只是确认“四条已提交工作”和“四条同时执行工作”是不同数量。环境申请与命令申请的对应关系由上面的真实源码确定，不由教学脚本假造。

本次实际 CPU 运行得到上述结果，输出保存在 env-service-verification/results/container-job-relationship-20261026/cpu-demo.jsonl。本文及总索引的 115 个本地链接、行号范围和章节锚点检查通过，新增源码位置逐项核对。全仓 pre-commit 在隔离副本仍因既有风格违规失败，日志保存在同结果目录，自动改动未回写。此次没有修改生产代码或启动 GPU 实验，文档在主目录未提交，未推送或合并。

## 9. 多 rollout 意图纠正与 OFF 的同容器能力

**用户的目标是同一道题多次独立 rollout，不是同一个容器的多个 GPU 命令。** rollout 在这里指一次从题目开始、包含模型交互和环境执行的完整尝试；它不是一次尝试内部的一轮模型问答。该澄清修正实验拓扑，不改变已保存的原始日志。

```text
用户实际目标：同一道题
    ├─ rollout A → session A → 环境 A 的主容器
    ├─ rollout B → session B → 环境 B 的主容器
    ├─ rollout C → session C → 环境 C 的主容器
    └─ rollout D → session D → 环境 D 的主容器
      并发受配置限制；ON 为各环境建立独立租约，跨租约调度 GPU

此前误选的组件实验：一次固定脚本 episode
    └─ 一个主容器 → 并发提交四条 CIFAR GPU 命令
```

图例：图里每个部分对应的源码位置。

| 路径 | 实现与源码 |
|---|---|
| 每次 rollout 的独立会话 | [SciAccelAgentLoop 创建 session 并调用环境](../../examples/sciaccel_rl/agent_loop.py#L171)，每次尝试产生自己的会话身份 |
| 独立环境及 ON 租约 | [_run_harbor_in_thread](../../examples/sciaccel_rl/agent_loop.py#L358) → [runner 申请租约](../../examples/sciaccel_rl/runner.py#L346) → [Harbor Job](../../examples/sciaccel_rl/runner.py#L440)；不能拿当前关闭服务分支代替固定 OFF |
| 单容器实验的命令提交 | [NaturalGpuAgent 的普通 exec](env-service-verification/scripts/science_natural_gpu_agent.py#L37) 与 [gather](env-service-verification/scripts/science_natural_gpu_agent.py#L47)，不是四次真实模型 rollout |
| 固定 OFF 的设备映射 | [冻结 OFF runner](env-service-verification/results/science-natural-gpu-20261022/source/off-runner.py#L192) 使用静态 GPU compose override，基线为 0eeead861ffafed6bf31c962d259fd847775dcf9，运行修复另行披露 |

这些代码如果有理解上的问题，可以随时追问。独立尝试通常需要独立可写环境，避免不同轨迹互相修改代码、影响奖励。容器集合还可能包含评分器与网络辅助服务，因此上图只列主容器。

**OFF 可以让同一容器的多个 GPU 命令重叠执行。** 在固定基线及已测实验条件下，静态设备映射给进程提供 GPU 访问，实验代理对同一个 environment 并发调用普通 exec，没有当前 ON 的 GPU 命令串行协议。命令是调用方并发提交的，不是 OFF 自动把顺序代码改成并发。若调用方逐条 await，仍然顺序执行。

**Results：已有实机证据。** [四命令低压力实验](science-natural-gpu-concurrency.md#11-results高低显存压力实测) 中，OFF 在一个容器内完成四条训练命令，GPU 训练进程峰值为 4，命令阶段 55.00 秒；ON 峰值为 1，耗时 156.93 秒。它证明该实验中 OFF 的同容器 CUDA 工作可以重叠，不保证任意负载都放得下，也不表示 GPU 每个计算内核始终同时执行。高压力下 OFF 四条失败已保留，不能称其具有容量调度保护。

固定 OFF 仍有进程内 episode 信号量，限制能同时运行多少次尝试；它不是同容器 GPU 命令锁，也不是按显存决定是否放行的跨任务调度器。当前工作区关闭 Env Service 的路径还包含其他 GPU gate，回答固定 OFF 行为时不得将两者混用。

**跟着图读源码。** 多 rollout 路径为每次 run 创建 session，经过 episode 准入后调用 runner 并创建自己的任务环境；ON 在这一层分配独立 lease，之后各环境的 GPU 命令进入共享 Router。相比之下，固定代理路径只有一次 runner 调用，在其返回前 gather 四条 exec。四个 GPU 进程不代表四条 rollout 轨迹，也不能产生四份真实语言模型探索数据。

后续若执行原始验证，应保持同一道题、相同 rollout 数与并发上限、各自独立环境、相同命令资源需求及固定 OFF 提交，比较跨容器的完成率、排队、显存不足与总耗时。已有 command GPU 就覆盖这个调度层次，不需要先实现同容器并发。此前组件数据不能作该场景的性能结论；本次只核对源码和历史证据，没有启动新 rollout 或 GPU 实验。

教学复现仍可用第 8 节的 explain_same_container_lock.py 观察并发提交与串行执行的差别；它不运行 rollout、容器或 GPU，不能替代上述实机证据或修正实验拓扑。

## 10. episode 与 command 的区别及通用性边界

本节核对的是主目录当前实现。episode 与 command 都是启用 EnvRouter 后的策略，均不是固定 OFF 的别名。episode 表示一次完整尝试，command 表示其中一次显式 GPU 执行。核心区别是**显存额度从什么时候承诺，到什么时候允许转给别人**。

```text
[G1] 同一道题的 rollout A、B，各自创建会话、租约和环境
         ↓
[G2] 资源准入：检查 CPU、主存、环境名额与 GPU 需求
         ├─ episode：此时承诺 GPU，整段会话持有
         └─ command：先建环境，需要执行 GPU 程序时才申请
                          ↓
[G3] 执行入口 → 排队 → 额度足够 → 授权设备 → 运行 GPU 程序
                          ↓
[G4] 程序结束 → 结果可供模型使用
         ├─ episode：即使暂时没有 GPU 程序，额度仍保留
         └─ command：节点撤权、核实句柄及 GPU 进程归还
                       ├─ 核实期间仍占预算
                       └─ 确认后释放预算，可服务另一个 rollout
         模型规划可以与归还检查重叠，不等于可以提前释放预算
                          ↓
         再次需要 GPU → 回到 G3
                          ↓
[G5] 整条 rollout 完成／取消 → 回收所属环境 → 释放会话资源
```

图例：图里每个部分对应的源码位置。

| 图中节点／箭头 | 当前函数及源码 | 输入、输出或状态变化 |
|---|---|---|
| G1 独立尝试 | [agent_loop.run](../../examples/sciaccel_rl/agent_loop.py#L171)、[runner](../../examples/sciaccel_rl/runner.py#L346) | 每次尝试自己的 session，申请自己的 lease |
| G2 两种持有策略 | [Ledger.try_admit](../../psrl/workers/env_service/ledger.py#L105)、[gpu_active 初始化](../../psrl/workers/env_service/ledger.py#L182)、[_fits_now](../../psrl/workers/env_service/ledger.py#L376) | episode 立即计 GPU 预算，command 暂不计活动 GPU 预算 |
| G3 工具入口 | [agent._execute_commands](../../examples/sciaccel_rl/agent.py#L104) → [dispatch_gpu_tool](../../examples/sciaccel_rl/gpu_tools.py#L28) → [start_gpu_job](../../psrl/workers/env_service/docker.py#L156) | 显式 psrl-gpu 请求转成受管理的作业 |
| G3 等待与执行 | [exec_gpu](../../psrl/workers/env_service/docker.py#L92) → [GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L49) → [grant_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L139) | 额度与物理占用检查通过后授权，当前同租约仍只允许一条待执行／活动命令 |
| G4 输出与异步归还 | [exec_gpu 的 overlap_return](../../psrl/workers/env_service/gpu_execution.py#L55)、[revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153) | 输出就绪与物理归还分开，后台确认完成前继续计费 |
| G5 最终清理 | [ManagedDockerEnvironment.stop](../../psrl/workers/env_service/docker.py#L238)、[runner finally](../../examples/sciaccel_rl/runner.py#L555) | 处理在途命令、回收所属容器、释放租约 |

这些代码如果有理解上的问题，可以随时追问。跟着图读源码时，先读 G1 的申请与 G2 的初始状态，再沿 G3 看工具到授权，接着读 G4 的双分支，最后读 G5。G4 的模型请求和归还检查是并发关系，不能读成先释放预算再让模型思考。异常也会直接进入 G5，不保证每条命令都正常返回。

| 问题 | episode 现状 | command 现状 |
|---|---|---|
| GPU 使用权持有多久 | 整次会话 | 每次受管理 GPU 执行及其归还确认期间 |
| 等模型时一定交还显存吗 | 不交还额度 | 仅在上一 GPU 阶段真实结束且归还确认后交还 |
| 高显存压力如何处理 | 会话准入时排队 | GPU 执行准入时排队，已建环境可继续非 GPU 工作 |
| 低压力可以多任务同时运行吗 | 可以，需求与宿主容量足够即可 | 不同容器／租约可以，同容器目前串行 |
| 适合哪种 GPU 程序 | 常驻服务、跨工具调用保留 GPU 状态的程序 | 能独立启动、结束并释放资源的前台程序 |
| 显存是硬隔离吗 | 否，声明是预算 | 否，设备权限控制也不是显存配额硬限制 |

例如，预算 80 GiB，两次尝试各申报 20 GiB，CPU 和容器槽位够时 episode 也能同时运行。把 episode 理解成“一张卡只跑一个任务”会误判其代价。反过来，command 也不会把一个总需 90 GiB 的程序变得能装入 80 GiB。

**之前说 command 不通用，需要准确限定为“不能对任意现有 GPU 任务透明开启”。** 调度队列本身没有限定某一道 science 题，已有显式 GPU 工具也可复用，限制在以下执行契约。

1. **必须知道真正的 GPU 生命周期。** 某个 Python 服务加载 6 GiB 模型后一直不退出，收到下一条请求才计算。它等待 10 秒时仍占显存，GPU 利用率为零只表示没执行计算，不表示内存空闲。CUDA 上下文是进程使用 GPU 的运行状态，也可能继续持有设备句柄。当前归还协议检查句柄及 GPU 进程，不能凭 shell 返回或模型请求发出就确认释放。
2. **GPU 工作必须经过受管理入口。** 普通终端可能运行后台程序、tmux 会话或长期 Python 解释器。任务在初始化或评分器中也可能调用 GPU。只包装代理的一条工具调用，无法覆盖这些位置。当前默认 Science 代理已接入显式 psrl-gpu 工具，但普通 terminal、任意自定义代理和所有 verifier 尚未统一自动接入，不能说成默认代理完全没有支持。
3. **必须能确定和回收归属。** 超时的是观察请求，可能不是底层程序。取消一条作业，需要终止其所属进程并确认资源释放。当前依赖 Docker 主容器、节点进程视图和设备访问控制，要求预检能力通过。[DeviceCgroupFence](../../psrl/workers/env_service/command_gpu_node.py#L17) 当前面向 cgroup v1 设备控制，不是任意容器平台都已有同等实现。cgroup 是 Linux 将进程分组并施加资源／访问限制的机制，这里控制设备访问，没有自动搬走已占用的显存。
4. **资源描述和入口还未完全统一。** [demand_from_task](../../examples/sciaccel_rl/env_demand.py#L23) 从 task.toml 取 CPU、主存和 GPU 需求，显存来自覆盖值或 Router 默认值。当前命令仍复用 lease 的显存需求，不能逐命令分别申报。模式在 [EnvUnit](../../psrl/workers/env_service/types.py#L49) 上配置，不能直接在同一 unit 中任意混用不同持有策略。
5. **存在独立的集成边界。** Science 的 [runner](../../examples/sciaccel_rl/runner.py#L402) 及 [gpu_grading](../../examples/sciaccel_rl/gpu_grading.py#L12) 仍有 LAPS CUDA 的归属检查和奖励接入分支。它们不是通用调度算法的必要条件。旧 [EnvWorker._allocate_gpu_indices](../../psrl/workers/env_worker/worker.py#L195) 仍维护另一条整卡分配路径，尚未因为 Science 接入就自动统一。vLLM 推理副本的同卡借还还有另一套协议，[当前类型检查](../../psrl/workers/env_service/types.py#L54) 禁止与 command 任意组合。

因此不能通过识别命令字符串里有没有 python／cuda 来解决通用性，也不能把所有任务都改成“等待模型即归还”。正确的边界是：谁持有 GPU 状态，谁必须提供可验证的结束或释放边界。没有这样的边界，完整会话持有仍是合法且必要的策略。

## 11. 通用 GPU 执行服务的具体设计

**本节是完整设计提案。执行后端及标准 Science 作业入口已完成[第一阶段实现](unified-gpu-execution.md)，本节的任务级配置、混合范围账本及动态状态策略尚未完成。** 目标是新任务声明资源与生命周期，再接标准执行适配器，即可复用调度与控制优化。首期保持每个环境一次最多执行一条 GPU 命令，各独立 rollout 可以共享 GPU 容量。同容器并发见 [ENV-GPU-SAME-CONTAINER TODO](gpu-jobs-and-task-telemetry.md#7-尚未完成的目标)，不作为这项设计的前置条件。

```text
新 GPU 任务：资源声明 + 生命周期声明 + 启动／工具／评分入口
                         ↓
[G1/G3 扩展] 统一任务适配器，覆盖整份环境的 GPU 执行入口
                         ↓
[G2 扩展] 统一 GPU 使用权服务，共用一份容量账本与等待队列
       ├─ session 范围：完整会话持有，对应现在 episode
       └─ invocation 范围：一次可独立回收的执行，对应现在 command
                         ↓
[G3/G4 扩展] 执行后端：授权 → 启动 → 观察／取消 → 确认物理归还
                         ↓
[G5 扩展] 会话结束：收拢未完成执行、清理环境、释放主机资源
```

图例：这是目标协作图，尚不存在名为“统一任务适配器”的完整实现。G1 至 G5 对应上一节的已核对源码，改动落点如下，不能将这些落点当成新功能已经具备。

| 模块 | 建议改动与现有落点 | 新任务得到什么 |
|---|---|---|
| 任务声明 | 扩展 [EnvDemand／EnvUnit](../../psrl/workers/env_service/types.py#L49) 与 [demand_from_task](../../examples/sciaccel_rl/env_demand.py#L23) | 声明显存、GPU 数量、持有范围及必要能力，无需按题目名分支 |
| 统一执行适配器 | 从 [gpu_tools](../../examples/sciaccel_rl/gpu_tools.py#L28)、[runner](../../examples/sciaccel_rl/runner.py#L387) 提取通用接入层 | 标准代理和任务初始化／评分器都能提交 GPU 阶段，任务不用重写调度器 |
| 统一使用权账本 | 重构 [Ledger](../../psrl/workers/env_service/ledger.py#L325) 与 [CommandScheduler](../../psrl/workers/env_service/command_gpu.py#L49) | 长期和短期使用权都参加相同容量准入，避免两套账各自认为有空间 |
| 执行与回收后端 | 抽象 [ManagedDockerEnvironment](../../psrl/workers/env_service/docker.py#L92) 和 [node 授权／撤权](../../psrl/workers/env_service/command_gpu_node.py#L139) | 复用执行身份、取消、物理归还和异常隔离，后续其他后端实现同一契约 |
| 任务语义验证 | 从 [gpu_grading](../../examples/sciaccel_rl/gpu_grading.py#L30) 提取可配置验证插件 | 正确性和奖励留在任务侧，资源安全与容量调度不依赖固定题目名称 |

**统一的是服务，保留两种持有范围。** session 表示完整会话，适合常驻状态或不能切分的程序。invocation 表示一次独立执行，适合前台程序结束后只留文件的任务。它们应是请求所有者的属性，物理 unit 负责描述卡、容量及后端能力。当前模式属于 unit，混合两种范围必须改类型、账本和准入状态机，不是简单加一个 task.toml 字段就完成。

建议的最小任务描述如下，**这是接口草案，不是当前可直接使用的配置**：

```yaml
gpu_execution:
  scope: invocation
  memory_mb: 6144
  device_count: 1
  process_lifetime: foreground
  queue_timeout_s: 300
  execution_timeout_s: 120
```

这里 memory_mb 是包含运行时开销和峰值余量的承诺需求，device_count 是执行需要几张卡。foreground 表示程序和它所属的子进程不会在受管理执行结束后继续偷偷使用 GPU。队列超时与执行超时分开，避免把排队等待误报成程序算得太慢。入口由标准适配器注册，cwd、环境变量、输入输出位置需显式传递，不依赖另一个终端的临时变量。实际实现仍需决定配置字段所属位置与兼容迁移方式。

对新接入的一个求解器任务，流程应当是：声明需求 → 采用标准代理／执行适配器 → 通过统一 submit 入口运行自己的程序 → wait 取结果 → 任务自己的评分逻辑判正确性。显存够时其他 rollout 的 GPU 阶段可同时获准，不够时排队。任务作者只提供程序、资源描述和评分，不必另写显存轮询、设备授权、排队或清理逻辑。若初始化或独立 verifier 也用 GPU，必须分别注册入口及所属环境，不能以主容器已受管为由漏掉它们。

对于加载模型后常驻的服务，新任务选择 session 后，同样获得容量准入、队列、身份管理和结束清理，但不会因模型交互而腾出它仍在使用的显存。若想切得更细，程序需要提供保存状态、退出 GPU 进程、随后恢复的协作协议，并验证恢复成本。普通“暂停计算”不等于释放内存，也不能仅将空闲利用率当作归还证据。无法满足 invocation 能力的任务应在预检时明确报不支持，由配置选择 session，不能静默降级后仍宣称享有相同收益。

**哪些加速设计可以推广，哪些有条件。**

| 能力 | 通用服务如何复用 | 必要条件或边界 |
|---|---|---|
| 容量足够并行，不足排队 | 长短两种使用权共用账本与队列 | 需求覆盖真实峰值，多 GPU 必须逐卡判定，不能只比较总显存 |
| 利用命令之间的空档 | invocation 归还后让其他 rollout 执行 | GPU 状态确实释放，且确有等待者 |
| CPU 准备与 GPU 等待重叠 | 标准 prepare 阶段读数据、编译等 | 任务明确保证准备阶段不初始化 GPU，不能任意提前运行未知 Python 导入 |
| 结果返回与归还确认重叠 | 区分 output_ready 与 released 两个状态 | 结果已稳定，后续 CPU／模型工作不依赖仍在运行的 GPU 程序 |
| 预热与控制路径优化 | 复用节点服务、Docker 查询连接和 NVML 连接 | 初始化只做一次并按节点失效重建，硬件状态仍需新鲜查询 |
| 自由切换任意执行后端 | 统一授权、执行、取消、归还接口 | 每个后端仍须实现真实的权限和归属检查，接口抽象本身不创造这些能力 |

NVML 是读取 NVIDIA GPU 状态的管理接口。复用连接能省重复初始化，但不能缓存旧的空闲显存并一直据此放行。异步也不是提前把预算清零，而是让独立 CPU 工作不必等确认过程结束。

**推广前要修正一处容量语义。** 当前 [_effective_gpu_mb](../../psrl/workers/env_service/ledger.py#L357) 取声明占用与实测占用的较大值。它能覆盖部分情况，但不是“外部占用 + 尚未兑现承诺”的完整合成。教学反例：卡容量 16 GiB，外部已占 8 GiB，A 已获准 6 GiB 但还没实际分配，B 又申请 6 GiB。当前这一步得到 max(6, 8) + 6 = 14，可能放行，但 A、B 后续都达到承诺需求时总量是 20 GiB。这个反例是公式分析，不是本次实机 OOM 结果。

通用方案应按每张卡计算：当前总实测占用 U，加上每个已批准所有者尚未实际分配的承诺 max(bᵢ − mᵢ, 0)，再加新请求和安全余量，不超过可用容量。其中 bᵢ 是承诺，mᵢ 是已可靠归属到该所有者的实测显存。上述反例应得到 8 + (6 − 0) + 6 = 20，从而等待。这个公式依赖一致、新鲜且不重复计数的归属数据，不能拿不同时间或不能归属的读数机械相加。无法可靠归属时采用保守上界或停止准入，显式报告原因，不能把缺失值补成零。它仍不是硬隔离，任务超出声明或外部进程突增显存仍可能造成内存不足。

**状态与异常也必须统一。** 使用权至少有 waiting（等待）、reserved（承诺已记账、尚未运行）、running（运行）、returning（确认归还）和 released（可再分配）状态。授权前先原子地记入承诺，授权失败走所属资源回收。returning 不提前扣减预算，客户端观察超时不直接认定程序结束，取消由服务端继续负责，回收无法证实则隔离该资源。队列初期可保留先进先出，后续若允许小请求越过大请求，必须同时解决大请求长期等待，不能把这一优化混称当前已有能力。

**实施顺序与最小验收。**

1. 先确定资源与生命周期契约，把请求身份、逐执行预算、长短持有范围加入共同账本，修复外部占用合成。用 CPU 模型／针对性单元验证共享预算、原子承诺、取消及异常归还，不要求同容器并发。
2. 再统一 Science 的启动、显式 GPU 工具和 GPU verifier 入口，移走按题目名称判断的资源策略。选第二种非 LAPS GPU 任务，通过声明和标准适配器接入，验证无须修改调度器或题目白名单。旧 EnvWorker 后端另行实现同一执行契约，不继续复制一套预算逻辑。
3. 在具备设备权限与进程归属能力的节点，做有界实机验证：同题多个独立 rollout、每个独立环境，覆盖显存充足并行、容量不足排队、结果与归还重叠、取消及最终物理释放。针对常驻 GPU 程序验证 session 可用、错误 invocation 契约被拒绝。
4. 性能比较分别报告固定 OFF、ON episode、ON command，使用相同工作量和计时边界，拆开准备、排队、GPU 执行、模型等待、归还与清理。只要能回答目标就停止，不以跑更大规模或必须得到漂亮加速比为验收条件。

高压力下少一些显存不足失败和重试，可能提高总完成速度。细粒度归还还可能利用空档缩短总时间。但仅有“高压力”并不足以保证 command 更快：若所有程序始终占着 GPU、没有其他等待任务，或者每段极短而授权开销很高，收益可能很小，甚至变慢。vLLM 同卡借还、自动迁移、抢占、同容器并发及未知峰值预测不属于上述首期方案已经解决的能力。

## 12. Results：持有范围的 CPU 教学验证

本例只模拟两次独立 rollout 的资源时间线，不调用模型、不创建容器、不运行 CUDA，也不模拟固定 OFF 的实际 OOM。它用于解释机制及检查边界，不构成生产实现或实际加速证据。

**教学假设**：A、B 是同一道题的两个独立环境，宿主 CPU、主存及环境槽位允许同时存在。每个执行阶段需要 6 GiB，先执行 GPU 5 秒，再等待模型／做 CPU 工作 10 秒，最后再执行 GPU 5 秒。模型使用其他资源，第一阶段结束可完整释放环境 GPU，授权与回收开销暂设为零，GPU 阶段并行也暂不考虑计算竞争。

| GPU 容量 | 完整会话持有 | 按执行阶段归还 | 原因 |
|---|---:|---:|---|
| 8 GiB，只能容纳一个 6 GiB 阶段 | 40 秒 | 25 秒 | 后者允许 B 使用 A 的等待空档 |
| 16 GiB，能容纳两个 6 GiB 阶段 | 20 秒 | 20 秒 | 两者本来都能并行，没有额外的等待可消除 |
| 8 GiB，但程序跨阶段常驻 GPU | 40 秒 | 拒绝该契约 | 不能假装等待期间资源已释放 |

8 GiB 的时间线可手算：episode 下 A 在 0 至 20 秒持续持有 6 GiB，B 在 20 秒后才进入，40 秒结束。command 下 A 在 0 至 5 秒用卡，B 在 5 至 10 秒用卡，A 在 15 至 20 秒再用卡，B 在 20 至 25 秒再用卡。10 至 15 秒两者都等模型，卡即使空闲也没有可执行工作。40 到 25 秒是本例的 37.5% 总耗时下降，不是硬件实测，也不是对所有任务的预测。

运行 [model_gpu_hold_scope.py](env-service-verification/scripts/model_gpu_hold_scope.py)：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/model_gpu_hold_scope.py
```

脚本使用整数逻辑时间，循环上限 101 次，没有真实 sleep。G2/G3 对应准入与执行，G4/G5 对应阶段结束、确认归还及会话结束。每个 rollout 自有状态，断言总承诺不超过容量，最终全部释放。它省略真实设备权限、过程取消、遥测波动和控制开销，真实系统必须由第 11 节的验收补足。

本次 CPU 运行实际得到表中 40、25、20、20、40 秒的逻辑结果，并拒绝常驻 GPU 程序使用 invocation 的契约。原始输出保存在 env-service-verification/results/gpu-policy-generalization-20261026/cpu-demo.jsonl。本次仅记录 TODO、核对源码、形成设计与运行 CPU 模拟，没有修改生产调度代码或启动 GPU／真实训练。通用任务接入与性能收益仍待按第 11 节落地和实机验证。

本次编辑文档及总索引共 394 个本地链接、源码行号范围和章节锚点检查通过，新增源码引用已逐项核对。教学脚本 Ruff 检查及格式检查通过。全仓 pre-commit 在隔离副本运行，仍因其他文件的既有 Ruff／格式及 prose-style 问题失败，日志保存在上述结果目录，自动修订没有回写主目录。此次改动保持在 psrl 主目录未提交状态，未推送或合并。

## 13. 常驻 GPU 状态与释放恢复成本

**需要保留执行状态，不等于物理上必须整个 episode 都占着 GPU。** 上文“常驻任务使用 session”是针对当前缺少安全卸载协议的执行方式，不是说这类任务永远不能让出显存。未来可以在一次会话中多次保存、释放、重新准入和恢复，但那需要可恢复状态的契约及成本判断。

先区分两个参与者：环境里的 GPU 程序执行任务，vLLM 负责生成代理下一条指令。二者可能使用不同 GPU，也可能轮流使用同一张 GPU。以下图中 H2/H4 的通用状态保存与恢复是提案，当前 command 并不替任意程序实现它。

```text
不同 GPU，或两者能够同时驻留：
[H1] 环境产出稳定观察 → 发起 vLLM 请求 ───────→ 新指令就绪
             └→ [H2] 保存状态、释放并确认
                           ↓
                   [H3] 其他 rollout 使用环境 GPU
                           ↓
                   [H4] 重新准入、恢复状态 → 执行新指令
   H2 可与生成重叠。H4 只有状态不依赖新指令且容量可用时才可提前。

同一 GPU，且环境与 vLLM 放不下：
环境保存／释放 → 确认 → vLLM 恢复 → 生成
  → vLLM 释放 → 确认 → 环境恢复 → 执行新指令
   这些互斥阶段不能全部重叠。
```

图例：图里各部分的实现与未实现边界。

| 节点／箭头 | 源码位置 | 边界 |
|---|---|---|
| H1 模型请求边界 | [TruncatingTerminus2._query_llm](../../examples/sciaccel_rl/agent.py#L125) | 配置 lending 时先切 model 阶段，再请求模型，finally 切回 env |
| H2 command 当前归还 | [exec_gpu](../../psrl/workers/env_service/docker.py#L92)、[revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L153) | 已支持结束与确认，不实现任意活进程的状态卸载 |
| H3 排队及再次申请 | [GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L49) | 当前按容量及先来先服务准入，没有预测模型完成时间的恢复预约 |
| H4 通用保存／恢复 | 尚无通用实现，拟接入第 11 节执行后端 | CPU 教学模型只计算时间，不证明可恢复任意 CUDA 上下文 |
| 同卡交接 | [GpuLendingBinding._transition](../../psrl/workers/env_service/lending.py#L51)、[freeze_gpu_lease](../../psrl/workers/env_service/lending_node.py#L79) | 先确认环境 GPU 已归还，再交回模型。暂停容器不是把显存搬到主存 |
| 模型侧释放／恢复 | [prepare_env_gpu](../../psrl/workers/gen/gpu_handoff.py#L64)、[return_env_gpu](../../psrl/workers/gen/gpu_handoff.py#L92) | 调用模型侧 sleep/wake，并关闭／恢复生成准入，不是环境进程通用检查点 |

这些代码如果有理解上的问题，可以随时追问。跟着图读源码时，独立环境卡沿 H1 与 H2 两个并发分支，再走 H3/H4。当前 H4 只是在后续命令中重新运行程序，通用恢复尚无实现。同卡分支从 agent 的阶段通知进入 lending，再到节点确认和模型侧交接。失败会保留隔离状态，不能当作成功释放。当前 [EnvUnit 检查](../../psrl/workers/env_service/types.py#L54) 仍禁止 command 与绑定推理副本的同卡借还直接组合。

**哪些任务可能有这种状态？** 下表是任务形态示例，不是声称当前 science 数据集已经逐一包含或实测过这些任务。

| 任务形态 | 跨模型交互保留什么 | 可行的释放边界 |
|---|---|---|
| 代理控制一个持续训练会话，每轮决定下一阶段参数 | 模型权重、优化器状态、训练进度，必要时随机数及数据迭代状态 | 在完整训练步骤边界保存，再恢复。不能只存模型权重就保证继续训练语义一致 |
| 持续物理／机器人仿真 | 位置、速度、仿真器内部状态和 GPU 缓冲区 | 仿真器支持状态导出／恢复时在合法步边界切换，不支持时不能透明重启 |
| 常驻图像、视频或其他模型工具服务 | 权重、可复用特征及服务状态 | 可卸载部分张量或重启并重载，需要工具自身支持 |
| 长期交互式 Python／Notebook 程序 | 跨命令 GPU 张量、模型及库内部状态 | 必须明确哪些状态可序列化，不能仅根据终端命令返回判断可释放 |
| 每次独立运行一个 CUDA 求解器，结束后只留下文件 | 下一阶段需要的文件，GPU 进程本身不再有用 | 程序自然退出即是边界，通常无需新增活状态搬运机制 |

此前 CIFAR 组件实验使用独立前台 train.py 程序，不是一个持续训练进程跨多轮模型交互保留状态。它也没有真实语言模型代理，因此不能拿其启动耗时证明本节常驻状态切换的成本。

训练检查点是保存继续执行所需状态的一份记录。PyTorch 官方教程明确要求继续训练时除模型外还保存优化器等状态，见[保存与加载模型](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。是否需要其他状态由任务语义决定。本地通用恢复功能尚未实现，不能把“能保存权重”扩大成“任意时刻无损恢复完整程序”。

需要进一步区分：容器文件和 CPU 内存中的状态、GPU 上的应用数据、CUDA 上下文，以及 vLLM 自己的生成缓存。CUDA 上下文是进程使用 GPU 的运行状态，不是语言模型的对话上下文。恢复应用数据通常会重建 GPU 运行状态，而不是复制任意 CUDA 上下文就能继续。vLLM 的 KV cache 是注意力计算的键和值缓存，用于减少重复计算，也不等于容器里环境程序的状态。

释放未使用的缓存也不能替代卸载活状态。[PyTorch empty_cache 文档](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.empty_cache.html)针对未使用的缓存内存，不会把仍然引用的模型或张量自动搬到主存。当前 command 更严格地等待设备句柄和 GPU 进程消失，因此即使一个常驻 Python 程序搬出了大部分张量，也未必符合现有完整归还契约。要支持部分释放，需要另设可核实的剩余驻留预算，不能冒充全部归还。

**如果 vLLM 生成很久，可以做什么？** 在另一张卡上生成时，环境可继续保留状态以降低本任务延迟，也可在安全边界卸载给其他 rollout 腾位置。无等待者且容量不紧张时，来回搬运通常没有资源收益。存在等待者、可用空档长且恢复成本可接受时，卸载更有价值。生成已经等待了 60 秒不代表未来还会等 60 秒，决策应估计剩余等待时间，而不是只看已等待时间。

同卡且显存互斥时，环境若一直不释放，模型可能根本无法恢复生成。此时需要已验证的阶段切换，或者将两者放到不同资源上，不能用无限等待解决。这里的释放首先关系到是否能推进，而不只是吞吐优化。共享推理副本服务多个请求时，不能为了单个环境恢复就随意让整个副本睡眠，需要副本级准入与协调。

模型自身的专用 sleep 能力也不等于通用任务卸载。[vLLM 官方 Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)说明 level 1 会将权重移到 CPU 并丢弃 KV cache，恢复时有数据搬回和缓存重新建立的成本。该说明用于解释机制，不意味着当前安装版本支持文档中的全部新接口，更没有测出本节点的恢复时间。

**怎样比较成本：首先明确谁的时间。** 单条 rollout 更快，与一批 rollout 总完成时间更短，是两个指标。让出资源可能使 A 晚几秒恢复，同时让 B 提前几十秒完成。实际策略需要明确允许 A 增加多少延迟，不能用总吞吐改善掩盖个别请求无限等待。

以下公式为简化教学模型：观察结果已稳定，vLLM 在另一张卡上，可从时刻 0 开始生成，容量只允许 A、B 一个驻留，忽略带宽竞争与预测误差。

- W：从现在起到新指令可用的模型等待时间。
- S：A 保存状态、释放资源并完成归还确认的时间。
- B：让出的资源被其他任务使用到确认归还的时间，包含必要切换开销。
- R：A 获准后恢复到能执行下一阶段的时间，包括数据搬回及必要初始化。
- Q：A 申请恢复之后额外等待的时间，不重复计算已计入 B 的占用。

若保持驻留，A 最早在 W 时刻继续。若等指令回来才申请恢复，A 最早继续时刻为：

```text
T_reactive = max(W, S + B) + Q + R
额外延迟 = max(0, S + B - W) + Q + R
```

可见，S 即使完全藏在模型生成中，R 仍可能落在生成完成之后。只判断 S + R < W，不足以推出没有延迟，也不足以证明值得切换。

如果下一条指令到来之前就能安全恢复不变的状态，且重新获准，则可用提前恢复模型：

```text
T_prefetch = max(W, S + B + Q + R)
不拖慢 A 的条件：S + B + Q + R <= W
```

这里的 Q 从提前申请恢复开始计，两个策略的排队时刻不同，真实队列等待也会不同。这是理想时间线而非现有功能。提前恢复占回显存，必须服从容量与公平策略，不能把 B 抢掉。若新指令决定换模型或改状态，提前恢复还可能成为无用功。

**Results：两个可手算且已执行的 CPU 教学例子。** 为比较整批完成时间，再设 A 收到指令后需要 5 秒 GPU 工作，然后整个 episode 结束，B 只有一次 GPU 工作。

| 教学输入与结果 | 切换便宜、等待长 | 切换昂贵、等待短 |
|---|---:|---:|
| W / S / B / R / Q，单位秒 | 60 / 3 / 20 / 4 / 0 | 10 / 8 / 5 / 12 / 0 |
| 保持驻留，A 可继续时刻 | 60 | 10 |
| 按需恢复，A 可继续时刻 | 64 | 25 |
| 提前恢复，A 可继续时刻 | 60 | 25 |
| 保持驻留，两任务全部结束 | 85 | 20 |
| 让出后按需恢复，两任务全部结束 | 69 | 30 |
| 让出后提前恢复，两任务全部结束 | 65 | 30 |

第一列：A 在 0 至 3 秒释放，B 在 3 至 23 秒执行并归还，模型到 60 秒才给新指令。A 等指令后恢复需要到 64 秒，再执行 5 秒，69 秒全部结束。若安全提前恢复，则可到 60 秒直接继续，65 秒全部结束。第二列：保存和恢复太贵，反而从 20 秒增加到 30 秒，说明盲目释放确实会适得其反。

同卡互斥另算：教学设环境释放 3 秒、模型恢复 2 秒、生成 60 秒、模型释放 2 秒、环境恢复 4 秒，则环境在 71 秒后才能继续。没有把这些成本全部藏在 60 秒中。该例仅说明串行依赖，当前没有验证任意环境程序能执行其中的状态保存与恢复。

这些数值来自 [model_gpu_suspend_cost.py](env-service-verification/scripts/model_gpu_suspend_cost.py)，不是真实硬件性能。运行方式如下，使用标准库，无 GPU、容器或真实等待：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/model_gpu_suspend_cost.py
```

脚本对应 H1/H2 的并发、H3 的借用及 H4 的恢复，用断言验证上述时刻，并展示额外排队可使恢复变晚。模型输出保存在 env-service-verification/results/gpu-suspend-cost-20261026/cpu-demo.jsonl。它不验证保存正确性、CUDA 状态重建或带宽行为。

**对通用方案的补充建议，尚未实现。** 支持“继续驻留”“自然结束后归还”“协作卸载后恢复”三种行为，调度器先检查合法性，再考虑收益。任务提供 safe point（允许保存并恢复的执行边界）、可释放显存、剩余驻留量、保存恢复接口和费用估计。调度器结合等待者、剩余模型等待时间、恢复排队及单请求延迟预算决定是否切换。没有状态接口时不能强制卸载，但可以推迟或降低新 episode 准入。

初期采用可解释的固定策略与保守阈值，记录每次保存、确认、借出、重获准入和恢复的真实时间。引入最短驻留／冷却时间，避免在阈值附近反复搬运。CPU 主存容量、保存文件 I/O 和复制带宽也要计入，不能把 GPU 压力转成主存耗尽。迁移到其他 GPU 需要任务和通信后端支持，不能视为重新准入自动具备。

当前 command 的主要目标仍是利用程序自然结束后本来就存在的空档，而不是为一个持续运行程序反复制造昂贵的销毁重建。前者的性能对照应只算相对 baseline 新增的开销，若 OFF 本来每次也启动同一个 Python 程序，就不能把双方共有的全部加载成本只归因于 ON。真正的状态卸载恢复，需要新契约和实测后再决定何时值得做。

本节新增脚本实际 CPU 运行及 Ruff／格式检查通过，文档与总索引的 172 个本地链接、行号范围及章节锚点检查通过。全仓 pre-commit 在隔离副本仍因其他文件的既有 Ruff／格式与 prose-style 问题失败，日志保存在本节结果目录。未修改生产调度代码、未运行 GPU、未提交或推送。
