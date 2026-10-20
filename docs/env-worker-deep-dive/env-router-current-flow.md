# EnvRouter 当前流程、固定基线差异与验证边界

> **核心结论**：当前 EnvRouter 将容器生命周期的 CPU／内存预算与 GPU 命令阶段预算分开。显存够用时允许多个命令并行，不够时排队；命令结束后经实际执行节点确认归还，预算可用于其他容器，而原容器继续 CPU／模型交互。直接 Docker 查询和 NVML 连接复用降低重复控制开销，节点异步预热减少首次服务启动的关键路径等待。
>
> 固定 OFF 基线为 `0eeead861ffafed6bf31c962d259fd847775dcf9`。命令阶段复用与 vLLM 同卡借还是不同配置分支，尚不能任意组合。最新实机数字维护在[预热与 OFF／ON 对照报告](router-baseline-control-validation.md#10-results私有节点预热与固定-off-复验)。本页解释当前实现，不把合成 CUDA 小实验写成真实训练收益。

## 目录

- [1. 完整流程与角色](#1-完整流程与角色)
- [2. 相比固定 baseline 改变了什么](#2-相比固定-baseline-改变了什么)
- [3. 用一张卡说明准入、并发和回填](#3-用一张卡说明准入并发和回填)
- [4. 预热的实现与生命周期](#4-预热的实现与生命周期)
- [5. 跟着图读源码](#5-跟着图读源码)
- [6. 当前限制与可复现实验](#6-当前限制与可复现实验)

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
| 1 | [训练器初始化](../../psrl/trainer/ppo/ray_trainer.py#L549) 调用 [create_env_router](../../psrl/workers/env_service/router.py#L407)；提交预热后继续模型工作，首请求按需等待节点 |
| 2 | [AgentLoopManager._refresh_env_capacity](../../psrl/workers/agent_loop/manager.py#L909) 更新调度容量，[get_dispatch_plan](../../psrl/workers/agent_loop/manager.py#L947) 分配任务组；这是正式训练入口，本轮合成实验直接调用 runner |
| 3 | [EnvServiceClient.acquire](../../psrl/workers/env_service/client.py#L159)、[EnvUnitLedger.try_admit](../../psrl/workers/env_service/ledger.py#L105)、[runner](../../examples/sciaccel_rl/runner.py#L253)：会话需求变为带归属标签的容器配置 |
| 4 | [ManagedDockerEnvironment.exec_gpu](../../psrl/workers/env_service/docker.py#L52)、[GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L47)：请求标识、队列、显存预算与授权确认 |
| 5 | [begin_return](../../psrl/workers/env_service/command_gpu.py#L109)、[_return](../../psrl/workers/env_service/command_gpu.py#L124)：服务持有后台归还任务，锁外检查，锁内提交配额释放 |
| 6 | [EnvNodeService](../../psrl/workers/env_service/node.py#L19)、[grant_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L120)、[revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L131)：同节点 Docker 身份校验、GPU 权限和实时状态确认 |
| 7 | [runner 等待延迟归还](../../examples/sciaccel_rl/runner.py#L498)、[Node.cleanup](../../psrl/workers/env_service/node.py#L68)、[Ledger.block_gpu_cleanup](../../psrl/workers/env_service/ledger.py#L332)：错误不能变成成功，配额过期不等于物理资源已消失 |
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

表中的直接 Docker 接口、NVML 连接复用和预热，都是为了降低 ON 自身安全协议的成本。不能说 baseline 原本也花了这些检查时间，然后把节省次数直接算作相对 OFF 的收益。相对 OFF 的主要收益需要竞争、失败补采或更好任务放置等实际条件来体现；低负载仍可能付出净管理成本。

## 3. 用一张卡说明准入、并发和回填

本次实验条件是一张约 95.6 GiB 的 H20，unit 预算 80 GiB，最多四个存活容器。每条任务申报比 CUDA 分配多 1 GiB，为上下文等开销留余量。

低负载每条分配 1 GiB、申报 2 GiB，四条累计 8 GiB，预算允许四条 GPU 命令并行。unit 的短暂检查锁用于协调读取和承诺额度，不覆盖命令完整运行时间。

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

环境服务启用时，默认配置为 `prewarm.enable: True`，启动预算 30 秒、最多四个节点并行准备。[create_env_router](../../psrl/workers/env_service/router.py#L407) 等待 [begin_prewarm](../../psrl/workers/env_service/router.py#L328) 确认提交，随后训练器继续初始化模型；它不等待全部节点启动完成。[wait_prewarm](../../psrl/workers/env_service/router.py#L335) 提供显式等待接口，实验用它测量串行与并行准备。

[_node_task](../../psrl/workers/env_service/router.py#L316) 保存每节点唯一启动任务，预热、准入、GPU 借用绑定均复用它。一个请求取消不会取消所有人共用的启动任务。[_start_node](../../psrl/workers/env_service/router.py#L296) 的超时覆盖并发名额等待和实际准备；失败会销毁本次刚创建的服务，失败结果留给后续消费者，不自动无限重试。

[Node.prepare](../../psrl/workers/env_service/node.py#L49) 在执行节点探测 Docker，首次读取 GPU 身份和实时占用，使该服务的 NVML 连接就绪。它不创建租约、不授权容器，也不把这次状态读数缓存成以后 GPU 空闲的证明。节点只读准备与模型加载可以重叠，真正的借用 GPU 仍要求模型 sleep 后再次确认。

[shutdown_nodes](../../psrl/workers/env_service/router.py#L341) 拒绝丢弃活动租约；租约结束后取消未完成启动、关闭服务接口与连接、销毁所属服务 actor。训练器正常停止以及模型初始化异常的退出路径都调用环境服务停止入口。实际 CUDA、容器、后台评分器的释放仍单独核验，不能用服务 actor 消失替代物理清理检查。

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
