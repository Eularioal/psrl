# vLLM 与环境同卡共享：先用剩余容量，再利用长空闲窗口

> **核心结论**：以整批任务完成时间为目标，先使用未被训推占用的其他 GPU，包括同机的独立 GPU。只有这些资源紧张时，才考虑 vLLM 正常服务时的安全余量。再放不下且有足够长的真实空闲窗口时，才评估睡眠借卡。不能把设备都已分配给训推理解成每张卡都没有物理余量，也不能把物理余量当作不影响模型的保证。
>
> 本文是 2026-10-27 的方案分析及阶段实现记录。设备级优先级已接入 worker 放置、派发及准入，CPU 回归与真实 CUDA 验证见第 9 节。显式共享 command 已加入推理余量与模型控制门禁，并支持已睡眠窗口，见第 10 节。统一训练峰值预约、空窗预测与按总时长决策仍未实现，没有同卡训练性能实测。计算竞争与控制等待的机制、已有实验的证据边界见第 11 节。

## 目录

- [1. 全局关系与已有源码](#1-全局关系与已有源码)
- [2. 先确定要缩短谁的时间](#2-先确定要缩短谁的时间)
- [3. 第一层：模型保持服务，环境使用安全余量](#3-第一层模型保持服务环境使用安全余量)
- [4. 第二层：在长空闲窗口内释放模型显存](#4-第二层在长空闲窗口内释放模型显存)
- [5. 预测失败、提前唤醒与大任务等待](#5-预测失败提前唤醒与大任务等待)
- [6. 当前实现与建议推进顺序](#6-当前实现与建议推进顺序)
- [7. 跟着图读源码](#7-跟着图读源码)
- [8. 可运行教学模拟与验证边界](#8-可运行教学模拟与验证边界)
- [9. 独立 GPU 优先的实现与实机验证](#9-独立-gpu-优先的实现与实机验证)
- [10. 常驻共享、KV 上限与已睡眠窗口](#10-常驻共享kv-上限与已睡眠窗口)
- [11. 计算竞争与控制等待：已经验证到哪里](#11-计算竞争与控制等待已经验证到哪里)
- [12. MPS、SM 与时分复用：怎样理解多进程共享](#12-mpssm-与时分复用怎样理解多进程共享)

## 1. 全局关系与已有源码

先把模型、环境和调度器串起来。下图是**建议结构**，不是当前已经存在的完整调用链。推理副本是一套能完成模型生成的实例，可能使用多张 GPU。GPU 命令是环境明确提交的一段计算，不等于整个容器生命周期。

```text
[1] 模型请求队列、环境命令队列、任务前后依赖
                  ↓
[2] 共享决策层（待实现）
    读取逐卡显存预约、计算干扰、下一次需要模型的时间
    目标：同一批任务更早全部完成，限制模型额外等待
       │
       ├─ 其他独立 GPU 可以接纳任务 → 优先放到独立 GPU
       │
       ├─ 余量足够且干扰可接受
       │    → [3] command 准入 → 环境执行 → 物理归还
       │         vLLM 始终保持服务，不触发 sleep
       │
       └─ 余量不足且有长空窗
            → [4] 协调器确认可借，关闭新生成准入，必要时 sleep
            → [5] 环境命令执行，持有本轮授权
            → [6] 停新准入、确认 GPU 归还、恢复模型和路由
                  ↓
             回到 [1]，根据真实延迟修正下一次决策

若两条路径都不合适：继续等待或选择其他有容量的 GPU。
失败清理期间保留使用权，不把超时当作物理释放。
```

图例：图里每个部分对应的源码位置。

| 编号 | 可复用实现 | 当前缺口 |
|---|---|---|
| 1 | [EngineStats](../../../psrl/workers/gen/stats_collector.py#L21)、[ModelTurnRecord](../../../psrl/utils/profiling/records.py#L140) | 有引擎统计和轮次记录，不等于已有跨模型与 Docker 的完整依赖图及未来空窗预测 |
| 2 | [accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9)、[账本有效占用](../../../psrl/workers/env_service/ledger.py#L405) | 有实测加未兑现预约的算法，尚未把活跃 vLLM 的未来峰值纳入同一预约协议，也没有总完成时间优化器 |
| 3 | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L58)、[命令归还](../../../psrl/workers/env_service/command_gpu_node.py#L207) | 已有命令预算、排队和设备权限，当前按独立环境卡设计，不提供模型性能隔离 |
| 4 | [协调器借卡入口](../../../psrl/workers/gen/rollout_coordination/coordinator.py#L272)、[prepare_env_gpu](../../../psrl/workers/gen/gpu_handoff.py#L105) | 已有生成准入关闭、等待在途请求、一级睡眠。尚无“空窗足够且值得借”的前置决策 |
| 5 | [GpuLendingBinding._enter_env](../../../psrl/workers/env_service/lending.py#L63) | 睡眠后检查逐卡容量并恢复容器，当前仅一个借卡会话 |
| 6 | [物理归还检查](../../../psrl/workers/env_service/lending_node.py#L87)、[模型恢复](../../../psrl/workers/gen/gpu_handoff.py#L133)、[最终清理](../../../psrl/workers/env_service/lending.py#L98) | 旧互斥执行机制，预测性提前归还仍待实现。新共享路径见第 10 节 |

以上源码如果有理解上的问题，可以按编号继续追问。

## 2. 先确定要缩短谁的时间

目标是 **makespan，即固定一批任务从开始到最后一个任务完成的总时长**。环境命令更早开始不必然降低这个指标，因为模型回答可能是多个后续任务的前置条件。

教学例子：模型请求 M 需要 2 秒 GPU，回答后才能开始 10 秒 CPU 工作 A。另有独立 Docker GPU 命令 B，需要 10 秒。假设显存都够，忽略切换开销。

| 策略 | 时间线 | 全部完成 |
|---|---|---|
| 先完成模型 | M 在 0 至 2 秒。A 在 2 至 12 秒，B 同时在 2 至 12 秒 | 12 秒 |
| 先让 Docker 用卡 | B 在 0 至 10 秒。M 在 10 至 12 秒，A 在 12 至 22 秒 | 22 秒 |
| 模型与 Docker 同时提交，教学上假设争用期间各为半速 | M 到第 4 秒结束，A 到第 14 秒结束。B 前 4 秒做完 2 秒工作，剩余 8 秒，故到第 12 秒结束 | 14 秒 |

第三行说明，即使没有 sleep、没有显存不足，共享计算也可能使总时长更长。半速只是反例假设，不代表真实 GPU 的公平分配或性能比例。

“关键路径”指决定最后完成时间的依赖链。调度不能只看模型队列长度或 Docker 等待时长，也应考虑某个模型回答能解锁哪些后续任务。无需一开始实现全局最优求解器，可以先用可观测的模型响应时间、环境队列年龄和阶段时长作保守近似。

## 3. 第一层：模型保持服务，环境使用安全余量

把 vLLM 看成外部显存压力，在**容量记账**上是合理的。现有账本已经将物理总占用与尚未真正分配的环境预约相加。但是活跃 vLLM 还有未来内存变化与计算竞争，不能完全等同于“分配一块张量后不再计算”的压力注入器。

先区分两种余量：物理卡上真正未分配的显存，和 vLLM 已经分配的缓存池中尚未装入有效内容的空间。后者仍属于 vLLM，Docker 不能直接再分配使用。KV cache 是模型为避免重复计算保存的注意力中间结果，减少其容量可能降低可同时服务的请求数或增加重算。vLLM 的内存比例和缓存大小设置用于规划本实例内存，并不是跨进程共享仲裁器，见[官方引擎参数](https://docs.vllm.ai/en/stable/configuration/engine_args/#cacheconfig)。本地会传递 `gpu_memory_utilization`，见 [vllm_async_server.py:366](../../../psrl/workers/gen/vllm_async_server.py#L366)。不能把调低该比例后释放的容量当作对推理完全无代价。

建议在每张卡上按以下方式准入，**这是扩展设计，不是当前已实现的 vLLM 预约**：

```text
实测总占用 U
+ 已批准环境任务尚未兑现的预约 R_env
+ vLLM 本阶段尚未兑现的峰值预约 R_model
+ 新命令峰值需求 D
+ 安全余量 S
<= 可用总容量 C
```

如果某个 owner 的峰值预约为 B，同一次可信采样测得它已占 A，则它尚未兑现的预约为 `max(0, B - A)`。这种逐 owner 的差额避免重复计算已经包含在 U 中的占用。现有环境算法就是这个形状，扩展点是把模型也作为已登记的资源使用者。归属缺失时不能借别人的显存抵扣预约，共享模式的关键观测过期时应停止新准入。它仍是准入承诺，不是 CUDA 的显存硬限额，必须约束参与者遵守申报范围。

教学数字：80 GiB 卡，模型实测 56 GiB、阶段峰值 60 GiB。已有环境任务实测 8 GiB、申报峰值 12 GiB，安全余量 4 GiB。

- 物理实测合计是 `56 + 8 = 64 GiB`。
- 模型与环境各还有 4 GiB 未兑现预约。
- 新任务需要 4 GiB 时，`64 + 4 + 4 + 4 + 4 = 80 GiB`，正好满足教学预算。
- 新任务需要 8 GiB 时，总计 84 GiB，应等待。只看当前 16 GiB 空闲会低估将来的占用。

峰值预算必须覆盖选定的模型服务阶段，包括缓存配置、临时工作区及相关状态切换。已经预约进缓存池的空间不能因为逻辑请求较少就再次当作增长空间累加，具体分项必须与本地版本的分配方式一致。模型重新配置、加载或权重同步需要扩大预算时，应先经过同一仲裁层，不能事后才用遥测发现冲突。训练、验证和其他使用同卡的角色也必须登记，不能只考虑 vLLM 与环境两方。

容量允许后还要检查**计算干扰**。多个进程会争用 GPU 计算、显存带宽及传输资源，可能时间片轮转，不能把“同时存在”理解为 GPU 内核必然并行。NVIDIA 对普通 CUDA 上下文与 MPS 多进程服务的调度差异有[官方说明](https://docs.nvidia.com/deploy/mps/latest/architecture.html)。当前设备权限控制只能决定能否访问 GPU，不能保证模型拿到某个计算份额。本方案不以部署 MPS 为前提。

起步策略可以是限制共享命令数量和持续时间，按真实模型延迟判断是否继续批准新命令。已经在运行的普通 CUDA 命令不能假定可瞬间撤回。对持续长、计算密集的任务，即使显存足够，也可能应当等模型较空闲时再进入。

## 4. 第二层：在长空闲窗口内释放模型显存

“空闲”需要拆开看：

| 当前状态 | 能否直接把释放后的容量交给环境 |
|---|---|
| 模型仍在生成，只是某个环境在等待它 | 不是空闲，环境等待恰好说明模型是前置条件 |
| 目标副本暂时无请求，但权重和缓存仍在卡上 | 可以评估剩余空窗，不能只看队列此刻为零就睡眠 |
| vLLM 已经睡眠，卡也没有其他 owner 使用 | 可以评估回填，已经发生的 sleep 成本不重复计入本次增量成本 |
| vLLM 因训练阶段而睡眠，卡随后交给训练 | 卡并不空闲，必须把训练预约计入，不能从 vLLM 状态单独推断 |

预测应针对目标副本的**剩余空闲时间**，包含其他会话何时会发模型请求，而不是当前 Docker 会话自身的等待时间。任务未来依赖也是输入。如果模型请求正在等待本次大环境命令的结果，两者的时间不能视为相互独立的外部到达。

offload 在这里指把模型 GPU 状态转移到主机内存以释放设备容量。本地借卡调用 `nixl_sleep(level=1)`。vLLM 一级睡眠会备份模型权重到 CPU 并丢弃 KV cache，恢复时要搬回模型状态。二级睡眠不保留相同的权重备份，需要另外恢复权重。详见[vLLM 睡眠文档](https://docs.vllm.ai/en/stable/features/sleep_mode/)。这与推理过程中按层持续搬运权重的 CPU offload 配置是不同使用方式，不能把后一种当作无代价空窗借卡。

本地还执行 NIXL 内存注册的注销与恢复。NIXL 是这里用于跨进程传输模型数据的通信组件，注册状态要与当前 GPU 内存一致，见 [nixl_sleep / nixl_wake_up](../../../psrl/workers/gen/vllm_async_server.py#L1284)。实际成本应包括等待在途请求、sleep、主机备份容量、传输带宽、环境清理、wake、传输注册及必要的缓存重算，而不仅是某个 API 返回耗时。

一个保守的空窗准入条件是：

```text
等待在途请求和 offload 的增量成本
+ 环境命令运行时间上界
+ 物理归还检查时间
+ 恢复到模型真正可服务的时间
+ 预测安全余量
<= 剩余空闲窗口下界
```

上界、下界如果来自历史分位数，只是经过校准的估计，仍可能失准。这个条件是保护模型恢复时机的筛选条件，不能单独证明整批任务总时长必然下降。还应比较等待其他 GPU 或继续排队的方案，计入模型被延迟后对下游的影响。

教学例子：预计模型至少还有 60 秒不用。offload 3 秒，Docker 命令 30 秒，清理确认 2 秒，恢复服务 5 秒，安全余量 5 秒，合计 45 秒，能装进窗口。

```text
0       3                 33    35      40              60
| offload | Docker GPU计算 | 清理 | 恢复 | 已就绪、等待新请求 |
```

若下一次模型请求确实在第 60 秒到达，它不用等 Docker。如果实际只有 15 秒空闲，就不应接纳这个不能提前结束的命令。若 offload 与恢复变成 20 秒和 15 秒，总预算为 72 秒，即使预计空闲 60 秒也不值得按这个条件借出。所有数字均为教学假设。

若模型本来就已睡眠，应从当前状态算剩余释放与恢复成本，不能为了套用统一流程先 wake 再 sleep。现有 `prepare_env_gpu` 在引擎已经睡眠时会先唤醒、必要时恢复权重，再做一级睡眠。这是当前为获得可恢复备份的流程，说明“利用既有睡眠窗口”需要状态感知改造，不能把它直接当成零 offload 成本路径。二级睡眠或权重更新后的恢复代价尤其需要单独计算。

## 5. 预测失败、提前唤醒与大任务等待

沿用上述例子，如果请求在第 8 秒就到来，Docker 命令还要到第 33 秒完成，清理与恢复后模型第 40 秒才能服务，额外等了 32 秒。当前普通命令没有通用的无损抢占能力，不能先唤醒模型，再希望环境自行让出容量。

需要明确两类承诺：

- **尽量不拖慢模型**：允许预测误差，监控额外等待，停止新准入并在已有命令的安全边界回收。收益需要实测。
- **严格限制模型最多多等几秒**：环境命令必须有可信的最大剩余时间，或支持有界取消/检查点及物理释放。只有预测值而没有这些执行条件，无法保证硬上限。

检查点是程序主动保存可恢复状态的机制。杀掉 GPU 进程能释放资源，但可能丢失已完成的计算，不能叫作无损抢占。取消到物理释放的耗时也要计算。恢复前禁止新 GPU 命令进入，旧授权必须失效，防止一边回收、一边批准新任务。

“大型任务”至少有两个维度：占显存大、运行时间长。一个需要 60 GiB 但仅运行 5 秒的任务，可能适合借空窗。一个占 10 GiB 却运行数十分钟且不能提前结束的任务，反而难以承诺及时还卡。长时间没有合适窗口时，需要其他卡、任务拆分或有计划的资源时段，不能无限等待，也不能为避免等待就默认打断繁忙模型。

## 6. 当前实现与建议推进顺序

当前 `GpuLendingBinding` 是环境请求驱动的 ENV / MODEL 切换，没有先证明模型有长空窗。`prepare_env_gpu` 等待在途生成而不是强制杀掉它，但关闭准入后仍可能推迟新的模型请求。已有互斥交接保障可以保留为第二层的执行机制，不宜成为所有同卡任务的默认策略。

普通 command unit 仍不能冒充受协调的模型共享。显式 `shared_rollout_replica_rank` 路径已经接入推理增长余量和生命周期门禁，详见第 10 节，性能收益仍需专门对照。旧的互斥借卡配置 `rollout_replica_rank + command` 仍被拒绝，见 [EnvUnit 校验](../../../psrl/workers/env_service/types.py#L67)。

| 顺序 | 建议实现 | 最小验收问题 |
|---|---|---|
| 1 | 建立逐卡统一 owner 和预算，先接入模型常驻时的 command 余量共享，不主动 sleep | 相同工作量下是否避免 OOM，模型响应与整批完成时间是否可接受 |
| 2 | 消费协调器已确认的睡眠窗口，识别实际释放状态和其他 GPU owner | 是否避免重复 offload，能否在恢复期限前清理并恢复模型 |
| 3 | 预测长空窗，只有净收益成立时主动 offload，加入提前请求处理 | 预测失败时是否有界，交接总成本和整批完成时间是否改善 |

第一阶段的重点不是实现预测模型，而是完成容量、实际归还和性能观测。后续再决定是否需要复杂的代价估计器。不要为验证共存直接改弱模型缓存配置或替换固定 baseline 后宣称提速，模型配置变化本身需要单独对照。

真实验证可用固定少量模型请求和环境 GPU 命令，先比较模型优先串行与余量共享，再另测长空窗。记录固定工作量、模型响应、环境完成时间、总完成时间、OOM、offload / wake / 清理时间和物理释放。达到问题的证据边界后停止，不需要先跑完整训练。第 10 节先验证真实共存和物理归还，尚未进行该性能对照。若继续做历史 science ON/OFF 对照，OFF 仍固定为 `0eeead861ffafed6bf31c962d259fd847775dcf9`，不能把同版本策略对照改称该 baseline。

## 7. 跟着图读源码

以下按建议的数据流阅读，明确区分已有函数与缺失的连接。图中异步队列、命令归还和模型请求不是一条全局串行线程。

| 步骤 | 编号与函数 | 输入、输出或状态变化 |
|---|---|---|
| 获取信号 | 1，[EngineStats](../../../psrl/workers/gen/stats_collector.py#L21) | 已有引擎队列等状态，仍需加入环境依赖和空闲窗口信号 |
| 计算容量 | 2，[accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9) | 实测总量加每个预约未兑现部分，显式共享路径另外加入模型增长余量，全阶段统一 owner 预算仍未完成 |
| 授权命令 | 3，[acquire](../../../psrl/workers/env_service/command_gpu.py#L58) | 队列、容量及遥测合格后授权，现有代码不会检查本方案的模型干扰目标 |
| 归还命令 | 3，[revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L207) | 禁新设备打开，检查旧句柄与 GPU 工作，确认后才释放预算 |
| 判断长空窗 | 2 → 4，无生产函数 | 本方案新增决策点，不把已有吞吐估计器冒充空窗预测器 |
| 借出整副本 | 4 → 5，[prepare_env_gpu](../../../psrl/workers/gen/gpu_handoff.py#L105)、[_enter_env](../../../psrl/workers/env_service/lending.py#L63) | 关闭生成准入、睡眠、采集基线与核对容量，随后恢复容器 |
| 恢复模型 | 6，[_freeze_gpu_lease](../../../psrl/workers/env_service/lending_node.py#L87)、[return_env_gpu](../../../psrl/workers/gen/gpu_handoff.py#L133) | 旧互斥路径检查本租约归还后唤醒，共享路径另用命令票据聚合控制 |
| 会话结束 | 6，[_finish](../../../psrl/workers/env_service/lending.py#L98) | 清理容器、确认归还、释放预约，失败时保持隔离 |

## 8. 可运行教学模拟与验证边界

脚本：[vllm_env_sharing_demo.py](../env-service-verification/scripts/vllm_env_sharing_demo.py)。只使用 Python 标准库，无 GPU、Docker、Ray、网络或真实等待。它模拟上述容量计算和时间线，不调用生产调度器，不是性能预测器。

从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/vllm_env_sharing_demo.py
```

`memory_admission` 对应图中 2，`dependency_orders` 展示图中 1 的前后依赖，`idle_window` 对应图中 4 至 6。输出 JSON 记录准入、模型额外等待和全部完成时间。真实实现还需要多会话、多卡、遥测异常、任务取消和通信故障控制，教学脚本没有模拟这些机制。

**Results：2026-10-27 CPU 教学运行。** 原始输出保存在 [vllm-env-sharing-demo.json](../env-service-verification/results/vllm-env-sharing-demo.json)。这些数值只用于检验示例推导，不是 GPU 实测。

| 场景 | 预期观察 |
|---|---|
| 80 GiB 预算与增长预约 | 新申请 4 GiB 获准，8 GiB 被拒绝 |
| 模型是后续 CPU 工作前置条件 | 模型优先 12 秒，Docker 优先 22 秒，假设共享期间各半速则 14 秒 |
| 60 秒空窗，交接预算 45 秒 | 固定工作量参考顺序总计 102 秒，空窗回填 72 秒，模拟节省 30 秒 |
| 原预测 60 秒，模型实际第 8 秒回来 | 模型多等 32 秒，总时间从参考顺序的 50 秒变为 52 秒，反而增加 2 秒 |
| 只有 15 秒窗口，或交接预算增至 72 秒 | 拒绝借出 |

空窗场景的参考顺序明确为：保持模型常驻直到下一次请求完成，再 offload、执行同一个 30 秒 Docker 命令、清理并恢复。两种策略都包含模型请求后的 10 秒 CPU 工作以及最终模型恢复。参考顺序只是本教学模型中的一种可行策略，不代表生产系统的最优 baseline。复现脚本中有这些时间关系的断言。

已有真实借卡与独立环境卡训练结果仍按[状态总览](../experiments/env-status-and-roadmap.md)各自的验收范围理解，不能视为本方案已经验证。关于当前 command 与借卡为何被隔开的源码说明，见[现有组合边界](../experiments/env-status-and-roadmap.md#为什么同卡借还还不能与-command-组合)。

## 9. 独立 GPU 优先的实现与实机验证

**本轮先落地设备选择顺序。** `prefer_other` 按已核实的物理 GPU 归属区分优先与后备资源，不再把一台机器上的所有 GPU 都视作正在被训推占用。它仍是放置偏好，不是模型共享许可或性能隔离。默认 `cpu_only` 不自动放行普通训推占用卡，已核实独立的同机 GPU 可以使用。`avoid` 仍排除整台训推节点，`any` 仍表示不施加这项偏好。

下面是本轮已接入的路径，与上文尚未实现的共享决策层分开阅读。

```text
[A] Trainer 查询已初始化 worker 的节点和全部 GPU ID
      ↓ 把设备归属发布给 Router
[B] NodeRules：同机独立卡也属于优先资源，未知归属保持保守
      ↓ 带优先级的逐 unit 容量快照
[C] Manager 的 EnvAwareDispatcher 先选有容量的独立资源
      ↓ 任务交给固定节点上的 worker
[D] Router 根据新鲜节点状态和账本再次选择 unit
      ↓ 独立卡放得下则优先，放不下才考虑已允许的后备 unit
[E] command 申请设备权限 → Docker 执行 → 确认归还
      ↓ 容量恢复后，新会话重新优先使用独立卡
[F] 会话结束：清理容器和租约
```

图例：图里每个部分对应的源码位置。

| 编号 | 函数与源码 | 状态变化 |
|---|---|---|
| A | [gpu_placements_from_worker_groups](../../../psrl/workers/env_service/node_policy.py#L278)、[Trainer 发布入口](../../../psrl/trainer/ppo/ray_trainer.py#L608) | 核对实际节点与所有已分配 GPU，不能只记录一个 worker 的首张卡 |
| B | [overlaps_train_rollout / tier](../../../psrl/workers/env_service/node_policy.py#L89)、[发布归属](../../../psrl/workers/env_service/router.py#L134) | 精确归属优先，缺失时按训推节点保守判断。有活动租约时拒绝替换归属表 |
| C | [节点容量](../../../psrl/workers/env_service/ledger.py#L304)、[_assign_resources](../../../psrl/workers/env_service/placement.py#L188) | 优先级先于 CPU 负载和轮询顺序。同题轨迹仍按组放置，临时扣除本批已派出的容量 |
| D | [Ray Router.try_admit](../../../psrl/workers/env_service/router.py#L537)、[账本选择](../../../psrl/workers/env_service/ledger.py#L106) | 所有健康候选共同参与选择，不按节点遍历顺序抢先批准。遥测 URL 与实际选择节点一致 |
| E | [command 准入](../../../psrl/workers/env_service/command_gpu.py#L58)、[归还检查](../../../psrl/workers/env_service/command_gpu_node.py#L207) | 仍经真实命令预约和设备授权。容器获准排队不等于 GPU 已授权 |
| F | [Router.release](../../../psrl/workers/env_service/router.py#L220) | 按原有物理归还协议收尾，再释放租约 |

跟着图读源码：先从 A 的运行时 GPU 清单进入 B，得到每个 unit 的优先级，再看 C 如何把实际需求和整组轨迹数量用于放置。D 是最终准入，必须再次看当前容量，不能把 C 的快照当作已保留资源。随后沿 E 进入实际命令授权与归还，最后走 F。不同 worker 和环境命令可以并发，C 与 D 之间容量可能变化，因此该偏好不是跨所有会话的原子最优调度。以上源码可以继续按编号追问。

还修正了两个会使优先级落空的条件：`prefer_other` 创建 worker 时保留后备节点，不再在初始化阶段永久过滤掉它们。该策略要求每个候选环境节点至少有一个 worker，数量不足时明确报错。Env Service 启用时 worker 使用固定节点放置，保证实际启动 Docker 的节点与调度记录一致。

command 会话允许在 GPU 满时先保留主机资源并排队，但“能排队”不能压过另一张卡“现在就有容量”。本轮让派发和 unit 选择都查看 GPU 当前余量，避免把有空容器槽位误认为有可执行 GPU 容量。如果所有候选都暂时满了，仍可在优先资源上排队，实际 GPU 授权继续等待准入。

**可用范围。** 在显式配置的 unit、允许的节点和实际存在的 worker 范围内选择。没有实现扫描整个平台并自动申请未分配给本作业的 GPU，也没有把已运行的 Docker 容器迁移到其他节点。同题轨迹仍保持整组派发，所以可容纳单条但不能容纳整组的节点，不保证排在能容纳整组的节点前。

**训推占满所有设备时。** 这只表示所有设备都有训推归属，还要另看物理余量、未来峰值与计算负载。本轮不会自动建立模型的显存峰值承诺，也没有检测计算干扰后授权活跃训推共享。`prefer_other` 的既有后备路径不能被当作这个安全协议，不能仅靠切换该配置就宣称“训推不紧张时自动共存”已经完成。实际模型 GPU 仍须通过明确的共享预算或已接通的借卡协议使用。本轮没有新增 sleep / offload 调用。

**Results：2026-10-27 有界真实验证。** 使用首选实验节点、两张 H20、真实 Ray Router、Harbor/Docker 与 CUDA。每个 unit 申报预算 4096 MiB，每条命令申报 2048 MiB，容器实际执行 64 MiB 小型 CUDA 内核。后备实验卡常驻注入 1 GiB 压力。该卡仅被标为受保护的实验资源，**没有加载 vLLM 或训练模型**。压力是容量条件，不是假装运行模型。

| 阶段 | 条件 | 实际选择与执行 |
|---|---|---|
| 独立卡优先 | 独立卡空闲，后备卡有 1 GiB 常驻压力 | 选择独立卡，GPU UUID 与预期一致，CUDA 返回 42 |
| 容量不足回退 | 独立卡另注入 3 GiB 压力，不能再容纳 2 GiB 命令预约 | 选择后备卡，CUDA 返回 42 |
| 恢复后返回 | 释放独立卡压力，后备卡仍保留原占用 | 再次选择独立卡，CUDA 返回 42 |

成功轮总计 84.42 秒，三条命令退出码均为 0，租约归零，两张卡回到约 0.19 MiB 应用显存基线，GPU 计算进程和所属容器清空。外层运行预算 360 秒，工作阶段预算 210 秒，压力进程自带 240 秒寿命并由父进程按归属回收。达到三项选择结论后结束，没有追加训练或吞吐实验。

首次尝试已选对独立卡、CUDA 也返回 42，但探针写入不存在的 `/logs/artifacts` 目录而退出 1。先确认 GPU 与容器清理，再补建产物目录，在一次有界重试中完成三阶段。原始失败没有删除，也没有计作完整成功。

脚本为 [real_gpu_preference.py](../env-service-verification/scripts/real_gpu_preference.py)。复现前先按运行指南核对节点和镜像，在已激活的项目环境中设置 `NODE_IP`、`CUDA_IMAGE`、`PROTECTED_GPU`、`DEDICATED_GPU` 和全新的 `OUTPUT_DIR`，分别表示执行节点、已有 Python/CUDA 探针镜像、两张现场空闲卡的遥测 index 及结果目录。然后执行：

```bash
timeout -s INT -k 30s 360s python \
  docs/env-worker-deep-dive/env-service-verification/scripts/real_gpu_preference.py \
  --node-ip "$NODE_IP" --image "$CUDA_IMAGE" \
  --protected-gpu "$PROTECTED_GPU" --dedicated-gpu "$DEDICATED_GPU" \
  --output "$OUTPUT_DIR"
```

原始[成功证据](../env-service-verification/results/gpu-preference-20261027-r2/run/evidence.json)、[首轮失败](../env-service-verification/results/gpu-preference-20261027/run/evidence.json)保留了代码哈希、依赖来源、设备身份、选择结果与清理数据。[独立收尾复查](../env-service-verification/results/gpu-preference-20261027-r2/physical-release.json)确认所属容器、网络、隔离 Ray 进程及 GPU 计算进程均已清空，Env Service 源码哈希与执行时一致。CPU 回归为 Env Service 全目录 309 项通过，之后新增全局 Router 选择与 URL 对应检查，定向 10 项通过，两次统计有重叠。跨两台真实机器的回退、完整 Trainer 启动、活跃 vLLM/训练共存和吞吐收益不在本次实机覆盖范围。


## 10. 常驻共享、KV 上限与已睡眠窗口

本节描述显式 `shared_rollout_replica_rank` 路径。前文第 1 节的预测决策层仍是提案，下面是已接入的执行链。共享不是让环境获得模型的缓存，而是在同一物理 GPU 上另外分配自己的显存。

```text
[1] Trainer 核对完整推理副本的 GPU，排除其他模型角色重叠
    → [2] worker 报告推理临时内存 profile，Router 校验用户预留余量
    → [3] 环境 command 请求，先取得模型状态票据
         ├─ 模型醒着 → 不暂停生成，保留完整推理余量
         ├─ 模型已睡 → 不唤醒、不额外 sleep，保留安全余量
         └─ 控制操作已排队或正在执行 → 本次等待
    → [4] 新鲜逐卡遥测 + 环境未兑现预约 + 模型余量，足够才打开设备权限
    → [5] Docker 执行 CUDA，醒着的 vLLM 可同时生成
    → [6] 关闭设备权限，确认旧 GPU 句柄和进程已退出
    → 释放模型票据，允许排队的唤醒或权重更新继续

[5] 超时分支 → 删除本租约的容器 → 物理归还确认 → 释放票据
    任一清理确认失败 → 隔离该 unit，保留模型门禁，不冒险唤醒
```

图例：图里每个部分对应的源码位置。

| 图中编号 | 职责与函数 | 源码 |
|---|---|---|
| 1 | `_init_gpu_lending` 核对设备及角色 | [ray_trainer.py](../../../psrl/trainer/ppo/ray_trainer.py#L608) |
| 2 | `env_sharing_profile` 导出瞬时容量测量，`validate_env_sharing` 校验下限 | [vllm_extension.py](../../../psrl/workers/gen/vllm_extension.py#L29)、[gpu_handoff.py](../../../psrl/workers/gen/gpu_handoff.py#L66) |
| 3 | `acquire_env_command` 原子检查模型状态和控制队列 | [gpu_handoff.py](../../../psrl/workers/gen/gpu_handoff.py#L76) |
| 4 | `_grant` 使用遥测、预约及逐卡余量 | [command_gpu.py](../../../psrl/workers/env_service/command_gpu.py#L106) |
| 5 | 设备授权和环境命令执行 | [command_gpu_node.py](../../../psrl/workers/env_service/command_gpu_node.py#L192)、[真实实验脚本](../env-service-verification/scripts/real_gpu_sharing.py) |
| 6 | `_return` 先物理归还，`release_env_command` 再放开控制 | [command_gpu.py](../../../psrl/workers/env_service/command_gpu.py#L215)、[gpu_handoff.py](../../../psrl/workers/gen/gpu_handoff.py#L94) |
| 超时分支 | `_expire_shared` 删除所属容器，保留未确认的隔离状态 | [command_gpu.py](../../../psrl/workers/env_service/command_gpu.py#L171) |

这些代码如果有理解上的问题，可以随时追问。

**常驻占用为什么不等于峰值。** GPU 中同时存在几类东西：模型权重、KV 缓存池、运算临时张量、CUDA 图和第三方算子工作区。KV 是注意力机制保存的键和值，用来避免每生成一个 token 都重新计算完整前文。临时张量则是本次推理运算产生、之后可以释放的中间结果。启动后空闲时测到的占用，未必包含下一次长输入计算所需的全部临时空间。

本环境核对的版本为 `0.22.1.dev0+g0b3ba88f1.d20260729.cu128`。安装来源是外置 editable 依赖，不在本仓库中。下面给出该版本的文件、函数及核对位置，不能把网上最新版本的新功能直接当成本机能力：

| 依赖源码位置 | 实际行为 |
|---|---|
| `vllm/v1/worker/gpu_worker.py:433`，`determine_available_memory` | 自动模式用 dummy forward 测量非 KV 占用，从计划内存中扣除权重、激活峰值等，决定 KV 池容量 |
| 同文件 `:445` | 显式设置 `kv_cache_memory_bytes` 时跳过该内存测量，不遵循 `gpu_memory_utilization` 的自动 KV 规划 |
| 同文件 `:506` | 保存 `non_torch_memory`、`peak_activation_memory` 和图内存估计，作为本次共享校验输入 |
| `vllm/v1/worker/gpu_model_runner.py:6926`，`_allocate_kv_cache_tensors` | 根据已经确定的大小创建 GPU tensor，物理缓存池在初始化时分配 |
| `vllm/v1/core/sched/scheduler.py:933`，`_preempt_request` | 释放被抢占请求的 KV 块，将已计算 token 计数清零并放回等待队列，后续重新计算 |

用已激活的运行环境执行 `python -c "import importlib.util; print(importlib.util.find_spec('vllm').origin)"` 可以定位实际依赖，再核对上述文件。这里的依赖行号对应本轮读取版本，不承诺升级后不变。一般机制也可对照[官方优化说明](https://docs.vllm.ai/en/stable/configuration/optimization/#preemption)与[官方缓存参数](https://docs.vllm.ai/en/stable/configuration/engine_args/#cacheconfig)。

**KV 增长究竟会发生什么。** 教学假设：物理 KV 池预先分配了 20 GiB，初始有效 token 只占相当于 2 GiB 的块。生成变长后，逻辑使用量从 2 增到 15 GiB，通常仍在这 20 GiB 物理池内，没有再向 GPU 申请 13 GiB。剩余块不够时，当前 V1 的普通路径优先等待或抢占重算。`recompute` 就是释放某个请求的缓存，把它暂时退回队列，以后重新算前缀来恢复缓存，代价是额外计算和延迟。它不是默认自动扩大池，也不是默认自动换出到 CPU。

`offload` 表示把数据搬到 CPU 等其他存储，需要显式的相关配置或缓存连接器支持。模型 sleep 的权重卸载与 KV 压力下的请求重算也是两个不同机制。更重要的是：如果 Docker 把推理临时张量所需的物理显存占掉，后续 PyTorch/CUDA 分配仍可能直接 OOM。重算机制管理的是缓存块，不能把跨进程显存不足自动变成可恢复排队。

**本次实际采用的准入公式。** 对每张卡分别计算，单位使用 MiB：

```text
已承诺占用 = 实测总占用
           + Σ max(0, 已有环境承诺 - 已核验归属的环境实占)

醒着时：已承诺占用 + 新环境需求 + model_headroom_mb ≤ 可用预算
睡眠时：已承诺占用 + 新环境需求 + safety_margin_mb ≤ 可用预算
可用预算 = min(unit 配置容量, 设备物理容量)
```

多卡情况下，现有实现不能精确归属的部分继续保守全额计入预约，不把多张卡平均后的一张富余用于掩盖另一张不足。模型余量每张卡只留一份，多个环境命令不会重复扣掉多份完整模型余量。

`model_headroom_mb` 是在**当前物理占用之外**保留的增长空间，不是把模型常驻显存扣掉后得到的环境容量，也不是模型总显存硬上限。绑定时要求它至少覆盖全部 worker 中最大的 `激活峰值 + 非 PyTorch 增长 + CUDA 图估计 + 安全余量`。使用最大值是为了不依赖 worker 返回顺序和物理 GPU 顺序的偶然一致。已经存在于遥测中的工作区可能再次被保守计入，这会少放一些任务，但不会借用别人尚未兑现的预算。

教学假设：卡总容量 80 GiB，当前占用 58 GiB，已有环境承诺中尚未分配的部分 4 GiB，为模型再留 8 GiB。那么最多再放 10 GiB 的新需求。申请 12 GiB 必须等待，不能因为 `80−58=22` 就放行。若遥测时恰好处于推理峰值，仍再留 8 GiB，属于有意保守。

这不是对任意模型输入的数学峰值证明。自动 profile 是指定模型配置下的测量，额外安全余量应结合真实最大批次、长度、算子和图配置验证。修改这些配置、启用额外 GPU 缓存组件或更换模型时，需要重新测量和配置。缺少 profile 时明确拒绝共享，尤其不把显式 KV 大小路径的缺失字段默认为零。环境申报显存也不是 CUDA 硬隔离，越预算的任意 CUDA 程序仍可能破坏容量假设。

**两类共享如何与模型控制协调。** `GpuControlGate` 是一个异步门禁，使用条件变量把“模型状态检查”和“命令取得票据”连成不可交错的操作。正常生成不经过这个控制门禁，因此可以继续服务。睡眠、唤醒和权重更新等操作必须等票据全部归还。一个控制操作开始等待后，后来的共享命令不能插队，避免环境源源不断地延迟模型。

已睡眠模式直接利用观察到并在门禁内锁定的状态，既不先唤醒再睡眠，也不在命令结束时自行决定恢复生成。原本负责模型状态的协调器继续拥有唤醒决策。若协调器提前要唤醒，等待当前命令物理归还，新的共享命令等待。这种方案没有预测未来空窗，也没有为环境主动 offload 模型。

默认命令服务期限为 30 秒，配置上限 300 秒，不适合把无法中断的长任务直接塞进共享池。到期会终止**本租约所属环境容器**，确认 GPU 归还后才释放门禁。正常归还同样先关闭设备权限并检查残留句柄、进程。控制等待除了命令剩余期限，还可能包括清理耗时。如果物理释放无法确认，宁可保留隔离状态，也不能声称“到时间一定可以安全唤醒”。

**配置示意，数字是教学值。** 在现有 `env_service.units` 中为完整推理副本声明共享 unit，具体设备和预算由部署决定：

```yaml
node_policy: prefer_other
units:
  - unit_id: shared-rollout-0
    node_ip: ${oc.env:ENV_NODE_IP}
    cpu_quota: 4
    memory_mb: 8192
    gpu_indices: [0]
    gpu_memory_mb: 81920
    max_containers: 2
    gpu_reservation: command
    shared_rollout_replica_rank: 0
    model_headroom_mb: 8192
gpu_sharing:
  safety_margin_mb: 1024
  max_command_s: 30
  reuse_sleeping: true
```

这里不能同时填写旧的 `rollout_replica_rank`。旧字段表示 episode 级互斥借卡，新字段表示 command 级共享。独立 GPU 仍需单独声明 unit，并由既有优先级优先选择。已经启动的 Docker 不会因为后来出现空闲卡而自动迁移。Trainer 拒绝没有显式绑定却与模型设备重叠的 unit，拒绝共享副本与训练、验证或其他模型角色占同一设备。共享训练设备所需的完整训练峰值协议仍未实现。

**跟着图读源码。** 按真实执行路径阅读：

1. 图 1：[ray_trainer.py:608，`_init_gpu_lending`](../../../psrl/trainer/ppo/ray_trainer.py#L608) 输入所有 worker 的设备位置，检查完整副本及其他角色重叠，然后选择共享绑定。
2. 图 2：[router.py:158，`bind_shared_rollout_gpu`](../../../psrl/workers/env_service/router.py#L158) 调用模型 `validate_env_sharing`，后者通过集体远程调用收集每个模型 worker 的 profile。校验通过才解除共享 unit 的初始阻塞。
3. 图 3：[command_gpu.py:58，`acquire`](../../../psrl/workers/env_service/command_gpu.py#L58) 在 unit 锁内调用 `acquire_env_command`。这不是推理的全局串行锁，其他生成请求仍在引擎内并行推进。
4. 图 4：[command_gpu.py:106，`_grant`](../../../psrl/workers/env_service/command_gpu.py#L106) 拉取新鲜设备遥测，根据票据对应的 resident 或 sleeping 状态计算余量，预算满足才授权 cgroup 设备访问。
5. 图 5：调用方执行 Docker 命令，服务端同时保留期限任务。命令和模型生成是并发分支，不能读成先后顺序。
6. 图 6：[command_gpu.py:215，`_return`](../../../psrl/workers/env_service/command_gpu.py#L215) 等待节点撤销设备权限和确认旧进程退出，再调用 `release_env_command` 通知模型控制门禁。失败分支由 `_expire_shared` / [command_gpu.py:315，`finish`](../../../psrl/workers/env_service/command_gpu.py#L315) 执行所属资源清理并保留必要隔离。

**可运行 CPU 协议实验。** 在项目已配置的 Python 环境中执行：

```bash
python -m pytest tests/env_service/test_gpu_sharing.py -q
```

该脚本调用真实模型门禁和命令调度器，但用测试替身模拟设备遥测与物理清理，所以只验证协议。例子中的容量 80、模型余量 35、命令需求 51，会在醒着时等待，睡眠后保留安全余量 4 则可放行。另一个用例故意让物理归还确认等待，观察模型唤醒仍不能完成。真实容器与模型验证使用 [real_gpu_sharing.py](../env-service-verification/scripts/real_gpu_sharing.py)，执行命令参数包括模型目录、节点地址、输出目录，模型只生成少量 token，不运行训练。

**Results 与验证边界。** 后台实机脚本 `real_gpu_sharing.py` 使用 Qwen2.5-3B-Instruct、单张 H20、eager 模式、最多 512 token 上下文和每次最多 16 个输出 token。自动 profile 给出的临时空间为 248 MiB，本轮配置额外预留 2296 MiB，睡眠状态仍保留 1024 MiB。整个功能实验含初始化和清理约 248.76 秒，退出码 0。

| 验收问题 | 本次观察 |
|---|---|
| 超预算命令是否被拒绝授权 | 返回 `wait`，未打开设备权限 |
| 模型常驻时能否执行环境 GPU 命令 | `resident` 准入成功，真实 CUDA 分配 64 MiB、内核读回 42、设备 UUID 匹配 |
| 命令活动期间模型是否能生成 | 观测到模型与环境 CUDA 进程同时存在，生成完成，引擎保持醒着 |
| 已睡眠路径是否可用 | 第二条命令记录为 `sleeping`，CUDA 内核读回 42，没有为取得票据先唤醒 |
| 提前唤醒是否越过归还 | 命令活动时唤醒仍等待，命令归还后完成 |
| 模型恢复是否有效 | 恢复后的 16 个确定性输出 token 与初始请求相同 |
| 是否清理干净 | 脚本无清理错误、无所属容器，应用显存回到初始 3.5 MiB 且无 GPU 进程。独立后检 `nvidia-smi` 显示 0 MiB、无计算进程及本轮网络/Ray 进程 |

原始证据位于本地 `env-service-verification/results/gpu-sharing-20261027/`，包含执行脚本快照、源码校验和、配置、输出、退出码、失败过的初轮 CPU 回归日志及独立清理记录，结果目录默认不随仓库发布。该目录名用于定位证据，不是新的 baseline。

任务遥测报告包含缺少 host procfs 的归属错误，所以没有以其缺失数值计算模型/环境各自的显存峰值或利用率。命令准入没有因此获得额外归属抵扣，最终物理释放由独立整卡检查补充确认。这轮证明命令协议和真实模型/容器共存，不证明长时间同时运行计算内核时无性能干扰，不证明完整 PSRL 训练、NIXL 权重同步或全链路 SMG 路由已通过，也没有触发 KV 耗尽来实测重算。没有比较 Science 训练吞吐，没有重新定义固定历史 baseline。

新增的 7 项共享协议 CPU 回归全部通过。初轮唯一失败是测试误读了准入返回的租约副本，改为检查服务端实时租约后通过，未为此修改调度逻辑。实机结束后另补充了“归还回调不能把已经隔离的超时命令改成成功”的竞态保护，仅通过针对性 CPU 回归验证，不为这项失败分支重复加载模型。实验脚本另外修正了初始显存快照与 resident 命令结果的同名字段，原始已执行脚本单独保留。


补充检查：涉及的 12 个 Python 文件通过 Ruff 检查与格式检查，本文和两个知识索引共 156 个本地链接通过路径、章节锚点或行号范围校验。额外全量 `tests/env_service` 在依赖导入阶段长期等待文件系统请求，约 10 分钟后中断，结果为 `no tests ran`，不能写成全量通过。隔离副本中已执行 `pre-commit run --all-files`，仍有既有绘图脚本长行和配置注释长度等错误，没有把这些无关自动修复复制回主目录。

## 11. 计算竞争与控制等待：已经验证到哪里

**显存放得下，只回答“能否同时留在卡上”，没有回答“同时运行会不会变慢”。** 当前共享协议没有给环境划分固定比例的计算单元，也没有部署 MPS。MPS 是 NVIDIA 的多进程服务，用于协调多个 CUDA 进程的执行，不是显存准入公式的一部分。

先把两种影响放在同一张图里。下图上半部分是当前调用关系，下半部分标为外部机制的部分由 CUDA 驱动实现。

```text
[A] vLLM generate ───────────────→ 模型 CUDA 上下文 ─┐
                                                   ├→ [D] 驱动调度 GPU
[B] 环境 command 准入 → Docker → 环境 CUDA 上下文 ─┘      SM / 显存带宽 / 缓存
          │                                            （外部机制，计算竞争）
          └→ 命令结束 → [C] 撤销设备权限、确认物理归还
                                     │
                                     ↓ 释放共享票据、通知等待者
[E] wake_up / pause_for_sync → GpuControlGate → 进入模型控制操作
                                     ↑
                     共享票据未归还时在这里等待
                     已有控制操作等待时，拒绝新共享命令
```

图例：图里每个部分对应的源码位置。

| 图中部分 | 职责与真实入口 |
|---|---|
| A | [vllm_async_server.py:945，`generate`](../../../psrl/workers/gen/vllm_async_server.py#L945)，不使用控制门禁装饰器，因此常驻共享期间可以继续生成 |
| B | [gpu_handoff.py:76，`acquire_env_command`](../../../psrl/workers/gen/gpu_handoff.py#L76) 发共享票据，实际命令准入见 [command_gpu.py:58](../../../psrl/workers/env_service/command_gpu.py#L58) |
| C | [command_gpu.py:215，`_return`](../../../psrl/workers/env_service/command_gpu.py#L215) 确认归还，再通过 [gpu_handoff.py:94，`release_env_command`](../../../psrl/workers/gen/gpu_handoff.py#L94) 释放票据 |
| D | CUDA 驱动的外部执行机制，PSRL 没有 SM 分区实现，不能给这个节点配一个伪造的本地调度函数 |
| E | [wake_up:735](../../../psrl/workers/gen/vllm_async_server.py#L735)、[pause_for_sync:770](../../../psrl/workers/gen/vllm_async_server.py#L770) 经 [GpuControlGate.operation:21](../../../psrl/workers/gen/gpu_handoff.py#L21) 等待独占控制条件 |

这些代码如果有理解上的问题，可以随时追问。

**计算竞争究竟竞争什么。** SM（Streaming Multiprocessor，流式多处理器）是 GPU 中执行大量并行线程的计算单元。CUDA 上下文可以理解为一个进程提交 GPU 工作时使用的执行状态和资源集合。模型进程与 Docker 内的计算进程通常拥有不同上下文。

在常规、未启用 MPS 的多进程执行中，不同上下文通常在 GPU 上轮流获得执行时间。因而不应理解成“系统预先切出 20% 的 SM 永久给环境，剩下 80% 给 vLLM”。启用 MPS 后可以出现跨进程同时调度，但本项目本轮并未部署它。[NVIDIA MPS 架构说明](https://docs.nvidia.com/deploy/mps/latest/architecture.html)区分了同上下文并发、不同上下文的时间片执行以及 MPS 的作用。

环境运行期间，模型可能少拿到执行时间，也可能受显存带宽、缓存及调度开销影响。显存**容量**表示能放多少数据，显存**带宽**表示单位时间能搬多少数据。即使容量还剩很多，频繁读写也可能使带宽繁忙。MPS 也不自动提供完整带宽隔离或模型延迟保证。

结果可能是首个 token 更晚返回、后续 token 间隔变大、每秒生成 token 数下降。若后续环境步骤必须先拿到模型输出，整个任务链也跟着推迟。只看单卡显存余量或者 GPU 利用率不能推导这个损失是多少。

**控制等待在等什么。** 它是 PSRL 主动设置的安全门禁，不要求环境正在大量使用 SM 才发生。模型唤醒会重新建立显存占用，权重同步会改变模型状态。如果环境仍持有共享授权，先等它归还，避免两个控制动作破坏容量假设。已有控制请求排队时，不再发新的共享票据，防止后来环境命令不停延长等待。

例如，环境在第 0 秒开始，第 6 秒结束，第 6.3 秒完成物理清理。模型在第 2 秒要求唤醒，则门禁增加的等待为 `6.3 - 2 = 4.3` 秒，之后才开始模型自身的唤醒。**这是教学假设，不是实测延迟。** 环境这几秒即使只拿着 CUDA 上下文睡觉，等待也存在。默认 30 秒命令期限约束命令服务时间，不能把它宣传成故障下保证 30 秒恢复：清理仍需时间，无法确认释放时仍会保留隔离。

**跟着图读源码。** 先读 B 的共享准入，获得票据后有两条并发分支：A 继续向 vLLM 提交生成，B 执行环境命令。D 在二者提交 GPU 工作时决定实际执行次序。E 若收到唤醒或同步请求，等待 C 释放票据。C 正常归还后通知门禁；超时走 [command_gpu.py:171，`_expire_shared`](../../../psrl/workers/env_service/command_gpu.py#L171) 清理分支。不能把这张图读成先生成完、再执行环境、最后才检查控制请求的串行调用。

**一个可以手算、可以运行的任务依赖例子。** 假设模型工作需要独占 GPU 2 秒，它的输出会启动一个 CPU 任务，后者耗时 10 秒。同时有一个独立环境 GPU 作业，需要独占 GPU 10 秒。

| 教学策略 | 中间过程 | 全部完成 |
|---|---|---|
| 模型先执行 | 模型 0–2 秒，后续 CPU 与环境 GPU 都在 2–12 秒运行 | 12 秒 |
| 环境先执行 | 环境 0–10 秒，模型 10–12 秒，后续 CPU 12–22 秒 | 22 秒 |
| 同时共享，假设重叠期间双方各半速 | 模型 0–4 秒，环境剩余工作独占运行到 12 秒，后续 CPU 4–14 秒 | 14 秒 |

共享比“环境先执行”好，但比“让模型尽早解锁后续 CPU”差。这里的半速是计算方便的假设，不是 CUDA 的固定时间片比例，也不是实测减速。现有 [vllm_env_sharing_demo.py:19](../env-service-verification/scripts/vllm_env_sharing_demo.py#L19) 实现这个教学模型，仅需 Python 标准库：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/vllm_env_sharing_demo.py
```

本次实际运行输出 `model_first_makespan_s=12`、`docker_first_makespan_s=22`、`shared_makespan_s=14`，解释的是依赖关系如何改变总时长。它没有调用 CUDA，不预测某个真实任务会慢几倍。真实门禁的 CPU 回归命令见第 10 节。

**Results：已有实测能证明什么。** 第 10 节的真实 GPU 功能实验中，[probe:49](../env-service-verification/scripts/cuda_driver_probe.py#L49) 只启动一次很小的内核并读回 42，随后 [probe:65](../env-service-verification/scripts/cuda_driver_probe.py#L65) 在 CPU 上 `time.sleep`。实验调用方把保留时间设为 6 秒，并在命令活动期间提交短生成。

| 问题 | 已有证据 | 结论边界 |
|---|---|---|
| 两个进程能否持有 GPU 资源，模型能否继续生成 | resident 命令成功，模型返回少量 token | 已验证功能共存 |
| 环境持续高负载计算时，模型慢多少 | 没有持续计算负载与等工作量 A/B 延迟数据 | **未验证计算竞争幅度** |
| 未归还时唤醒是否被阻挡 | [实验脚本:147](../env-service-verification/scripts/real_gpu_sharing.py#L147) 等 0.2 秒后检查唤醒尚未完成，归还后成功 | 已验证等待顺序，**没有完整等待耗时统计** |
| 含初始化和清理约 249 秒能否表示共享开销 | 只有整个功能实验的总时长 | 不能，据此不能声称模型无减速或训练加速 |

若后续要回答性能幅度，应固定模型、输入 token、输出长度、预热和 GPU，分别测模型独占、环境仅持有显存、环境持续计算三组。记录首 token 延迟、生成总时长、输出 token 数、命令耗时，并单独记录控制请求到达至门禁放行的时间。需要触发相同工作量和明确的计算重叠，不能把持有显存的休眠当作计算负载。本次没有启动这项额外模型性能实验。

## 12. MPS、SM 与时分复用：怎样理解多进程共享

**核心结论**：显存共存、不同进程的 GPU 内核重叠、整个任务并发是三件事。普通 CUDA 上下文在同一 GPU 计算引擎上采用时间片调度，MPS 可使不同进程的工作具备并发执行机会。但时间片调度不等于一个任务跑到结束才轮到下一个，MPS 也不保证所有负载加速。

**先从一条真实命令理解，2026-10-29 补充。** 用户在容器 A 运行 Python，Python 仍是 CPU 上的进程。当它调用矩阵乘法时，CUDA 库向 GPU 提交计算内核，GPU 执行后把完成状态返回。容器 B 也可以做同样的事。Router 决定是否允许这两份工作进入，驱动决定它们提交的内核何时执行。容器进程存活、命令已提交、数据还在显存中，都不意味着这个进程此刻正在 SM 上计算。

```text
EnvRouter 容量准入 → 容器 A/B 获准执行
                         ↓
任务 A 进程 → CUDA 上下文 A ─┐
                            ├→ 驱动 / GPU 计算调度 → SM 执行内核
任务 B 进程 → CUDA 上下文 B ─┘
      │
      └→ A、B 的数据可以同时常驻显存，不必随每次计算切换搬走

普通上下文：计算时间片 A → B → A → B
启用 MPS ：满足资源条件时，A 与 B 的内核可在同一时间推进
EnvRouter：在上述执行之前检查容量、决定允许或等待，结束后核验归还
```

图中上下文与 MPS 属于 NVIDIA 驱动机制，不是 PSRL 自己实现的 SM 调度器。图例与源码对应如下：

| 图中部分 | 对应实现或资料 |
|---|---|
| 容量检查与授权 | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L58)、[显存记账](../../../psrl/workers/env_service/gpu_memory.py#L9) |
| 模型进程提交生成 | [vLLM generate](../../../psrl/workers/gen/vllm_async_server.py#L945) |
| 模型控制与归还 | [共享票据](../../../psrl/workers/gen/gpu_handoff.py#L76)、[命令归还](../../../psrl/workers/env_service/command_gpu.py#L215) |
| 上下文时间片与 MPS | [NVIDIA 架构说明](https://docs.nvidia.com/deploy/mps/595/architecture.html)，属于外部机制，PSRL 无对应 SM 调度源码 |

跟着图读源码：先沿容量检查取得命令授权，随后环境与模型分别提交工作，由驱动决定计算的实际交叠。命令结束后沿归还路径释放票据，唤醒或权重更新才可继续。模型生成与环境执行是并发分支，不是同一个 Python 顺序调用。具体函数可继续追问，第 11 节给出了更完整的控制等待关系。

**显存和 SM 分别是什么。** 显存保存模型权重、输入、输出和中间张量。SM 是 Streaming Multiprocessor，中文常称流式多处理器，是执行大量 GPU 线程的计算单元。kernel 即内核，是提交到 GPU 上执行的一段程序，其线程块被调度到 SM。两个进程各占 20 GiB 显存，只说明数据同时存得下，不说明计算各拿到一半 SM。显存带宽还决定每秒能搬多少数据，它与容量、SM 数量也不同。

**“一个时刻只有一个进程跑”应怎样限定。** NVIDIA 对普通 CUDA 上下文的说明是：不同上下文提交到同一计算引擎的内核不能并发，采用时间片调度。同一上下文内的不同 stream 则可能并发。stream 是按顺序提交 GPU 操作的队列。不同进程通常各有自己的上下文，所以 Docker 隔离不会消除这层差异。以上不是说整个机器只运行一个进程，也不是说其他进程无法进行 CPU 工作、I/O 或保留显存，更不表示等待一个完整 episode 结束才切换。[官方依据](https://docs.nvidia.com/deploy/mps/595/architecture.html)

**MPS 改变的是跨进程提交与调度条件。** MPS 是 Multi-Process Service，即 NVIDIA 多进程服务。客户端仍是独立进程，现代 MPS 客户端有各自 GPU 地址空间，并可直接提交工作，由服务协调共享资源，不应理解成所有运算都经过一个 Python 中转队列。资源条件合适时，不同客户端的内核可以重叠。若每个内核单独已经吃满计算或带宽，重叠不代表凭空多出吞吐。[架构说明](https://docs.nvidia.com/deploy/mps/595/architecture.html)

**MPS 到底是一个什么东西？** 它是一套 NVIDIA 提供的软件运行服务，不是一块额外 GPU，也不是 PyTorch 的 batch 参数。控制程序 `nvidia-cuda-mps-control` 负责服务启动和连接协调，`nvidia-cuda-mps-server` 协调客户端共享 GPU 调度资源，应用通过 CUDA 驱动库 `libcuda.so` 接入。适用的程序通常不需要重写计算内核，但须在 CUDA 初始化时正确接入服务。服务运行在哪里与最终算在哪张物理 GPU 上，是两件事。[官方组件说明](https://docs.nvidia.com/deploy/mps/latest/index.html)

把“时间片”理解为轮流获得计算机会即可，不要假定固定几毫秒切一次，也不要假定轮到 A 就能用满 GPU。教学假设 GPU 有 100 个 SM，A 的内核只有足以用 20 个 SM 的并行工作，B 也只有 20 个，显存与带宽均足够：

```text
普通独立上下文，示意而非固定时间片长度：
时间段一：A 用 20 个 SM，另 80 个没有被这两个任务利用
时间段二：B 用 20 个 SM，另 80 个没有被这两个任务利用

MPS 允许有效重叠时的一种可能：
同一时间：A 用 20 个 + B 用 20 个，仍剩 60 个
```

MPS 的价值是在有合适工作时填补余量，而不是给每个程序复制整张卡。若 A、B 各自已经需要全部 100 个 SM，并且始终算满，总计算能力没有增加。两者同时推进可能各自更慢，总完成时间未必缩短。普通时间片切换也不等于把 A 的全部显存卸载到 CPU，因此不能把它与 episode 的 offload/reload 混为一谈。

**“分 SM”需要区分两种机制。**

| 方式 | 实际含义 | 不应推导成什么 |
|---|---|---|
| MPS 动态执行额度，如 active thread percentage | 限制某个客户端可使用的执行资源上限，减少其影响范围 | 50% 不等于永久独占固定的一半 SM，也不保证半速或一半显存带宽 |
| MPS 静态 SM 分区 | 支持的版本和硬件上，显式为客户端指定 SM 分区 | 不是旧版百分比开关的同义词，也不等于显存、带宽等全部资源都隔离 |

官方文档明确动态额度不预留专属资源，不同客户端的工作仍可能落到同一个 SM。CUDA 编程指南记载从 CUDA 13.1 开始支持显式 MPS 静态分区。是否能用要核对实际驱动、工具及 GPU 支持，不能把新文档能力直接当成本机已安装能力。[MPS 资源配置](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)、[CUDA 分区机制对比](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)

**为什么不能直接说“与 Router 串行排队一样快”。** 下面全部是教学假设，忽略切换和启动成本：

| 负载 | 整任务串行 | 允许任务共存 | 原因 |
|---|---|---|---|
| A、B 各持续占满 GPU 计算 5 秒 | 总计 10 秒 | 公平时间片下总计仍约 10 秒，可能额外有切换成本 | 没有空闲计算能力可挖掘，单个任务延迟也可能变大 |
| A、B 各做 GPU 2 秒、CPU 6 秒、GPU 2 秒 | 总计 20 秒 | 不需要内核重叠，也可排成 12 秒 | A 做 CPU 时 GPU 可交给 B，等待阶段能重叠 |
| A、B 各有一个只用部分 SM 的 5 秒内核，且其他资源够用 | 10 秒 | MPS 若使二者有效重叠，理想情况下接近 5 秒 | 填补单个上下文不能利用的计算余量，真实收益必须测量 |

第二个例子的合法时间线是：A 的 GPU 阶段 0到2秒、CPU 阶段2到8秒、GPU 阶段8到10秒。B 的 GPU 阶段2到4秒、CPU 阶段4到10秒、GPU 阶段10到12秒。任何时刻都没有两个 GPU 阶段同时执行，但两条任务比整任务串行更早完成。

这里还取决于 Router 排队粒度。如果 Router 已经在每次 GPU 阶段完成后让位，它也可以得到上述 12 秒调度，不应把这个收益误算成 MPS 独有。若一个 GPU 命令进程内部包含长时间 CPU 工作，Router 持有整条命令的额度，与驱动只在有实际 GPU 工作时执行的时间片调度又不同。比较前必须说明排队保护的是会话、命令还是内核。

**这对当前 Env 结果意味着什么。** 高压力训练的主要结果是 OOM 从 18 次降到 0 次，减少失败补采。它不要求两个环境内核同时运行，因此没有 MPS 不会推翻这项结论。低压力共享能否进一步提吞吐、模型会慢多少，则没有得到充分证明。已有共享 probe 只启动小内核，随后持有显存并在 CPU 上 sleep，不能据此声称持续计算时内核重叠或双方无减速。

对 Router 可以把问题分为三个层次：第一层是容量准入，保证不重复承诺同一份显存。第二层是进入后的计算共享，普通驱动时间片或 MPS 决定内核怎样推进。第三层是结束后的归属与清理，确认本任务退出后其他工作仍能继续。只开启 MPS 不会自动完成第一和第三层。当前 command 预算不够只放行一条时，MPS 也没有第二个已获准任务可重叠。

**接入 MPS 对当前代码还有实际适配工作。** 本地 [task_telemetry](../../../psrl/workers/env_service/task_telemetry.py#L25) 使用 GPU 进程身份做任务归属，[归还检查](../../../psrl/workers/env_service/command_gpu_node.py#L207) 检查所属进程、设备句柄与 GPU 活动。MPS 监控可能把活动归到共享 server，不能将 server 的全部占用算到某个容器，也不能因容器 PID 不在 GPU 列表中就判定已释放。这是结合源码与官方监控说明得出的接入风险，尚非本机故障实测。还需对照 MPS 客户端终止协议验证超时取消，不能将直接杀容器天然视为对其他客户端无影响。官方提供定向终止客户端上下文的机制，跨容器时需使用服务所见的进程编号。[监控与客户端终止要求](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)

**Docker 内可以使用 MPS，也可以运行 MPS 服务。** NVIDIA 提供了服务和客户端分别运行在容器里的示例。因此答案不是“容器一定不能开”，而是要把 GPU 设备、用户身份和进程间通信配置完整。IPC 是进程间通信，IPC namespace 是这类资源的隔离视图。服务与客户端需要能找到并使用彼此的通信资源。

```text
同一节点、同一物理 GPU
  ├─ MPS 服务：可以在宿主机，也可以在专用服务容器
  ├─ 任务容器 A ─┐
  └─ 任务容器 B ─┴→ 连接同一个 MPS 服务
                    共享所需通信目录与 IPC 视图
```

两种容器用法分别是：

- **容器接入已有服务**：把服务通信目录挂入任务容器，正确设置 `CUDA_MPS_PIPE_DIRECTORY`，匹配用户身份及 GPU 可见性，并按部署方式配置 IPC。不是在每个任务容器内各开一个服务就会自动合并。
- **在专用容器里开服务**：服务容器运行 MPS 工具，任务容器连接它。NVIDIA 的 Compose 示例使用共享通信卷和 `ipc: container:mps-daemon`。该示例较旧，仅用于证明确有这种拓扑，不能原样当作当前集群配置。

官方示例还将“修改 GPU 计算模式”放在单独具备额外权限的容器中，这说明管理 GPU 模式的权限与运行 MPS 服务不是同一件事。单用户 MPS 有普通用户启动路径，容器内 root 则不等于宿主管理权限。是否适用于腾讯节点，仍需检查实际驱动、工具、设备和平台授权。本次没有连接节点核验或启动服务。[NVIDIA 容器示例](https://gitlab.com/nvidia/container-images/samples/-/raw/master/mps/docker-compose.yml)、[单用户部署](https://docs.nvidia.com/deploy/mps/common-tasks.html)、[通信目录](https://docs.nvidia.com/deploy/topics/topic_5_2_2.html)

MPS 不是“装了 CUDA 就必定已开启”。需要服务已启动且客户端真正接入。平台可能预配置服务，也可能没有，所以需现场核对版本、服务进程、客户端连接、容器目录挂载与用户身份。容器内看不到宿主服务进程也不能单独证明没开。只设置一个额度环境变量或看到两条 nvidia-smi 进程记录，都不是接入或内核并行的证据。[官方启动与接入检查](https://docs.nvidia.com/deploy/mps/quick-start.html)

**可运行的 CPU 教学计算。** 以下只检验上面的时间线算术，不模拟 CUDA 时间片长度、不检测 MPS、更不是性能实测：

```bash
python3 - <<'PY'
# 图中的 A/B：内存可以共存，GPU 计算阶段仍不重叠。
gpu, cpu = 2, 6
whole_task_serial = 2 * (gpu + cpu + gpu)
a_first_end = gpu
b_first_end = a_first_end + gpu
a_second_start = max(a_first_end + cpu, b_first_end)
a_end = a_second_start + gpu
b_second_start = max(b_first_end + cpu, a_end)
b_end = b_second_start + gpu
print("whole_task_serial_s:", whole_task_serial)
print("interleaved_without_kernel_overlap_s:", b_end)
assert (whole_task_serial, b_end) == (20, 12)
# 满负载的理想下界，与未占满时 MPS 的理想重叠假设分开。
print("two_saturated_jobs_total_work_s:", 5 + 5)
print("ideal_small_kernel_overlap_s:", max(5, 5))
PY
```

输出为 20、12、10、5。最后的 5 依赖两个小内核确实能有效重叠，不是启用 MPS 后必然达到的结果。后续若做实机对照，应分别测单任务、Router 串行、无 MPS 多进程、MPS 多进程，固定成功工作量并记录总时间、单任务响应和内核时间线。小内核与持续满载应分开，模型与环境的依赖也要保留，不能只看整卡利用率或显存占用。
