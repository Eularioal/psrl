# Docker 借用 rollout GPU：Ray 配额、vLLM sleep/wake 与归还协议

> **核心结论**：不必把 Docker 与训推 GPU 永久分成两个互斥资源池。可以让 Ray 保留 GPU 的逻辑归属，在 vLLM 睡眠期间把同一批物理卡借给 Docker。但需要统一的借还状态机，保证容器确实释放资源后才允许唤醒。当前已接入受控的 science 命令批次与模型调用交替借卡，详见[实现与验证](#12-命令批次与模型调用交替借卡的实现)。CPU 回归覆盖交接协议；2026-10-15 已完成两轮真实同卡 Docker CUDA / vLLM 交接，详见[GPU 结果](#14-真实-gpu-验证结果2026-10-15)。该实验未接入训练侧 NIXL 和完整 SMG 路由。
>
> 本文更新日期：2026-10-19。第 15 节为普通任务阶段租约、抢占与预测回填的设计分析及 CPU 教学模拟，其中命令阶段协议的落地实现见[命令阶段配额](command-gpu-reservations.md)。第 1–11 节保留实现前的设计分析，其中“拟新增/尚未实现”描述的是当时的状态，当前接入范围以第 12 节为准。第 12 节记录实现与 CPU 回归，第 13 节解释 unit 配额，第 14 节记录真实 GPU 验证；各节的验证时间与范围不可混用。本文补充[现有 Env Service 讲解](env-service-implementation.md)中尚未实现的借还协议，单独展开其跨调度器约束；原文主要描述静态 unit，不应把其中早期“没有实机验证”的状态当作今天的总结果。

> 当前能力及调用图见[第 16 节](#16-当前交接能力与-procfs-报错的关系)：显式借还已实现，但不同于普通 command 配额释放，不会自动卸载任意环境 CUDA 状态。任务 procfs 归因报错也不能直接等同于交接协议不可用。

## 目录

- [1. 全局流程与源码图例](#1-全局流程与源码图例)
- [2. 已有实验究竟证明了什么](#2-已有实验究竟证明了什么)
- [3. 为什么不必永久分卡](#3-为什么不必永久分卡)
- [4. sleep 和 wake 到底改变什么](#4-sleep-和-wake-到底改变什么)
- [5. 当前实现具备什么与缺什么](#5-当前实现具备什么与缺什么)
- [6. 建议的借还协议与数字例子](#6-建议的借还协议与数字例子)
- [7. science 多轮任务的等待陷阱](#7-science-多轮任务的等待陷阱)
- [8. 跟着图读源码](#8-跟着图读源码)
- [9. 可运行的 CPU 教学模拟](#9-可运行的-cpu-教学模拟)
- [10. 最小实机验收边界](#10-最小实机验收边界)
- [11. 参考资料与结论边界](#11-参考资料与结论边界)
- [12. 命令批次与模型调用交替借卡的实现](#12-命令批次与模型调用交替借卡的实现)
- [13. 分配粒度和 unit 配额](#13-分配粒度和-unit-配额)
  - [配置为什么需要事先确定](#配置为什么需要事先确定)
  - [每个字段是在规定什么](#每个字段是在规定什么)
  - [配额怎样算出能放几个任务](#配额怎样算出能放几个任务)
  - [14 GiB 任务剩下 13 GiB 是不是 unit 太粗](#14-gib-任务剩下-13-gib-是不是-unit-太粗)
  - [实际配置时怎么定这些数值](#实际配置时怎么定这些数值)
  - [纯 CPU 可运行例子](#纯-cpu-可运行例子)
- [14. 真实 GPU 验证结果（2026-10-15）](#14-真实-gpu-验证结果2026-10-15)
- [15. 普通任务的阶段租约与预测回填](#15-普通任务的阶段租约与预测回填)
- [16. 当前交接能力与 procfs 报错的关系](#16-当前交接能力与-procfs-报错的关系)

## 1. 全局流程与源码图例

先串起“谁决定、谁执行、谁确认”。**下面是接入方案图，不是已经实现的完整调用链**；标有“拟接入”的箭头是缺口。

```text
                       Ray 的 GPU 配额 / placement group 始终保留
                                            │
[1] ElasticExecutor + GPU 拓扑 ──SLEEP──> [2] RolloutCoordinator
       统一决定卡的用途（拟扩展）                 │ 停派请求，处理在途请求
             │                                 v
             │                         [3] vLLM server / 全部 GPU ranks
             │                           sleep ACK + 实测释放显存
             │                                 │
             ├────拟接入：借用许可 <──────────────┘
             v
[4] EnvRouter / Gate A ──lease──> [5] science runner / Harbor / Docker
       预算检查 + 拟新增借用状态             指定节点、物理 GPU，执行工具
             ^                                  │
             │                              工具结束 / 撤销借用
             │                                  v
             └────拟接入：归还确认──── [6] 容器清理 + 节点物理资源检查
             │
             └────无借用者、显存足够──> [3] wake + NIXL 内存重新注册
                                                │
                                  [7] 恢复权重 / 确认模型版本
                                                │
                                                └──> [2] 恢复派发
```

图例：图里每个部分对应的源码位置。

| 编号或箭头 | 职责及当前实现 | 源码位置 |
|---|---|---|
| 1 | 弹性执行器按角色、实例管理睡眠/唤醒；尚未纳入 Docker 借用者 | [`ElasticExecutor._scale_up_instance`](../../../psrl/utils/elastic_rm/elastic_executor.py#L516)、[`_scale_down_instance`](../../../psrl/utils/elastic_rm/elastic_executor.py#L644) |
| 1 的 GPU 拓扑 | 维护实例对应的 GPU 集合，以及 ASLEEP/AWAKEN 两态 | [`GPUSlot` / `ClusterTopology`](../../../psrl/utils/elastic_rm/cluster_topology.py#L17) |
| 1 → 2 → 3 | SLEEP 关闭协调器派发，WAKE_UP 调用引擎后恢复派发 | [`command_loop.py`](../../../psrl/workers/gen/rollout_coordination/command_loop.py#L151)、[`_do_sleep_instance`](../../../psrl/workers/gen/rollout_coordination/coordinator.py#L318) |
| 3 | 引擎 sleep/wake；NIXL 关联的内存注销与重新注册 | [`vllm_async_server.py`](../../../psrl/workers/gen/vllm_async_server.py#L726)、[`nixl_wake_up` / `nixl_sleep`](../../../psrl/workers/gen/vllm_async_server.py#L1285) |
| 4 | 静态 unit 的准入预算；借用许可目前没有实现 | [`EnvServiceClient.acquire`](../../../psrl/workers/env_service/client.py#L159)、[`EnvUnitLedger._fits_now`](../../../psrl/workers/env_service/ledger.py#L376) |
| 4 → 5 | 先申请 lease，再把 GPU 和标签写入 Compose overlay，交给 Harbor | [`runner.py`](../../../examples/sciaccel_rl/runner.py#L344)、[`_apply_env_lease`](../../../examples/sciaccel_rl/runner.py#L99)、[`build_lease_overlay`](../../../psrl/workers/env_service/overlay.py#L139) |
| 5 → 6 | runner 的 finally 清理 Compose 后释放 lease；节点服务负责孤儿清理 | [`runner.py`](../../../examples/sciaccel_rl/runner.py#L535)、[`EnvNodeService.cleanup`](../../../psrl/workers/env_service/node.py#L47) |
| 6 → 1 → 3 | 以物理清理确认阻止过早 wake | **拟新增协议，无现成连线**；现有清理函数本身不阻止 ElasticExecutor 唤醒 |
| 3 → 7 | NIXL 重新注册内存；从参数服务器拉权重 | [`nixl_register_after_wake_up`](../../../psrl/workers/gen/vllm_extension.py#L178)、[`pull_model_for_sync`](../../../psrl/workers/gen/vllm_async_server.py#L1378)、[`pull_model`](../../../psrl/workers/gen/vllm_async_server.py#L1424) |

这些代码如果有理解上的问题，可以随时追问；第 8 节还按执行路径列出阅读顺序。图中的借用控制应落在已有调度职责中，不能让 EnvRouter 和 ElasticExecutor 各自独立宣布同一张卡可用。

## 2. 已有实验究竟证明了什么

以下均为 **2026-10-13 留存的历史实机结果**，不是本次重新运行。总入口是[验证报告](../env-service-verification/README.md)与[汇总 JSON](../env-service-verification/results/fixed-validation/summary.json)。

| 证据 | 已验证的结论 | 没有证明的结论 |
|---|---|---|
| [GPU oracle](../env-service-verification/results/fixed-gpu-oracle/evidence.json) | Harbor 容器绑定目标 GPU，能执行 CUDA 小内核；GPU index 与设备 minor 不一致时仍正确 | 没有同时运行 vLLM 并完成借还 |
| [一卡多容器](../env-service-verification/results/fixed-gpu-shared/evidence.json) | 两个容器看到同一个 GPU UUID，均执行小内核并返回 42；第三个申请等待 | 第三个等待主要证明容器数量上限；没有证明大显存任务共存，也没有硬性显存隔离 |
| [资源边界](../env-service-verification/results/fixed-admission-boundaries/summary.json) | CPU、主机内存、显存预算的独立准入边界 | 预算声明无法限制任意 CUDA 程序的实际峰值 |
| [跨节点检查](../env-service-verification/results/fixed-multinode/evidence.json) | Gate A 读取目标节点占用；Gate B 查询远端执行节点，受限等待约 2.46 秒 | 不代表容器和 vLLM 的共同调度协议已经存在 |
| [故障清理](../env-service-verification/results/fixed-crash-recovery/evidence.json) | 租约过期、实际 worker 被杀后，带归属标签的容器可清理，含 verifier / sidecar | 没有覆盖借用途中控制器重启、失联节点恢复和 wake 竞争 |
| [EnvWorker GPU](../env-service-verification/results/fixed-env-worker-gpu-v3/evidence.json) | Path A 的 worker GPU 身份、CUDA 执行与清理 | 不代表 Path A 与 Harbor 已统一后端或共享一个借还账本 |
| [单步训练 v6](../env-service-verification/results/fixed-one-step-training-v6/training-evidence.json) | 退出 0，轨迹进入训练，两个 rank 执行训练和权重推送；有效 token 为 44,697 | **不证明 GPU Docker 与 rollout 同卡**；也不证明模型效果提升 |

特别核对了 v6 原始配置日志（本地资料，未随仓库发布）：`node_policy=cpu_only`，unit 名为 `science-cpu`，没有 `gpu_indices`。所以这轮是 **CPU science 容器 + GPU 训推** 的功能链路。不能把“真实 Docker 与真实训练一起跑过”写成“它们共享 GPU 已通过”。

v6 的首步学习率非零，但训练采用的两条 `reward_repair` 都为 0，因此策略梯度为 0。它回答原来的崩溃修复和执行链路问题；本次 GPU 调度问题不需要再训练来补一个非零梯度。v7 已按用户要求停止。

## 3. 为什么不必永久分卡

先区分两个容易混在一起的东西：

- **Ray 的逻辑资源**：调度账本，决定一个 actor（持续运行的远程进程）能否被安排到某台节点。`num_gpus=1` 是申请一个 GPU 份额。placement group 是把一组资源一起预留的机制，便于多卡实例成组启动。
- **GPU 的物理状态**：此刻哪些 CUDA 进程持有显存、提交计算。Docker 通过设备挂载或 GPU 运行时让容器进程访问设备，通常不会自动向 Ray 登记它的显存使用。

例如一张卡已由 Ray 预留给 vLLM。vLLM sleep 后，进程和 Ray actor 仍活着，资源组也仍存在，因此 **Ray 的 GPU 配额仍被占用，但物理显存可以空出大部分**。Docker 若获准访问这张卡，就能使用空出的物理资源。反过来，再创建一个申请 `num_gpus=1` 的独立 Ray actor，仍可能因为逻辑配额没有释放而一直排队。

这正是“可以共享”的基础，也是“不能只改配置就放开”的原因：两边并没有天然共享的内存准入账本。[Ray 官方资源说明](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)明确区分逻辑配额和物理使用限制；GPU 可见性通过 `CUDA_VISIBLE_DEVICES` 设置，不能据此认为其他经授权打开设备的进程无法使用它。

| 调度方式 | 同一张卡能否被两边使用 | 适用判断 |
|---|---|---|
| 永久分池 | 各用各的卡 | 实现最直接，但闲置卡不能借出 |
| 按阶段借用 | 推理睡眠时给 Docker，归还后恢复推理 | **本场景优先考虑**；资源池不永久互斥，交接仍需协调 |
| 同时驻留、共同计算 | vLLM 和 Docker 都常驻同卡 | 可以进一步考虑，但必须容纳两边峰值显存，并评估计算、带宽干扰 |

“不互斥”若指**不永久分卡**，答案是可以；若指**双方随时分配显存、独立唤醒且无需协调**，答案是不成立。`num_gpus=0.5` 只改变 Ray 的份额记账，不会给进程硬切一半显存，也不会让 Docker 自动遵守另一半。

## 4. sleep 和 wake 到底改变什么

**rollout** 指让策略模型生成回答或多轮轨迹。vLLM instance 是执行这些生成请求的一套推理引擎，可以由多张 GPU 协作组成。

模型驻留 GPU 时，除了权重，还会保存 **KV cache（注意力键值缓存）**：模型把已经处理过的 token 的中间信息存下来，下次生成时复用。比如对话已有 100 个 token，缓存允许下一步复用这 100 个的结果；丢掉缓存后，恢复时可能需要重新处理对话前缀。

| 操作 | 物理资源与模型内容 | 借用的影响 |
|---|---|---|
| 暂停派发 / pause | 可以停止接新计算，但不能据此推断显存已释放 | 不是可借用确认 |
| level 1 sleep | 权重转到主机内存，KV cache 丢弃 | 适合稍后恢复同一模型；主机内存也要有预算 |
| level 2 sleep | 权重和 KV cache 内容丢弃；部分模型 buffer 保存到 CPU | 恢复显存分配后仍需重新装载有效权重 |
| wake | 重新提供相关显存，恢复可恢复的内容 | 不能越过 Docker 的归还检查；level 2 还需恢复权重 |

这与 [vLLM 官方 sleep 说明](https://docs.vllm.ai/en/latest/features/sleep_mode/)的基本语义一致。官方最新文档包含更多选项，不能直接假定本地版本都支持。

本次实际读取的共享环境版本是 `vllm 0.22.1.dev0+g0b3ba88f1.d20260729.cu128`、`ray 2.49.1`。本地 vLLM 有 CUDA graph 和通信器的额外处理，因此不承诺“sleep 后显存为零”或固定切换耗时。应以目标节点逐卡遥测确认可用容量。该依赖来源只是本次源码检查，不是历史实机运行依赖完全相同的证明。具体实现可读本地 vLLM GPU worker 的 sleep/wake（本地资料，未随仓库发布）；此链接指向工作区外的共享依赖，不是本仓库自带文件。

PSRL 的 [`_get_sleep_level`](../../../psrl/workers/gen/rollout_coordination/coordinator.py#L318) 当前返回 **2**。裸 `nixl_wake_up()` 只唤醒分配并重新注册内存，没有顺带调用权重拉取。**此外，`pull_model_for_sync()` 在本地版本号已达到目标版本时会提前返回**：如果 level 2 已丢弃权重，而版本号仍是 7，再要求“同步到 7”可能跳过真实复制。借用实现必须增加“权重内容有效”的状态，或显式强制重载，不能只比较版本数字。这是源码可见的接入风险；本次没有把它宣称为已实机触发的故障。

**NIXL** 是这里用于跨进程/节点传输模型张量的通信库。它需要先登记可传输的内存区域。sleep/wake 后虚拟地址可能没变，背后的物理显存页已经换了；就像同一个门牌换了住户，旧的访问登记不能继续用。所以应复用 [`nixl_register_after_wake_up`](../../../psrl/workers/gen/vllm_extension.py#L178) 这条路径，而不是仅调用裸 vLLM wake。当前 PSRL `nixl_sleep` 的实际顺序是先 `sleep`、再 `nixl_deregister`；本文没有把它写成相反顺序。

## 5. 当前实现具备什么与缺什么

**已经具备的部分**是 ElasticExecutor 的实例拓扑与睡眠/唤醒、EnvRouter 的容器准入账本、Docker 的设备映射、租约标签及清理、目标节点遥测。缺口是让两侧认同一个 GPU 的当前用途，并对借出、撤销、唤醒作出统一承诺。

当前 [`node_policy=cpu_only`](../../../psrl/trainer/config/psrl/env_service.yaml#L30) 按**节点**拦截 GPU episode：只要节点承载训推，就不让 GPU 容器过去。因此它甚至比“逐卡隔离”更粗。`any` 和 `prefer_other` 可以放宽节点资格，但它们没有自动加入 sleep ACK、撤销借用或 wake 前检查。仅改成 `any` 不能得到安全复用。

**Gate A** 是容器启动前的准入闸门。当前显存规则为 `max(现有租约声明总量, 实测总占用) + 新需求 <= unit 预算`。它能拦截不少已知占用，但看不到 vLLM 下一刻 wake 的分配，也不为并行发生的外部显存增长预留完整未来峰值。测量缺失/过期时会退回声明值；借用 GPU 时应要求有效观测，不能把缺失当作卡空。多卡 unit 当前使用显存总和，未来借用还需要逐卡约束，不能靠“另一张卡有富余”弥补这一张卡已经放不下。

**Gate B** 是延迟模型回复的软节流，源码在 [`_wait_for_gpu_idle`](../../../psrl/workers/gen/session_router.py#L120)。共享卡模式只看利用率，查不到遥测或等待超时会放行。它不能证明显存已经释放，也不控制 ElasticExecutor 的 wake，因而不能充当借还锁。

还要区分两条环境执行路径：

- **Harbor / Env Service 路径**：runner 先拿 lease，再启动 Docker。目前可以在此增加借用许可与撤销接口。
- **Path A / EnvWorker 路径**：当前 [`EnvWorkerManager`](../../../psrl/workers/env_worker/manager.py#L74) 按 `num_gpus=float(gpu_slots)` 创建 worker。vLLM 保留配额时，这样的独立 GPU worker 未必能被 Ray 调度。若统一借用，应由已有 GPU 所有者委托启动容器，或使用无需再次申请 GPU 配额的节点控制 actor；其物理 GPU 访问必须受同一个借还协议约束，不能只把 `num_gpus` 改成 0 就绕过管理。

## 6. 建议的借还协议与数字例子

**以下为设计提案，未接入生产代码。** 保留 Ray 资源组作为整套实例的逻辑所有者，把 Docker 当作这个所有者允许的临时借用者。扩展已有 ElasticExecutor / 拓扑，使所有唤醒入口都必须确认没有冲突借用；EnvRouter 只能在获得明确许可的 GPU 上准入。训练、验证等其他会使用同卡的角色也必须遵守这份状态，不能只拦住某一个 wake API。

```text
ROLLOUT → DRAINING → LENDABLE → ENV → RETURNING → WAKING → ROLLOUT
           停派/收尾   释放确认    借用   禁止新借用    恢复权重后才接请求
任何阶段状态不明或失败 → QUARANTINED（隔离待核实，不再借出或自动唤醒）
```

租约 **lease** 是一个有身份和期限的借用凭证；**epoch** 是这次借用周期的递增编号，用来挡住迟到消息。例如第 7 次许可已经撤销，第 6 次的心跳或启动请求不能重新获得权限。`TTL` 是凭证超过多久未续期就失效；它只能说明账本该回收，不能证明容器已经退出。

各状态的必要条件如下：

1. **DRAINING**：冻结该实例的新增派发；有界地等待在途请求完成，或通过既有取消/重试机制处理。当前协调器的 SLEEP 还会暂停其路由循环，不应假定已经做到“只停一个实例且完全不影响其他实例”。本地 AsyncLLM.sleep（本地资料，未随仓库发布） 的默认模式是 `abort`，而 PSRL 未传入替代模式；不能把当前 SLEEP 理解成自动无损等待所有生成完成。
2. **LENDABLE**：实例所有参与进程 sleep 成功，逐卡容量符合预算，拓扑没有其他活跃角色占用，才发布带 epoch 的借用许可。
3. **ENV**：EnvRouter 同时检查许可和每卡显存预算，可在同一张已借出的卡上安排多个小容器。许可还需绑定节点、GPU UUID、期限和本次任务身份；不能把 Ray 局部 GPU 编号直接当成 `/dev/nvidiaN`。
4. **RETURNING**：原子地关闭新借用，再通知现有借用者在期限内结束 GPU 阶段。到期按约定终止本次 GPU 子进程/容器，收回所有相关 verifier / sidecar；只执行 `docker pause` 不能视为显存已归还。
5. **WAKING**：确认借用者退出、没有未完成的 GPU 工作且逐卡显存够恢复后，才调用 wake；恢复内存登记、有效权重与正确版本，最后恢复模型请求。

控制器重启或网络失联后，必须先核对节点上仍然活着的进程/容器，再恢复权限。仅更换 epoch 不能让旧 CUDA 进程消失。多卡实例的 **TP（张量并行）** 表示同一个模型运算切到多个 GPU 合作执行；例如 TP=2 的两张卡要一起进入睡眠和唤醒，不能睡一张就当整个实例可借。借出时可以把空出的卡分别分给小任务，但归还要满足整个实例的恢复条件。

**教学假设，非实测：** 有两套单卡 rollout 实例 R0、R1，每卡 96 GiB。R0 继续服务，R1 平时占 82 GiB、sleep 后残留 2 GiB，安全余量取 8 GiB。两个 Docker GPU 阶段各需 24 GiB。

| 时刻 | R1 所在卡的状态 | 手算与结果 |
|---|---|---|
| 起初 | R1 醒着，空闲利用率也可能是 0% | 若直接塞入一个容器：82 + 24 + 8 = 114 > 96，放不下 |
| 睡眠确认 | Ray 仍保留该卡；物理占用只剩假设的 2 GiB | 借给两个容器：2 + 24 × 2 + 8 = 58 ≤ 96 |
| 模型需要扩容 | 先关闭新借用，两个容器结束 GPU 阶段 | 账本清空后还要确认物理显存回到允许基线 |
| 恢复 | R1 wake、恢复权重并重新接请求 | 82 + 8 = 90 ≤ 96；卡没有永久移出 rollout 池 |

若不先归还就 wake，则 `82 + 48 + 8 = 138 > 96`。这说明“sleep 可释放显存”支持借用，但不能省掉归还顺序。

收益也要看时间尺度。**另一组教学假设**：空闲窗口 30 秒，sleep、wake、权重同步及检查共花 5 秒，窗口内最多剩约 25 秒执行环境 GPU 工作；若窗口只有 3 秒，频繁切换可能反而更慢。当前没有测量这套组合的切换成本，不能把借用可行直接等同于吞吐提升。初始策略宜使用足够长的空闲窗口、最短驻留时间和有界 GPU 阶段，避免每个短命令都切一次大模型。

## 7. science 多轮任务的等待陷阱

典型交互是：`模型生成命令 → 容器执行 → 把输出给模型 → 模型生成下一条命令`。当前 runner 在 episode（一次完整任务交互）开始前申请 lease，在结束清理后释放，未按每条 GPU 工具命令拆分。

如果只有一个推理实例，把它睡掉后将 GPU 借给整个 episode，就可能变成：

```text
容器拿着 GPU lease，等模型给下一条命令
模型等容器归还 GPU，才能 wake
容器只有 episode 结束才释放 lease
```

这是**循环等待**：双方各自等对方先释放资源，任务无法前进。即使 GPU 利用率为 0，仍然可能存在这个问题。

有两类可行处理方式：保留其他有能力接管请求的 rollout 实例，并确保会话能路由过去；或者把容器生命周期与 GPU 借用期拆开，在 GPU 工具阶段结束、真实释放 CUDA 资源后归还卡，再向模型请求下一条命令。CPU shell、文件和任务状态可以保留，但后台 GPU 进程若仍持有资源，就不能谎报归还。

因此第一批适合接入的是**不需要在持卡期间反复询问模型、执行时间可界定的 GPU 工具或 verifier**。任意 shell 命令可能起后台任务，不能只因为前台命令返回就认定卡空。若场景只有一个实例、GPU 阶段不可终止且又必须等待模型，那么当前借用条件就不成立，应让它等待其他容量。

## 8. 跟着图读源码

以下串联的是目标执行路径：现有函数可以复用的部分给出源码；缺失的连线明确标为提案。实际系统有 actor RPC（远程过程调用）和多个异步循环，不是一个顺序函数自动完成所有步骤。

| 步骤 → 图中编号 | 文件、函数 | 输入 / 输出或状态变化 |
|---|---|---|
| 1 → 1 | [`register_role`](../../../psrl/utils/elastic_rm/elastic_executor.py#L109)、[`ClusterTopology.register`](../../../psrl/utils/elastic_rm/cluster_topology.py#L68) | 注册实例对应的 GPU 集合；未来补充统一的物理 UUID 映射 |
| 2 → 1、2 | [`_scale_down_instance`](../../../psrl/utils/elastic_rm/elastic_executor.py#L644)、[`SLEEP 分支`](../../../psrl/workers/gen/rollout_coordination/command_loop.py#L151) | 发送 SLEEP 命令；暂停协调器派发；当前包含最小活跃实例数检查 |
| 3 → 3 | [`_do_sleep_instance`](../../../psrl/workers/gen/rollout_coordination/coordinator.py#L318)、[`nixl_sleep`](../../../psrl/workers/gen/vllm_async_server.py#L1291) | 调用 level 2 sleep、注销内存；成功后执行器更新 ASLEEP；尚无向 EnvRouter 发布许可的步骤 |
| 4 → 4 | [`acquire`](../../../psrl/workers/env_service/client.py#L159)、[`try_admit`](../../../psrl/workers/env_service/ledger.py#L105) | 容器需求与本节点显存观测进入 router；当前只检查静态资格/预算；拟新增许可和 epoch 检查 |
| 5 → 5 | [`_apply_env_lease`](../../../examples/sciaccel_rl/runner.py#L99)、[`build_lease_overlay`](../../../psrl/workers/env_service/overlay.py#L139) | lease 中的设备映射成 Docker 配置；Harbor 启动并运行任务 |
| 6 → 6、1 | [`runner 的 finally`](../../../examples/sciaccel_rl/runner.py#L535)、[`EnvNodeService.cleanup`](../../../psrl/workers/env_service/node.py#L47) | 当前清理和 lease 释放；拟新增撤销通知、逐卡物理确认以及超时隔离 |
| 7 → 3、7 | [`nixl_wake_up`](../../../psrl/workers/gen/vllm_async_server.py#L1285)、[`pull_model`](../../../psrl/workers/gen/vllm_async_server.py#L1424) | 恢复显存和传输登记；level 2 必须实际重载权重，即使版本数字未变 |
| 8 → 2 | [`WAKE_UP 分支`](../../../psrl/workers/gen/rollout_coordination/command_loop.py#L160) | 当前 wake 后立即恢复协调器派发；拟改为借用清零且权重有效后才恢复 |

并发边界要特别看：当前 ElasticExecutor 的伸缩循环与 EnvRouter 的准入循环是独立的；直接串几次 API 调用不能保证其他 wake 入口不会插队。需要一个按 GPU 集合序列化状态转移的权威控制者，并要求所有入口遵守它。

## 9. 可运行的 CPU 教学模拟

脚本：[gpu_lending_demo.py](../env-service-verification/scripts/gpu_lending_demo.py)。**这是协议教学模拟，不是生产实现测试，也不是 Ray/Docker/vLLM 的实机替代品。** 使用标准库，固定 1 张虚拟卡、最多 2 个虚拟容器，没有网络、GPU、子进程或真实等待。

从工作区根目录执行：

```bash
python3 docs/env-worker-deep-dive/env-service-verification/scripts/gpu_lending_demo.py
```

`sleep_ack()` 对应图 1→2→3，把物理占用从假设的 82 改为 2；`borrow()` 对应 4→5，只接受当前周期许可；`begin_return()` 先关新借用；`wake()` 要求逻辑租约和物理占用同时允许；`serve()` 要求权重内容有效。

模拟故意制造“账本已释放、物理显存尚未回收”以及“level 2 wake 已分配显存、版本仍为 7 但权重未恢复”两种情况，展示单靠 lease 数量和版本号会漏掉什么。它按顺序模拟这些交错，不验证真实异步竞争、分布式故障、CUDA 释放时序或性能。

本次实际运行输出保存在 [gpu-lending-demo.json](../env-service-verification/results/gpu-lending-demo.json)。实际结果与预期一致：最终回到 `ROLLOUT`，Ray 保留份额始终为 1，模型版本仍为 7；共阻止 7 种不合规则的动作：醒着借卡、第三个容器、过期许可、归还途中新借用、仍有租约时 wake、仍有物理占用时 wake、未恢复权重就推理。模拟正常回到 ROLLOUT 后立即结束。

## 10. 最小实机验收边界

**以下仅为后续实现后的验收建议，本次没有启动。** 问题是“同卡可否安全交接并恢复正确推理”，因此不用完整训练，不需要奖励差异、非零梯度或收敛。

在完成借还实现和节点前置检查后，可固定 **1 张 GPU、1 个已准备好的小模型、1 个已有 CUDA 探针镜像、一次借还循环**。预设总运行预算如 180 秒、不自动重试；模型/镜像未准备好则记录阻塞并结束，不把下载或构建变成长实验。

必要证据是：睡眠前短推理正常；sleep 后显存下降且 Ray 配额仍保留；容器在同 UUID GPU 上完成小内核；借用期间 wake 被协议阻止；清理确认后可恢复同一模型的有效推理；失败路径不遗留容器或本次 GPU 进程。level 2 还应显式检验“版本号相同也实际恢复权重”。前后短推理可使用固定输入和确定性设置辅助判断，不把逐字一致当作所有推理实现的无条件保证。

达到这个边界就停止并清理。这个实验仍不证明 TP 多卡原子交接、控制器崩溃恢复、任意长 GPU 工具的可抢占性或吞吐收益；这些不能悄悄追加成第一轮的验收目标。

## 11. 参考资料与结论边界

- [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)：基础睡眠级别和恢复语义；最新选项须按实际安装版本核对。
- [Ray Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)：逻辑资源、可见设备和小数 GPU 配额的含义。
- [Ray Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)：成组预留资源；sleep 不等于销毁 actor 或移除资源组。
- [现有 Env Service 实现说明](env-service-implementation.md)：静态 unit、准入、租约与节点策略的背景。
- [验证报告](../env-service-verification/README.md)：实机结果及失败边界，以具体运行条件为准。

实现前的分析结论是**机制上可行，已有代码可以复用，但当时联合借还尚未实现、尚未实机验证**。当前代码与验证状态见第 12 节。建议优先借用 rollout 的空闲实例；训练卡还涉及梯度、优化器状态和分布式同步，应与既有训推共置流程统一处理，不能把 vLLM sleep 直接当作训练进程的释放接口。


## 12. 命令批次与模型调用交替借卡的实现

**可以实现 Docker → vLLM → Docker，而且不用每次删掉容器。** 本次接入在 science agent 的下一次模型查询边界交还 GPU，在模型查询结束后再次借出。一次模型回复可能包含多条终端命令，因此准确粒度是“两个模型调用之间的命令批次”，不是强行按每条 shell 字符串切换。

实际接入链路如下。2026-10-14 完成代码与 CPU 回归，2026-10-15 的真实 GPU 子链路验证见第 14 节：

```text
[A] runner：拿任务 lease，进入 ENV，再创建 Harbor 容器
       │
       v
[B] 容器执行一批命令，文件、shell、工作目录继续保留
       │ agent 准备请求下一次模型回答
       v
[C] 节点服务：暂停本 lease 的容器，检查 GPU UUID、CUDA PID 和显存
       │ 还有 GPU 工作：恢复容器，有限等待后再检查
       │ 已回到 sleep 后基线：容器保持暂停
       v
[D] vLLM：wake、恢复 level 1 权重备份及 NIXL 登记
       │
       v
[E] agent：发模型请求，等待完整回复
       │ 无论模型正常返回还是报错，都执行阶段收尾
       v
[F] vLLM：关闭生成准入，sleep(level=1)，确认逐卡容量
       │ 恢复本 lease 的容器
       └──────────────────────────────> [B]

[G] episode 结束 / 阶段超时 / 交接失败：清理本 lease 容器
    确认物理归还后恢复推理、释放调度保留；无法确认则隔离该 unit
```

图例：图里每个部分对应的源码位置。

| 步骤 | 函数及源码 | 输入、输出与状态 |
|---|---|---|
| A | [runner 的 lending 接入](../../../examples/sciaccel_rl/runner.py#L386) | 初始 `ENV` 成功后才调用 Harbor，agent 与 environment 收到同一个 lease 身份 |
| B → C → E → F | [`TruncatingTerminus2._query_llm`](../../../examples/sciaccel_rl/agent.py#L92) | 请求模型前切到 MODEL，finally 中再切到 ENV |
| C | [`freeze_gpu_lease`](../../../psrl/workers/env_service/lending_node.py#L95)、[`returned_to_baseline`](../../../psrl/workers/env_service/lending_node.py#L25) | 暂停本 lease 的所有容器，校验身份、额外 CUDA 进程和显存回落 |
| D、F | [`GpuHandoffMixin`](../../../psrl/workers/gen/gpu_handoff.py#L56) | 关闭生成准入，执行 level 1 sleep / wake 与 NIXL 处理 |
| 全部阶段 | [`GpuLendingBinding`](../../../psrl/workers/env_service/lending.py#L7)、[`EnvRouterCore.set_gpu_phase`](../../../psrl/workers/env_service/router.py#L131) | 串行处理阶段变化，拒绝旧序号，设置超时和隔离状态 |
| G | [`GpuLendingBinding.finish`](../../../psrl/workers/env_service/lending.py#L80) | 移除任务容器、检查显存、解除推理引擎与协调器的保留 |
| 创建与 G 竞争 | [`ManagedDockerEnvironment.start`](../../../psrl/workers/env_service/docker.py#L22) | 登记异步创建中的容器，创建未结束时禁止还卡；过期 lease 不能再启动 verifier |
| 弹性调度与 D/F 竞争 | [`sleep_instances_for_control`](../../../psrl/workers/gen/rollout_coordination/coordinator.py#L292)、[`GpuControlGate`](../../../psrl/workers/gen/gpu_handoff.py#L8) | 协调器在暂停全局派发前跳过已保留实例，引擎串行化实际显存控制 |

这些代码如果有理解上的问题，可以随时追问。

**跟着图读源码：**

| 执行顺序 → 图中编号 | 源码入口 | 输入、输出或状态变化 |
|---|---|---|
| 1 → A | [`runner`](../../../examples/sciaccel_rl/runner.py#L386) | 取得带借卡标记的 lease，提交序号 0 的 ENV，再创建 Harbor 任务 |
| 2 → B/C | [`_query_llm`](../../../examples/sciaccel_rl/agent.py#L92) → [`set_phase`](../../../psrl/workers/env_service/lending.py#L27) | 下一次模型查询递增序号，校验租约仍有效，再串行切换到 MODEL |
| 3 → C/D | [`_transition`](../../../psrl/workers/env_service/lending.py#L51) → [`freeze_gpu_lease`](../../../psrl/workers/env_service/lending_node.py#L95) → [`return_env_gpu`](../../../psrl/workers/gen/gpu_handoff.py#L92) | 暂停容器并确认 GPU 已归还后，唤醒模型 |
| 4 → E/F/B | [`_query_llm` 的 finally](../../../examples/sciaccel_rl/agent.py#L92) → [`_enter_env`](../../../psrl/workers/env_service/lending.py#L63) | 模型完成或报错后关闭生成准入、排空在途请求、sleep 1，逐卡检查容量，再解冻容器 |
| 5 → G | [`finish`](../../../psrl/workers/env_service/lending.py#L80) | 任务结束时移除归属容器，确认物理归还，解除实例保留 |
| 并发分支 → G | [`node.cleanup`](../../../psrl/workers/env_service/node.py#L47)、[`begin_gpu_container`](../../../psrl/workers/env_service/router.py#L151)、[`sleep_instances_for_control`](../../../psrl/workers/gen/rollout_coordination/coordinator.py#L292) | 分别处理租约/阶段到期、创建与回收竞争、弹性调度竞争，不能把这些分支当成单线程顺序调用 |


**为什么不直接删容器？** 例如 agent 执行 `python solve.py`，脚本退出、CUDA 上下文释放后，GPU 可以交给模型，但 `/workspace/answer.json` 和 shell 的工作目录仍有用。这里保留容器，只在 MODEL 阶段冻结它。模型返回新命令后，先确认 vLLM 已睡眠，再解冻容器。`docker pause` 自己不释放显存，必须与 GPU 进程和显存检查配合。

**后台 GPU 程序怎么处理？** 如果 agent 用 `nohup ... &` 启动了任务，或者 Python 服务虽然没有计算但仍持有显存，系统不会把“前台命令已经返回”当作还卡成功。它有界等待，仍无法归还就终止本次任务容器并走物理清理。Docker 不可达、遥测缺失或清理失败时保持隔离，不冒险 wake，也不会让新的申请永久轮询这个已知隔离池。这个策略不保存被终止 GPU 程序的内存状态。

**容器创建结果不明时，不会自动宣称恢复成功。** 例如 Docker 启动命令被取消，而 daemon 可能仍在创建容器，创建登记会保留，unit 进入隔离；需要核实实际启动结果再处理。当前没有自动对账恢复接口。节点上的冻结、解冻和删除按 lease 串行，删除后保留已退役 lease 身份，阻止迟到 RPC（远程过程调用）再次解冻。控制器重启后的持久化恢复仍未实现，这不等同于完整的分布式故障容错。

**已经补上的控制约束：**

- 借用阶段的正常 sleep/wake、权重同步等实际 GPU 操作共用控制门，借出时不能被另一条 wake 路径插入。
- 整个 episode 保留调度归属，模型回答阶段不会被普通弹性 SLEEP 把唯一实例再次睡掉。SLEEP 返回实际睡眠的实例列表，执行器不会把跳过的实例误记为 ASLEEP。
- 借出先使用 `mode="wait"` 等待在途生成完成，再采用 level 1，把权重保存在主机内存，减少每次完整重载。主机内存必须容纳这份备份。KV cache 仍会丢弃，对话前缀可能重新计算。
- 对已有 level 2 路径增加权重有效标记。即使版本数字不变，丢弃过的权重也不能跳过实际同步。对应回归执行了生产 `pull_model_for_sync` 的函数体。
- 借用的完整实例必须与配置逐卡匹配。训练 worker 新增 GPU 位置查询，以便检查与训练、验证、奖励等其他 owner 的重叠。无法证明不重叠时启动报错。
- 物理资源交接用有限超时，模型/容器任何一侧状态不明时保留保护，不能仅靠 lease 到期自动唤醒。

**配置方式。** 默认没有任何借用 unit，因此不会自动让 Docker 使用 rollout 卡。以下为教学配置片段，IP、GPU 列表、显存预算必须替换成实际放置，合并到已有训练配置中，不是独立可运行的训练命令：

```yaml
psrl:
  env_service:
    enable: true
    node_policy: cpu_only
    units:
      - unit_id: rollout-command-phase
        node_ip: 192.168.1.10
        gpu_indices: [0]
        rollout_replica_rank: 0
        max_containers: 1
        cpu_quota: 4
        memory_mb: 8192
        gpu_memory_mb: 24000
    gpu_lending:
      transition_timeout_s: 60
      max_env_phase_s: 300
      return_tolerance_mb: 16
      safety_margin_mb: 1024
```

`rollout_replica_rank` 指定借用哪套 rollout 实例，`gpu_indices` 必须覆盖这套实例的全部 GPU；当前 unit 属于单节点。`cpu_only` 仍限制普通 GPU unit，但显式配置且验证过的借用 unit 可以通过。`max_containers=1` 是当前合同，保证一个命令批次和一个模型调用的交替关系，不支持多个 episode 同时借这套实例。

`transition_timeout_s` 限制一次交接，`max_env_phase_s` 限制一次 ENV 阶段，即使任务仍有心跳，到期也会回收。`return_tolerance_mb` 允许相对 sleep 后基线的小幅显存变化，但不能抵消额外 CUDA PID。`safety_margin_mb` 是每卡预留余量，不是驱动层的显存硬限额。vLLM 必须启用 `enable_sleep_mode`，当前不允许 `profile.fix_weight=True`。

训练/验证等独立 owner 与目标卡重叠、跨节点实例、多个 episode 同时借用、任意其他 agent 类型都不在本次支持范围。stock `terminus-2` 会被替换为本项目带阶段钩子的 `TruncatingTerminus2`，已有该子类也可直接使用。Path A 的 EnvWorker 没有接入这个命令批次协议。

**运行与验证。** 从 `psrl/` 仓库根目录、激活指定环境后执行：

```bash
PYTHONPATH=.:../.verification-deps python -m pytest \
  tests/env_service/test_gpu_lending.py \
  tests/env_service/test_runner_env_service.py \
  tests/env_service/test_node_cleanup.py -q
```

这是生产状态机、science 请求钩子、生产同步函数体及节点检查逻辑的 CPU 回归。测试中的 Ray actor、GPU 和 Docker 结果使用适配器或替身，没有启动真实 GPU 服务。`../.verification-deps` 是工作区已存在的隔离依赖补丁目录，用于整个 Env Service 回归中的张量测试，本次未修改共享 veRL 安装。没有该目录时，先按[集群检查指南](../../psrl/psrl-run-guide/cluster-preflight.md)确认本地依赖。

完整 Env Service 与 science agent 回归为 **171 passed**；最后加入竞态保护后，针对性回归为 **37 passed**，两批存在重叠，不能相加作为独立用例数。修改文件的 Ruff 静态检查通过。

原始结果见 [gpu-lending-implementation](../env-service-verification/results/gpu-lending-implementation/summary.json)，包含完整 Env Service / agent 回归与最后修改后的针对性回归。既有教学脚本仍用于解释概念，其模拟结果不替代这些生产函数回归，更不等同于真实 GPU 交接验收。

2026-10-14 的实现验证到 CPU 回归与静态检查结束。后续同卡循环已有第 14 节实机证据；训练侧完整集成、主机内存备份峰值、稳定的切换性能以及分布式故障恢复仍未验证。

## 13. 分配粒度和 unit 配额

**仍然按已有 unit 准入，unit 的配额事先配置。** unit 是“一台节点上的一包资源”，并不天然等于一张 GPU；它可以只包含 CPU，也可以包含一张或多张指定 GPU。lease 是一次任务对这包资源的使用登记。借卡功能没有另建一套绕过 unit 的准入账本，而是在已有租约下面增加 ENV / MODEL 阶段。

```text
[1] 配置 units：节点、GPU 集合、CPU/内存/显存预算、并发上限
       │ 启动时建立固定资源池
       v
[2] 任务声明 demand ───> [3] EnvRouter / Ledger 检查配额并授予 lease
                              │ 若为借卡 unit，绑定完整 rollout 实例
                              v
                       [4] 同一个 lease 保持占位
                        ENV ↔ MODEL ↔ ENV
                              │ 任务结束，物理清理确认
                              v
                       [5] release，名额重新可用
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责、函数和源码 |
|---|---|
| 1 | [`EnvUnit`](../../../psrl/workers/env_service/types.py#L20) 定义资源包；[`EnvUnitLedger.__init__`](../../../psrl/workers/env_service/ledger.py#L65) 建立静态资源池 |
| 2 | [`demand_from_task`](../../../psrl/workers/env_service/harbor_resources.py#L22) 从任务读取 CPU/内存，GPU 显存需求由覆盖值或默认值补齐 |
| 3 | [`try_admit`](../../../psrl/workers/env_service/ledger.py#L105)、[`_fits_now`](../../../psrl/workers/env_service/ledger.py#L376) 做准入；[`_init_gpu_lending`](../../../psrl/trainer/ppo/ray_trainer.py#L608) 校验 unit 等于整个实例的 GPU 集合 |
| 4 | [`GpuLendingBinding.set_phase`](../../../psrl/workers/env_service/lending.py#L27) 只切换物理使用阶段，不释放任务租约；[`_enter_env`](../../../psrl/workers/env_service/lending.py#L63) 在真实 sleep 后逐卡检查容量 |
| 5 | [`EnvRouterCore.release`](../../../psrl/workers/env_service/router.py#L204) 完成物理清理后释放账本名额 |

这些代码如果有理解上的问题，可以随时追问。

**跟着图读源码：** 从 1 的 YAML 转 `EnvUnit` 开始，2 将任务资源需求转为 `EnvDemand`；3 按节点策略过滤，再检查是否能放入空 unit、当前名额和累计预算，成功输出 `EnvLease`。4 持续使用同一 lease，节点服务与推理 actor 异步执行 sleep / 检查 / wake。5 在清理确认后撤销登记。心跳、超时回收和其他任务申请与 4 并发，阶段切换本身不使第二个任务获得名额。

| 层面 | 当前粒度 | 哪些事先定好，哪些运行时变化 |
|---|---|---|
| unit 资源池 | 节点上的完整 GPU 集合及资源预算 | `node_ip`、`gpu_indices`、`cpu_quota`、`memory_mb`、`gpu_memory_mb`、`max_containers` 来自启动配置；没有自动拆分、合并或按遥测扩容 |
| 任务准入 | 一个 episode 的 lease | 每任务的 CPU/内存需求、GPU 显存声明参与账本计算；未指定 GPU 显存时使用 `default_gpu_memory_mb` |
| 普通 GPU unit | 可有多个 lease 共享同一批卡 | 并发由名额和预算共同限制 |
| 借用 rollout 的 unit | 一套完整实例、一个借卡 episode | 显式指定 `rollout_replica_rank`，当前要求 `max_containers=1`；不是按任意几 GB 切片 |
| 时间交接 | 两次模型查询之间的一批命令 | ENV / MODEL 和实际显存动态变化，unit 容量和 episode 占位不变 |

这里的 `max_containers` 是沿用字段名：实际账本按 episode 的 lease 计数，不是每次 `docker ps` 看到的容器数量。一个 episode 可能带有主容器、独立 verifier 或 sidecar，归属标签让节点服务一起暂停或清理它们；配置中的一个名额不能直接解读为“物理上永远只有一个容器”。

**用这次真实实验的配额算一遍。** 借卡 unit 配置为 2 CPU、1024 MiB 主机内存预算、GPU 0、1024 MiB 显存预算、1 个名额。任务 A 声明 1 CPU、512 MiB 主机内存、512 MiB 显存：这些数值都在 unit 上限内，因此获得 lease。任务 B 声明完全相同，虽然两份声明之和仍未超 CPU/内存预算，但 2 个 lease 超过名额上限 1，所以实测返回 `wait`。

这张 H20 的物理总显存是 97871 MiB，unit 显存预算只是 1024 MiB。**配置预算不等于把显卡硬切成 1 GiB，也不是自动把整张卡容量填进 unit。** vLLM 工作时约占 7209 MiB，这部分不能计成 Docker 已用掉了 unit 的 1024 MiB，否则连首次申请都进不来。因此借卡 unit 的准入先看 Docker 的声明预算；等 vLLM 确实睡眠后，再检查物理容量。本次睡眠后为 593 MiB，检查式为：

```text
睡眠后实际占用 593 + 任务声明 512 + 安全余量 1024 = 2129 MiB
2129 <= 物理总容量 97871，允许进入 ENV
```

普通 unit 使用声明占用与有效遥测占用的较大值；借卡 unit 的特殊处理见 [`_effective_gpu_mb`](../../../psrl/workers/env_service/ledger.py#L357)。GPU 显存声明仍是准入预算，不会阻止 CUDA 程序超额分配。CPU/主机内存 unit 上限也是账本预算，容器实际运行限制由 Harbor/Docker 的环境配置另行执行，不能把 unit 当作一个覆盖所有容器的硬件隔离分区。

如果 rollout 实例跨两张 GPU，当前 unit 必须包含这两张卡，不能只借一张；如果实例跨节点，当前绑定校验不接受。多卡借用还会保守地要求**每张卡**都能容纳完整任务声明和余量，不假设任务自动平均分摊显存。若以后需要多个任务共享同一套借卡实例、运行时重划 unit 或按实际占用调整预算，需要另加调度策略，当前没有实现。

### 配置为什么需要事先确定

unit 是配置者声明的“允许环境任务使用的资源池”。真实机器有多少资源，是物理事实；愿意给环境任务开放多少，是调度政策。两者不必相等。显卡还有空闲显存，并不能说明这些显存已经授权给环境任务，也不能说明其他进程不会在下一刻使用它。因此需要先确定容量上限，再在运行时检查占用、更新租约。

资源还必须在同一节点上配套：一个 Docker 任务不能把节点 A 的 GPU 与节点 B 的普通主机内存拼成自己的本地运行环境。unit 把节点、设备集合和各项预算放在一起，准入时整体选择一个可承载任务的资源包，不会先拿到 GPU、再发现同一节点没有给任务预留 CPU/内存预算。

写下一个 unit 不会立即创建 CUDA 上下文、分配显存或绑定 CPU 核；它先建立准入账本中的额度。模型的 Ray 预留和实际内存分配、Harbor 的容器创建是其他执行步骤。

当前 `EnvUnit` 是冻结的数据对象，router 初始化时把配置读入账本；没有自动生成 unit 或运行时调整容量的接口。修改 YAML 文件本身不会改变已经运行的 router，通常需要在任务清理后以新配置重新创建服务。动态变化的是任务占用和物理遥测，不是这份容量定义。

### 每个字段是在规定什么

下面是**普通 GPU unit 的教学配置**，不是本次 H20 实验参数，也不是借用 rollout 的配置。假设机器有一张 80 GiB GPU，我们决定只开放 60 GiB 给环境任务，留出 20 GiB 余量；同时分配 8 CPU、64 GiB 主机内存预算。字段名使用项目的 `*_mb`，这里按 1024 换算 GiB。

```yaml
psrl:
  env_service:
    enable: true
    node_policy: any
    default_gpu_memory_mb: 20480
    units:
      - unit_id: example-gpu0
        node_ip: 192.0.2.10
        gpu_indices: [0]
        cpu_quota: 8
        memory_mb: 65536
        gpu_memory_mb: 61440
        max_containers: 4
```

这只是要合并到启动配置的片段，示例 IP 必须替换成实际执行节点。字段定义见 [`EnvUnit`](../../../psrl/workers/env_service/types.py#L20)，默认配置入口见 [`env_service.yaml`](../../../psrl/trainer/config/psrl/env_service.yaml#L47)。

| 配置项 | 含义 | 不表示什么 |
|---|---|---|
| `unit_id` | 资源池的唯一名字，用于记账和归属标记 | 不会创建一台机器或一张卡 |
| `node_ip` | 本资源池所在执行节点 | 不是让任意节点的 Docker 自动迁移过去 |
| `gpu_indices: [0]` | 本 unit 内 GPU 任务获准访问的设备集合 | 不会把一张卡切成多个硬件分区 |
| `cpu_quota: 8` | 本 unit 内所有有效 lease 的 CPU 声明合计不超过 8 | 不等于自动绑定 8 个固定物理核 |
| `memory_mb: 65536` | 所有 lease 的主机内存声明合计上限 | 不是给每个任务都分配 64 GiB |
| `gpu_memory_mb: 61440` | GPU 准入预算，总额度 60 GiB | 不是驱动对每个 CUDA 进程强制设置的上限 |
| `max_containers: 4` | 允许最多 4 个有效任务租约，仍须满足其他预算 | 不保证一定能跑满 4 个任务 |
| `default_gpu_memory_mb: 20480` | 单个 GPU 任务未声明显存时，按 20 GiB 记账 | 不是自动测出来的任务峰值，也不是 unit 总额 |

任务的 CPU 和主机内存需求来自 `task.toml` 的 `[environment]`；显存需求来自任务覆盖值，未提供时使用上述默认值。`EnvDemand` 就是“一次任务需要多少”的声明，和 `EnvUnit` 的“整个池允许多少”分开。任务资源缺失时当前 loader 还有默认处理，所以配置并不等于已完成任务资源画像，具体见 [`demand_from_task`](../../../psrl/workers/env_service/harbor_resources.py#L22)。

### 配额怎样算出能放几个任务

继续上面的**教学假设**：每个任务声明 2 CPU、16 GiB 主机内存和 20 GiB 显存，没有额外占用或节点限制。

```text
CPU 最多支持：    8 / 2  = 4 个
主机内存支持：  64 / 16 = 4 个
显存预算支持：  60 / 20 = 3 个
任务名额上限：             4 个

实际并发上限 = min(4, 4, 3, 4) = 3 个
```

这不是一个额外的“先计算并发数”函数，而是逐项累计检查自然得到的结果：前三个任务获准，第四个加入后显存声明会达到 80 GiB，超过 60 GiB，因而返回 `wait`。第一个任务退出、归还其 20 GiB 声明后，第四个重试就能获准。单任务若要求 80 GiB，超过一个空 unit 的 60 GiB 上限，返回 `reject`，因为再等其他任务退出也无济于事。

不同任务大小不一致时，不再套一个固定任务数，而是每次计算“当前累计声明 + 新任务声明”。普通 GPU unit 还会用有效遥测占用与声明占用的较大值参与判断。若同一策略优先级内有多个合适 unit，代码比较加入新任务后各资源的占用比例，取其中最大比例作为负载，优先选该负载较小的 unit；它不是只按空闲显存选卡。对应 [`_fits_now`](../../../psrl/workers/env_service/ledger.py#L376) 和 [`_load`](../../../psrl/workers/env_service/ledger.py#L392)。

### 14 GiB 任务剩下 13 GiB 是不是 unit 太粗

**会出现剩余容量，但要区分任务本身放不下和人为划分造成的浪费。普通 unit 是资源池，不是每个任务必须整块拿走的固定份额。** 现有实现按每个任务的 `EnvDemand` 累加声明，不会把 14 GiB 自动向上取整成一个 unit。

以下先用**教学数字**说明，尚不是实验测量：一张卡开放 55 GiB 预算，每个任务声明 14 GiB，CPU、主机内存和容器名额均充足。

```text
55 GiB 预算
├── 任务 A：14 GiB
├── 任务 B：14 GiB
├── 任务 C：14 GiB
└── 剩余：  13 GiB

第四个 14 GiB 任务：42 + 14 = 56 > 55，必须等待。
另一个 10 GiB 任务：42 + 10 = 52 <= 55，可以申请余量。
```

把记账精度改成 1 GiB 或 1 MiB，都不能让 14 GiB 装进 13 GiB。这属于任务不可拆分带来的装箱尾部余量。若工作队列只有同样大小的任务，等待是合理结果；若有较小任务，当前准入可以让它利用余量，并不要求先来的大任务一定先执行。不过这还不是带公平性保证的最优装箱调度器，持续到来的小任务可能影响大任务等待时间。实际还有 CUDA 上下文等开销，普通 unit 使用“累计声明、有效实测显存的较大者”，因此账面剩余 13 GiB 不等于物理上可以恰好再分配 13 GiB。实现见 [`_effective_gpu_mb`](../../../psrl/workers/env_service/ledger.py#L357)、[`_fits_now`](../../../psrl/workers/env_service/ledger.py#L376)。

**人为粒度确实可能增加浪费。** 例如物理卡足够且可以安全开放 60 GiB，却只配置 55 GiB，第四个 14 GiB 任务会因为人为预算被挡住；`max_containers=1` 也会让普通 unit 中剩余显存无法被其他任务使用。这些限制可能是有意保留安全余量或限制计算竞争，需要依据真实峰值调整，不能见到空闲就删掉。当前 rollout 借卡 unit 的 `max_containers=1` 是交接协议的实现约束，不能直接套用普通 unit 的多任务共享能力。

建议普通单卡任务以一张物理卡建立一个 unit、按任务峰值填写显存需求，并在 CPU、内存、容器数都允许时复用剩余容量。多卡 unit 当前对显存做聚合检查，而非逐卡任务放置；把两张卡各 13 GiB 余量相加成 26 GiB，不能保证一个需要单卡 14 GiB 的任务能执行。这是后续逐卡需求向量和放置检查需要解决的问题，不能用账本总和替代 CUDA 的单卡分配约束。

补充的 [14 / 10 GiB CPU 演示](../env-service-verification/scripts/unit_fragmentation_demo.py)已实际运行：三个 14 GiB 获准、第四个等待，随后 10 GiB 获准，释放后第四个重试成功，最终租约为 0。该脚本调用真实账本，输出见[结果 JSON](../env-service-verification/results/unit-fragmentation-demo.json)，不涉及真实显存分配。

本次训练尝试、未完成边界和成功的真实 GPU 容器对照图表另见[资源账本对照实验](../experiments/science-env-router-comparison.md)。它使用 55 GiB / 14 GiB 的固定资源任务来检验准入，不把任务求解能力或模型效果作为验收目标。

### 实际配置时怎么定这些数值

1. **先定设备归属和交接范围。** 普通 unit 不应随意指向一个正在独立使用的模型 GPU；借卡 unit 显式绑定完整 rollout 实例，由交接协议控制时间使用权。同节点不同 unit 不能重复声明同一张 GPU，配置检查会拒绝重复设备。
2. **再定愿意开放的资源总预算。** CPU、主机内存需为其他服务和模型的主机内存备份留空间；同节点多个 unit 的 CPU/内存预算不要把同一份容量重复许诺。当前 unit 校验不会替配置者完成整机资源实测和总预算分摊。
3. **用任务峰值估计单任务需求。** 取任务加载和计算阶段可能达到的资源量，并为波动留余量，不能只用空闲时占用。需求填写越小，账本越容易放行，但真实 CUDA 分配不会因此缩小。未画像任务的默认显存值应明确视为估计，而不是实测。
4. **最后设置并发名额。** 预算之外，还可用名额限制 Docker 启动、磁盘和 CPU 竞争。普通 unit 可设多个任务；当前 rollout 借卡 unit 必须为 1，这是本版调度协议约束，不能仅因显存充足就改成 4。

这些步骤是配置建议，当前实现不会自动完成上述容量规划。也不建议把 GPU 都写入 unit 后仅靠 `node_policy: any` 解决冲突；该策略只允许放置，不执行 vLLM 借还。

### 纯 CPU 可运行例子

配套脚本 [unit_budget_demo.py](../env-service-verification/scripts/unit_budget_demo.py) 使用真实 `validate_env_service_config` 与 `EnvUnitLedger`，输入采用上面的教学数字，不启动 Ray、Docker 或 GPU。步骤 1 对应配置校验，步骤 2/3 对应申请与准入，步骤 5 对应释放；硬件交接步骤 4 不在这个账本演示内。

从工作区根目录、激活项目环境后运行：

```bash
PYTHONPATH=./psrl python \
  docs/env-worker-deep-dive/env-service-verification/scripts/unit_budget_demo.py
```

实际运行结果为 `granted, granted, granted, wait`，过大任务为 `reject`，释放后重试为 `granted`，收尾有效租约数为 0。输出保存在 [unit-budget-demo.json](../env-service-verification/results/unit-budget-demo.json)。这是生产准入逻辑的 CPU 演示，不是用教学数字重新做了一次 GPU 实验。

## 14. 真实 GPU 验证结果（2026-10-15）

**结果：两轮同一物理 GPU 的 Docker CUDA → vLLM → Docker 交接通过，实验已停止并确认清理。** 使用一张 H20、现有 Qwen2.5-3B-Instruct、512 token 上下文上限和每次最多 16 个生成 token。两轮 Docker 命令，加一次初始模型对照，共三次短生成，没有训练、反向传播或效果评估。脚本内主要流程上限 480 秒，外层进程上限 620 秒，首次尝试即通过，脚本记录总耗时 325.588 秒，包含模型和 actor 初始化与清理。

这次实机链路对应第 13 节的 1 → 5：真实 Ray GPU actor 保留配额，真实 EnvRouter 授予租约，真实节点服务读 GPU 遥测并暂停/恢复容器，Harbor 创建实际 Docker 容器，vLLM 在实际 GPU 执行生成。验证脚本通过同一个 `GpuPhaseClient` 显式驱动阶段，没有让 science agent 自主选择命令。

| 观察点 | 同一张卡的显存占用 | 检查结果 |
|---|---:|---|
| 初始模型生成 | 7209 MiB | 生成 16 个 token，保存对照输出 |
| vLLM 睡眠、借出前 | 593 MiB | 建立 UUID、进程和显存基线 |
| 第一次 Docker CUDA | 978 MiB | 新增容器 CUDA 进程，小内核返回 42 |
| 第一次归还给 vLLM | 7191 MiB | 容器被冻结，额外 CUDA 进程消失，模型生成 token 与对照一致 |
| 第二次 Docker CUDA | 978 MiB | 同一 GPU UUID，内核再次返回 42，之前文件仍存在 |
| 第二次归还给 vLLM | 7191 MiB | 再次生成成功且 token 一致 |
| 再次借出 | 593 MiB | vLLM 休眠后恢复容器 |
| 实验清理后 | 首次记录 3 MiB，独立复查 0 MiB | GPU 无计算进程，本次容器为空，本次 Ray/模型进程为空 |

Ray 的可用逻辑 GPU 数在以上运行阶段始终为 0，因为唯一 GPU 份额仍由模型 actor 持有。Docker 并未向 Ray 重新申请一个 GPU actor，而是经授权使用同一设备。每轮模型阶段都检查了容器暂停，每轮恢复后都确认容器内文件仍存在。这证明本配置下可以保留容器状态反复交接，不需要每轮删除重建容器。

**覆盖边界必须保留。** 脚本使用生产 `GpuHandoffMixin`、`PSRL_vLLMHttpServer` 的 sleep/wake/pause/resume 控制、真实生成准入 servicer，以及生产 EnvRouter、账本、借卡状态机、节点服务和 ManagedDockerEnvironment。为了隔离训练，它使用 [`InferenceOnlyServer`](../env-service-verification/scripts/real_gpu_lending.py#L35) 直接加载已有模型；其中 `nixl_sleep/nixl_wake_up` 只调用真实引擎 sleep/wake，未执行 NIXL 权重传输内存的注销/重新登记。绑定时未使用生产 RolloutCoordinator，未启动 SMG 网络路由，也未经过 trainer 的实例位置校验。这些生产集成仍只有此前部分 CPU 回归，不能称为完整训练链路已实测。16 token 输出一致只是本次恢复后生成的对照，不是全部权重校验或模型质量评价。

**证据与复现：**

- [汇总与边界](../env-service-verification/results/gpu-lending-real-20261015/summary.json)、[原始阶段证据](../env-service-verification/results/gpu-lending-real-20261015/evidence.json)、运行日志（本地资料，未随仓库发布）、退出码 0（本地资料，未随仓库发布）。
- [独立清理核实](../env-service-verification/results/gpu-lending-real-20261015/cleanup-verified.json)，不是仅凭 lease 计数归零判断清理成功。
- [可复用 GPU 脚本](../env-service-verification/scripts/real_gpu_lending.py)、[本次实际执行的脚本快照](../env-service-verification/results/gpu-lending-real-20261015/executed-script.py)。快照保留执行版本；可复用脚本随后只整理导入和将只读探针改为不阻塞事件循环，没有为这些调整重跑 GPU 实验。

在目标节点激活项目环境，先按集群指南确认 Docker 与 GPU 0 空闲，再执行下列命令。`MODEL_PATH` 指已有小模型目录，`NODE_IP` 是路由和 Ray 核对过的本机 IP；`VERIFY_ROOT`、`PYTHONPATH` 都指本次工作区，不沿用共享环境覆盖的路径。每次使用新的输出目录，避免覆盖证据。

```bash
task_root="$PWD"
export PYTHONPATH="$task_root/psrl:$task_root/.verification-deps"
export TMPDIR=/tmp RAY_TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_HOST_IP="$NODE_IP" RAY_NODE_IP_ADDRESS="$NODE_IP"
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RAY_ADDRESS
VERIFY_ROOT="$task_root/docs/env-worker-deep-dive/env-service-verification"
timeout -k 30s 620s python -u "$VERIFY_ROOT/scripts/real_gpu_lending.py" \
  --model "$MODEL_PATH" --node-ip "$NODE_IP" \
  --environment "$VERIFY_ROOT/results/gpu-smoke-task/environment" \
  --output "$VERIFY_ROOT/results/gpu-lending-new-run"
```

预期观察是日志依次出现 `model_initial → env_initial → docker_cuda_0 → model_0 → env_0 → docker_cuda_1 → model_1 → env_1 → released_model_awake`，证据中两个 CUDA 结果均为 42，`second_admission.status=wait`，最终 GPU 无进程且容器为空。确认这些结论后结束；不追加训练或扩大 GPU 数量。只想理解 unit 准入时，可运行已有 [CPU 账本演示](../../../examples/sciaccel_rl/diagnostics/env_service_demo.py)，无需启动 GPU。


## 15. 普通任务的阶段租约与预测回填

> 本节为设计分析。已落地的显式命令申请、物理归还确认和重新排队协议见[命令阶段配额](command-gpu-reservations.md)。它采用 cgroup v1 设备访问控制，限独立环境 GPU，未包含预测器或与 vLLM 借卡的组合。

**核心结论：用户提出的“预测下次使用间隔，把空闲 GPU 交给其他任务”合理。首先需要把容器生命周期与 GPU 使用权分开，再用预测改善选择。预测不是物理资源释放证明，也不能让任意 CUDA 程序自动变得可抢占。** 本节是针对普通 unit 的设计分析；没有把本次 CPU 模拟冒充已落地调度器。此前 Docker/vLLM 借还实现已具备部分交接机制，但仍由一个任务租约持有 unit，不能直接推导为多任务交替复用已经完成。

先看完整串联关系。P1/P2/P4 有部分已有实现，P3 的跨任务借用与预测决策尚待实现：

```text
任务 A 的持久容器 / 文件 / 对话状态（保留）
  |
[P1] GPU 命令结束，进入模型调用或其他等待阶段
  |
[P2] 禁止 A 发起新 GPU 工作 → 确认 CUDA 进程/显存已归还
  |
[P3] Router：只归还 GPU 阶段配额，仍记录容器/主机内存占用
  |   预测空闲窗口 + B 命令耗时 + 交接余量 → 决定是否回填 B
  +------→ B 获得带 epoch 的执行许可 → B 运行 GPU 命令
  |
[P4] A 再次需要 GPU → 请求重新准入
      若 B 已结束：确认归还后交给 A
      若 B 尚未结束：A 等待安全边界，不能同时放行
      若 B 可检查点恢复：才考虑协作式提前回收
```

图例：图里每个部分对应的源码位置与实现边界。

| 编号 | 当前对应源码 | 已实现与缺口 |
|---|---|---|
| P1 | [TruncatingTerminus2._query_llm:103](../../../examples/sciaccel_rl/agent.py#L103) | 借卡模式已有 model/env 边界；不是所有普通容器命令都会自动上报 |
| P2 | [GpuLendingNodeMixin._freeze_gpu_lease:103](../../../psrl/workers/env_service/lending_node.py#L103)、[returned_to_baseline:25](../../../psrl/workers/env_service/lending_node.py#L25) | 已有暂停归属容器、核对 GPU UUID/进程/显存的受控借卡路径；暂停本身不等于显存归还 |
| P3 当前限制 | [GpuLendingBinding.set_phase:27](../../../psrl/workers/env_service/lending.py#L27)、[EnvUnitLedger._fits_now:376](../../../psrl/workers/env_service/ledger.py#L376) | 前者保留单租约 owner，后者仍累计整个 demand；没有普通任务 GPU 阶段子租约和预测回填器 |
| P3 单位定义 | [EnvUnit:20](../../../psrl/workers/env_service/types.py#L20)、[try_admit:168](../../../psrl/workers/env_service/ledger.py#L168) | unit 是资源池，lease 属于 session，并非 worker 永久独占一整个 unit |
| P4 | [GpuPhaseClient.set_phase:120](../../../psrl/workers/env_service/lending.py#L120) | 已有单租约阶段序号；跨任务 epoch 校验、重排队优先级和有界回收仍需设计 |
| 普通任务当前释放点 | [runner finally:534](../../../examples/sciaccel_rl/runner.py#L534) | 清理整个任务容器后才释放租约，因此空闲 GPU 配额仍被保留 |
| 教学协议 | [phase_backfill_demo.py](../env-service-verification/scripts/phase_backfill_demo.py) | 只模拟独占 GPU 时间线、原任务提前返回与旧许可拒绝，不操作容器 |

这些代码如果有理解上的问题，可以随时追问。

**先纠正“unit 粒度太粗”的含义。** unit 是配置好的资源集合，例如一张卡 80 GiB 预算、16 CPU、32 GiB 主机内存、四个容器名额。一个任务申请的是其中的配额，比如 51 GiB、4 CPU、16 GiB 主机内存，而不是一定拿走整个 unit。任务 lease 是这笔配额的所有权记录；worker 可以同时承载多个任务，各自具有 session 与 lease。

这里要分清两个维度。此前“任务需要 14 GiB，卡上只剩 13 GiB”的例子是**空间上放不下**：即便释放与调度完全即时，13 也不足以满足 14。教学例子：一个 69 GiB 的预算池放四个 14 GiB 任务后剩 13 GiB，第五个同规格任务只能等待；一个 8 GiB 小任务却可以继续放入。把同一张卡切成更多固定小 unit 通常不能凭空增加总容量；跨卡的两个小空洞也不能直接满足要求单卡连续使用的程序。减少这种剩余容量，需要大小任务混合、按实际峰值合理申报，或者任务本身支持分片/卸载，不能靠预测安全地将 13 当作 14。

而你本次提出的“多轮之间暂时不用，先借给别人”针对的是**时间粒度**：整个 episode（从创建环境到最终评分清理的一次任务执行）持有配额。任务可能先使用 GPU 45 秒，再等模型、编译 CPU 代码或执行其他工具几分钟，这段时间显存已空闲但账本仍计费。多轮任务之后确实还可能使用 GPU，保留配额可以避免未经协调的重新争抢，也简化状态管理；这不是物理上必须整段占有。

**本次真实 GPU 已观察到的例子。** 2026-10-19 的等完成量 science 对照中，两条任务的 50 GiB CUDA 阶段分别结束于时间戳 `1792344062.7972255` 和 `1792344065.3695402`。后续快照 `1792344066.5814188` 显示 GPU 6、7 的 `nvidia-smi` 显存占用均为 0 MiB，但两条 session 的账本仍各申报 52224 MiB（51 GiB），节点另有两个等待者。注意等待者数量在每个 unit 快照重复出现，不能相加成四个。这两条任务还分别持续约 219 秒、217 秒才完成评分和清理。原始 [idle-reservation-example.json](../env-service-verification/results/science-fixed-work-20261019/idle-reservation-example.json)保存了时间戳、会话和全部设备读数。这是已实测的“物理空闲但租约未释放”，不是预测器收益，也不是 CUDA 内存分配器碎片；模型在另一张卡上继续推理。后续总体对照见[等完成量实验](../experiments/science-gpu-pressure-ab.md#11-四题各四条-rollout-的等完成量对照)。

旧版为何能在任务未结束时使用已释放的显存、为何仍不算主动安全回填，已有[实际 A/B 时间线与当前准入公式的逐项解释](../experiments/science-gpu-pressure-ab.md#12-off-的空档复用与-on-的整任务保留)。其中同时区分历史旧版 OFF 和当前整卡锁 OFF，不能混用。

**应该拆什么。** 建议保留两层资源记录：一层是会话/容器租约，记录文件系统、容器名额和仍在占用的主机内存；另一层是短期 GPU 阶段租约，只在实际 CUDA 阶段持有显存配额及执行许可。归还 GPU 不代表删除容器，也不代表主机内存已释放。CPU 配额通常可以调整，已有内存数据却不会因为账本清零而消失；各资源必须分别确认。比如 A 已让出 GPU，但仍占 16 GiB 主机内存，B 再需要 16 GiB，而该池主机预算只有 32 GiB，就最多容纳这两个持久任务；不能只看 GPU 空闲就再放入第三个。现有代码尚未实现这层拆分。

**抢占、插队与回填不是同一个动作。**

| 操作 | 含义 | 普通容器的适用边界 |
|---|---|---|
| 排队优先级 / 插队 | 资源下一次空闲时先给某个请求 | 最容易实现；必须加等待时长加权和最大等待约束，避免小任务一直抢先 |
| 空闲窗口回填（backfilling） | A 明确暂时不用 GPU 时，让短任务 B 利用空档 | 非抢占式版本就有价值，但必须阻止 A 直接绕过账本再次执行 |
| 协作式抢占 | B 在约定检查点保存状态、主动释放 GPU，之后恢复 | 工具本身要提供 checkpoint/restore 或可安全重跑契约 |
| 强制暂停/终止 | 停止进程或容器来回收 | `docker pause` 不释放已经分配的显存；杀进程会丢 GPU 状态，不能泛称无损抢占 |

vLLM 的请求级抢占可释放该请求的 KV cache（保存历史 token 注意力键值的缓存），把请求放回队列，后续由 token 历史重新计算。当前本地 vLLM 的 `_preempt_request` 中确实有 `kv_cache_manager.free(request)`、`num_computed_tokens = 0` 和重新入队。其代码位于共享依赖，不属于本仓库：/lhy/psrl/third_party/vllm 的实际源码（本地资料，未随仓库发布）。该位置是本次导入来源，不保证未来环境不变。

请求级 KV 回收主要服务于 **vLLM 内部缓存池重分配**，不能直接认定物理显存已经让给 Docker。向其他进程交还大块显存需要类似 sleep 的显式卸载/释放路径，再做物理核验。参考 [vLLM 官方抢占说明](https://docs.vllm.ai/en/v0.20.0/configuration/optimization/#preemption)和[官方 Sleep Mode 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/sleep_mode.md)。普通科学程序的张量、中间计算、随机数状态和外部文件副作用由各程序自行管理，Router 无法像 vLLM 重算 KV 那样通用地重建它们。

**一种更清晰的执行边界。** 如果工具能适配，可以让会话容器保存文件和 CPU 状态，GPU 命令交给单独的短生命周期执行容器，通过共享工作目录交换输入/结果；执行容器结束并清理后再归还 GPU。这样会话容器本身没有随时启动 CUDA 的设备权限，较容易建立执行围栏。代价是镜像、依赖、文件副作用和启动成本需要明确，不能直接无损搬走已有 Python 进程中的 GPU 张量。另一条路是保留现有容器但在交接时暂停、核实所有 CUDA 使用已结束；如果还想让它继续 CPU 工作，就需要更细的 GPU 命令入口隔离，不能仅靠它自觉不访问 GPU。两种路径都是设计选项，本次没有切换实验执行方式。

**预测器可以做什么。** 它适合回答“这次借用值得吗”，不能回答“资源现在安全可用吗”。建议首先用任务历史分位数、阶段类型和当前队列状态，未必需要大模型。输入可以包括：任务/工具类型、最近几轮 GPU 空闲间隔、模型调用排队长度、提示长度、已生成 token 数、后续命令的历史时长；输出应当是剩余空档与借用时长的分布或保守边界，而不只是一个平均数。已等待一段时间时，应预测**剩余**空档，不是无条件重复预测完整间隔。

**项目里已有的相关基础。** [ThroughputProfileLoader](../../../psrl/utils/elastic_rm/scaling_policy.py#L58)能用运行队列长度估计模型吞吐，它不是容器下一次使用 GPU 的间隔预测器。[ModelTurnRecord](../../../psrl/utils/profiling/records.py#L140)与 [collector](../../../psrl/utils/profiling/collector.py#L73)已有模型轮次和环境间隔的记录结构，可作为补充特征的入口，但不能假定它们已覆盖 science 的全部后台 GPU 命令。[ThunderAgentScheduler](../../../psrl/workers/gen/rollout_coordination/session/thunder_agent.py#L27)按 KV 容量决定会话挂起/继续，也不是 Docker 显存回填器。本次检索的 Env Service 实现中没有直接可用的容器 GPU 空闲窗口预测器。

一个保守准入条件可写成：

```text
B 的预计运行时长上界 + 借出/归还开销 + 安全余量
    <= A 剩余空闲时长下界
```

教学数字：A 很可能还有 20 秒才需要 GPU，B 命令保守估计 8 秒，借出/归还各 1 秒，安全余量 2 秒，合计 12 秒，适合回填。若 B 可能运行 30 秒，则不应只因为均值很短就放入。所谓上界/下界若来自经验分位数，仍然可能预测失败，不是数学硬保证；需测量区间覆盖率和负载变化后的失准情况。取 97.5% 的 B 时长上分位与 2.5% 的 A 空档下分位，在校准成立时可用并集界限制两端失准概率，但不能直接把少量历史样本当成这种保证。

预测对象应当是具体 session 的 GPU 阶段租约，而非整个 worker：一个 worker 可以承载多个并发任务，其中一个等待模型时，另一个仍可能在使用 GPU。当前 [client.acquire](../../../psrl/workers/env_service/client.py#L159)采用轮询退避，账本记录等待者但没有在 `try_admit` 中实施严格先到先得或优先级回填。要做可控插队，还需明确队列优先级、等待时长加权、借用时间上限，以及多 GPU 请求的原子准入。

**预测错时怎么办。** 假设 A 实际仅等 4 秒就回来，而 B 还要跑 6 秒。正确行为是让 A 排队，待 B 完成并确认释放后再恢复；预测失误只增加 A 的等待，不应变成两个任务同时用满卡而 OOM。只有 B 声明支持有界取消或检查点，才允许更早回收。不能先归还账本，再寄希望于 B 及时停止。需要测量原任务额外等待的 P95/P99（95%/99% 请求不超过的等待值）、预测失误率、重做成本与总完成量，而非只看 GPU 利用率。

**如果 vLLM 与 Docker 共用同一卡，窗口还要重新判断。** A 的容器暂时不访问 GPU，不代表物理 GPU 空闲：若它正在等待同一卡上的 vLLM 生成回复，这张卡此时可能正在做推理。把卡睡眠后借给 B，会延迟 A 的模型回复，改变原来的空档预测，不能将这段时间当成无代价重叠。本次训练实验中模型卡与环境卡分开，所以环境卡的空档与这个情形不同。统一控制器必须同时考虑 vLLM 当前工作与借出后的影响，避免任务等待被自己暂停的模型。

**执行许可与旧消息。** epoch 是每次资源转交递增的代号。比如 A 原来持有第 7 代许可，借给 B 后变为第 8 代；A 延迟到达的“用第 7 代许可执行 GPU 命令”必须被拒绝。现有单租约 sequence 防止旧阶段请求，但还不是完整的跨租约执行围栏。命令接口返回也不总等于底层进程结束，例如后台命令或终端轮询式执行。所有 GPU 命令需要经过可校验入口；容器里遗留的后台 CUDA 进程、脱离 shell 的子进程和异步 kernel 都要纳入控制和归还验证。否则即使预测完全正确，也无法安全交接。

跟着图读源码：P1 先看 science 代理在模型请求前后调用阶段接口；P2 看 `freeze_gpu_lease` 如何暂停并检查实际资源；P3 看 `GpuLendingBinding.owner` 为什么拒绝另一个 lease，再看账本为何继续累计 `demand.gpu_memory_mb`；P4 看阶段序号的当前范围。普通任务仍沿 runner finally 整体释放。建议的跨任务回填是对这些环节的扩展，图中的预测器和阶段子租约没有对应生产函数，不能把概念图当成现有调用链。

CPU 教学模拟可以在工作区根目录直接运行：

```bash
python3 docs/env-worker-deep-dive/env-service-verification/scripts/phase_backfill_demo.py
```

本次实际运行的合成时间线：A GPU 0–4 秒，模型等待到 24 秒，最后 GPU 24–28 秒；B 需要 8 秒，每次交接计 1 秒。整任务持有资源时 B 到 29 秒才开始、37 秒结束；允许回填后 B 在 5–13 秒执行，全部到 28 秒结束。若 A 提前在第 8 秒回来，则等待到第 14 秒，额外等 6 秒，仍无重叠准入；旧许可被拒绝。保守预测短空档时则不回填。完整输出在 [phase-backfill-demo.json](../env-service-verification/results/science-fixed-work-20261019/phase-backfill-demo.json)。这些秒数和收益只是教学模拟，不是 GPU 实测或训练加速结果。

落地顺序建议：先为可控短命令实现阶段申请/归还与执行围栏，量化真实空闲窗口；再加入按历史分位数选择短命令的非抢占回填；最后才对明确支持检查点的工具加协作抢占和更复杂预测器。这样每一步都能单独验证，也能区分“预测更准”和“交接本身正确”。本次请求的旧版/Router 等完成量训练保持现有调度策略，不能中途加预测器混淆对照。

## 16. 当前交接能力与 procfs 报错的关系

**现有实现支持受控的 vLLM 与 Docker 同卡轮流使用，尚不支持任意环境程序的透明显存卸载恢复。** 本节以当前源码为准，区别于第 1 节的早期方案图。vLLM 是执行语言模型推理的引擎，环境是运行 science 工具命令的容器。两者都需要显存，因此同卡借还必须有明确的先后关系。

```text
[1] trainer 绑定完整推理副本，runner 为 science agent 注入阶段客户端
  ↓
[2] agent 即将请求模型 → Router.set_gpu_phase("model")
  ↓
[3] 借还状态机 → 节点暂停容器并核对 GPU 归还
  │ 未归还：暂时恢复容器等待工作退出，超时则隔离并清理
  ↓ 确认 GPU 身份、进程集合和显存回到基线
[4] vLLM wake → 恢复生成入口 → 执行模型请求
  ↓ 请求结束，agent 的 finally 请求 env 阶段
[5] 协调器/状态机 → 停止新生成、等在途请求结束 → vLLM sleep(level=1)
  ↓ 确认睡眠后实测剩余容量足够
[6] 恢复原容器 → 执行下一批环境命令 → 回到 [2]
  ↓ episode 结束
[7] 删除所属容器 → 确认物理归还 → 恢复模型并解除借用记录
```

图例：图里每个部分对应的源码位置。

| 编号 | 函数与职责 | 源码 |
|---|---|---|
| 1 | `_init_gpu_lending` 校验并绑定副本，runner 注入阶段客户端 | [ray_trainer.py:608](../../../psrl/trainer/ppo/ray_trainer.py#L608)、[runner.py:375](../../../examples/sciaccel_rl/runner.py#L375) |
| 2 | `_query_llm` 在模型调用前后切换阶段，Router 校验当前租约 | [agent.py:125](../../../examples/sciaccel_rl/agent.py#L125)、[router.py:123](../../../psrl/workers/env_service/router.py#L123) |
| 3 | `_transition` 请求归还，`_freeze_gpu_lease` 暂停并核查物理状态 | [lending.py:51](../../../psrl/workers/env_service/lending.py#L51)、[lending_node.py:87](../../../psrl/workers/env_service/lending_node.py#L87) |
| 4 | `return_env_gpu` 唤醒引擎并恢复生成 | [gpu_handoff.py:92](../../../psrl/workers/gen/gpu_handoff.py#L92) |
| 5 | `prepare_env_gpu` 关闭生成入口，等待并执行一级睡眠 | [gpu_handoff.py:64](../../../psrl/workers/gen/gpu_handoff.py#L64) |
| 6 | `_enter_env` 采集基线、检查容量并恢复容器 | [lending.py:63](../../../psrl/workers/env_service/lending.py#L63) |
| 7 | `_finish` 清理容器、确认归还并结束借用 | [lending.py:98](../../../psrl/workers/env_service/lending.py#L98) |

跟着图读源码：从 [1] 的配置绑定进入 [2] 模型调用钩子，参数是租约身份、目标阶段和递增序号。Router 把请求转给 [3]，状态机使用异步锁串行化阶段变化。[3] 的物理检查成功才调用 [4]，并非两个动作并发执行。模型请求结束后，`finally` 触发 [5]，已在途的其他模型请求可能导致等待。随后 [6] 读取睡眠基线，容量足够才解除容器暂停。任务结束走 [7]，失败分支保留隔离状态，不能只清账本就宣称 GPU 空闲。以上源码如有理解上的问题，可以沿编号继续追问。

**教学例子：** 假设 GPU 有 24 GiB，vLLM 活跃时占 18 GiB，环境命令需 10 GiB，两者同时执行需要 28 GiB，装不下。假设 vLLM 睡眠后残留 1 GiB，环境阶段只需要约 11 GiB，便可执行。环境进程退出、回到睡眠基线后，再唤醒模型使用约 18 GiB。这些是假设数字，展示的是分时复用的容量收益，不是吞吐实测。总耗时还包含等待在途生成、睡眠、恢复及核查开销，不能保证每个任务都更快。

如果环境使用一个长期 Python 进程持有 10 GiB 张量，那么 `docker pause` 只是暂停执行，张量仍占显存。当前实现会等待 GPU 进程退出及显存归还，超时后拒绝正常交接并进入隔离/清理流程。它不会自动把任意环境进程的 CUDA 上下文转存到 CPU，再无损恢复。GPU 程序自行释放张量也未必足够，仍需满足计算进程集合回到基线的条件，见 [returned_to_baseline](../../../psrl/workers/env_service/lending_node.py#L26)。

| 能力 | 当前状态 |
|---|---|
| 显式绑定推理副本，science 模型调用前后交接 | 已有代码，要求启用引擎睡眠，并绑定完整副本设备集合 |
| 一个借卡 unit 的多个 episode 同时借用 | 当前不支持，要求 `max_containers=1` |
| `command` 模式自动借用 vLLM GPU | 当前不支持与 `rollout_replica_rank` 组合，两者不是同一开关 |
| 任意新任务无需适配便自动卸载/恢复 GPU 状态 | 未实现，需要任务生命周期能力及状态保存契约 |
| 私有节点当前原生 science 完整交接训练 | 本次未执行，不能由历史验证或普通容器 CUDA 成功推导 |

配置限制来自 [EnvUnit.__post_init__](../../../psrl/workers/env_service/types.py#L51)。当前共享执行后端只注入非 lending 分支，见 [runner.py:387](../../../examples/sciaccel_rl/runner.py#L387)，因此统一 episode/command 执行接口不等于已经统一同卡借还协议。

**Results 与验证边界：** 第 14 节保留两轮真实 H20 / Qwen2.5-3B / Docker CUDA 交接结果，原始[汇总](../env-service-verification/results/gpu-lending-real-20261015/summary.json)明确未覆盖 NIXL 参数传输、完整生产协调器与 SMG 请求路由、trainer 初始化和自主 science agent。不能把该记录写成本次私有节点验证。当前 procfs 报错发生于原生任务的进程归因检查，见[归因诊断结果](gpu-jobs-and-task-telemetry.md#88-results当前-procfs-报错与驱动能力的区别)。借还检查使用整卡基线和进程集合，不调用同一个任务归因入口，但需要单独验收它自己的全部条件。

从仓库根目录可运行已有 CPU 教学模拟，无需模型、Docker 或 GPU：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/gpu_lending_demo.py
```

脚本演示 [3] 至 [7] 的顺序约束：账本释放但显存尚未回收时拒绝唤醒，物理归还且权重恢复后才重新服务。它是早期设计模拟，包含两个借用者及二级睡眠后恢复权重，与生产当前单借用者、一级睡眠有区别。此次运行输出保存在 [handoff-cpu-demo.json](../env-service-verification/results/procfs-handoff-20261026/handoff-cpu-demo.json)，只能证明模拟中的约束，不是本次 GPU 交接验收。
