# Router 修复、遥测与清理交接

> **核心结论**：本篇聚焦「Router 修复、遥测与清理交接」及其相邻完整章节。原文的例子、推导和证据保留，历史条件不代表当前部署。

## 目录

- [12. 2026-10-27 修复实现与验收边界](#12-2026-10-27-修复实现与验收边界)
- [13. 物理资源遥测与清理交接确认](#13-物理资源遥测与清理交接确认)

[返回专题索引](../router-paired-latency-and-design-review.md)

### 11.1 已修复及不能继续使用的旧结论

| 项目 | 本次判断 | 证据与保留边界 |
|---|---|---|
| 外部占用与未兑现预约被 `max` 漏算 | **已修复** | [accounted_gpu_memory_mb](../../../../psrl/workers/env_service/gpu_memory.py#L9)使用 `M + Σ max(R_i-A_i,0)`，未知归属不抵扣。现行 CPU 反例改为等待；[三步高压训练](../science-single-gpu-training-control.md#12-results三步后台对照)中 ON 零 OOM，OFF 18 次。不是所有 OOM 的永久免疫 |
| B1/B2：驱动库匹配与 PTX 编译库 | **实现已修复，有真实 CUDA 证据** | [resolve_driver_libs](../../../../psrl/workers/env_service/overlay.py#L93)解析匹配内核版本的库，仍须逐节点预检 |
| B3/B4：固定 GPU 0、进程内准入 | **ON 路径已替代** | 动态租约设备映射和共享账本生效；用户固定 OFF 保持历史静态映射与进程信号量，不能篡改 baseline 的语义 |
| B5/B6：verifier 归属清理、缺少标签 | **已有主要修复及历史回收证据** | 主容器、独立 verifier 与 sidecar 可按归属回收。物理清理失败和资源总预算仍有下节边界 |
| science GPU 工具完全未接入 | **该结论已过时** | runner 注入 `gpu_jobs`，agent 分派 `psrl-gpu start/status/wait/cancel/telemetry`。这是显式工具，不透明识别任意 shell GPU 工作 |
| 完全没有进程归因实现 | **该结论已过时，但部署未闭环** | 已有容器进程身份核验、逐进程显存／利用率、任务活动评分门槛。平台缺正确 procfs 时无法提供可信结果，`observe` 只标记未验证并保留原任务评分 |
| 两种 GPU 持有策略没有共用执行层 | **已共用作业执行层** | [GpuExecutionBackend](../../../../psrl/workers/env_service/gpu_execution.py#L84)统一执行和取消，episode／command 保留不同预约范围；这不等于 Phase 0 的 EnvWorker／Harbor 两套执行后端已合并 |

三步训练回答了高压力下重复工作能否覆盖初始化差异：ON 总墙钟减少 27.59%。它使用 CPU science 修复题加受控 CUDA 阶段，不能证明原生 `laps-accel-cuda` 的 GPU 奖励归因，也不能证明下列故障分支已经安全。

### 11.2 仍存在的正确性与故障恢复问题

**A. B7：迁移异常后路由可能停在暂停状态。建议优先修复。** 当前方法在 310 行暂停，在 312 行等待 ABORT、320 行等待中断请求返回，324 行恢复，恢复没有 `finally` 保护。这里的“路由”指把模型生成请求送往推理实例的控制循环，不是环境容器本身。

本次从真实源码抽取该方法，仅把外部调用替换为 CPU 故障注入：正常路径记录 `[false,true]`；ABORT 抛异常或等待抛超时，两者均只记录 `[false]`。这验证了方法层面的异常恢复漏洞，没有对运行中的集群制造超时。只有启用相关迁移策略并进入此分支才触发，不能说所有训练现在都会挂。

建议将暂停状态纳入有明确归属的恢复作用域，成功、异常、取消均完成收尾，同时隔离状态不确定的实例并传播原异常。不能不检查其他暂停原因就无条件覆盖全局状态。最小回归需覆盖 ABORT 失败、等待失败、取消，以及健康实例能继续接受请求。

**B. 排除项：不能只看账本就判定 TTL 会提前放行。** TTL 是“多久没有心跳就认为租约过期”的存活时限，它不是容器已经消失的证据。`_reap` 对普通 CPU／episode 删除租约后将物理清理排队，单独调用 Ledger 确实能看到容量立即可用；但生产 Ray actor 覆写了准入入口，先 `reap()`，再等待 `service.cleanup()`，清理未完成或仍有待清理记录就返回 `wait`，最后才调用 `Ledger.try_admit(..., reap=False)`。

教学例子：unit 最多一条会话，预算 4 CPU、8 GiB，A 十秒没有心跳后过期。第 11 秒 B 请求相同资源，单独调用账本会获准；通过实际 Router 方法，在节点报告尚有清理工作时仍等待，只有确认清理且无新待清理记录后才放行。当前节点服务在 Docker 删除失败或仍能列出容器时保留待清理项并重试。

本次 CPU 机制检查分别执行了账本与真实 `RayEnvRouter.try_admit` 方法。后者使用替代的节点清理响应，结果为：`pending → wait`、`confirmed → granted`、`new_reap_after_cleanup → wait`。因此**不把 TTL 先删账本列为当前完整运行链路的未修复漏洞**。这是控制代码检查，不代表已验证控制器崩溃、网络分区及全部 Docker 故障。后续重构也不能绕开这一生产入口保护。

**C. 原生 CUDA 评分仍缺平台能力与完整验收。** 最新平台记录见 [procfs 障碍](../host-proc-blocker.md)。procfs 是 Linux 暴露进程信息的文件系统；GPU 驱动给出宿主进程号，但嵌套容器内 `/proc` 可能是另一套编号。若不能把 GPU 进程对应到任务容器，就不能证明“这次 GPU 活动属于这个任务”。

`required` 会拒绝缺少可信能力的运行；`observe` 允许继续并写入 `gpu_attribution_enforced=0`，保留的 `reward_gpu` 仍可能来自整卡活动污染。已有作业与评分代码不能等同于平台条件已修复。举例：A 在 GPU 计算，B 在 CPU 求解，B 读取整卡 70% 利用率就可能误认为自己用了 GPU。旧 Gate B 只能延迟回复，不能原子地给 B 独占计算窗口；本次 CPU 审计也确认两条空闲观察者都放行、持续繁忙到上限后仍放行。

需要补的是可信宿主 procfs 或宿主采集服务，以及原生任务的正反例验收：GPU 求解器活动应通过，CPU 捷径在 GPU 邻居存在时仍应失败，且独立 verifier 的数值结果与活动证据一致。本次未重新查询平台挂载，部署状态来自 10 月 26 日已保存的预检，不能将其写成 10 月 27 日新测。

**D. 多卡 unit 仍只有总显存预算，没有逐卡峰值约束。** [EnvDemand](../../../../psrl/workers/env_service/types.py#L126)只有单个 `gpu_memory_mb`，[账本](../../../../psrl/workers/env_service/ledger.py#L343)将多张卡占用相加，命令准入也将物理容量求和。教学例子：两张各 40 GiB 的卡合计 80 GiB，一个只在单卡分配 60 GiB 的程序仍可能获准。新的显存公式修复“怎么算占用”，没有增加“分配到哪张卡”的需求向量。当前单环境卡三步实验不覆盖此项；可先保持一张卡一个 unit，扩展时增加逐卡需求与映射验收。

**E. 修复前的容器集合预算缺口。** 本段记录第 11 节检查时的问题，第 12 节已补实现。修复前 overlay 给附属服务追加的是标签，没有统一为 sidecar 分配 CPU／主存硬限制。当前 [overlay](../../../../psrl/workers/env_service/docker.py#L122)已应用容器集合预算，不能拿这条历史问题判断当前仍无侧车限制。主容器和 verifier 自身的限制也不自动成为“整个任务集合共享一个总限额”。`containers_running` 实际是租约数，`cpu_used` 是需求声明之和，不能当作实际容器计数和 CPU 繁忙度。需要明确 unit 管的是会话预算还是物理容器集合，并覆盖附属服务的最坏同时占用。

**F. 设备所有权与控制器恢复仍有边界。** unit 不重叠检查只覆盖同一个 Router 的配置，另一个 Router、手工 Docker、未绑定的模型进程不自动登记在同一份权威账本中。新的公式能保守处理已看见的占用，但不能禁止外部进程随后增长。普通 `node_policy=any` 也不等于平台给了物理独占。建议校验训练／推理和环境的设备分配，跨服务使用时明确唯一所有者或借还绑定。账本和节点待清理集合仍在内存中，尚无控制器重启后重建租约、接管孤儿容器和网络分区恢复的完整验收。

### 11.3 尚未完成的设计目标与非目标

| 原稿或扩展目标 | 当前状态及影响 | 建议验收方式 |
|---|---|---|
| 运行中健康检查、坏节点剔除与恢复 | 启动时有 Docker／GPU 预检，启动失败任务被保存；没有完整自动更新健康候选与恢复重试流程 | 节点运行后故障，停止新派发；恢复后受控加入，已有任务按归属处理 |
| 按真实负载放置 | `assign` 只接收 GPU 布尔值并逐组扣槽位；缺少每任务 CPU／主存／逐卡显存向量及实际 CPU 负载 | 两节点异构需求下，不把任务发送到永远无法满足其需求的节点；组内多轨迹实际容量一起计算 |
| B8 与坏放置重派发 | worker 重试循环存在，默认 `retry_limit: 1` 仍没有第二次本地尝试；manager 有补采和连续失败熔断 | 按失败类别设置有界重试、避开已知坏节点、确认清理后重派发，不把模型答错作为无限重试理由 |
| Phase 0 统一后端 | EnvWorker 持久 shell 和 science Harbor 专用线程仍同时存在 | 先验证 MLGym 跨轮目录、变量、后台进程等状态语义，再迁入共同后端；不能直接删旧 shell |
| 原生 science 工具完整使用 | 显式工具已接入，但 GPU 作业是单独 shell，不继承 tmux 的目录和局部变量；模型要传 `cwd`／`env` | 使用原始 science GPU 任务和真实模型，通过标准工具执行全套检查，验证评分与收尾，不以合成压力替代 |
| command 与 vLLM 同卡借还组合 | [配置](../../../../psrl/workers/env_service/types.py#L54)明确禁止 command 绑定借卡副本 | 若产品确需组合，另定同卡所有权协议及训练权重同步验收；现有独立路径通过不能拼成组合通过 |
| 小请求回填和同容器 GPU 并发 | FIFO 队首阻塞和每容器执行锁仍存在，是吞吐策略边界 | 在实际队列数据证明收益后再做有限回填／饥饿保护；不能为追求并发绕过归还检查 |

原稿已经排除或延期的活容器迁移、任意 CUDA 上下文抢占恢复、容器复用和 MPS，不应列为“必须修复才能签收此次显存修复”的 bug。cgroup v2 支持与更多平台适配也应按部署需求单列。

### 11.4 可复现检查与建议顺序

以下命令只读源码并运行 CPU 机制审计，不创建 Ray 集群、容器或 GPU 工作：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/audit_remaining_env_design.py
PYTHONPATH=. python docs/env-worker-deep-dive/env-service-verification/scripts/audit_gate_contracts.py
```

第一个脚本对实际 `check_and_migrate` 方法进行 AST 抽取和外部调用替换，并调用真实 Ledger 的过期与准入代码。AST 是 Python 的抽象语法树，此处用于只执行目标函数，避免导入整套训练依赖。替换的是外部故障和假时钟，不能视为真实分布式故障注入。第二个脚本调用真实 Gate A/B，用人工遥测验证准入修复与软节流边界。

本次实际输出：[迁移／TTL／多卡机制审计](../../env-service-verification/results/env-design-review-20261027/mechanism-audit.json)、[Gate A/B 审计](../../env-service-verification/results/env-design-review-20261027/gate-audit.json)。作业接口、归因策略、显存公式和 Gate B 的定向 CPU 回归 **32 项通过**，见[测试日志](../../env-service-verification/results/env-design-review-20261027/targeted-tests.log)；补充节点清理和启动入口 **10 项通过**，见[清理入口测试](../../env-service-verification/results/env-design-review-20261027/cleanup-entry-tests.log)。通过项不覆盖 B7 的异常恢复；TTL 则通过补查生产入口避免了误报。

建议先修 **B7 异常恢复**，并打通 **宿主归属能力与原生 CUDA 任务评分验收**。随后完善 **逐卡需求、附属容器预算、健康派发与有界重试**，最后再评估 FIFO 回填和自动选择持有策略。首要目标是错误和故障不会变成错误放行、资源重叠或全局暂停，而不是继续补跑较长训练追求更大的提速数字。

本次仅更新审计文档和两个 CPU 审计脚本，保留原始历史结果，没有修复上述生产问题，没有新增训练。主工作区已有其他未提交修改及文档分类迁移；本次记录不代表提交、推送、合并或部署完成。

## 12. 2026-10-27 修复实现与验收边界

本节对应用户授权后的生产代码修改，第 11 节对应修改前审计。目标是让迁移异常能够释放自己的暂停，并使准入、容器限制和派发使用一致的资源边界。可信进程归属不在本轮修改范围内。

### 12.0 简明解释：sidecar 与四项修复

**sidecar 就是陪着主容器工作的辅助容器。** 例如 science 任务的主容器负责运行 agent 写的程序，Harbor 可能另外启动一个网络辅助容器，控制它能访问哪些外部地址。这个网络辅助容器就是 sidecar。任务自己声明的数据库等辅助服务，也属于本轮需要统计和限制的附属容器。具体任务有没有它，要看任务配置和 Harbor 的网络设置。

verifier 是检查答案、执行测试和评分的组件。在独立 verifier 模式下，它会使用另一套环境，也可能带自己的 sidecar，因此预算里要分别考虑。

```text
任务资源说明 → [2] 选择节点 → [3/4] 检查健康与资源 → [5] 启动任务
                                                       ├─ 主容器：运行程序
                                                       ├─ sidecar：网络等辅助工作
                                                       └─ 独立 verifier 环境：检查结果

模型请求调度同时受 [7] 迁移暂停控制，异常后要回收本次暂停。
```

图例与源码入口：

| 编号 | 对应实现 |
|---|---|
| 2 | [get_dispatch_plan](../../../../psrl/workers/agent_loop/manager.py#L984)选择运行节点 |
| 3/4 | [节点检查](../../../../psrl/workers/env_service/router.py#L336)、[准入检查](../../../../psrl/workers/env_service/ledger.py#L425)决定是否允许启动 |
| 5 | [完整容器预算](../../../../psrl/workers/env_service/container_budget.py#L7)、[最终容器限制](../../../../psrl/workers/env_service/docker.py#L122)覆盖主服务与附属服务 |
| 7 | [RoutingPauseController](../../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L7)记录并释放暂停持有者 |

下面数字都是教学假设，不是训练实测。

**① 迁移异常恢复：避免一次失败把后续请求一直堵住。** 迁移过程需要先暂停发送新请求，再中断、转移已有请求。原先中间一步报错，就可能跳过恢复操作。现在无论成功、失败还是取消，都会收回这次迁移提出的暂停。

但收回自己的暂停不等于立即启动：假设模型同步也要求暂停，迁移结束后仍应继续暂停。可以理解为记录“谁还要求暂停”，只有条件允许时才继续派发。这修的是卡住问题，并不直接缩短一次正常计算。

**② 逐卡显存预算：每张卡都必须装得下。** 两张各 40 GiB 的 GPU，总计虽然有 80 GiB，但一个只在第一张卡申请 60 GiB 的程序还是会显存不足。新检查分别看每张卡的容量、占用和任务需求，不能用第二张卡的空闲弥补第一张卡的不足。

这是在启动前拦住明显放不下的任务。它不会自动把程序拆成多卡运行，也不保证程序实际使用量永远不超过声明。

**③ 完整容器预算：记账和限制都覆盖辅助容器。** 假设主容器上限是 4096 MiB，网络 sidecar 上限是 128 MiB，至少要给这一套环境预留 4224 MiB。只记主容器会少算 128 MiB，多条任务并发时少算部分会累积。现在先把允许占用的上限加起来，再分别限制每个容器。若还有独立 verifier 环境，也一起预留。

代价是预算更保守，可能让同时运行的任务变少。它解决的是资源少记、辅助容器不限额的问题，不能据此直接宣称训练更快。

**④ 健康派发与有限重试：不把新任务反复送到坏节点，也不无限等待。** 节点 A 的 Docker 不响应时，即使它账面上有空位，也先退出候选。节点 B 健康且资源合适，就选择 B。所有候选都不可用时，只在规定次数内重试，仍失败就把错误传给训练端。

同时，一组有两条采样任务，就按两条任务计算需求。例如两条各需 4 GiB，优先选择能容纳 8 GiB 的节点，不能只按一条算。节点恢复后，还要确认旧容器清理完成，才能重新接任务。本轮没有实现把已经运行到一半的整个任务自动搬到另一台机器。

这四项主要提高故障处理和资源管理的可靠性。简单、低压力、单卡且没有故障的任务，不一定因这些修复直接提速。上一轮实际运行的是 CPU 控制逻辑回归，没有新增训练速度对照。

源码阅读顺序和可运行的 CPU 示例继续使用 [12.4](#124-跟着图读源码与可运行验证)，不用重新启动训练。源码细节有不清楚的地方，可以按上面的编号继续追问。

### 12.1 全局执行链与源码图例

```text
[1] task.toml / extra_info → Harbor 资源计划
                         ↓
[2] AgentLoopManager：按每组实际 rollout 数选择健康节点
      ↑ 节点状态和逐 unit 资源快照
[3] EnvRouter → 节点 Docker/GPU 探活 → 物理清理确认
                         ↓
[4] Ledger：CPU、主存、会话槽位、逐卡显存准入
                         ↓ lease 携带完整容器预算
[5] runner → Harbor → ManagedDockerEnvironment
                         ├─ 主容器：CPU / 主存 / 单副本限制
                         ├─ verifier：继承同一预算
                         └─ sidecar：独立限制、数量上限与归属标签
                         ↓
[6] command 模式执行前再次逐卡检查 → 执行 → 确认归还 → 清理

模型调度旁路：[7] check_and_migrate → 临时暂停持有者
                  ABORT / 等待中断请求回流
                  finally 只释放自己的暂停
```

图例：图里每个部分对应的源码位置。

| 图中部分 | 职责与真实源码 |
|---|---|
| 1 | [harbor_resources.py：demand_from_task](../../../../psrl/workers/env_service/harbor_resources.py#L22)统一解析任务与 verifier 需求，供 manager 与 runner 使用 |
| 2 | [manager.py：get_dispatch_plan](../../../../psrl/workers/agent_loop/manager.py#L984)统计组内条数，[placement.py：_assign_resources](../../../../psrl/workers/env_service/placement.py#L184)按资源形状放置 |
| 3 | [router.py：_check_node](../../../../psrl/workers/env_service/router.py#L334)检查节点并保留清理状态，[try_admit](../../../../psrl/workers/env_service/router.py#L444)在确认后准入 |
| 4 | [ledger.py：_could_ever_fit](../../../../psrl/workers/env_service/ledger.py#L396)、[_fits_now](../../../../psrl/workers/env_service/ledger.py#L425)区分永远放不下与当前需等待 |
| 5 | [runner.py](../../../../examples/sciaccel_rl/runner.py#L149)把预算放入 Harbor 环境参数，[docker.py](../../../../psrl/workers/env_service/docker.py#L122)为最终全部 Compose 服务施加限制 |
| 6 | [command_gpu.py：acquire](../../../../psrl/workers/env_service/command_gpu.py#L50)在设备授权前检查每张卡，不修改可信进程识别机制 |
| 7 | [check_and_migrate](../../../../psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py#L290)使用 [RoutingPauseController](../../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L7)管理暂停持有者 |

这些源码如果有理解上的问题，可以沿编号继续追问。

### 12.2 暂停归属和逐卡预算怎样工作

**迁移暂停不是无条件恢复。** 暂停控制器维护“其他流程期望的运行状态”和当前临时持有者集合。实际运行条件是：期望运行，并且没有临时持有者。只在写状态时持短锁，不把锁跨越整个 ABORT 命令等待，避免命令队列中的休眠或唤醒请求与迁移互相等待。

教学例子：初始路由运行，迁移 M 获得暂停，实际状态变为暂停。此时休眠流程要求继续暂停，控制器记录这个要求。即使 M 的 ABORT 报错，M 退出只删除自己的持有者，路由仍按休眠要求暂停。若没有其他暂停要求，则恢复运行。两条临时操作重叠时，只有最后一个持有者退出才可能恢复。

暂停请求已经生效但响应丢失，也会进入清理。清理独立于调用者取消执行，重复取消不会让清理任务失去等待者，但恢复操作本身仍有 30 秒超时。网关持续不可达时会抛出错误，不能声称一定能恢复一个无法访问的外部服务。ABORT、回流等待和取消的异常不会被当作迁移成功吞掉。

**多卡总显存不能代替单卡容量。** `EnvUnit.gpu_memory_per_device_mb` 与 `gpu_indices` 按位置对应，多卡配置必须显式提供，且合计等于 `gpu_memory_mb`。单卡配置可由总预算直接确定该卡预算。节点预检还会检查配置是否超过物理设备容量。

教学例子：卡 0 和卡 1 各有 40 GiB，总预算 80 GiB。

- 未声明分片的 60 GiB 请求被拒绝，因为任意单卡都容纳不了它。不会假定程序自动把显存平均分配到两张卡。
- 显式声明逐卡 10、10 GiB 的请求，在卡 0 已占 35 GiB 时等待，因为第一张卡需要 45 GiB，虽然总数 55 GiB 小于 80 GiB。
- 卡 0 的压力释放后，同一请求可以获准。仅缺失卡 1 的遥测，不会抹掉卡 0 已知的高占用。

未提供逐卡需求向量时，按“全部需求可能集中到任意可见卡”检查每张卡，因此偏保守。提供 `EnvDemand.gpu_memory_per_device_mb` 时，其总和必须与该请求的总需求一致。命令准入同时检查账本预算与实时物理容量。

当前可信归属接口只给出整个请求的占用总量，不能凭这个总量猜测多卡上的分布。因此多卡检查暂不按总归属量逐卡抵扣，可能保守地重复计费。单卡仍可使用已有可信归属抵扣。这个边界交由逐卡归属数据完善后再放宽，不改成平均分摊来追求并发。

### 12.3 容器集合与健康派发的具体变化

**CPU 和主存申报覆盖整个正常 Harbor 容器集合。** 资源计划读取主环境和 verifier 声明，取足以容纳两者的主服务规格。独立 verifier 按最多两套环境预留，附属服务按任务 Compose 声明的最大数量，加一个 Harbor 可能创建的网络 sidecar 预留。即使当前 Harbor 某条路径顺序执行两套环境，也不依赖这个时序来少记预算。

教学例子：agent 主容器需 2 CPU、4096 MiB，独立 verifier 需 4 CPU、8192 MiB，任务有一个数据库 sidecar，还可能有一个网络 sidecar。默认每个 sidecar 为 0.25 CPU、128 MiB：

```text
每套环境：4 + 2 × 0.25 = 4.5 CPU
          8192 + 2 × 128 = 8448 MiB
两套总预留：9 CPU、16896 MiB
```

最终 Compose 中，main、verifier 的 main、数据库和网络服务都有明确 CPU、主存及单副本限制。实际服务数多于预留时拒绝启动。任务可通过 `[metadata.env_service]` 的 `sidecar_cpu`、`sidecar_memory_mb` 调整附属服务规格。主存声明缺失时 main 默认限制为 2048 MiB，不再按零主存申报。

这采用“分配到各容器的上限之和不超过申报”方式，不是创建一个跨所有容器共享且可自由借用的父 cgroup。它约束正常 Harbor 生命周期与最终 Compose 配置，不是对恶意 Docker 操作的安全隔离，也未覆盖镜像构建过程的资源消耗。`max_containers` 和快照中的 `containers_running` 仍按会话租约计数，不等于现场物理容器数。

**派发按真实任务和实际条数估算。** manager 与 Gate A 共用资源解析函数。每组有 2 条 rollout 时，虚拟放置扣两条任务的 CPU、主存和槽位。CPU 与 GPU 任务落到同一个 unit 时，共用该 unit 的主机资源，不能分别重复消费同一份空闲槽位。

教学例子：节点 A 只有 100 MiB 空闲，节点 B 有 1000 MiB，同一组两条任务各需 60 MiB。只看“有槽位”会选择 A，但两条合计需 120 MiB。新派发优先选择 B。没有节点能同时容纳整个组，但单条任务可以放下时，允许把组放到合格节点排队，不把派发提示误当成已经获得租约。

**节点故障与普通资源等待分开。** Docker/GPU 探活失败后，该节点退出新派发候选。恢复必须重新探活并确认待清理集合已清空。连续失败达到配置上限后停止恢复尝试，准入返回拒绝，不无限重建 actor。默认健康检查间隔 5 秒、单次 30 秒超时、连续失败上限 3 次，检查由容量刷新或准入驱动。

manager 的派发默认最多尝试 3 次，间隔 5 秒。没有新鲜健康快照时不退回盲目轮询。耗尽预算会将错误传给训练与验证的结果等待者，避免只有后台循环报错、前台训练一直等待。尚未派发的请求不会提前标为 RUNNING。

### 12.4 跟着图读源码与可运行验证

| 步骤 | 图中编号 | 函数与输入／输出、状态变化 |
|---|---|---|
| 1 | 1 → 2 | `demand_from_extra_info` 解析任务路径，调用 `demand_from_task` 生成同一份 `EnvDemand` |
| 2 | 2 ↔ 3 | `get_dispatch_plan` 统计组内条数，读取节点资源和健康状态，给每组选择 worker。节点探活是异步共享任务，不是每个调用者各启动一套服务 |
| 3 | 3 → 4 | `RayEnvRouter.try_admit` 等待健康检查和清理确认，再调用 `Ledger.try_admit(reap=False)`。检查后新出现的待清理项仍会阻止放行 |
| 4 | 4 → 5 | `runner._apply_env_lease` 将租约和 `ContainerBudget` 传给环境。Harbor 的独立 verifier 继承环境 kwargs，最终 overlay 覆盖全部服务 |
| 5 | 5 → 6 | command 模式重新取得每卡遥测，通过后才授权设备访问。episode 模式在会话准入时预留显存 |
| 6 | 6 → 3 | GPU 归还与节点清理保留原有确认边界。预算释放不替代物理回收，节点服务关闭前会取消并等待本轮健康探测 |
| 并行旁路 | 7 | `check_and_migrate` 在控制器的暂停作用域内执行 ABORT 和回流等待，退出时回收自己的暂停持有者 |

前表给出了对应函数的源码跳转。CPU 教学审计可以直接从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/audit_remaining_env_design.py
PYTHONPATH=. pytest -q tests/env_service/test_design_remediation.py
```

第一个脚本使用标准库，执行真实迁移函数、账本和 Router 准入函数，外部网关、Docker 和 GPU 使用教学替身。预期迁移正常、ABORT 失败、回流超时三种情况都记录 `[false, true]`，异常类型仍保留。逐卡教学输入返回 `reject`。清理未确认和清理后又出现过期记录都返回 `wait`，清理确认返回 `granted`。

第二条命令需要项目测试依赖，但不创建 Ray 集群、容器或 GPU 工作。它进一步覆盖重复取消、重叠暂停、真实命令授权前的逐卡判断、verifier 需求、容器限制、健康恢复和有限重试。旧审计 JSON 保留原样，新结果单独存入 [env-design-remediation-20261027](../../env-service-verification/results/env-design-remediation-20261027)。

### 12.5 Results 与未覆盖边界

相关 CPU 回归 **291 项通过**，覆盖完整 `tests/env_service`、manager 失败熔断和 science agent loop 测试。另补 Harbor 配置序列化与独立 verifier 继承路径检查，相关 **38 项通过**，与前述回归有重叠，不相加计数。预算在配置边界传普通字典，环境构造时重建经过校验的对象，避免配置落盘再加载后丢失限制方法。定向 Ruff 与格式检查通过，检查结果见 [validation.txt](../../env-service-verification/results/env-design-remediation-20261027/validation.txt)，完整日志与源码指纹一同留档。全仓 pre-commit 在隔离副本执行，仍因已有其他文件的 Ruff／注释规范问题失败，自动改动没有应用回主目录。这里的故障注入是 CPU 控制逻辑验证，不能写成真实 Docker 限额验收、多卡 CUDA OOM 实测或训练吞吐提升。没有为了这些修复重跑 science 训练。

以下仍保持未完成或独立任务状态：

- 可信进程归属及其正反例验收由另一个进程处理。
- 控制器重启后的持久化账本重建、跨 Router 的唯一设备所有权和网络分区恢复未实现。
- 运行中的整个 episode 失败后，先确认清理再把同一请求搬到另一个 worker 的协议未实现。本轮重试限于派发前选择和节点探活，不盲目重跑已产生副作用的任务。
- 多卡需求向量是准入契约，不会强制程序按该向量分配显存，仍不是 CUDA 显存硬隔离。外部进程也可能在准入后增加占用。
- EnvWorker／Harbor 后端统一、FIFO 小请求回填和同容器多 GPU 作业并发未包含在本轮。

所有修改位于 `psrl/` 主工作区，保留了开始前其他进程的修改。未提交、未推送、未合并，也未改变固定 baseline `0eeead861ffafed6bf31c962d259fd847775dcf9` 或已完成的三步实验快照。

## 13. 物理资源遥测与清理交接确认

本轮选择补原提案 §6.6、Phase 1 的物理资源观测，以及 Phase 3 的负载派发输入。另修复清理交接中的丢单窗口。以下是实际实现关系，方括号编号对应源码图例。

```text
[1] Manager 按需求和容量选择 worker ← [4] 缓存的新鲜节点负载
                 ↓                           ↑
[2] Router / Ledger 准入并记录租约      [3] 节点独立查询 Docker / 主机负载
                 ↓                           ↑
Harbor → 主环境 + 辅助容器、独立验证环境 + 辅助容器
                 ↓ 结束或失去心跳
[5] Ledger 撤销活跃租约，保留待清理记录
                 ↓ 可重复领取
[6] Node 删除所属容器 → 查询确认无残留 → 向 Router 确认 → 删除待办
                 ↓
[7] 新任务准入：探活成功且待清理记录为空，才允许进入

故障分支：[6] 确认前 Node 被杀 → [5] 待办仍在 → 新 Node 再领取 [6]
遥测 [3] 独立异步运行，读取失败不会直接触发 [5] 或判任务失败。
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责及函数 | 源码 |
|---|---|---|
| 1 | `_assign_resources` 先筛资源形状，再比较节点负载 | [placement.py](../../../../psrl/workers/env_service/placement.py#L184) |
| 2 | `snapshot` 区分租约计数、声明资源和物理观测 | [ledger.py](../../../../psrl/workers/env_service/ledger.py#L268) |
| 3 | `resource_snapshot` 调用 `read_node_resources`，按 unit 标签查询实际容器 | [node.py](../../../../psrl/workers/env_service/node.py#L59)、[resource_telemetry.py](../../../../psrl/workers/env_service/resource_telemetry.py#L24) |
| 4 | `_poll_node_resources` 异步采集，`_resource_state` 控制新鲜度，`node_capacity` 提供负载 | [router.py](../../../../psrl/workers/env_service/router.py#L343)、[新鲜度](../../../../psrl/workers/env_service/router.py#L290) |
| 5 | `retire` 保留待办，`pending_reaped` 重复交付，`acknowledge_reaped` 精确删除 | [ledger.py](../../../../psrl/workers/env_service/ledger.py#L231) |
| 6 | `cleanup` 删除、确认物理消失、再确认账本记录 | [node.py](../../../../psrl/workers/env_service/node.py#L83) |
| 7 | `_check_node` 处理有界探活和安全重建，`try_admit` 检查未完成清理 | [router.py](../../../../psrl/workers/env_service/router.py#L395)、[准入](../../../../psrl/workers/env_service/router.py#L524) |

这些代码若有不清楚的地方，可以按图中编号继续追问。

### 13.1 原提案还缺什么

| 原提案内容 | 本轮处理后的状态 | 下一步或边界 |
|---|---|---|
| §6.6、Phase 1：物理占用与声明账本分开 | 已增加真实容器计数、主存、CPU 使用量和主机负载，独立周期采集 | 未读取到的数据是未知，不能显示成零 |
| Phase 3.1：根据实测负载选择节点 | 已接入资源感知派发的偏好 | 未做多节点吞吐实验，不能宣称训练提速 |
| Phase 1：TTL 回收并实际停止容器 | 已补“确认删除后才销账”，允许节点服务重领待办 | 仅 Router 仍存活的恢复，不是持久化账本 |
| Phase 0：统一 EnvWorker / Harbor 后端 | CPU 单容器已有共享适配器，整体统一未完成 | 先通过下述执行契约门槛，再接远程池及统一账本，最终迁移调用方并删除旧路径 |
| Phase 3.2：运行失败的 episode 清理后跨 worker 重派发 | 未完成 | 需请求身份、清理确认、尝试上限及防重复副作用协议 |
| B8：episode 级重试默认次数 | 未在本轮调大 | 不能把派发前探活重试当作 episode 执行重试 |
| GPU 进程归属及严格归因 | 不在本轮实现范围 | 由另一进程维护，需它自己的真实验收证据 |
| 容器池、状态迁移、持久账本与跨 Router 唯一所有权 | 未实现 | 前两项在原稿延期或排除，后两项不能冒充已完成能力 |

**实施优先级与验收门槛**如下。这里的优先级是后续实施计划，不代表所有条目已经实现。

| 优先级 | 工作 | 完成门槛 |
|---|---|---|
| P0.1 | 统一后端的执行语义 | 持久 shell 跨轮保留目录、变量、函数和文件。普通 Harbor 命令保持独立 shell。超时、取消或执行状态不确定后整个环境不可复用 |
| P0.2 | 远程 worker 池、资源所有权及调用方迁移 | 文件传输不依赖客户端本地路径，资源只记一次账，清理确认后释放。覆盖 MLGym 评分隔离、Harbor 多服务及 GPU，再删除被替代路径 |
| P1.1 | Phase 3.2 的 episode 重派发 | 稳定请求身份、独立 attempt 身份、明确可重试错误、旧尝试清理确认、次数上限及单次结果提交共同生效 |
| P1.2 | B8 的默认重试次数 | 依赖 P1.1。仅重试明确允许重放的任务，无法判断外部副作用的任务不得自动重放 |
| P2 | 多节点负载派发实验 | 相同任务与资源预算下比较耗时、失败及资源占用，实测后再判断收益 |
| 独立架构阶段 | 持久账本、Router 重启和跨 Router 唯一所有权 | 需要持久权威状态及旧所有者失效机制，不能由内存租约或节点重建替代 |
| 保持延期 | 容器池及运行状态迁移 | 不作为当前正确性工作的前置依赖 |

GPU 进程归属沿用其独立维护与验收范围。P0.2 的 GPU 接入依赖该部分的验收，不重复实现归因机制。

P0.1 的接口位于 [EnvWorkerEnvironment](../../../../psrl/workers/env_worker/harbor.py)：
`exec_shell` 调用 `EnvWorker.exec`，使用容器的持久 shell，返回合并输出、退出码与超时标记。
`exec` 调用 `EnvWorker.exec_process`，每次创建独立进程，返回分离的 stdout 和 stderr。
持久 shell 从镜像的用户、目录和容器启动环境开始，不套用 Harbor 单次命令的 cwd、user 或 scoped env。
调用方应通过 shell 命令明确建立跨轮状态，不能把两个接口无条件互换。

例如第一轮 `exec_shell` 执行 `export VALUE=42` 并切换目录、写文件，第二轮仍能读取变量、目录和文件。
同一环境的普通 `exec` 可以读取该文件，但不会继承持久 shell 中的变量或函数。
普通命令返回非零退出码后仍可继续。超时、取消或 shell 退出会使适配器失效，后续命令和文件操作被拒绝，
`stop` 仍可重试物理清理。失效标记本身不代表清理已经成功。

执行契约的 CPU 回归使用真实本地 Bash，只替换 Docker 创建、删除及执行传输：

```bash
python -m pytest tests/env_worker/test_harbor_backend.py tests/env_worker/test_airs_regressions.py \
  tests/env_worker/test_shell.py tests/env_worker/test_worker.py tests/environments/test_mlgym_env.py -q
```

这组检查验证跨轮状态、独立命令与持久 shell 的状态隔离、非零退出码、超时、取消、shell 退出及失败后拒绝复用。
它不是远程 Ray、真实 Docker 或完整 MLGym 迁移验收。P0.2 尚未完成，science 默认路径和两套资源账本仍需统一。

**P0.1 Results：** 上述命令的 60 项 CPU 回归通过，修改文件的 Ruff、格式与注释规范检查通过。
全仓 pre-commit 在隔离副本执行，因未修改文件的行长、格式与注释规范问题失败，不能视为全仓检查通过。

### 13.2 两类资源数值与清理确认的意义

**声明预算是提前预留，物理观测是此刻实际使用。** 一条 science 任务可能启动两套环境，每套包含 main 和 sidecar。main 是执行任务的主容器，sidecar 是同环境的辅助容器，例如网络代理。本轮实验的 sidecar 只运行一个小 Python 常驻进程，验证资源限制和回收机制，不冒充真实网络代理功能。

以下数字是本轮容器实验配置：两套环境各预留 `0.5 CPU + 128 MiB` 给 main，`0.25 CPU + 64 MiB` 给 sidecar。总声明为 `2 × (0.5 + 0.25) = 1.5 CPU`、`2 × (128 + 64) = 384 MiB`。实际启动 **4 个容器，但只有 1 条活跃租约**。因此接口使用 `leases_active=1` 表示租约，`containers_running=4` 表示物理容器。`max_containers` 仍是租约槽位上限，不可拿物理容器数直接与它比较。历史结果文件中的同名旧字段不改写，历史绘图程序保留原格式。

`cpu_used`、`memory_used_mb` 仍是已申报预算。新增 `measured_memory_mb` 来自 Docker 的内存用量，可能包含缓存，并非单个 Python 对象大小。`measured_cpu_cores` 来自容器累计 CPU 时间与宿主累计 CPU 时间的差值。教学例子：区间内容器增加 100 单位、宿主增加 1000 单位，宿主有 4 个逻辑核，则使用量为 `100 / 1000 × 4 = 0.4 核`。缺任一必要计数就返回未知。

派发使用 `node_cpu_load_1m / node_cpu_count`。其中 load 是一分钟平均可运行及不可中断等待任务数量，不是 CPU 利用率，也不是该 unit 独占的负载。教学例子：A、B 都能容纳两条新轨迹，同为 8 核，load 分别为 16、4，则优先 B。若 B 内存不够，低 load 不能绕过容量检查。未知或过期负载不当作零，完全同分时保留轮询顺序。该策略还没有真实跨节点性能验收。

清理修复解决的是另一件事：TTL（租约存活期限）到期只能说明心跳断了，不能证明四个容器已消失。如果 Router 把待清理记录交出去就删除，Node 在删容器前被杀，接班者就没有清理名单。因此 Router 必须等 Node 删完容器并再次查询确认为空，才删除对应待办。重复删除已经不存在的容器是安全的，确认回包丢失时重新领取即可。确认某一条不会删除同时出现的其他待办。正常 release 同样走确认，关闭节点服务也拒绝丢弃尚未确认的清理。

自动重建节点服务还要求没有活跃租约、没有绑定的模型借卡状态、没有待清理 GPU 租约。涉及 GPU 撤权、旧句柄或模型状态的故障继续保守阻塞，不能把本轮 CPU 故障恢复外推到 GPU 状态恢复。连续失败仍受原有次数上限约束。

### 13.3 跟着图读源码与复现

| 执行步骤 | 图中编号 | 输入、输出或状态变化 |
|---|---|---|
| 选择候选 worker | 1，`placement.py:184` `_assign_resources` | 需求、组大小、节点容量及负载，输出 worker 序号 |
| 申请租约并启动容器 | 2、7，`router.py:524` `try_admit` | 探活和清理通过后授予 lease，Harbor 使用预算覆盖文件 |
| 并发采集物理资源 | 3、4，`resource_telemetry.py:24`、`router.py:343` | Docker 样本缓存到 Router，不在每次准入中同步等待资源统计 |
| 任务结束或心跳中断 | 5，`ledger.py:243` `retire` | 活跃租约移入待清理集合，物理容器可能仍存在 |
| 清理与确认 | 6，`node.py:83` `cleanup` | 删除容器、确认标签查询为空，回调 `acknowledge_reaped` |
| 故障后的新准入 | 7，`router.py:395`、`:524` | 探活可恢复且待办清空后才允许新租约 |

表中源码链接在上一张图例表。并行遥测、TTL 定时器与 Node 清理器是独立异步分支，不应理解成一条串行线程。

在仓库根目录、项目 Python 环境中运行 CPU 回归，无需 Ray 集群或 GPU：

```bash
python -m pytest tests/env_service/test_resource_telemetry.py tests/env_service/test_node_cleanup.py tests/env_service/test_design_remediation.py tests/env_service/test_node_prewarm.py -q
```

这些测试包含故障替身：模拟缺失测量、清理确认失败、节点失效、负载不同等输入，验证未知值不会变成空闲，失效节点不能接新任务，确认前待办可重复领取。它们不能单独证明 Docker 内核限额生效。

真实实验脚本：[real_resource_recovery.py](../../env-service-verification/scripts/real_resource_recovery.py)。在目标节点预检通过、已有 Python 容器镜像的前提下运行：

```bash
read -r -p '目标节点 IP: ' ENV_TEST_IP
read -r -p '已有 Python 镜像: ' ENV_TEST_IMAGE
run_dir=$(mktemp -d /tmp/env-resource-check.XXXXXX)
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" nohup timeout 300 python \
  docs/env-worker-deep-dive/env-service-verification/scripts/real_resource_recovery.py \
  --node-ip "$ENV_TEST_IP" --image "$ENV_TEST_IMAGE" --output "$run_dir" \
  > "$run_dir/driver.log" 2>&1 < /dev/null &
```

脚本起隔离 Ray 集群、真实 Harbor 环境和四个受限 CPU 容器。预期先看到 `leases_active=1`、`containers_running=4`，再看到 64 MiB sidecar 中 256 MiB 分配触发 Docker OOM 事件，最后看到 owner 死亡、节点交接中被杀、新节点恢复清理，且新租约在清理完成后才授予。OOM 指内存不足，不涉及 GPU 显存。`--skip-memory-probe` 用于已有内存限额证据时只补故障恢复。脚本使用唯一标签清理自身容器，外层超时时仍需按 `ownership.json` 检查残留。

### 13.4 Results 与验证边界

本轮在用户指定的私有节点后台执行真实 Harbor/Docker 和独立 Ray 实验，没有启动模型或训练，也没有使用 GPU。代码为 `b52a82d` 加主工作区补丁，镜像、依赖导入来源、节点身份和源码 SHA-256 记录在结果文件中。本轮不做 ON/OFF 性能比较，固定 baseline `0eeead861ffafed6bf31c962d259fd847775dcf9` 保持不变。

| 验收项 | 实测结果 | 证据 |
|---|---|---|
| 租约与真实容器计数分开 | 1 条租约、4 个运行容器。第二次运行观测主存 69.25 MiB，声明预算 384 MiB | [资源与限额记录](../../env-service-verification/results/resource-recovery-20261027-r2/evidence.json) |
| 所有容器的资源限制 | 两个 main 各 0.5 核、128 MiB，两个 sidecar 各 0.25 核、64 MiB，Docker 实际配置吻合 | 同上 `containers` |
| 内存硬限制 | 64 MiB sidecar 尝试分配 256 MiB，进程退出 137，Docker 记录该容器的 `oom` 事件 | [事件原始记录](../../env-service-verification/results/resource-recovery-20261027-r2/oom-events.jsonl) |
| 清理交付后节点被杀 | 待办仍在 Router，旧四个容器仍存在时新请求为 `wait` | [恢复实验](../../env-service-verification/results/resource-recovery-20261027-r4/evidence.json) |
| 替代节点接手清理 | 节点 actor 已替换，旧容器全部删除、待办确认后新请求获准，最终租约和实际容器均为 0 | 同上 `node_replaced`、`recovery_admissions`、`final_sample` |
| 最终物理释放 | 本轮各次运行所属容器、网络和隔离 Ray 进程均已退出 | [收尾核验](../../env-service-verification/results/resource-recovery-20261027-r4/physical-release.json) |

失败记录也保留：第一次在 Harbor 的整数 CPU 配置校验处退出，尚未启动容器。第二次资源计数和限额已通过，但脚本读取的容器内 cgroup 根路径失败计数不增加，断言中断，随后由 Docker OOM 事件补齐内存限制证据。第三次已验证“交付后仍保留”和“清理前阻塞”，但实验设置的 2 秒健康检查期限耗尽重试预算，未完成接手清理。[第三次日志](../../env-service-verification/results/resource-recovery-20261027-r3/driver.log)保留该失败。最终运行使用默认 30 秒健康检查期限，暂停点返回未完成清理数，杀掉节点后恢复成功，退出码 0。最终运行跳过已验证的内存超限探针，不重复制造 OOM。

全套 Env Service 回归在第一阶段通过 291 项，后续针对新增故障分支和遥测的 33 项通过，两组包含重复测试，不能相加当作独立用例数。新增生产文件、测试和实验脚本的定向 Ruff 检查及格式检查通过。全仓 `pre-commit run --all-files` 在隔离副本执行，仍因 `docs/conf.py`、历史实验脚本等已有风格问题失败，未将自动修改复制回主目录，见[检查日志](../../env-service-verification/results/resource-recovery-20261027-r4/precommit.log)。

本次是实际容器限额与节点清理恢复的组件验收。两套 Harbor 环境由脚本直接启动来覆盖完整容器集合，不是运行真实 science verifier 的评分流程。没有证明 CPU 运行耗时随配额变化、真实多节点负载派发提速、GPU 状态重建、Router 进程重启恢复、跨节点 episode 重试或完整 science 训练收益。可信进程归属逻辑仍由另一进程维护。以上修改保存在主目录，未提交、推送或合并。
