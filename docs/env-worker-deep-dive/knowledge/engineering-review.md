# Env 工程问题审查与修复顺序

> **核心结论**：已有统一 GPU 执行、命令借还、MPS 并发、物理归还确认、资源遥测和失败监督。下一步优先补配置一致性与故障恢复闭环，再做队列效率和性能决策。自动 command/episode、通用卸载和跨节点迁移属于后续能力，不应与当前正确性问题混为一谈。
>
> 审查日期为 2026-11-05，主目录 HEAD `0326d46`，包含已有未提交文档与教学脚本。本文是有边界的源码 review，不是部署验收或全仓安全审计。核对 Env Service 账本、节点健康、MPS、GPU 作业管理及 KernelBench verifier，其他任务沿已有报告标注证据范围。本次不修改生产实现、不启动 Ray 集群、Docker 或 GPU 实验，不查询远端合并和实际部署状态。

## 目录

- [1. 全局流程与审查范围](#1-全局流程与审查范围)
- [2. 已完成的旧问题](#2-已完成的旧问题)
- [3. 优先解决的工程问题](#3-优先解决的工程问题)
- [4. 后续能力与验证缺口](#4-后续能力与验证缺口)
- [5. 建议修复包与验收条件](#5-建议修复包与验收条件)
- [6. 跟着图读源码与 CPU 验证](#6-跟着图读源码与-cpu-验证)
- [7. 文档维护范围](#7-文档维护范围)

## 1. 全局流程与审查范围

```text
[1] runner / 训练启动方 → 创建或复用 EnvRouter
                              ↓
[2] 节点健康检查 → Ledger 准入 → 会话绑定 unit
                              ↓
[3] native EnvWorker / Compose 创建容器 → 提交受管 GPU 作业
                              ↓
[4] direct 命令队列 或 MPS 作业队列 → 设备授权 → GPU 执行
                              ↓
[5] 结果 / 评分 → 撤权与物理回收 → 归还预算
                              ↓
[6] 故障分支：暂停准入、隔离未确认资源 → 重试清理或人工恢复
```

图例：图里每个部分对应的源码位置。

| 图中步骤 | 函数、职责与源码 |
|---|---|
| 1 | [create_env_router](../../../psrl/workers/env_service/router.py#L704)，命名 actor 创建或复用 |
| 2 | [RayEnvRouter.try_admit](../../../psrl/workers/env_service/router.py#L657)、[EnvUnitLedger.try_admit](../../../psrl/workers/env_service/ledger.py#L112)，检查节点与预算后发放租约 |
| 3 | [ManagedGpuExecution._init_gpu_execution](../../../psrl/workers/env_service/managed_execution.py#L17)，两种容器提供者的公共 GPU 执行入口 |
| 4 direct | [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L58)，按 unit 排队和准入 |
| 4 MPS | [MpsDomain.admit](../../../psrl/workers/env_service/parallel_gpu.py#L208)、[launch](../../../psrl/workers/env_service/parallel_gpu.py#L267)，预算检查与启动 |
| 5 | [MpsDomain.cleanup](../../../psrl/workers/env_service/parallel_gpu.py#L342)、[release](../../../psrl/workers/env_service/parallel_gpu.py#L408)，确认回收并释放记录 |
| 6 | [RayEnvRouter._check_node](../../../psrl/workers/env_service/router.py#L516)、[_health_task](../../../psrl/workers/env_service/router.py#L557)，失效隔离和有界重试 |

actor 是 Ray 管理的常驻服务进程，unit 是配置中的资源单元，lease 是一次获准会话的资源与身份凭据。进程之间通过异步调用协作，图不表示所有步骤单线程顺序执行。这些源码如果有理解上的问题，可以随时追问。

## 2. 已完成的旧问题

| 旧问题 | 本次核对后的状态 | 不能扩大的结论 |
|---|---|---|
| EnvWorker 与 Harbor 的 GPU 执行逻辑分散 | 已共用 ManagedGpuExecution 和作业接口 | 创建容器、网络和所有旧任务入口并没有全部合并 |
| 同容器只能执行一个 GPU 作业 | MPS 后端已有逐作业并发、取消、回收 | direct 仍同环境串行，不是任意命令自动并发 |
| 只能等整个 episode 才归还 GPU | direct command 支持逐命令借还和异步归还 | 命令内部等待模型不自动触发归还 |
| 缺乏显存归属与物理资源遥测 | 已有 eBPF 身份、进程显存归属、容器资源采样 | 不等于任务树显存硬隔离或计算利用率精确归因 |
| 失败完全没有恢复 | 已有重试、整组补采、执行监督和健康排除 | 不等于原请求跨 worker 迁移或 Router 重启恢复 |
| MPS 收尾管理连接竞争 | 已有按 daemon 串行的控制通道与定向训练回归 | 不等于所有服务失联和驱动异常已验收 |
| KernelBench 不能走 command verifier | `0517547` 已接入显式受管 verifier | 当前要求专用单卡 direct、一个会话，不支持并发测速 |

依据分别见[当前设计](current-design.md)、[统一执行层](unified-gpu-execution.md#10-当前结构与前两阶段的对照)、[恢复专题](worker-retry-and-redispatch.md)和 [KernelBench 校验入口](../../../examples/kernel_bench/verifier.py#L8)。旧审查中的 TODO 必须按后端和版本重新解释。

## 3. 优先解决的工程问题

优先级是本次建议：P1 表示扩大使用范围前应补齐，P2 表示在现有受控范围内可继续使用但会限制效率或长会话。没有将本次未复现的风险写成线上事故。

### R1：同名 Router 复用缺少配置一致性核对，P1

**静态确认的入口缺口。** `create_env_router` 使用 `get_if_exists=True` 获取同名 actor，之后只调用 `ping` 和可选预热，没有比较现有服务的 unit、预算、后端或节点策略。源码见图中 1。

教学例子：同一个 Ray 命名空间里，A 用名字 X 启动预算 80 GiB 的 Router。B 改成 40 GiB 或切换 direct/MPS，但仍使用 X。复用既有 actor 的路径没有主动拒绝配置差异，调用方可能以为新配置已经生效。没有运行真实双 driver 复现，本项不宣称当前实验已受影响。

建议在复用握手中返回规范化配置指纹、协议版本和实际资源配置。不一致应明确拒绝，不能直接热改有活动租约的服务。验收必须覆盖同配置复用、预算差异、模式差异和共享设备身份差异。

### R2：安全隔离之后缺少完整恢复通道，P1

**确认的能力限制，不是要求撤掉安全检查。** `_health_task` 达到默认 3 次失败后停止探测，`force=True` 也不会绕过预算。已有测试明确要求此行为，见 [test_unhealthy_node_exhausts_retry_budget_without_spinning](../../../tests/env_service/test_design_remediation.py#L323)。因此 Docker 恢复可用不代表该 Router 会重新接纳节点，目前缺少公开的受控重新探测入口。

另一个更复杂的分支是节点 actor 已死，但仍有活动租约、借卡绑定或 GPU 清理待办。`_check_node` 此时保留旧句柄，不盲目重建。MPS `register` 发现孤儿服务也拒绝接管，见 [MpsDomain.register](../../../psrl/workers/env_service/parallel_gpu.py#L45)。这些保护避免遗忘物理资源，但不提供完整恢复流程。

建议先做只读诊断与管理员触发的有限恢复：列出阻塞原因和所属资源，执行物理核对，确认归属后清理，最后才允许新探测。不能用清空字典、直接解除隔离或全局重启 Docker 代替。已有 CPU 容器清理接手实验不覆盖 GPU 状态重建。

### R3：Router 重启与跨 Router 设备唯一所有权，扩展部署前 P1

**确认的架构边界。** 账本的租约、待清理记录、预算和测量保存在内存字典，见 [EnvUnitLedger.__init__](../../../psrl/workers/env_service/ledger.py#L66)。一个账本内校验配置，不等于不同 Router、不同 Ray 集群之间有统一设备所有权。

教学例子：A、B 两个独立 Router 都配置同一张 80 GiB GPU，各自在尚未实际分配显存时承诺 60 GiB。单个账本均未超额，但总承诺达 120 GiB。实测显存不能替代跨控制器的预约协调。本次未运行双 Router GPU 竞争实验。

如果仍是一套受控实验独占资源，可先明确单控制器部署约束。若要支持多用户或长期服务，则需要设备所有权协调、可恢复的租约状态，以及防止旧控制器在失去所有权后继续授权的代际凭据。不能只添加自动重启。

### R4：MPS 服务故障与真实接入证明仍缺验收，P1 验证项

`launch` 检查容器身份、IPC 和管道挂载，配置 MPS 环境后放行作业。它没有在运行期间持续证明该客户端仍由目标 MPS 服务管理。参见图中 4 和 [MPS 的故障边界](mps-parallel-execution.md#8-原理与当前效果应该怎样理解)。这是已知验证缺口，不能写成“已经测出静默退回普通 CUDA”。

建议先限定环境专用卡，覆盖服务在启动前消失、连接过程中失联、执行中退出、取消回执缺失四类故障。验收看作业是否明确失败、预算是否保留、相邻作业是否受影响、最终物理释放是否确认。还要保留 cgroup v1、可信任务域、客户端显存限额按进程生效等限制，不把该实现当作恶意多租户隔离。

### R5：严格队首准入会闲置可用资源，P2

direct 的 `acquire` 和 MPS 的 `admit` 都先找队首，后来的请求不能越过它。FIFO 指先进先出，这是确定的策略，不是竞态或账本错误。

教学例子：unit 总预算 80 GiB，活动作业占 50 GiB。队首先来一个需要 40 GiB 的 A，后面 B 只需要 10 GiB。A 放不下，B 虽然放得下也要等，30 GiB 余量暂时闲置。MPS 并发机制无法自动解决这个上层排队问题。

建议考虑有限回填：允许能放下的小任务先运行，同时限制绕过次数、等待年龄或为队首保留容量，避免大任务一直等不到。是否缩短整组完成时间必须用相同工作量验证，不能仅看瞬时 GPU 占用。目标仍可保持整组反馈完成时间，不必另引入公平性优化目标。

### R6：作业数量上限是会话累计值，P2

[GpuJobManager.start](../../../psrl/workers/env_service/gpu_jobs.py#L61) 根据 `len(self.jobs)` 限制默认 128 项，已完成作业仍保留。MPS `submit` 也按同 lease 的累计记录数限制 128 项，见 [submit](../../../psrl/workers/env_service/parallel_gpu.py#L133)。所以即使每次只跑一条，第 129 次提交也会被拒绝。这是有意的有界保护，不是已确认内存泄漏。

短 episode 通常不会触发，但长会话或持久执行器池需要明确产品契约。若支持更长会话，应把活动上限与历史保留上限分开，提供终态确认和有界归档，保留去重与取消所需身份，不能直接删除所有完成记录或无限调高上限。

## 4. 后续能力与验证缺口

| 方向 | 当前状态 | 建议 |
|---|---|---|
| 自动 command/episode | 静态配置，部分任务有显式 command 接入 | 先建立任务能力声明，说明是否常驻 GPU、命令边界是否可靠、是否可重跑 |
| episode 自动卸载与恢复 | 没有通用实现 | 先选支持显式保存恢复的一个真实任务，测保存、加载、重建成本，再决定是否让位 |
| 下一条命令自动换卡或跨节点 | 会话仍绑定 unit，动态执行池是提案 | 先解决文件版本和执行输入输出，再做同节点多卡，最后考虑跨节点 |
| MPS 自动选择并发数 | 准入关注容量，没有预测并发吞吐收益 | 先记录排队、初始化、执行、回收分段耗时，再做小规模任务特征分析 |
| 性能评分与并发干扰 | KernelBench command 已限制为专用 direct | 不为提高吞吐拆掉评分隔离，正确性评分和计时评分分开处理 |
| vLLM 共卡 | 有显式共享和睡眠借卡分支 | 环境专用 MPS 验证不能替代真实训推共存与模型性能边界验证 |
| 原请求跨 worker 重派发 | 有重试与补采，缺完整转移协议 | 先确认旧执行终止，再引入尝试身份、旧结果隔离和有界重跑 |
| 多节点派发吞吐与长期运行 | 有健康过滤、容量与负载偏好 | 不能将源码接入或单卡组件验收等同于长期多节点收益 |

已有 MPS 真实训练证据是训练段约一成收益，短流程整体不一定更快。高显存压力下减少 OOM、MPS 计算共享、完整训练提速是三个不同结论。具体数字继续以[实验报告](../experiments/mps-training-comparison.md)为准，不为 review 追加训练。

## 5. 建议修复包与验收条件

1. **配置与恢复入口**：先做 R1、R2。配置不一致能够明确报错，节点故障有可读原因，恢复必须经过新的物理核对。范围有限、能直接降低下一次实验误判和人工排障成本。
2. **MPS 故障矩阵**：处理 R4。先补服务失联和清理回执缺失验证，再根据失败点修改实现。不是先增加并发数。
3. **长会话与队列效率**：有实际需求时做 R5、R6。保护大任务，同时验证整组完成时间和有界历史存储。
4. **长期或多用户部署基础**：启用该部署目标前必须完成 R3，不能排在跨节点大规模实验之后。
5. **动态调度研究**：建立能力声明和分段成本数据之后，再推进自动模式、动态放置和卸载预测。把新策略与故障恢复分开验收。

没有要求现在把这些全部实现。用户可以按近期目标选修复包，单控制器实验与多用户服务的优先级不同。

## 6. 跟着图读源码与 CPU 验证

阅读顺序为：图 1 的创建入口检查配置 → 图 2 的健康状态和账本准入 → 图 3 的每环境作业管理器 → 图 4 两个队列分支 → 图 5 的回收确认 → 图 6 的失败隔离。函数链接见图例。MPS 作业通过异步任务并发执行，健康检查和遥测也独立推进，不能把这条阅读路径理解成全局串行锁。

以下命令在已激活项目环境、仓库根目录执行，不创建 Docker 容器、不运行 GPU 内核：

```bash
PYTHONPATH=. python -m pytest tests/env_service/test_design_remediation.py tests/env_service/test_parallel_gpu.py tests/env_service/test_gpu_jobs.py tests/env_service/test_mps.py -q
python docs/env-worker-deep-dive/env-service-verification/scripts/model_concurrent_gpu_jobs.py
```

测试使用故障替身，核对清理未确认时保留预算、并行作业独立取消、输出与归还分离、健康重试耗尽后拒绝。教学脚本对应图 4、5 的预算与回收，使用模拟资源，预期低压力四项可进入、高压力只进入一项、回收确认前不能重新分配。它不是 R1 双 driver、R3 跨 Router 或 R4 服务失联的真实验证。

本次 51 项针对性 CPU 回归通过，耗时 8.77 秒。教学脚本输出低压力准入 4 项、高压力准入 1 项，确认前保留预算为真，取消 B 后 A/C/D 保留授权。审查中未重跑历史 GPU 性能对照，不将历史测试数算作本次通过数。

四篇新增或编辑文档的 182 个本地链接及源码行号范围检查通过，新增目录锚点通过。全仓 pre-commit 在临时隔离副本执行，Ruff 与 prose 检查仍因已有代码问题失败，未将格式化改动带回主目录。本次未修改任何生产代码或测试文件。

## 7. 文档维护范围

本文汇总跨组件工程问题，不取代[当前设计](current-design.md)的架构入口。旧 [2026-10-27 审计](../experiments/router-paired-latency-and-design-review.md)和 [AIRS 修复前审查](implementation-and-gaps.md)保留固定阶段证据，通过索引跳转到当前结论。MPS 专题已有服务故障、隔离和性能边界，无需重复修改其架构描述。当前设计增加本次 review 入口，不整体刷新其历史核对日期。
