# EnvRouter 低负载配对实验与 Gate A/B 设计审视

> **最新修复状态（2026-10-27）**：资源遥测和清理交接见[第 13 节](#13-物理资源遥测与清理交接确认)。关于“重派发尚未完成”，应准确理解为同请求跨 worker 转移协议未完成，框架早已有同 worker 重试和整组补采。新的[Worker 恢复专题](../knowledge/worker-retry-and-redispatch.md)核对了固定 baseline 中的实现，并记录本轮执行监督、熔断修复和真实 Ray/Docker 故障验证。这次 worker 改动尚未提交、推送或合并。第 1 至 13 节保留对应阶段的审计与实验条件。

> **核心结论**：当前“会话资源准入 + 命令阶段 GPU 预算 + 确认归还”的方向适合受控的多轮任务。Gate A 的预算拦截、共享准入和高压力下减少失败已有真实证据。旧 Gate B 是基于遥测的软节流，没有证明它能保证计算互斥或奖励归因。当前命令模式主动跳过旧 Gate B，不能将这条新路径的收益记到 Gate B 上。
>
> 本文对照用户指定的原始设计稿，区分原提案、当前实现、历史证据、配对实测和 CPU 教学输入。低负载数值来自此前同期 OFF → ON → ON → OFF 两对实验，见第 2 节。2026-10-21 新增第 9、10 节源码审计：统一后端、健康重派发与活动归因仍未闭环；原生 CUDA science 题尚需适配终端工具的真实 GPU 作业生命周期。本次没有运行模型训练，也没有改动被测生产代码。

## 目录

- [1. 原设计与当前执行链](#1-原设计与当前执行链)
- [2. Results：低负载同期配对](#2-results低负载同期配对)
- [3. Gate A/B 的收益究竟验证了什么](#3-gate-ab-的收益究竟验证了什么)
- [4. 为什么 Gate B 不能保证奖励归因](#4-为什么-gate-b-不能保证奖励归因)
- [5. 当前方案的合理部分与适用边界](#5-当前方案的合理部分与适用边界)
- [6. 建议的改进顺序](#6-建议的改进顺序)
- [7. 跟着图读源码与复现](#7-跟着图读源码与复现)
- [8. Gate B 与上一轮 ON 能否同时使用](#8-gate-b-与上一轮-on-能否同时使用)
- [9. 原设计剩余工作审计](#9-原设计剩余工作审计)
- [10. Science 原生任务能否使用 command](#10-science-原生任务能否使用-command)
- [11. 2026-10-27 剩余问题复核](#11-2026-10-27-剩余问题复核)
- [11.1 已修复及不能继续使用的旧结论](#111-已修复及不能继续使用的旧结论)
- [11.2 仍存在的正确性与故障恢复问题](#112-仍存在的正确性与故障恢复问题)
- [11.3 尚未完成的设计目标与非目标](#113-尚未完成的设计目标与非目标)
- [11.4 可复现检查与建议顺序](#114-可复现检查与建议顺序)
- [12. 2026-10-27 修复实现与验收边界](#12-2026-10-27-修复实现与验收边界)
- [12.0 简明解释：sidecar 与四项修复](#120-简明解释sidecar-与四项修复)
- [12.1 全局执行链与源码图例](#121-全局执行链与源码图例)
- [12.2 暂停归属和逐卡预算怎样工作](#122-暂停归属和逐卡预算怎样工作)
- [12.3 容器集合与健康派发的具体变化](#123-容器集合与健康派发的具体变化)
- [12.4 跟着图读源码与可运行验证](#124-跟着图读源码与可运行验证)
- [12.5 Results 与未覆盖边界](#125-results-与未覆盖边界)
- [13. 物理资源遥测与清理交接确认](#13-物理资源遥测与清理交接确认)
- [13.1 原提案还缺什么](#131-原提案还缺什么)
- [13.2 两类资源数值与清理确认的意义](#132-两类资源数值与清理确认的意义)
- [13.3 跟着图读源码与复现](#133-跟着图读源码与复现)
- [13.4 Results 与验证边界](#134-results-与验证边界)

## 1. 原设计与当前执行链

```text
[1] AgentLoopManager 选择执行 worker / 节点
                ↓
[2] runner → EnvRouter → Ledger：会话准入，获得租约
                ↓
[3] Harbor 在该 worker 所在节点创建容器
                │
                ├─ episode 模式：显存声明随整条任务保留
                │      → [B] 模型响应出口读取 GPU 遥测
                │          空闲或超时 → 放行回复 → agent 执行工具
                │
                └─ command 模式：CPU、主存和容器配额保留
                       → 普通 CPU / 模型交互
                       → [4] 显式 exec_gpu → 命令预算检查与设备授权
                       → CUDA 命令完成
                       → [5] 后台撤权、确认旧句柄和 CUDA 进程退出
                       → 释放 GPU 配额，其他命令可进入
                       → CPU / 下一轮规划可与归还检查重叠
                ↓
[6] 评分、清理所属容器、释放租约

独立分支 [L]：绑定整组 vLLM 副本，确认 sleep → Docker → 确认归还 → wake
              当前不能与 command 模式混用
```

图例：图里每个部分对应的源码位置：

| 图中部分 | 调用、职责与源码 |
|---|---|
| 1 | [get_dispatch_plan](../../../psrl/workers/agent_loop/manager.py#L984) 根据容量快照选择 worker，实际准入仍须经过账本。配对小实验直接调用 runner，没有测跨节点放置 |
| 2 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L258)、[Ledger.try_admit](../../../psrl/workers/env_service/ledger.py#L105)：检查可承载资源、登记会话 |
| 3 | [_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L98)：生成设备映射，选择 ManagedDockerEnvironment，再交给 Harbor |
| B | [SessionRouter 响应出口](../../../psrl/workers/gen/session_router.py#L444) 调用 [_wait_for_gpu_idle](../../../psrl/workers/gen/session_router.py#L120)。[runner 的模式分支](../../../examples/sciaccel_rl/runner.py#L154) 对命令模式与模型借卡模式传入空绑定地址，跳过此 Gate |
| 4 | [exec_gpu](../../../psrl/workers/env_service/docker.py#L73) 调用 [GpuCommandScheduler.acquire](../../../psrl/workers/env_service/command_gpu.py#L47)，节点 [grant_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L120) 开启已分配设备访问 |
| 5 | [begin_return / _return](../../../psrl/workers/env_service/command_gpu.py#L109)、[revoke_gpu_command](../../../psrl/workers/env_service/command_gpu_node.py#L131)：预算保持到物理确认完成，失败进入隔离状态 |
| 6 | [runner finally](../../../examples/sciaccel_rl/runner.py#L550)、[Node.cleanup](../../../psrl/workers/env_service/node.py#L68)：正常与异常路径按归属回收 |
| L | [GpuLendingBinding.set_phase](../../../psrl/workers/env_service/lending.py#L27)、[配置约束](../../../psrl/workers/env_service/types.py#L51)：整副本的协作借还，独立于命令级并发 |

原始设计见工作区的 [2026-09-21-env-service-design.md](../../../../2026-09-21-env-service-design.md)，该文件是本地历史设计材料。其第 6.3 节把 Gate A 定义为容器创建前的容量准入，把 Gate B 定义为模型回复出口的有界节流。这里的“准入”就是决定一个请求现在能否开始，租约则是获准资源的归属登记。

当前实现已经超出该提案的最初范围：Harbor 的环境子类参与 GPU 命令执行前后，Router 接收每条命令的控制请求，但不转发命令数据流。原设计“只在 Job.create 前做一次资源控制”“GPU 显存必须一直保留到 episode 结束”的描述，只适用于 episode 模式。命令模式将显存承诺缩短到明确的执行阶段，并通过设备权限和物理检查约束交接。

## 2. Results：低负载同期配对

本轮运行于 2026-10-20，采用 **OFF → ON → ON → OFF** 的顺序平衡设计，避免四次都让 ON 吃到后运行的缓存优势。两对均使用真实 science runner、Harbor、Docker 和 CUDA，属于合成容器工作负载，不包含模型生成或训练。每组 8 条任务、16 条 GPU 命令全部成功，无重试，合计 32 条任务、64 条命令。

| 指标 | OFF | ON | ON 相对 OFF |
|---|---:|---:|---:|
| 第一对，8 任务工作阶段 | 89.46 s | 77.83 s | 少 13.00% |
| 第二对，8 任务工作阶段 | 82.23 s | 83.02 s | 多 0.96% |
| 两对工作阶段合计 | 171.68 s | 160.84 s | 少 6.31% |
| 单任务平均延迟，每边 16 条 | 41.741 s | 39.045 s | 少 6.46% |
| GPU 命令平均调用延迟，每边 32 条 | 3.827 s | 3.718 s | 少 2.84% |
| CUDA 脚本平均时间，含固定持有 | 2.989 s | 2.993 s | 多 0.13% |
| 两对驱动总时长合计，含初始化及收尾 | 213.96 s | 217.15 s | 多 1.49% |

![同期配对的任务阶段、含初始化的总时间、任务和命令延迟](../env-service-verification/results/router-paired-review-20261020/paired-latency.png)

**应怎样解读。** 修复后的低负载 ON 已接近 OFF，本次工作阶段合计更短，但两对的差异方向不一致，不能把 6.31% 宣布为稳定提速。这里仅有两对运行，16 条任务共享运行环境，不能当成 16 次独立重复试验，更没有可靠的尾部百分位或统计显著性结论。图中黑线是样本平均值，散点是每条观测。

**冷启动仍有成本。** 第一对初始化／工作前准备为 OFF 18.97 秒、ON 28.50 秒，第二对分别为 17.68 秒、22.60 秒。ON 除 Ray 外还需创建 Router 和节点服务，首次预热超过 8 秒独立准备窗口约 1.54 秒。因此，包含初始化和驱动收尾后的总时间，第一对为 111.65 对 108.76 秒，第二对为 102.31 对 108.38 秒，两对合计 ON 仍多 1.49%。长期复用服务可摊薄这笔开销，但不能仅展示工作阶段而隐藏短作业启动成本。此处驱动总时长不包含 Python 导入前的进程启动和驱动结束后的独立物理释放复核。

**控制路径的长尾确有改善迹象。** 工作阶段事件循环最大停顿，第一对由 3.408 秒变为 0.526 秒，第二对由 4.353 秒变为 0.441 秒。OFF 的采样栈仍捕获同步 Docker 探测和进程创建。ON 第二对也捕获进程创建停顿，说明不能把所有这类长尾都宣称已消除。CUDA 脚本平均时间基本相同，与“改进主要在宿主控制路径”这一解释一致，但并非每一秒差异都可由这些采样唯一归因。

这比较的是用户指定旧提交与当前完整实现，ON 包含同步准备移出事件循环等优化，所以不能把净差异全部解释为账本算法本身的收益。两组 ON 均观察到最多 4 条活动 GPU 命令，所有账本采样均未越过声明预算。本轮低负载没有触发超额排队，不能据此再次声称验证了过载保护。

原始统计见 [paired-analysis.json](../env-service-verification/results/router-paired-review-20261020/paired-analysis.json)，生成入口为 [analyze_paired_router.py](../env-service-verification/scripts/analyze_paired_router.py)。逐组及最终清理已通过，最后 [physical-cleanup-final.json](../env-service-verification/results/router-paired-review-20261020/physical-cleanup-final.json) 中本次容器、网络、actor 和私有 Ray 进程为空，目标卡无 CUDA 进程，应用显存约 0.19 MiB。到达预定两对后即停止，没有追加训练或补跑。

**实验约束。** 固定 OFF 为 `0eeead861ffafed6bf31c962d259fd847775dcf9`，保留其静态 GPU 映射和进程内信号量，没有添加整卡锁。必要运行补丁为传递 `override_gpus`、移除节点级镜像 prune，以及已有依赖／HTTP 运行修复，全部保存在 [off.patch](../env-service-verification/results/router-paired-review-20261020/off.patch)。最新 ON 是 `37a06ea8d32b05bbbf385ad40cb404b30ba01680` 加同步准备优化，冻结在独立 checkout，见 [on.patch](../env-service-verification/results/router-paired-review-20261020/on.patch)。

四组均为同一张 H20、相同镜像、八任务、四并发、两批，每条任务执行两次真实 CUDA 命令。每次分配 1 GiB、持有 2 秒，中间 CPU 等待 8 秒，ON 每条申报 2 GiB，unit 预算 80 GiB。工作量只做小型 CUDA 写入读回及持有，不代表持续高强度 GPU 运算。各组使用独立 Ray 进程，相同的 8 秒独立准备工作、同一套计时与栈采样，ON 在该工作期间预热节点服务，未能完全重叠的部分计入准备时间。

每组工作预算 240 秒、进程预算 360 秒，总实验预算 1800 秒。预先固定两对，不根据结果继续补跑。每组结束先核验本次容器、网络、actor、Ray 进程与目标 GPU 释放，再开始下一组。节点其他 GPU 上存在其他任务，本次没有独占整个节点，因此顺序平衡只能减轻时间漂移，不能消除所有 CPU、存储和驱动干扰。

单任务端到端延迟包含容器准备、命令执行、评分、runner 清理和驱动读取结果的小开销。GPU 命令调用延迟包含 ON 的相应准入及返回控制，不能解释成单纯 CUDA 计算时间。任务批次时间不含 Ray 初始化和最终核验，初始化及最终清理另列。四并发的总时长不能除以任务数当作单任务 latency。

## 3. Gate A/B 的收益究竟验证了什么

| 目标 | 核对到的证据 | 可成立的结论与边界 |
|---|---|---|
| Gate A 防止声明预算超订 | 历史 [fixed-admission-boundaries](../env-service-verification/results/fixed-admission-boundaries/summary.json) 的真实容器 GPU、CPU、主存单约束检查均通过。GPU 预算 1200 MiB、每条申报 512 MiB，第三条等待，释放后恢复 | 已验证容量准入功能，容器上限设为 8 排除了数量限制这个混杂因素 |
| 共享 GPU 预算与安全回填减少失败 | 历史同卡等完成量 [pressure 对比](../env-service-verification/results/router-baseline-control-20261020/comparison.json)：两边均完成 16 任务，OFF 15 次 OOM、266.52 秒，ON 零 OOM、207.84 秒 | 已有高压力执行收益，属于容量准入、命令阶段复用和控制实现的组合效果，不是纯 Gate A 或旧 Gate B 单独消融 |
| Gate B 共享卡不被常驻显存误阻塞 | 早期[报告第 3.3 节](../env-service-verification/README.md#33-gate-b邻居在同一张卡上时回复要等多久)记载 8 秒等待上限下，共享卡空闲常驻显存由约 8.4 秒变为 0 秒。本次 CPU 检查亦验证共享模式只需一次读取 | 原始完整独立日志未找到，历史秒数仅按报告记载引用。本次 CPU 检查验证判定逻辑，不替代真实 GPU 性能复测 |
| Gate B 查询正确执行节点 | 历史 [fixed-multinode/evidence.json](../env-service-verification/results/fixed-multinode/evidence.json)：远端保留分配时，两端调用均等待约 2.45 秒，`remote_gate_b_passed=true` | 已验证真实跨节点遥测方向与受限等待，未覆盖完整模型 HTTP 出口到评分的归因链 |
| Gate B 防止后台邻居被计入本任务奖励 | 原设计第 9 节要求 `nohup` 邻居、第二任务和逐进程 `pmon` 对照。已有验证报告[未覆盖项](../env-service-verification/README.md#97-不能据此签收的设计范围)及原始结果中没有完成这一验收的证据 | **未证明奖励归因收益，更未保证互斥。** 本轮 CPU 检查揭示空闲同时放行和超时仍放行边界 |
| 租约过期后真的回收容器 | 历史 [fixed-crash-recovery](../env-service-verification/results/fixed-crash-recovery/evidence.json) 中 worker 被杀后 `after_worker_reaper=[]` | 已验证这一历史故障恢复路径，不能代替控制器重启、网络分区或所有分布式故障验证 |
| Docker 与 vLLM 同卡借还 | 历史 [gpu-lending-real-20261015](../env-service-verification/results/gpu-lending-real-20261015/summary.json)：两次同卡 Docker/CUDA 与真实 vLLM 交替，生成三次、每次 16 token，物理资源已清理 | 功能循环通过，使用推理适配器绕过 NIXL，未证明长期吞吐、训练权重同步或任意多任务借卡收益 |

本轮另外执行当前 Gate A/B 相关 CPU 回归 **47 项通过**，见[测试日志](../env-service-verification/results/router-paired-review-20261020/gate-tests.log)。这些回归使用可控输入，不能当成 GPU 训练收益。新的低负载配对实验也没有触发 OOM 或旧 Gate B，更没有测模型能力改善。

**澄清：Gate B 有专门设计的测试和实验，不是完全没有验证。** “没有证明奖励归因收益”不能简写为“Gate B 没有实验”。现有材料分为：

| 入口 | 实际覆盖 | 执行证据与限制 |
|---|---|---|
| [test_gate_b.py](../../../tests/env_service/test_gate_b.py) | 11 个测试，覆盖空闲放行、繁忙后转空闲、多卡等待、共享卡常驻显存、遥测缺失、超时放行和绑定解析 | 使用脚本化遥测，本轮已与 Ledger 测试合计执行 47 项通过，不是 47 个 Gate B 测试 |
| [gate_b_experiment.py](../env-service-verification/scripts/gate_b_experiment.py) | 专门的真实 GPU 实验，一种邻居只占约 22 GiB 显存、不计算，另一种邻居真正计算约 5 秒，测量 Gate B 等待时间，等待上限设为 8 秒 | 实验脚本存在，历史报告记载了结果，但没有找到对应独立完整原始日志，不能补称为本轮新测。脚本直接调用等待函数，没有启动两个完整评分任务，也没有比较奖励误判率 |
| [multinode_check.py](../env-service-verification/scripts/multinode_check.py) | 两台节点调用 Gate B，确认都查询到指定远端节点的占用 | 保存了真实原始结果。采用 `shared=False` 与常驻分配触发等待，证明远端查询和有界等待，不是对计算争用或奖励归因的验证 |
| [audit_gate_contracts.py](../env-service-verification/scripts/audit_gate_contracts.py) | 用教学遥测调用真实函数，检查多个空闲观察者放行、共享显存判定、持续繁忙超时放行 | 本轮已运行 CPU 审计，不是实机吞吐或奖励收益实验 |

具体区分可以用一个例子：邻居计算 5 秒，Gate B 等它结束后返回，证明“看见已在计算的邻居时会等待”。但要证明“B 的 GPU 奖励不会误算 A 的计算”，还必须让 B 走完整 agent、工具和 verifier 链路，对比 Gate B 开关下的实际重叠窗口与奖励结果，并覆盖两个请求同时看到空闲、后台任务遗留及超时放行。这后半部分是原设计已提出、目前没有找到完成证据的验收项。上一轮低负载配对采用 command 模式，跳过旧 Gate B，因此不能代替这项实验。

## 4. 为什么 Gate B 不能保证奖励归因

先看原设计担忧的具体问题。`reward_gpu` 是“正确性奖励 × GPU 活动判定”。如果活动判定读取整张卡的利用率，任务 A 在 GPU 上忙，任务 B 实际只用 CPU，B 也可能看到整卡繁忙，从而被误认为使用了 GPU。这是**活动归因**问题，与显存能否装下两条任务是两件事。

当前数据构造明确把 `laps-accel-cuda` 映射为 [reward_gpu](../../../examples/sciaccel_rl/prepare/build_dataset.py#L35)，最终 [compute_score](../../../examples/sciaccel_rl/reward.py#L44) 取指定奖励键。修复类 LAPS 任务使用 `reward_repair`，不能用其训练通过来证明 `reward_gpu` 的归因正确。

Gate B 只观察状态，没有“检查后立即为某条任务保留计算窗口”的原子操作。教学时序如下：

```text
时刻 0：整张 GPU 利用率低，A 和 B 的模型回复都在等待
A 读取到空闲 → 放行 A 的回复
B 也读取到空闲 → 放行 B 的回复
随后 A 开始 GPU 工作，B 开始自己的任务和活动采样
结果：两者仍可能在同一个采样窗口重叠
```

这里“原子”指检查与占用承诺不可被另一个竞争者穿插。现有 Gate B 没有这样的占用承诺。默认还会在约 600 秒上限后放行，遥测缺失时也放行，都是有意避免模型请求无限等待的软策略。因此它可以减少某些已知繁忙时段的重叠，但不能作为奖励正确性的充分条件。

本次 [audit_gate_contracts.py](../env-service-verification/scripts/audit_gate_contracts.py) 用**教学遥测**调用真实 `_wait_for_gpu_idle`，得到两条空闲观察者都放行，持续 100% 利用率在缩短的 0.05 秒上限后仍放行。输出见 [gate-contract-audit.json](../env-service-verification/results/router-paired-review-20261020/gate-contract-audit.json)。没有伪造实机性能，也没有启动外部服务。

当前命令配额解决的是“谁可以打开设备、谁占多少显存预算”，显存足够时允许多个 CUDA 命令并行。因此它也不会自动解决整卡利用率的归因问题。后续若评分确实依赖 GPU 活动，需要明确的独占评分窗口覆盖所有相关计算者，或能够关联容器／进程的活动证据。后台进程、独立 verifier 和外部任务也必须进入这个约束，不能只给主容器排队。

## 5. 当前方案的合理部分与适用边界

**合理之处是将资源生命周期分开。** 容器里还有文件、CPU 进程和下一轮对话状态，不等于它必须继续占用显存预算。以教学数字说明：一张预算 80 GiB 的卡，两条任务各申报 51 GiB，只能同时运行一条 GPU 命令。A 命令结束并确认归还后，A 继续 CPU／模型工作，B 此时使用这 51 GiB。A 下一条命令再排队。这正适合用户关心的整体完成时间目标。

**安全交接比预测空档更基础。** `revoke_gpu_command` 先禁止新打开设备，再检查旧句柄、映射和 CUDA 进程，确定交接后才释放预算。Python 协程返回、模型开始生成、利用率瞬间为零，都不能单独替代这一步。对仍持有 CUDA 上下文的任务，目前不会强行抢占并恢复其显存内容。原稿“等待任意容器完成或销毁”的保守前提，只有在任务提供明确前台 GPU 边界且释放成功时才能缩短。

还不能忽略以下边界：

| 边界 | 当前实际情况 | 对使用者的含义 |
|---|---|---|
| 普通 science 任务透明接入 | [TruncatingTerminus2](../../../examples/sciaccel_rl/agent.py#L94)已接入显式 `psrl-gpu` 作业工具，普通 shell 不会自动识别 CUDA。三步训练的压力代理仍只显式接入注入阶段 | 不能只改 `gpu_reservation: command` 就宣称任意终端命令、原生评分和 GPU verifier 都已完成验收 |
| 预算与硬限制 | [Ledger._effective_gpu_mb](../../../psrl/workers/env_service/ledger.py#L357)已改为实测占用加每条未兑现预约，并按可信当前归属抵扣。设备权限控制仍不是每容器显存上限 | 原 `max` 漏算已修复，但已授权命令低报或外部进程继续增长仍可能竞争 |
| 外部 GPU 使用者 | 多个 unit 不重叠的校验作用于同一个 Router 的配置，不能自动约束另一个 Router、手工 Docker 或未绑定的训练引擎 | 需要设备所有权约定或统一注册。读取实时总量不能替代所有权协议 |
| 多卡 unit | 普通准入使用多卡显存之和，GPU 需求没有逐卡分布向量 | 两张各 40 GiB 的空闲卡不等于一个只用单卡的 60 GiB 请求能运行。普通单卡任务优先用一张卡一个 unit |
| 队列与回填 | 命令队列先入先出，队首不满足时后面的小请求不会绕过它 | 教学例子中剩 13 GiB、队首需 14 GiB、后面需 10 GiB，后者仍等待。公平与装箱利用率之间尚无有限回填策略 |
| 平台与跨机 | 当前设备访问控制要求可写的 cgroup v1，运行中容器不会自动迁移到另一节点 | 本轮单节点结果不证明 cgroup v2 支持、跨机命令转移或大集群吞吐 |
| 模型借卡 | 借卡分支要求完整副本绑定、一个容器位置，不能和命令模式同时配置 | 两条功能各自通过，不等于任意组合已经通过 |
| 统一后端目标 | 原稿计划合并 EnvWorker 与 Harbor。当前通用 [EnvWorker](../../../psrl/workers/env_worker/worker.py#L156) 仍存在，science 走 runner／Harbor／节点服务 | 原设计的全部路线图尚未完成，不宜笼统标记为“全部落地” |

**已修复的历史账本反例。** CPU 教学输入为容量 80 GiB，A 申报 50 GiB、实际只用 20 GiB，外部另用 20 GiB，B 申请 30 GiB。10 月 20 日的旧实现按 `max(50,40)+30=80` 放行，将来合计可能达到 100 GiB。当前缺少归属时保守计算 `40+50+30=120`，B 等待；有可信 A 当前占用 20 GiB 时计算 `40+(50-20)+30=100`，B 同样等待。120 是安全记账上界，不是实测占用。原始历史输出保留，现行审计脚本已随公式更新，见第 11 节与[公式说明](../knowledge/gpu-memory-admission.md)。不能再将旧反例写成当前未修复漏洞。

## 6. 建议的改进顺序

1. **保留会话准入与命令确认归还，明确支持契约。** 对可提供前台 GPU 边界的任务使用命令模式，覆盖普通工具调用及 verifier 的适配以后再扩大适用范围。episode 模式仍适用于无法拆分的常驻 GPU 任务。
2. **将奖励正确性作为单独验收。** 旧 Gate B 保持可选软节流，不能充当正确性屏障。真正需要 `reward_gpu` 时优先验证 CPU 任务与 GPU 邻居、后台遗留进程和独立评分器的归因，而不是继续只比较延迟。
3. **补齐设备所有权和逐卡预算。** 单一权威 Router／明确协作借卡、外部占用归因、逐 GPU 需求比更复杂的预测调度更优先。对未协调的设备使用者保持明确边界。
4. **再依据队列观测优化利用率。** 可以评估有界回填、等待时长加权和资源可用通知。预测器只能估计何时值得等，不能预测完就提前释放尚未确认归还的资源，也不能替代容器状态迁移机制。

以上是审视后的改进建议，本轮没有修改生产协议或实现这些后续功能。低负载的成本与高压力的失败减少，应使用各自实验回答，不以“任何场景 ON 都更快”为方案合理性的唯一标准。

## 7. 跟着图读源码与复现

按真实执行路径阅读，不能把两条配置分支及后台归还伪装成单线程顺序调用：

| 步骤 | 图中编号与函数 | 输入、输出或状态变化 |
|---|---|---|
| 选择 worker | [1] [get_dispatch_plan](../../../psrl/workers/agent_loop/manager.py#L984) | 输入一批任务和容量快照，输出 worker 对任务的分组，本轮实验直接从下一步进入 |
| 任务进入与会话准入 | [2] [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L258) → [try_admit](../../../psrl/workers/env_service/ledger.py#L105) | 输入资源声明、节点约束，输出等待或租约，登记持久 CPU、主存和容器预算 |
| 环境配置与创建 | [3] [_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L98) | 租约变为设备映射、环境子类和 Harbor 参数 |
| episode 分支的回复出口 | [B] [_wait_for_gpu_idle](../../../psrl/workers/gen/session_router.py#L120) | 遥测用于延迟回复，不授予独占计算权，命令模式跳过此分支 |
| command 分支的每次调用 | [4] [exec_gpu](../../../psrl/workers/env_service/docker.py#L73) → [acquire](../../../psrl/workers/env_service/command_gpu.py#L47) | 请求进入队列，预算可用时登记活动声明、授权设备，再执行命令 |
| 并行归还与规划 | [5] [begin_return](../../../psrl/workers/env_service/command_gpu.py#L109) → [_return](../../../psrl/workers/env_service/command_gpu.py#L124) | 保留预算并启动后台撤权检查，调用方在服务接管后可继续 CPU 工作，确认成功才释放显存预算 |
| 分支汇合与清理 | [6] [runner finally](../../../examples/sciaccel_rl/runner.py#L550) | 后台错误在下一 GPU 请求或收尾汇合，任务结束回收所属容器并释放会话资源 |
| 独立模型借卡分支 | [L] [set_phase](../../../psrl/workers/env_service/lending.py#L27) | 输入副本绑定、阶段和序号，在模型 sleep/wake 与容器访问之间切换，不能与 command 混用 |

源码或并发边界有理解上的问题，可以随时追问。

在仓库根目录、项目环境中运行 CPU 审计：

```bash
PYTHONPATH=. python docs/env-worker-deep-dive/env-service-verification/scripts/audit_gate_contracts.py
python -m pytest tests/env_service/test_gate_b.py tests/env_service/test_ledger.py -q
```

预期看到已知两条 50 GiB 需求的第二条等待、释放后获准，两个 Gate B 空闲观察者都放行，共享卡空闲常驻显存不等待，以及持续繁忙超过缩短上限后放行。本次已执行。外部占用例子只验证账本计算结果，不分配显存。

读取本地保存的配对结果并生成图表：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/analyze_paired_router.py \
  docs/env-worker-deep-dive/env-service-verification/results/router-paired-review-20261020
```

真实 GPU 入口仍为 [profile_science_tail.py](../env-service-verification/scripts/profile_science_tail.py)，设备、镜像和 overlay 必须现场预检。运行脚本、冻结源码、补丁与预算保存在[本轮结果目录](../env-service-verification/results/router-paired-review-20261020)。不应直接把其中节点身份复制到其他环境执行。

本次新增诊断脚本的 Ruff 检查与格式检查通过，文档链接及源码行位置已核验。全仓 `pre-commit run --all-files` 在隔离副本执行，仍因既有 Ruff 和注释风格问题失败，详见 [pre-commit.log](../env-service-verification/results/router-paired-review-20261020/pre-commit.log)。自动修正没有写回共享工作区，不把全仓检查报告为通过。

## 8. Gate B 与上一轮 ON 能否同时使用

**ON 是整套资源管理方案的实验标签，不是某一个 Gate 的名字。** EnvRouter 启用后，episode 模式可同时使用 Gate A 和 Gate B。上一轮 ON 特指命令级 GPU 预算、后台确认归还、节点预热和同步准备优化的组合，主动跳过旧 Gate B。这是当前接线策略，不是两种机制在理论上不能共存。

```text
[1] 会话进入 EnvRouter，申请 CPU、主存和容器等资源
                         ↓
[2] 模型产生下一步规划 → SessionRouter 返回回复
                         │
         旧 Gate B 启用时：GPU 繁忙就先扣住回复
         上轮命令模式：跳过此等待，agent 可继续 CPU 工作
                         ↓
[3] 真正需要 GPU：exec_gpu → 命令预算与设备授权 → 执行
                         ↓
[4] 命令输出完成 → 服务接管后台归还 → CPU／后续规划继续
                            ↓
                 撤权、确认旧访问消失 → 释放预算
                         ↓
[5] 后续 GPU 请求／收尾汇合归还结果，最终清理容器与租约
```

图例与跟着图读源码：

| 步骤 | 对应函数 | 输入、输出与状态变化 |
|---|---|---|
| 1 | [Ledger.try_admit](../../../psrl/workers/env_service/ledger.py#L105) | 会话需求变为租约，command 模式先承诺宿主资源 |
| 2 的配置 | [_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L154) | command 或模型借卡模式传入空绑定地址，跳过 Gate B，episode 模式有有效 SessionRouter 地址时可绑定 |
| 2 的响应出口 | [SessionRouter 出口](../../../psrl/workers/gen/session_router.py#L444) → [_wait_for_gpu_idle](../../../psrl/workers/gen/session_router.py#L120) | 已有模型结果等待 GPU 遥测满足条件或超时后返回，不判断下一条工具命令究竟需要 CPU 还是 GPU |
| 3 | [exec_gpu](../../../psrl/workers/env_service/docker.py#L73) → [acquire](../../../psrl/workers/env_service/command_gpu.py#L47) | 具体 GPU 命令申请预算，允许符合预算的并发，授权后执行 |
| 4 | [begin_return](../../../psrl/workers/env_service/command_gpu.py#L109) → [_return](../../../psrl/workers/env_service/command_gpu.py#L124) | 后台归还与规划并行，成功确认后才修改活动预算，不因发出归还请求就提前释放 |
| 5 | [wait_for_gpu_return / stop](../../../psrl/workers/env_service/docker.py#L68)、[runner finally](../../../examples/sciaccel_rl/runner.py#L550) | 下一 GPU 调用或结束时汇合异步错误，按归属清理 |

沿 1 → 2 → 3 → 4 → 5 阅读，4 中的归还与调用方 CPU 工作并行。`overlap_return=True` 仅适用于输出已经完成、后续 CPU 工作不依赖确认结果的调用。源码中的并发关系可以随时追问。

| 对比 | 旧 Gate B | 上一轮 ON 的命令调度 |
|---|---|---|
| 核心问题 | 尽量避开已有繁忙计算，减少下一轮工具与其他任务评分窗口重叠 | 避免受管声明超订，并在 GPU 阶段结束后复用预算 |
| 等待位置 | 模型回复返回之前，CPU 下一步也一起被延后 | 显式 GPU 命令执行之前，CPU 与规划可继续 |
| 判断依据 | GPU 利用率，共享模式不以常驻显存阻塞 | 声明显存预算、实测显存、请求归属和设备权限 |
| 对并发的态度 | 看见整卡繁忙就倾向等待，不区分是否还装得下 | 显存预算足够时允许计算重叠，不因利用率高就自动拒绝 |
| 强制程度 | 软节流，超时或遥测缺失后放行，没有独占凭据 | 对接入的前台命令实行预算与设备访问协议，仍不是每容器显存硬上限 |
| 适合的场景 | 原有 episode 任务边界粗、工具尚未精确接入时，作为可选的尽力节流 | 多轮 CPU／模型与前台 GPU 命令交替，重视总完成时间且能够显式接入的任务 |

**为什么不直接串上旧 Gate B。** 以下是假设时间线，不是实测收益：一张卡预算 80 GiB，A、B 各需 51 GiB。时刻 0，A 已在 GPU 上运行，还要 10 秒，B 的模型回复已生成。B 下一步先做 8 秒 CPU 工作，再做 2 秒 GPU 工作。

| 路径 | B 的模型回复 | B 的 CPU 工作 | B 的 GPU 工作 | 两任务均完成 |
|---|---|---|---|---|
| 上轮命令模式 | 0 秒返回 | 0 到 8 秒 | 先等待 2 秒，再于 10 到 12 秒执行 | 12 秒 |
| 命令模式再串旧 Gate B | 等 A 空闲，到 10 秒返回 | 10 到 18 秒 | 18 到 20 秒执行 | 20 秒 |

两种路径都遵守一条 GPU 命令占用的预算约束，但第二种损失了 CPU 与 GPU 的重叠。Gate B 只延迟已生成回复，不会撤销已经完成的模型生成，不过 agent 无法收到下一步指令，后续工具和再下一轮模型交互仍会被推迟。

还有另一种冲突：若 A、B 各只需 20 GiB，容量 80 GiB，两个普通命令可以同时准入。旧 Gate B 看见 A 正忙仍会推迟 B 的回复。若目标是提高整体吞吐，这可能降低并发。实际同时计算也可能因算力、带宽竞争而变慢，不能仅靠显存能装下就断言并行一定更快。

**它们能否合理组合。** 理论上可串联，也可按任务启用不同策略，但现有 command 入口明确不绑定旧 Gate B，不能把修改分支或手工绑定当成已经验证的组合。若目标是评分归因或不受邻居干扰的性能测量，更适合把“计算独占”做成命令／评分窗口的调度约束，放到同一个资源调度器中。这是后续提案：普通 GPU 命令持有共享执行许可，独占评分请求等待现有普通计算退出，并阻止新普通计算进入，直到评分完成。后台进程、独立 verifier、模型借卡和其他使用者都必须遵守相同设备约束，独占窗口才有保证。

显存预算和计算独占不是同一概念。两条 20 GiB 命令放得下，只说明容量允许，不保证整卡利用率能正确归属于某一条任务。单纯限制“评分任务之间互斥”也不够，普通命令同样可能污染评分。若能可靠按容器／进程归因，可减少独占需要，但性能计时仍可能受到邻居算力与带宽竞争影响。

对于当前吞吐目标，继续使用命令模式更合适，不默认再串旧 Gate B。对于尚未接入命令协议的老任务，episode 加 Gate B 可提供尽力节流，但不能作为奖励正确性保证。Docker 与 vLLM 同卡交接另由 sleep/wake 借还协议处理，Gate B 的空闲观察不能代替物理归还。

**可运行教学模拟。** [simulate_gate_b_composition.py](../env-service-verification/scripts/simulate_gate_b_composition.py) 仅使用标准库，对上述时间线做确定性计算，不调用生产调度器、不启动 GPU 或外部服务：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/simulate_gate_b_composition.py
```

已运行，第一例输出两种路径总完成时间 12 秒和 20 秒，第二例输出 10 秒和 12 秒，见 [gate-b-composition-teaching.json](../env-service-verification/results/router-paired-review-20261020/gate-b-composition-teaching.json)。假设没有控制开销和并发减速，且 A 持续繁忙到第 10 秒。数字用于解释等待位置与重叠的关系，不是新一轮 GPU 性能实验，也没有据此证明真实组合策略的收益。

## 9. 原设计剩余工作审计

本节保留 2026-10-21 的审计截面，其中“没有逐进程归因实现”和“原生工具没有作业入口”已被后续实现更新；当前状态以第 11 节为准。

**审计日期：2026-10-21。** 对照用户指定的原始设计稿第 6、8、9、10 节与当前本地实现。这一节是只读源码及历史证据审计，没有新启动 GPU、容器或训练。区分“未实现”“已有实现但未完整验收”“实现路线已变化”和“原稿明确排除”，不按目录中出现过文件就判定全部完成。

| 原稿目标 | 当前状态 | 仍缺少什么及源码依据 |
|---|---|---|
| Phase 0：合并 EnvWorker 与 Harbor 执行后端 | **未完成统一** | [SciAccelAgentLoop 的 Harbor 专用线程](../../../examples/sciaccel_rl/agent_loop.py#L30)与 [EnvWorker.exec](../../../psrl/workers/env_worker/worker.py#L276)、`shell.py` 仍同时存在。还没有将所有任务迁入同一后端、完成 MLGym 跨轮 shell 状态等价验收并删除旧路径 |
| Phase 0：跨 worker 容器准入、归属与清理 | **核心能力已实现，架构不同于原稿** | ON 用共享 Router 账本替代各进程单独计数，并给容器集合加标签、执行归属清理。实际是 [EnvNodeService](../../../psrl/workers/env_service/node.py#L19) 每节点服务，不是每 unit 一个统一执行所有任务的 EnvWorker。不能把“没有按原稿类名合并”说成“没有共享限流” |
| Phase 1：静态 unit、Gate A、租约与回收 | **已实现，有历史实机证据** | [Ledger](../../../psrl/workers/env_service/ledger.py#L105)、[节点清理](../../../psrl/workers/env_service/node.py#L68)覆盖准入、心跳过期后的容器回收。控制器重启后的租约恢复、网络分区等完整容错范围尚未证明，内存账本不是持久恢复系统 |
| Phase 1：完整的资源遥测 | **部分实现** | 已有显存总量、GPU 计算 PID 和预算快照，但没有原稿要求的逐 PID 利用率 `measured_util_by_pid` 与实测 `cpu_load` 派发。注意 [snapshot](../../../psrl/workers/env_service/ledger.py#L257) 中 `containers_running=len(leases)`，`cpu_used` 是声明之和，不是现场 Docker 容器数与实测 CPU 繁忙程度 |
| Phase 1：健康探测 | **启动探测已实现，持续健康调度未完成** | [节点 prepare](../../../psrl/workers/env_service/node.py#L49)执行 Docker ping、设备与遥测检查，[节点启动](../../../psrl/workers/env_service/router.py#L296)还检查 command 所需能力。尚未形成“运行中识别坏节点 → 从派发候选剔除 → 恢复后重新加入”的完整自动闭环。启动失败会保存并传播，不能视为自动换机 |
| Phase 2：GPU 映射、驱动挂载、借还 | **实现存在，部分真实验证** | 动态 overlay、设备身份核对和命令归还已验证。vLLM 同卡借还的历史小实验通过，但绕过 NIXL 的推理适配器不是完整训练权重同步验收，见第 3 节。当前 command 与模型借卡不能组合，逐卡需求和外部 GPU 所有权边界仍见第 5 节 |
| Phase 2：Gate B 与 GPU 活动归因 | **软节流已实现，归因目标未完成** | Gate B 有 CPU 测试、实机等待与远端查询证据，但没有逐进程归因，以及原稿指定的后台邻居、第二任务、完整奖励对照验收。命令模式跳过旧 Gate B，不能用 command 的吞吐结果签收此目标 |
| Phase 3：负载感知放置 | **基础容量派发已实现，完整目标未完成** | [EnvAwareDispatcher.assign](../../../psrl/workers/env_service/placement.py#L84)按 CPU/GPU 类型、空闲槽位及快照派发，最终再准入。尚未按实际编译 CPU 压力、具体每条任务的 CPU／主存／逐卡显存需求精确放置，整组采样还共用一个放置决策 |
| Phase 3：坏放置后的有界重派发 | **通用恢复基础存在，环境专用闭环未完成** | [manager](../../../psrl/workers/agent_loop/manager.py#L630)已有失败组补采与连续失败熔断，但这不是“同一环境任务识别坏节点，清理后按明确尝试上限换到另一健康节点”。不应写成没有任何重试，也不能写成自动重派发已完整实现 |
| Phase 4：容器复用等延期功能 | **尚未实现，但原稿已明确延期** | 节点服务预热不等于跨任务保留并复用旧容器，主容器仍按任务创建和清理 |

**B1 到 B8 也不能笼统标记全修复。**

| 原稿问题 | 当前判断 |
|---|---|
| B1 驱动库版本、B2 PTX JIT 库 | [resolve_driver_libs](../../../psrl/workers/env_service/overlay.py#L93)解析匹配库，旧 sandbox 路径也引用它。真实 CUDA 有通过证据，但原稿“特定不匹配版本必定报指定错误码”的观察不能当作现在所有节点的事实 |
| B3 固定映射 GPU 0 | 启用 EnvRouter 后按租约动态生成设备映射。用户固定 OFF 仍保留其静态调度语义，不将 OFF 或其他未启用路径说成也受这份账本管理 |
| B4 进程内信号量 | ON 绕过本地信号量并使用共享账本，OFF 分支保留原有路径，见 [agent_loop](../../../examples/sciaccel_rl/agent_loop.py#L394) |
| B5 独立 verifier 泄漏、B6 无归属标签 | 已有 [环境标签覆盖](../../../psrl/workers/env_service/docker.py#L158)与 [runner 收尾](../../../examples/sciaccel_rl/runner.py#L550)，历史故障回收通过。标签覆盖不等于 sidecar 等整个集合都有独立的资源硬限制，也不保证所有 Docker 永久故障均能自动恢复 |
| B7 暂停路由后异常未恢复 | **原函数仍未彻底修复**。[check_and_migrate](../../../psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py#L290)暂停后执行 ABORT 和等待，恢复操作仍未放进 `finally`。中途抛异常可能跳过恢复，这是当前源码风险，本次没有在实机注入该故障。新的借卡协议有自己的保护，不能替代修复这个函数 |
| B8 默认只尝试一次 | [默认 retry_limit](../../../psrl/trainer/config/rollout/psrl_rollout.yaml#L154)仍为 1。[worker 重试循环](../../../psrl/workers/agent_loop/worker.py#L362)与 manager 补采存在，但实验代理的受控重试不等于默认 science 配方已启用同样策略 |

**实现方向发生过变化。** 原稿将每次 exec 加锁和同卡并发计算列为范围外，因为 shell 可以遗留后台计算。当前 command 增加了设备撤权与物理归还确认，才允许在合作式前台阶段之间复用，并允许预算内多个命令并发。它不是只补一把锁，也没有解决任意后台任务的自动暂停恢复。原稿“Router 不参与 exec 路径”现在也应区分控制面与数据面：命令需要请求 Router 授权，但命令输出不经过 Router 转发。

**不应列为欠账的原稿非目标。** 活容器跨节点迁移、任意 CUDA 进程抢占恢复、MPS 共享、复制多份任务取最快结果，本来就不在这一阶段承诺中。预测空档、严格 FIFO 后的小请求回填、cgroup v2 支持是后续扩展，不是把原稿某一阶段勾选完成所必需的全部条件。当前应优先补后端统一、原生工具接入、活动归因、健康重派发与 B7 异常恢复。

## 10. Science 原生任务能否使用 command

本节描述 2026-10-21 的源码与依赖快照。当前 agent 已有显式 GPU 作业工具，评分归因也已有实现与严格／观察两种策略，剩余部署和端到端验证边界见第 11 节，不能将下图的旧默认工具路径当作当前全部能力。

```text
[1] task.toml 与数据行资源声明
            ↓
[2] SciAccelAgentLoop → EnvRouter 准入 → runner 创建 Harbor 容器
            ↓
[3] 默认 TruncatingTerminus2 → Harbor Terminus2 → tmux 终端发送命令
            │                         ↑
            │              当前没有自动转为 exec_gpu
            ↓
[4] 任务编译、运行仿真、将数值结果和遥测写到 artifacts
            ↓
[5] 独立 verifier 读取已提交文件 → reward_key 选择训练奖励
            ↓
[6] 清理任务容器集合与租约

实验代理的已接入分支：
[3P] CommandPressureAgent → exec_gpu → CUDA 压力程序 → 确认归还
                          随后继续原生 CPU 修复流程
```

图例：图里每个部分对应的源码位置。外部任务仓库与实际安装的 Harbor 文件已只读复制并记录 SHA256，下面使用该次审计快照，避免依赖路径随环境变化。

| 编号 | 职责与源码 |
|---|---|
| 1 | [demand_from_task](../../../psrl/workers/env_service/harbor_resources.py#L22)从任务清单与数据提示读取 GPU 需求。模型训练本身用 GPU，不等于环境任务也需要 GPU |
| 2 | [SciAccelAgentLoop._run_harbor_in_thread](../../../examples/sciaccel_rl/agent_loop.py#L358)、[runner 注入 gpu_commands](../../../examples/sciaccel_rl/runner.py#L401)负责资源接线，不会分析任意 shell 的内容 |
| 3 | [TruncatingTerminus2](../../../examples/sciaccel_rl/agent.py#L34)没有通用 command 适配。Harbor [_execute_commands](../env-service-verification/results/env-design-status-20261021/source/harbor/terminus_2.py#L1199)调用 `session.send_keys(block=False)`，由 [tmux 非阻塞发送](../env-service-verification/results/env-design-status-20261021/source/harbor/tmux_session.py#L765)等待指定观察时间后返回 |
| 3P | [CommandPressureAgent](../env-service-verification/scripts/science_fixed_work_agent.py#L27)显式调用 `exec_gpu`，仅覆盖注入的 CUDA 阶段 |
| 4 | 原生 CUDA 题的 [gpu-wrap](../env-service-verification/results/env-design-status-20261021/source/cuda-task/gpu-wrap#L35)运行求解程序并记录整卡利用率、显存 CSV |
| 5 | 原生 CUDA 题的 [telemetry_ok](../env-service-verification/results/env-design-status-20261021/source/cuda-task/grade.py#L68)读取 CSV，[compute_score](../../../examples/sciaccel_rl/reward.py#L44)选取指定奖励键 |
| 6 | [runner finally](../../../examples/sciaccel_rl/runner.py#L550)回收容器集合并释放租约 |

**答案要分三类。**

| 当前任务类型 | 是否需要环境 GPU | 能否用当前 command |
|---|---|---|
| 原生 LAPS 修复题，例如 `laps-repair-coef-2d-rktmod-l17k1` | [任务清单](../env-service-verification/results/env-design-status-20261021/source/cpu-task/task.toml)申报 4 CPU、8192 MiB 主存，没有 GPU 声明，独立 verifier 也没有 GPU 声明 | 可以使用 EnvRouter 管理 CPU／主存和并发，但无需 GPU command。不能因训练模型用了 GPU，就把此题称为 GPU 容器任务 |
| 原生修复题加实验注入的 CUDA 压力阶段 | 注入阶段需要 GPU，后面的修复和评分仍是原任务逻辑 | **受控阶段已接入**，相关实机与训练实验验证的是这条显式路径，不代表普通模型生成的 GPU 命令都已接入 |
| 原生 `laps-accel-cuda` GPU 移植题 | [清单](../env-service-verification/results/env-design-status-20261021/source/cuda-task/task.toml#L46)明确申报一张 GPU | **任务结构适合进一步适配，但当前不能仅切配置就视为完整支持**。默认终端工具未接入真实命令生命周期，活动归因也未完成 |

本地 `laps-accel-cuda` 位于任务仓库的 `blocked-gpu` 作者目录，本次核查的是该目录的任务契约，不能据目录名就推断它已经编译、构建或完成真实训练。本文结论只针对本地 PSRL 示例、任务快照及已读取的依赖，不代表另一个 ScienceInfra 仓库的当前实现。

**为什么原生 CUDA 题在结构上适合。** 它要求模型修改 Fortran／CUDA 求解器，编译后运行仿真，将数值轨迹和 `telemetry.csv` 保存到 `/logs/artifacts`。这可以划为“CPU 编辑编译 → GPU 仿真 → 文件保留 → CPU 检查”，如果求解进程及相关 GPU 访问都完整结束，便具备阶段归还条件。

这道题的 verifier 主要比较数值文件并检查遥测，不是重新在 GPU 上跑求解器。[清单的 verifier](../env-service-verification/results/env-design-status-20261021/source/cuda-task/task.toml#L60)标为 separate，独立环境没有 GPU 声明。因此，此题当前最需要补的是 agent 执行阶段的命令生命周期和遥测归因，不能笼统说必须先把这个 CPU verifier 改成 GPU 命令。其他会在评分时重跑 CUDA 的任务，才需要另外适配其 GPU verifier。

**真正缺的适配不是改一个函数名。** 教学例子：模型向 tmux 输入一个需要 30 秒的 GPU 程序，工具只等待 1 秒就返回当前终端输出。若把这个“发送按键”的调用当成 `exec_gpu` 的全部阶段，1 秒后命令接口已返回，GPU 程序却还要运行 29 秒。当前归还检查应拒绝这种交接，无法完成确认时隔离相关 unit。正确适配需要一个可观察、可取消的 GPU 作业边界：获得授权后启动程序，允许模型读取其状态，程序及 GPU 访问结束后才发起归还。保持 tmux 的目录、变量和进程状态也要纳入适配，不能随意替换成一个全新的无状态 shell。

当前 [exec_gpu](../../../psrl/workers/env_service/docker.py#L73)要求 GPU 工作显式走这个入口；[command overlay](../../../psrl/workers/env_service/overlay.py#L183)默认不授予设备读写权限，获得命令授权时才开放。因此，把原生 GPU 题配置成 command，并不会自动识别普通 `exec` 中的 CUDA 程序或替它申请 GPU。实际可能表现为无法访问设备，不能据配置能够解析就宣称任务已经适配。

**遥测还涉及另一项独立问题。** 当前 `gpu-wrap` 采集整张可见 GPU 的活动，评分函数只要求每个检查至少 3 个有效样本且峰值利用率达到 10%。即使命令预算正确，两条合法并发命令仍可能使 CPU 求解器的遥测看起来活跃。必须在执行并采集遥测的窗口内做可靠进程归因或计算隔离。等到 CPU verifier 读取 CSV 时才加独占锁，不能修复此前已污染的样本。具体阈值来自本次源码读取，不是新一轮 GPU 测量结论。

**跟着图读源码。** 先在 1 核对 GPU 需求属于任务环境，沿 2 看配置变为租约和 Harbor 参数。随后分开读取 3 的默认终端路径与 3P 的实验代理路径，它们不是顺序调用。到 4 要辨别“工具返回”和“GPU 程序实际结束”，再到 5 确认评分使用的是哪些文件和奖励键，最后 6 核对清理。命令模式中的归还确认可与不依赖它的 CPU／模型工作并行，但下次 GPU 请求必须等待旧归还完成。这些代码如果有理解上的问题，可以随时追问。

**本次可复现的 CPU 静态审计。** 在项目 Python 环境、仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/audit_science_command_readiness.py
```

脚本读取保存的两个任务清单，并用 Python 抽象语法树查找调用位置，不导入 Harbor、不运行容器。实际输出见 [readiness.json](../env-service-verification/results/env-design-status-20261021/readiness.json)：修复题环境 GPU 声明为 0，CUDA 题为 1，两者 verifier GPU 声明都为 0。这里读取的是清单声明，不是对实际容器设备挂载的测量。默认代理及 Harbor 终端、评分器文件没有 `exec_gpu` 调用，实验代理有显式调用。静态调用检查不是完整运行验收，不能证明任意输入一定成功或失败。文件来源与指纹见 [manifest.json](../env-service-verification/results/env-design-status-20261021/manifest.json)。

本节新增脚本的 Ruff 与格式检查通过，本文及总览索引的本地链接路径、源码行号范围和本文目录锚点检查通过。全仓检查在隔离副本执行，仍因既有代码和注释风格问题失败，见 [本次 pre-commit.log](../env-service-verification/results/env-design-status-20261021/pre-commit.log)；自动修正未写回工作区。

## 11. 2026-10-27 剩余问题复核

对照原稿第 6、9、10、11 节、本文旧审计及当前主工作区源码，结论是：核心容量调度已有实机收益，但不能据此签收全部设计目标。`b52a82d` 上的本地未提交显存与评分策略改动也属于本次读取范围，不将它们描述为已提交或已部署。原始设计稿保持历史原文，本节在现有文档内补充状态。

```text
[1] Manager / Dispatcher 选择 worker 和节点
                  ↓
[2] Router / Ledger 会话准入、心跳、过期处理
                  ↓
[3] runner / Harbor 创建任务容器集合
                  ↓
[4] agent 显式 psrl-gpu → 作业执行 → 命令准入与归还
                  ↓
[5] 宿主 GPU 进程归属 → 任务活动门槛 → verifier 奖励
                  ↓
[6] 正常收尾或节点清理 → 容器消失 → 配额可复用

过期分支：
[2] 账本删除过期租约 → 生产 Router 等待 [6] 节点清理
                  └── 清理未完成或仍有待清理记录 → wait
[2] command 过期还会隔离 unit → [6] 确认后恢复

独立训推控制分支：
[M] 暂停路由 → ABORT / 等待请求返回 → 恢复路由
                         └── 异常会跳过恢复
```

图例：图里每个部分对应的源码位置。

| 编号 | 当前函数与职责 |
|---|---|
| 1 | [EnvAwareDispatcher.assign](../../../psrl/workers/env_service/placement.py#L84)、[get_dispatch_plan](../../../psrl/workers/agent_loop/manager.py#L984)：按组派发，读取空闲槽位和 GPU 类型提示 |
| 2 | [Ledger.try_admit](../../../psrl/workers/env_service/ledger.py#L105)、[_reap](../../../psrl/workers/env_service/ledger.py#L240)、[RayEnvRouter.try_admit](../../../psrl/workers/env_service/router.py#L381)：资源账本、租约过期与物理清理门槛 |
| 3 | [_apply_env_lease](../../../examples/sciaccel_rl/runner.py#L99)、[_docker_compose_paths](../../../psrl/workers/env_service/docker.py#L276)：动态映射与整组服务归属标签 |
| 4 | [TruncatingTerminus2._execute_commands](../../../examples/sciaccel_rl/agent.py#L104)、[dispatch_gpu_tool](../../../examples/sciaccel_rl/gpu_tools.py#L28)、[GpuExecutionBackend.execute](../../../psrl/workers/env_service/gpu_execution.py#L102)：显式作业、队列与生命周期 |
| 5 | [TaskGpuTelemetry.check_proc_root](../../../psrl/workers/env_service/task_telemetry.py#L111)、[apply_gpu_activity_gate](../../../examples/sciaccel_rl/gpu_grading.py#L46)：可信进程视图与活动评分策略 |
| 6 | [runner finally](../../../examples/sciaccel_rl/runner.py#L569)、[EnvNodeService.cleanup](../../../psrl/workers/env_service/node.py#L68)、[confirm_gpu_cleanup](../../../psrl/workers/env_service/ledger.py#L337)：正常结束及异步清理确认 |
| M | [check_and_migrate](../../../psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py#L290)：迁移期间暂停和恢复推理路由 |

跟着图读源码：先看 1 的输入只包含每组是否需要 GPU，再看 2 如何按具体需求准入和登记资源；3 将租约转换为环境配置。4 是 agent 显式选择的分支，普通终端命令仍走原 Harbor 工具；5 还依赖平台提供正确宿主进程视图。最后分别沿 6 的正常收尾和 2 → 6 的过期分支检查顺序。M 是独立的训推迁移控制，不是每个环境请求必经步骤。节点清理、命令归还和模型交互可以并发，不能将图解释为单线程逐行调用。这些代码如果有理解上的问题，可以随时追问。

### 11.1 已修复及不能继续使用的旧结论

| 项目 | 本次判断 | 证据与保留边界 |
|---|---|---|
| 外部占用与未兑现预约被 `max` 漏算 | **已修复** | [accounted_gpu_memory_mb](../../../psrl/workers/env_service/gpu_memory.py#L9)使用 `M + Σ max(R_i-A_i,0)`，未知归属不抵扣。现行 CPU 反例改为等待；[三步高压训练](science-single-gpu-training-control.md#12-results三步后台对照)中 ON 零 OOM，OFF 18 次。不是所有 OOM 的永久免疫 |
| B1/B2：驱动库匹配与 PTX 编译库 | **实现已修复，有真实 CUDA 证据** | [resolve_driver_libs](../../../psrl/workers/env_service/overlay.py#L93)解析匹配内核版本的库，仍须逐节点预检 |
| B3/B4：固定 GPU 0、进程内准入 | **ON 路径已替代** | 动态租约设备映射和共享账本生效；用户固定 OFF 保持历史静态映射与进程信号量，不能篡改 baseline 的语义 |
| B5/B6：verifier 归属清理、缺少标签 | **已有主要修复及历史回收证据** | 主容器、独立 verifier 与 sidecar 可按归属回收。物理清理失败和资源总预算仍有下节边界 |
| science GPU 工具完全未接入 | **该结论已过时** | runner 注入 `gpu_jobs`，agent 分派 `psrl-gpu start/status/wait/cancel/telemetry`。这是显式工具，不透明识别任意 shell GPU 工作 |
| 完全没有进程归因实现 | **该结论已过时，但部署未闭环** | 已有容器进程身份核验、逐进程显存／利用率、任务活动评分门槛。平台缺正确 procfs 时无法提供可信结果，`observe` 只标记未验证并保留原任务评分 |
| 两种 GPU 持有策略没有共用执行层 | **已共用作业执行层** | [GpuExecutionBackend](../../../psrl/workers/env_service/gpu_execution.py#L84)统一执行和取消，episode／command 保留不同预约范围；这不等于 Phase 0 的 EnvWorker／Harbor 两套执行后端已合并 |

三步训练回答了高压力下重复工作能否覆盖初始化差异：ON 总墙钟减少 27.59%。它使用 CPU science 修复题加受控 CUDA 阶段，不能证明原生 `laps-accel-cuda` 的 GPU 奖励归因，也不能证明下列故障分支已经安全。

### 11.2 仍存在的正确性与故障恢复问题

**A. B7：迁移异常后路由可能停在暂停状态。建议优先修复。** 当前方法在 310 行暂停，在 312 行等待 ABORT、320 行等待中断请求返回，324 行恢复，恢复没有 `finally` 保护。这里的“路由”指把模型生成请求送往推理实例的控制循环，不是环境容器本身。

本次从真实源码抽取该方法，仅把外部调用替换为 CPU 故障注入：正常路径记录 `[false,true]`；ABORT 抛异常或等待抛超时，两者均只记录 `[false]`。这验证了方法层面的异常恢复漏洞，没有对运行中的集群制造超时。只有启用相关迁移策略并进入此分支才触发，不能说所有训练现在都会挂。

建议将暂停状态纳入有明确归属的恢复作用域，成功、异常、取消均完成收尾，同时隔离状态不确定的实例并传播原异常。不能不检查其他暂停原因就无条件覆盖全局状态。最小回归需覆盖 ABORT 失败、等待失败、取消，以及健康实例能继续接受请求。

**B. 排除项：不能只看账本就判定 TTL 会提前放行。** TTL 是“多久没有心跳就认为租约过期”的存活时限，它不是容器已经消失的证据。`_reap` 对普通 CPU／episode 删除租约后将物理清理排队，单独调用 Ledger 确实能看到容量立即可用；但生产 Ray actor 覆写了准入入口，先 `reap()`，再等待 `service.cleanup()`，清理未完成或仍有待清理记录就返回 `wait`，最后才调用 `Ledger.try_admit(..., reap=False)`。

教学例子：unit 最多一条会话，预算 4 CPU、8 GiB，A 十秒没有心跳后过期。第 11 秒 B 请求相同资源，单独调用账本会获准；通过实际 Router 方法，在节点报告尚有清理工作时仍等待，只有确认清理且无新待清理记录后才放行。当前节点服务在 Docker 删除失败或仍能列出容器时保留待清理项并重试。

本次 CPU 机制检查分别执行了账本与真实 `RayEnvRouter.try_admit` 方法。后者使用替代的节点清理响应，结果为：`pending → wait`、`confirmed → granted`、`new_reap_after_cleanup → wait`。因此**不把 TTL 先删账本列为当前完整运行链路的未修复漏洞**。这是控制代码检查，不代表已验证控制器崩溃、网络分区及全部 Docker 故障。后续重构也不能绕开这一生产入口保护。

**C. 原生 CUDA 评分仍缺平台能力与完整验收。** 最新平台记录见 [procfs 障碍](host-proc-blocker.md)。procfs 是 Linux 暴露进程信息的文件系统；GPU 驱动给出宿主进程号，但嵌套容器内 `/proc` 可能是另一套编号。若不能把 GPU 进程对应到任务容器，就不能证明“这次 GPU 活动属于这个任务”。

`required` 会拒绝缺少可信能力的运行；`observe` 允许继续并写入 `gpu_attribution_enforced=0`，保留的 `reward_gpu` 仍可能来自整卡活动污染。已有作业与评分代码不能等同于平台条件已修复。举例：A 在 GPU 计算，B 在 CPU 求解，B 读取整卡 70% 利用率就可能误认为自己用了 GPU。旧 Gate B 只能延迟回复，不能原子地给 B 独占计算窗口；本次 CPU 审计也确认两条空闲观察者都放行、持续繁忙到上限后仍放行。

需要补的是可信宿主 procfs 或宿主采集服务，以及原生任务的正反例验收：GPU 求解器活动应通过，CPU 捷径在 GPU 邻居存在时仍应失败，且独立 verifier 的数值结果与活动证据一致。本次未重新查询平台挂载，部署状态来自 10 月 26 日已保存的预检，不能将其写成 10 月 27 日新测。

**D. 多卡 unit 仍只有总显存预算，没有逐卡峰值约束。** [EnvDemand](../../../psrl/workers/env_service/types.py#L126)只有单个 `gpu_memory_mb`，[账本](../../../psrl/workers/env_service/ledger.py#L343)将多张卡占用相加，命令准入也将物理容量求和。教学例子：两张各 40 GiB 的卡合计 80 GiB，一个只在单卡分配 60 GiB 的程序仍可能获准。新的显存公式修复“怎么算占用”，没有增加“分配到哪张卡”的需求向量。当前单环境卡三步实验不覆盖此项；可先保持一张卡一个 unit，扩展时增加逐卡需求与映射验收。

**E. 容器集合的预算与硬限制仍不完整。** 标签覆盖了 Compose 全部服务，但 [overlay](../../../psrl/workers/env_service/docker.py#L289)给附属服务追加的是标签，没有统一为 sidecar 分配 CPU／主存硬限制。主容器和 verifier 自身的限制也不自动成为“整个任务集合共享一个总限额”。`containers_running` 实际是租约数，`cpu_used` 是需求声明之和，不能当作实际容器计数和 CPU 繁忙度。需要明确 unit 管的是会话预算还是物理容器集合，并覆盖附属服务的最坏同时占用。

**F. 设备所有权与控制器恢复仍有边界。** unit 不重叠检查只覆盖同一个 Router 的配置，另一个 Router、手工 Docker、未绑定的模型进程不自动登记在同一份权威账本中。新的公式能保守处理已看见的占用，但不能禁止外部进程随后增长。普通 `node_policy=any` 也不等于平台给了物理独占。建议校验训练／推理和环境的设备分配，跨服务使用时明确唯一所有者或借还绑定。账本和节点待清理集合仍在内存中，尚无控制器重启后重建租约、接管孤儿容器和网络分区恢复的完整验收。

### 11.3 尚未完成的设计目标与非目标

| 原稿或扩展目标 | 当前状态及影响 | 建议验收方式 |
|---|---|---|
| 运行中健康检查、坏节点剔除与恢复 | 启动时有 Docker／GPU 预检，启动失败任务被保存；没有完整自动更新健康候选与恢复重试流程 | 节点运行后故障，停止新派发；恢复后受控加入，已有任务按归属处理 |
| 按真实负载放置 | `assign` 只接收 GPU 布尔值并逐组扣槽位；缺少每任务 CPU／主存／逐卡显存向量及实际 CPU 负载 | 两节点异构需求下，不把任务发送到永远无法满足其需求的节点；组内多轨迹实际容量一起计算 |
| B8 与坏放置重派发 | worker 重试循环存在，默认 `retry_limit: 1` 仍没有第二次本地尝试；manager 有补采和连续失败熔断 | 按失败类别设置有界重试、避开已知坏节点、确认清理后重派发，不把模型答错作为无限重试理由 |
| Phase 0 统一后端 | EnvWorker 持久 shell 和 science Harbor 专用线程仍同时存在 | 先验证 MLGym 跨轮目录、变量、后台进程等状态语义，再迁入共同后端；不能直接删旧 shell |
| 原生 science 工具完整使用 | 显式工具已接入，但 GPU 作业是单独 shell，不继承 tmux 的目录和局部变量；模型要传 `cwd`／`env` | 使用原始 science GPU 任务和真实模型，通过标准工具执行全套检查，验证评分与收尾，不以合成压力替代 |
| command 与 vLLM 同卡借还组合 | [配置](../../../psrl/workers/env_service/types.py#L54)明确禁止 command 绑定借卡副本 | 若产品确需组合，另定同卡所有权协议及训练权重同步验收；现有独立路径通过不能拼成组合通过 |
| 小请求回填和同容器 GPU 并发 | FIFO 队首阻塞和每容器执行锁仍存在，是吞吐策略边界 | 在实际队列数据证明收益后再做有限回填／饥饿保护；不能为追求并发绕过归还检查 |

原稿已经排除或延期的活容器迁移、任意 CUDA 上下文抢占恢复、容器复用和 MPS，不应列为“必须修复才能签收此次显存修复”的 bug。cgroup v2 支持与更多平台适配也应按部署需求单列。

### 11.4 可复现检查与建议顺序

以下命令只读源码并运行 CPU 机制审计，不创建 Ray 集群、容器或 GPU 工作：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/audit_remaining_env_design.py
PYTHONPATH=. python docs/env-worker-deep-dive/env-service-verification/scripts/audit_gate_contracts.py
```

第一个脚本对实际 `check_and_migrate` 方法进行 AST 抽取和外部调用替换，并调用真实 Ledger 的过期与准入代码。AST 是 Python 的抽象语法树，此处用于只执行目标函数，避免导入整套训练依赖。替换的是外部故障和假时钟，不能视为真实分布式故障注入。第二个脚本调用真实 Gate A/B，用人工遥测验证准入修复与软节流边界。

本次实际输出：[迁移／TTL／多卡机制审计](../env-service-verification/results/env-design-review-20261027/mechanism-audit.json)、[Gate A/B 审计](../env-service-verification/results/env-design-review-20261027/gate-audit.json)。作业接口、归因策略、显存公式和 Gate B 的定向 CPU 回归 **32 项通过**，见[测试日志](../env-service-verification/results/env-design-review-20261027/targeted-tests.log)；补充节点清理和启动入口 **10 项通过**，见[清理入口测试](../env-service-verification/results/env-design-review-20261027/cleanup-entry-tests.log)。通过项不覆盖 B7 的异常恢复；TTL 则通过补查生产入口避免了误报。

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
| 2 | [get_dispatch_plan](../../../psrl/workers/agent_loop/manager.py#L984)选择运行节点 |
| 3/4 | [节点检查](../../../psrl/workers/env_service/router.py#L336)、[准入检查](../../../psrl/workers/env_service/ledger.py#L425)决定是否允许启动 |
| 5 | [完整容器预算](../../../psrl/workers/env_service/container_budget.py#L7)、[最终容器限制](../../../psrl/workers/env_service/docker.py#L280)覆盖主服务与附属服务 |
| 7 | [RoutingPauseController](../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L7)记录并释放暂停持有者 |

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
| 1 | [harbor_resources.py：demand_from_task](../../../psrl/workers/env_service/harbor_resources.py#L22)统一解析任务与 verifier 需求，供 manager 与 runner 使用 |
| 2 | [manager.py：get_dispatch_plan](../../../psrl/workers/agent_loop/manager.py#L984)统计组内条数，[placement.py：_assign_resources](../../../psrl/workers/env_service/placement.py#L184)按资源形状放置 |
| 3 | [router.py：_check_node](../../../psrl/workers/env_service/router.py#L334)检查节点并保留清理状态，[try_admit](../../../psrl/workers/env_service/router.py#L444)在确认后准入 |
| 4 | [ledger.py：_could_ever_fit](../../../psrl/workers/env_service/ledger.py#L396)、[_fits_now](../../../psrl/workers/env_service/ledger.py#L425)区分永远放不下与当前需等待 |
| 5 | [runner.py](../../../examples/sciaccel_rl/runner.py#L149)把预算放入 Harbor 环境参数，[docker.py](../../../psrl/workers/env_service/docker.py#L280)为最终全部 Compose 服务施加限制 |
| 6 | [command_gpu.py：acquire](../../../psrl/workers/env_service/command_gpu.py#L50)在设备授权前检查每张卡，不修改可信进程识别机制 |
| 7 | [check_and_migrate](../../../psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py#L290)使用 [RoutingPauseController](../../../psrl/workers/gen/rollout_coordination/routing_pause.py#L7)管理暂停持有者 |

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

第二条命令需要项目测试依赖，但不创建 Ray 集群、容器或 GPU 工作。它进一步覆盖重复取消、重叠暂停、真实命令授权前的逐卡判断、verifier 需求、容器限制、健康恢复和有限重试。旧审计 JSON 保留原样，新结果单独存入 [env-design-remediation-20261027](../env-service-verification/results/env-design-remediation-20261027/)。

### 12.5 Results 与未覆盖边界

相关 CPU 回归 **291 项通过**，覆盖完整 `tests/env_service`、manager 失败熔断和 science agent loop 测试。另补 Harbor 配置序列化与独立 verifier 继承路径检查，相关 **38 项通过**，与前述回归有重叠，不相加计数。预算在配置边界传普通字典，环境构造时重建经过校验的对象，避免配置落盘再加载后丢失限制方法。定向 Ruff 与格式检查通过，检查结果见 [validation.txt](../env-service-verification/results/env-design-remediation-20261027/validation.txt)，完整日志与源码指纹一同留档。全仓 pre-commit 在隔离副本执行，仍因已有其他文件的 Ruff／注释规范问题失败，自动改动没有应用回主目录。这里的故障注入是 CPU 控制逻辑验证，不能写成真实 Docker 限额验收、多卡 CUDA OOM 实测或训练吞吐提升。没有为了这些修复重跑 science 训练。

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
| 1 | `_assign_resources` 先筛资源形状，再比较节点负载 | [placement.py](../../../psrl/workers/env_service/placement.py#L184) |
| 2 | `snapshot` 区分租约计数、声明资源和物理观测 | [ledger.py](../../../psrl/workers/env_service/ledger.py#L268) |
| 3 | `resource_snapshot` 调用 `read_node_resources`，按 unit 标签查询实际容器 | [node.py](../../../psrl/workers/env_service/node.py#L59)、[resource_telemetry.py](../../../psrl/workers/env_service/resource_telemetry.py#L24) |
| 4 | `_poll_node_resources` 异步采集，`_resource_state` 控制新鲜度，`node_capacity` 提供负载 | [router.py](../../../psrl/workers/env_service/router.py#L343)、[新鲜度](../../../psrl/workers/env_service/router.py#L290) |
| 5 | `retire` 保留待办，`pending_reaped` 重复交付，`acknowledge_reaped` 精确删除 | [ledger.py](../../../psrl/workers/env_service/ledger.py#L231) |
| 6 | `cleanup` 删除、确认物理消失、再确认账本记录 | [node.py](../../../psrl/workers/env_service/node.py#L83) |
| 7 | `_check_node` 处理有界探活和安全重建，`try_admit` 检查未完成清理 | [router.py](../../../psrl/workers/env_service/router.py#L395)、[准入](../../../psrl/workers/env_service/router.py#L524) |

这些代码若有不清楚的地方，可以按图中编号继续追问。

### 13.1 原提案还缺什么

| 原提案内容 | 本轮处理后的状态 | 下一步或边界 |
|---|---|---|
| §6.6、Phase 1：物理占用与声明账本分开 | 已增加真实容器计数、主存、CPU 使用量和主机负载，独立周期采集 | 未读取到的数据是未知，不能显示成零 |
| Phase 3.1：根据实测负载选择节点 | 已接入资源感知派发的偏好 | 未做多节点吞吐实验，不能宣称训练提速 |
| Phase 1：TTL 回收并实际停止容器 | 已补“确认删除后才销账”，允许节点服务重领待办 | 仅 Router 仍存活的恢复，不是持久化账本 |
| Phase 0：统一 EnvWorker / Harbor 后端 | 未完成 | 先验证跨轮 shell 状态，再考虑删除旧路径 |
| Phase 3.2：运行失败的 episode 清理后跨 worker 重派发 | 未完成 | 需请求身份、清理确认、尝试上限及防重复副作用协议 |
| B8：episode 级重试默认次数 | 未在本轮调大 | 不能把派发前探活重试当作 episode 执行重试 |
| GPU 进程归属及严格归因 | 不在本轮实现范围 | 由另一进程维护，需它自己的真实验收证据 |
| 容器池、状态迁移、持久账本与跨 Router 唯一所有权 | 未实现 | 前两项在原稿延期或排除，后两项不能冒充已完成能力 |

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

真实实验脚本：[real_resource_recovery.py](../env-service-verification/scripts/real_resource_recovery.py)。在目标节点预检通过、已有 Python 容器镜像的前提下运行：

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
| 租约与真实容器计数分开 | 1 条租约、4 个运行容器。第二次运行观测主存 69.25 MiB，声明预算 384 MiB | [资源与限额记录](../env-service-verification/results/resource-recovery-20261027-r2/evidence.json) |
| 所有容器的资源限制 | 两个 main 各 0.5 核、128 MiB，两个 sidecar 各 0.25 核、64 MiB，Docker 实际配置吻合 | 同上 `containers` |
| 内存硬限制 | 64 MiB sidecar 尝试分配 256 MiB，进程退出 137，Docker 记录该容器的 `oom` 事件 | [事件原始记录](../env-service-verification/results/resource-recovery-20261027-r2/oom-events.jsonl) |
| 清理交付后节点被杀 | 待办仍在 Router，旧四个容器仍存在时新请求为 `wait` | [恢复实验](../env-service-verification/results/resource-recovery-20261027-r4/evidence.json) |
| 替代节点接手清理 | 节点 actor 已替换，旧容器全部删除、待办确认后新请求获准，最终租约和实际容器均为 0 | 同上 `node_replaced`、`recovery_admissions`、`final_sample` |
| 最终物理释放 | 本轮各次运行所属容器、网络和隔离 Ray 进程均已退出 | [收尾核验](../env-service-verification/results/resource-recovery-20261027-r4/physical-release.json) |

失败记录也保留：第一次在 Harbor 的整数 CPU 配置校验处退出，尚未启动容器。第二次资源计数和限额已通过，但脚本读取的容器内 cgroup 根路径失败计数不增加，断言中断，随后由 Docker OOM 事件补齐内存限制证据。第三次已验证“交付后仍保留”和“清理前阻塞”，但实验设置的 2 秒健康检查期限耗尽重试预算，未完成接手清理。[第三次日志](../env-service-verification/results/resource-recovery-20261027-r3/driver.log)保留该失败。最终运行使用默认 30 秒健康检查期限，暂停点返回未完成清理数，杀掉节点后恢复成功，退出码 0。最终运行跳过已验证的内存超限探针，不重复制造 OOM。

全套 Env Service 回归在第一阶段通过 291 项，后续针对新增故障分支和遥测的 33 项通过，两组包含重复测试，不能相加当作独立用例数。新增生产文件、测试和实验脚本的定向 Ruff 检查及格式检查通过。全仓 `pre-commit run --all-files` 在隔离副本执行，仍因 `docs/conf.py`、历史实验脚本等已有风格问题失败，未将自动修改复制回主目录，见[检查日志](../env-service-verification/results/resource-recovery-20261027-r4/precommit.log)。

本次是实际容器限额与节点清理恢复的组件验收。两套 Harbor 环境由脚本直接启动来覆盖完整容器集合，不是运行真实 science verifier 的评分流程。没有证明 CPU 运行耗时随配额变化、真实多节点负载派发提速、GPU 状态重建、Router 进程重启恢复、跨节点 episode 重试或完整 science 训练收益。可信进程归属逻辑仍由另一进程维护。以上修改保存在主目录，未提交、推送或合并。
