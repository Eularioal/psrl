# EnvRouter 低负载配对实验与 Gate A/B 设计审视

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
| 1 | [get_dispatch_plan](../../psrl/workers/agent_loop/manager.py#L947) 根据容量快照选择 worker，实际准入仍须经过账本。配对小实验直接调用 runner，没有测跨节点放置 |
| 2 | [run_harbor_episode](../../examples/sciaccel_rl/runner.py#L258)、[Ledger.try_admit](../../psrl/workers/env_service/ledger.py#L105)：检查可承载资源、登记会话 |
| 3 | [_apply_env_lease](../../examples/sciaccel_rl/runner.py#L98)：生成设备映射，选择 ManagedDockerEnvironment，再交给 Harbor |
| B | [SessionRouter 响应出口](../../psrl/workers/gen/session_router.py#L444) 调用 [_wait_for_gpu_idle](../../psrl/workers/gen/session_router.py#L120)。[runner 的模式分支](../../examples/sciaccel_rl/runner.py#L154) 对命令模式与模型借卡模式传入空绑定地址，跳过此 Gate |
| 4 | [exec_gpu](../../psrl/workers/env_service/docker.py#L73) 调用 [GpuCommandScheduler.acquire](../../psrl/workers/env_service/command_gpu.py#L47)，节点 [grant_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L120) 开启已分配设备访问 |
| 5 | [begin_return / _return](../../psrl/workers/env_service/command_gpu.py#L109)、[revoke_gpu_command](../../psrl/workers/env_service/command_gpu_node.py#L131)：预算保持到物理确认完成，失败进入隔离状态 |
| 6 | [runner finally](../../examples/sciaccel_rl/runner.py#L550)、[Node.cleanup](../../psrl/workers/env_service/node.py#L68)：正常与异常路径按归属回收 |
| L | [GpuLendingBinding.set_phase](../../psrl/workers/env_service/lending.py#L27)、[配置约束](../../psrl/workers/env_service/types.py#L51)：整副本的协作借还，独立于命令级并发 |

原始设计见工作区的 [2026-09-21-env-service-design.md](../../../2026-09-21-env-service-design.md)，该文件是本地历史设计材料。其第 6.3 节把 Gate A 定义为容器创建前的容量准入，把 Gate B 定义为模型回复出口的有界节流。这里的“准入”就是决定一个请求现在能否开始，租约则是获准资源的归属登记。

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

![同期配对的任务阶段、含初始化的总时间、任务和命令延迟](env-service-verification/results/router-paired-review-20261020/paired-latency.png)

**应怎样解读。** 修复后的低负载 ON 已接近 OFF，本次工作阶段合计更短，但两对的差异方向不一致，不能把 6.31% 宣布为稳定提速。这里仅有两对运行，16 条任务共享运行环境，不能当成 16 次独立重复试验，更没有可靠的尾部百分位或统计显著性结论。图中黑线是样本平均值，散点是每条观测。

**冷启动仍有成本。** 第一对初始化／工作前准备为 OFF 18.97 秒、ON 28.50 秒，第二对分别为 17.68 秒、22.60 秒。ON 除 Ray 外还需创建 Router 和节点服务，首次预热超过 8 秒独立准备窗口约 1.54 秒。因此，包含初始化和驱动收尾后的总时间，第一对为 111.65 对 108.76 秒，第二对为 102.31 对 108.38 秒，两对合计 ON 仍多 1.49%。长期复用服务可摊薄这笔开销，但不能仅展示工作阶段而隐藏短作业启动成本。此处驱动总时长不包含 Python 导入前的进程启动和驱动结束后的独立物理释放复核。

**控制路径的长尾确有改善迹象。** 工作阶段事件循环最大停顿，第一对由 3.408 秒变为 0.526 秒，第二对由 4.353 秒变为 0.441 秒。OFF 的采样栈仍捕获同步 Docker 探测和进程创建。ON 第二对也捕获进程创建停顿，说明不能把所有这类长尾都宣称已消除。CUDA 脚本平均时间基本相同，与“改进主要在宿主控制路径”这一解释一致，但并非每一秒差异都可由这些采样唯一归因。

这比较的是用户指定旧提交与当前完整实现，ON 包含同步准备移出事件循环等优化，所以不能把净差异全部解释为账本算法本身的收益。两组 ON 均观察到最多 4 条活动 GPU 命令，所有账本采样均未越过声明预算。本轮低负载没有触发超额排队，不能据此再次声称验证了过载保护。

原始统计见 [paired-analysis.json](env-service-verification/results/router-paired-review-20261020/paired-analysis.json)，生成入口为 [analyze_paired_router.py](env-service-verification/scripts/analyze_paired_router.py)。逐组及最终清理已通过，最后 [physical-cleanup-final.json](env-service-verification/results/router-paired-review-20261020/physical-cleanup-final.json) 中本次容器、网络、actor 和私有 Ray 进程为空，目标卡无 CUDA 进程，应用显存约 0.19 MiB。到达预定两对后即停止，没有追加训练或补跑。

**实验约束。** 固定 OFF 为 `0eeead861ffafed6bf31c962d259fd847775dcf9`，保留其静态 GPU 映射和进程内信号量，没有添加整卡锁。必要运行补丁为传递 `override_gpus`、移除节点级镜像 prune，以及已有依赖／HTTP 运行修复，全部保存在 [off.patch](env-service-verification/results/router-paired-review-20261020/off.patch)。最新 ON 是 `37a06ea8d32b05bbbf385ad40cb404b30ba01680` 加同步准备优化，冻结在独立 checkout，见 [on.patch](env-service-verification/results/router-paired-review-20261020/on.patch)。

四组均为同一张 H20、相同镜像、八任务、四并发、两批，每条任务执行两次真实 CUDA 命令。每次分配 1 GiB、持有 2 秒，中间 CPU 等待 8 秒，ON 每条申报 2 GiB，unit 预算 80 GiB。工作量只做小型 CUDA 写入读回及持有，不代表持续高强度 GPU 运算。各组使用独立 Ray 进程，相同的 8 秒独立准备工作、同一套计时与栈采样，ON 在该工作期间预热节点服务，未能完全重叠的部分计入准备时间。

每组工作预算 240 秒、进程预算 360 秒，总实验预算 1800 秒。预先固定两对，不根据结果继续补跑。每组结束先核验本次容器、网络、actor、Ray 进程与目标 GPU 释放，再开始下一组。节点其他 GPU 上存在其他任务，本次没有独占整个节点，因此顺序平衡只能减轻时间漂移，不能消除所有 CPU、存储和驱动干扰。

单任务端到端延迟包含容器准备、命令执行、评分、runner 清理和驱动读取结果的小开销。GPU 命令调用延迟包含 ON 的相应准入及返回控制，不能解释成单纯 CUDA 计算时间。任务批次时间不含 Ray 初始化和最终核验，初始化及最终清理另列。四并发的总时长不能除以任务数当作单任务 latency。

## 3. Gate A/B 的收益究竟验证了什么

| 目标 | 核对到的证据 | 可成立的结论与边界 |
|---|---|---|
| Gate A 防止声明预算超订 | 历史 [fixed-admission-boundaries](env-service-verification/results/fixed-admission-boundaries/summary.json) 的真实容器 GPU、CPU、主存单约束检查均通过。GPU 预算 1200 MiB、每条申报 512 MiB，第三条等待，释放后恢复 | 已验证容量准入功能，容器上限设为 8 排除了数量限制这个混杂因素 |
| 共享 GPU 预算与安全回填减少失败 | 历史同卡等完成量 [pressure 对比](env-service-verification/results/router-baseline-control-20261020/comparison.json)：两边均完成 16 任务，OFF 15 次 OOM、266.52 秒，ON 零 OOM、207.84 秒 | 已有高压力执行收益，属于容量准入、命令阶段复用和控制实现的组合效果，不是纯 Gate A 或旧 Gate B 单独消融 |
| Gate B 共享卡不被常驻显存误阻塞 | 早期[报告第 3.3 节](env-service-verification/README.md#33-gate-b邻居在同一张卡上时回复要等多久)记载 8 秒等待上限下，共享卡空闲常驻显存由约 8.4 秒变为 0 秒。本次 CPU 检查亦验证共享模式只需一次读取 | 原始完整独立日志未找到，历史秒数仅按报告记载引用。本次 CPU 检查验证判定逻辑，不替代真实 GPU 性能复测 |
| Gate B 查询正确执行节点 | 历史 [fixed-multinode/evidence.json](env-service-verification/results/fixed-multinode/evidence.json)：远端保留分配时，两端调用均等待约 2.45 秒，`remote_gate_b_passed=true` | 已验证真实跨节点遥测方向与受限等待，未覆盖完整模型 HTTP 出口到评分的归因链 |
| Gate B 防止后台邻居被计入本任务奖励 | 原设计第 9 节要求 `nohup` 邻居、第二任务和逐进程 `pmon` 对照。已有验证报告[未覆盖项](env-service-verification/README.md#97-不能据此签收的设计范围)及原始结果中没有完成这一验收的证据 | **未证明奖励归因收益，更未保证互斥。** 本轮 CPU 检查揭示空闲同时放行和超时仍放行边界 |
| 租约过期后真的回收容器 | 历史 [fixed-crash-recovery](env-service-verification/results/fixed-crash-recovery/evidence.json) 中 worker 被杀后 `after_worker_reaper=[]` | 已验证这一历史故障恢复路径，不能代替控制器重启、网络分区或所有分布式故障验证 |
| Docker 与 vLLM 同卡借还 | 历史 [gpu-lending-real-20261015](env-service-verification/results/gpu-lending-real-20261015/summary.json)：两次同卡 Docker/CUDA 与真实 vLLM 交替，生成三次、每次 16 token，物理资源已清理 | 功能循环通过，使用推理适配器绕过 NIXL，未证明长期吞吐、训练权重同步或任意多任务借卡收益 |

本轮另外执行当前 Gate A/B 相关 CPU 回归 **47 项通过**，见[测试日志](env-service-verification/results/router-paired-review-20261020/gate-tests.log)。这些回归使用可控输入，不能当成 GPU 训练收益。新的低负载配对实验也没有触发 OOM 或旧 Gate B，更没有测模型能力改善。

**澄清：Gate B 有专门设计的测试和实验，不是完全没有验证。** “没有证明奖励归因收益”不能简写为“Gate B 没有实验”。现有材料分为：

| 入口 | 实际覆盖 | 执行证据与限制 |
|---|---|---|
| [test_gate_b.py](../../tests/env_service/test_gate_b.py) | 11 个测试，覆盖空闲放行、繁忙后转空闲、多卡等待、共享卡常驻显存、遥测缺失、超时放行和绑定解析 | 使用脚本化遥测，本轮已与 Ledger 测试合计执行 47 项通过，不是 47 个 Gate B 测试 |
| [gate_b_experiment.py](env-service-verification/scripts/gate_b_experiment.py) | 专门的真实 GPU 实验，一种邻居只占约 22 GiB 显存、不计算，另一种邻居真正计算约 5 秒，测量 Gate B 等待时间，等待上限设为 8 秒 | 实验脚本存在，历史报告记载了结果，但没有找到对应独立完整原始日志，不能补称为本轮新测。脚本直接调用等待函数，没有启动两个完整评分任务，也没有比较奖励误判率 |
| [multinode_check.py](env-service-verification/scripts/multinode_check.py) | 两台节点调用 Gate B，确认都查询到指定远端节点的占用 | 保存了真实原始结果。采用 `shared=False` 与常驻分配触发等待，证明远端查询和有界等待，不是对计算争用或奖励归因的验证 |
| [audit_gate_contracts.py](env-service-verification/scripts/audit_gate_contracts.py) | 用教学遥测调用真实函数，检查多个空闲观察者放行、共享显存判定、持续繁忙超时放行 | 本轮已运行 CPU 审计，不是实机吞吐或奖励收益实验 |

具体区分可以用一个例子：邻居计算 5 秒，Gate B 等它结束后返回，证明“看见已在计算的邻居时会等待”。但要证明“B 的 GPU 奖励不会误算 A 的计算”，还必须让 B 走完整 agent、工具和 verifier 链路，对比 Gate B 开关下的实际重叠窗口与奖励结果，并覆盖两个请求同时看到空闲、后台任务遗留及超时放行。这后半部分是原设计已提出、目前没有找到完成证据的验收项。上一轮低负载配对采用 command 模式，跳过旧 Gate B，因此不能代替这项实验。

## 4. 为什么 Gate B 不能保证奖励归因

先看原设计担忧的具体问题。`reward_gpu` 是“正确性奖励 × GPU 活动判定”。如果活动判定读取整张卡的利用率，任务 A 在 GPU 上忙，任务 B 实际只用 CPU，B 也可能看到整卡繁忙，从而被误认为使用了 GPU。这是**活动归因**问题，与显存能否装下两条任务是两件事。

当前数据构造明确把 `laps-accel-cuda` 映射为 [reward_gpu](../../examples/sciaccel_rl/prepare/build_dataset.py#L35)，最终 [compute_score](../../examples/sciaccel_rl/reward.py#L44) 取指定奖励键。修复类 LAPS 任务使用 `reward_repair`，不能用其训练通过来证明 `reward_gpu` 的归因正确。

Gate B 只观察状态，没有“检查后立即为某条任务保留计算窗口”的原子操作。教学时序如下：

```text
时刻 0：整张 GPU 利用率低，A 和 B 的模型回复都在等待
A 读取到空闲 → 放行 A 的回复
B 也读取到空闲 → 放行 B 的回复
随后 A 开始 GPU 工作，B 开始自己的任务和活动采样
结果：两者仍可能在同一个采样窗口重叠
```

这里“原子”指检查与占用承诺不可被另一个竞争者穿插。现有 Gate B 没有这样的占用承诺。默认还会在约 600 秒上限后放行，遥测缺失时也放行，都是有意避免模型请求无限等待的软策略。因此它可以减少某些已知繁忙时段的重叠，但不能作为奖励正确性的充分条件。

本次 [audit_gate_contracts.py](env-service-verification/scripts/audit_gate_contracts.py) 用**教学遥测**调用真实 `_wait_for_gpu_idle`，得到两条空闲观察者都放行，持续 100% 利用率在缩短的 0.05 秒上限后仍放行。输出见 [gate-contract-audit.json](env-service-verification/results/router-paired-review-20261020/gate-contract-audit.json)。没有伪造实机性能，也没有启动外部服务。

当前命令配额解决的是“谁可以打开设备、谁占多少显存预算”，显存足够时允许多个 CUDA 命令并行。因此它也不会自动解决整卡利用率的归因问题。后续若评分确实依赖 GPU 活动，需要明确的独占评分窗口覆盖所有相关计算者，或能够关联容器／进程的活动证据。后台进程、独立 verifier 和外部任务也必须进入这个约束，不能只给主容器排队。

## 5. 当前方案的合理部分与适用边界

**合理之处是将资源生命周期分开。** 容器里还有文件、CPU 进程和下一轮对话状态，不等于它必须继续占用显存预算。以教学数字说明：一张预算 80 GiB 的卡，两条任务各申报 51 GiB，只能同时运行一条 GPU 命令。A 命令结束并确认归还后，A 继续 CPU／模型工作，B 此时使用这 51 GiB。A 下一条命令再排队。这正适合用户关心的整体完成时间目标。

**安全交接比预测空档更基础。** `revoke_gpu_command` 先禁止新打开设备，再检查旧句柄、映射和 CUDA 进程，确定交接后才释放预算。Python 协程返回、模型开始生成、利用率瞬间为零，都不能单独替代这一步。对仍持有 CUDA 上下文的任务，目前不会强行抢占并恢复其显存内容。原稿“等待任意容器完成或销毁”的保守前提，只有在任务提供明确前台 GPU 边界且释放成功时才能缩短。

还不能忽略以下边界：

| 边界 | 当前实际情况 | 对使用者的含义 |
|---|---|---|
| 普通 science 任务透明接入 | 命令模式要求显式 `exec_gpu`。默认 [TruncatingTerminus2](../../examples/sciaccel_rl/agent.py#L34) 只接入模型借卡阶段钩子，没有将任意终端 CUDA 命令自动转换为该接口。训练实验由 [CommandPressureAgent](env-service-verification/scripts/science_fixed_work_agent.py#L27) 显式接入注入的 GPU 阶段 | 不能只改 `gpu_reservation: command` 就宣称任意现有 GPU 任务和 verifier 都能正常运行 |
| 预算与硬限制 | [Ledger._effective_gpu_mb](../../psrl/workers/env_service/ledger.py#L357) 使用声明与实测最大值。设备权限控制不是每容器显存上限 | 已授权的命令低报或实际增长超出声明，仍可能竞争 |
| 外部 GPU 使用者 | 多个 unit 不重叠的校验作用于同一个 Router 的配置，不能自动约束另一个 Router、手工 Docker 或未绑定的训练引擎 | 需要设备所有权约定或统一注册。读取实时总量不能替代所有权协议 |
| 多卡 unit | 普通准入使用多卡显存之和，GPU 需求没有逐卡分布向量 | 两张各 40 GiB 的空闲卡不等于一个只用单卡的 60 GiB 请求能运行。普通单卡任务优先用一张卡一个 unit |
| 队列与回填 | 命令队列先入先出，队首不满足时后面的小请求不会绕过它 | 教学例子中剩 13 GiB、队首需 14 GiB、后面需 10 GiB，后者仍等待。公平与装箱利用率之间尚无有限回填策略 |
| 平台与跨机 | 当前设备访问控制要求可写的 cgroup v1，运行中容器不会自动迁移到另一节点 | 本轮单节点结果不证明 cgroup v2 支持、跨机命令转移或大集群吞吐 |
| 模型借卡 | 借卡分支要求完整副本绑定、一个容器位置，不能和命令模式同时配置 | 两条功能各自通过，不等于任意组合已经通过 |
| 统一后端目标 | 原稿计划合并 EnvWorker 与 Harbor。当前通用 [EnvWorker](../../psrl/workers/env_worker/worker.py#L156) 仍存在，science 走 runner／Harbor／节点服务 | 原设计的全部路线图尚未完成，不宜笼统标记为“全部落地” |

**一个已复现的账本边界。** 以下全部是 CPU 教学输入，不是本次实机 OOM：容量 80 GiB，任务 A 申报 50 GiB、当前只用 20 GiB，未注册外部进程另用 20 GiB。实测总量为 40 GiB，账本算 `max(50, 40)=50`，因此仍允许 B 申报 30 GiB。将来 A、B 都达到各自声明，连同外部 20 GiB 就需要 100 GiB。上述审计脚本调用真实账本，确认 B 被放行。

这个例子说明：`max(声明, 实测)` 可以挡住显著占用，但不能同时区分“己方尚未使用的承诺”和“外部实际占用”。在所有 GPU 使用者都遵循同一份声明与生命周期协议时，预算才有更强保障。需要允许无关进程长期共卡时，应分别核算受管预留和已归因的外部占用，而非宣称现在已经能完全避免竞争。

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
| 选择 worker | [1] [get_dispatch_plan](../../psrl/workers/agent_loop/manager.py#L947) | 输入一批任务和容量快照，输出 worker 对任务的分组，本轮实验直接从下一步进入 |
| 任务进入与会话准入 | [2] [run_harbor_episode](../../examples/sciaccel_rl/runner.py#L258) → [try_admit](../../psrl/workers/env_service/ledger.py#L105) | 输入资源声明、节点约束，输出等待或租约，登记持久 CPU、主存和容器预算 |
| 环境配置与创建 | [3] [_apply_env_lease](../../examples/sciaccel_rl/runner.py#L98) | 租约变为设备映射、环境子类和 Harbor 参数 |
| episode 分支的回复出口 | [B] [_wait_for_gpu_idle](../../psrl/workers/gen/session_router.py#L120) | 遥测用于延迟回复，不授予独占计算权，命令模式跳过此分支 |
| command 分支的每次调用 | [4] [exec_gpu](../../psrl/workers/env_service/docker.py#L73) → [acquire](../../psrl/workers/env_service/command_gpu.py#L47) | 请求进入队列，预算可用时登记活动声明、授权设备，再执行命令 |
| 并行归还与规划 | [5] [begin_return](../../psrl/workers/env_service/command_gpu.py#L109) → [_return](../../psrl/workers/env_service/command_gpu.py#L124) | 保留预算并启动后台撤权检查，调用方在服务接管后可继续 CPU 工作，确认成功才释放显存预算 |
| 分支汇合与清理 | [6] [runner finally](../../examples/sciaccel_rl/runner.py#L550) | 后台错误在下一 GPU 请求或收尾汇合，任务结束回收所属容器并释放会话资源 |
| 独立模型借卡分支 | [L] [set_phase](../../psrl/workers/env_service/lending.py#L27) | 输入副本绑定、阶段和序号，在模型 sleep/wake 与容器访问之间切换，不能与 command 混用 |

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

真实 GPU 入口仍为 [profile_science_tail.py](env-service-verification/scripts/profile_science_tail.py)，设备、镜像和 overlay 必须现场预检。运行脚本、冻结源码、补丁与预算保存在[本轮结果目录](env-service-verification/results/router-paired-review-20261020)。不应直接把其中节点身份复制到其他环境执行。

本次新增诊断脚本的 Ruff 检查与格式检查通过，文档链接及源码行位置已核验。全仓 `pre-commit run --all-files` 在隔离副本执行，仍因既有 Ruff 和注释风格问题失败，详见 [pre-commit.log](env-service-verification/results/router-paired-review-20261020/pre-commit.log)。自动修正没有写回共享工作区，不把全仓检查报告为通过。

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
| 1 | [Ledger.try_admit](../../psrl/workers/env_service/ledger.py#L105) | 会话需求变为租约，command 模式先承诺宿主资源 |
| 2 的配置 | [_apply_env_lease](../../examples/sciaccel_rl/runner.py#L154) | command 或模型借卡模式传入空绑定地址，跳过 Gate B，episode 模式有有效 SessionRouter 地址时可绑定 |
| 2 的响应出口 | [SessionRouter 出口](../../psrl/workers/gen/session_router.py#L444) → [_wait_for_gpu_idle](../../psrl/workers/gen/session_router.py#L120) | 已有模型结果等待 GPU 遥测满足条件或超时后返回，不判断下一条工具命令究竟需要 CPU 还是 GPU |
| 3 | [exec_gpu](../../psrl/workers/env_service/docker.py#L73) → [acquire](../../psrl/workers/env_service/command_gpu.py#L47) | 具体 GPU 命令申请预算，允许符合预算的并发，授权后执行 |
| 4 | [begin_return](../../psrl/workers/env_service/command_gpu.py#L109) → [_return](../../psrl/workers/env_service/command_gpu.py#L124) | 后台归还与规划并行，成功确认后才修改活动预算，不因发出归还请求就提前释放 |
| 5 | [wait_for_gpu_return / stop](../../psrl/workers/env_service/docker.py#L68)、[runner finally](../../examples/sciaccel_rl/runner.py#L550) | 下一 GPU 调用或结束时汇合异步错误，按归属清理 |

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

**可运行教学模拟。** [simulate_gate_b_composition.py](env-service-verification/scripts/simulate_gate_b_composition.py) 仅使用标准库，对上述时间线做确定性计算，不调用生产调度器、不启动 GPU 或外部服务：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/simulate_gate_b_composition.py
```

已运行，第一例输出两种路径总完成时间 12 秒和 20 秒，第二例输出 10 秒和 12 秒，见 [gate-b-composition-teaching.json](env-service-verification/results/router-paired-review-20261020/gate-b-composition-teaching.json)。假设没有控制开销和并发减速，且 A 持续繁忙到第 10 秒。数字用于解释等待位置与重叠的关系，不是新一轮 GPU 性能实验，也没有据此证明真实组合策略的收益。

## 9. 原设计剩余工作审计

**审计日期：2026-10-21。** 对照用户指定的原始设计稿第 6、8、9、10 节与当前本地实现。这一节是只读源码及历史证据审计，没有新启动 GPU、容器或训练。区分“未实现”“已有实现但未完整验收”“实现路线已变化”和“原稿明确排除”，不按目录中出现过文件就判定全部完成。

| 原稿目标 | 当前状态 | 仍缺少什么及源码依据 |
|---|---|---|
| Phase 0：合并 EnvWorker 与 Harbor 执行后端 | **未完成统一** | [SciAccelAgentLoop 的 Harbor 专用线程](../../examples/sciaccel_rl/agent_loop.py#L30)与 [EnvWorker.exec](../../psrl/workers/env_worker/worker.py#L276)、`shell.py` 仍同时存在。还没有将所有任务迁入同一后端、完成 MLGym 跨轮 shell 状态等价验收并删除旧路径 |
| Phase 0：跨 worker 容器准入、归属与清理 | **核心能力已实现，架构不同于原稿** | ON 用共享 Router 账本替代各进程单独计数，并给容器集合加标签、执行归属清理。实际是 [EnvNodeService](../../psrl/workers/env_service/node.py#L19) 每节点服务，不是每 unit 一个统一执行所有任务的 EnvWorker。不能把“没有按原稿类名合并”说成“没有共享限流” |
| Phase 1：静态 unit、Gate A、租约与回收 | **已实现，有历史实机证据** | [Ledger](../../psrl/workers/env_service/ledger.py#L105)、[节点清理](../../psrl/workers/env_service/node.py#L68)覆盖准入、心跳过期后的容器回收。控制器重启后的租约恢复、网络分区等完整容错范围尚未证明，内存账本不是持久恢复系统 |
| Phase 1：完整的资源遥测 | **部分实现** | 已有显存总量、GPU 计算 PID 和预算快照，但没有原稿要求的逐 PID 利用率 `measured_util_by_pid` 与实测 `cpu_load` 派发。注意 [snapshot](../../psrl/workers/env_service/ledger.py#L257) 中 `containers_running=len(leases)`，`cpu_used` 是声明之和，不是现场 Docker 容器数与实测 CPU 繁忙程度 |
| Phase 1：健康探测 | **启动探测已实现，持续健康调度未完成** | [节点 prepare](../../psrl/workers/env_service/node.py#L49)执行 Docker ping、设备与遥测检查，[节点启动](../../psrl/workers/env_service/router.py#L296)还检查 command 所需能力。尚未形成“运行中识别坏节点 → 从派发候选剔除 → 恢复后重新加入”的完整自动闭环。启动失败会保存并传播，不能视为自动换机 |
| Phase 2：GPU 映射、驱动挂载、借还 | **实现存在，部分真实验证** | 动态 overlay、设备身份核对和命令归还已验证。vLLM 同卡借还的历史小实验通过，但绕过 NIXL 的推理适配器不是完整训练权重同步验收，见第 3 节。当前 command 与模型借卡不能组合，逐卡需求和外部 GPU 所有权边界仍见第 5 节 |
| Phase 2：Gate B 与 GPU 活动归因 | **软节流已实现，归因目标未完成** | Gate B 有 CPU 测试、实机等待与远端查询证据，但没有逐进程归因，以及原稿指定的后台邻居、第二任务、完整奖励对照验收。命令模式跳过旧 Gate B，不能用 command 的吞吐结果签收此目标 |
| Phase 3：负载感知放置 | **基础容量派发已实现，完整目标未完成** | [EnvAwareDispatcher.assign](../../psrl/workers/env_service/placement.py#L84)按 CPU/GPU 类型、空闲槽位及快照派发，最终再准入。尚未按实际编译 CPU 压力、具体每条任务的 CPU／主存／逐卡显存需求精确放置，整组采样还共用一个放置决策 |
| Phase 3：坏放置后的有界重派发 | **通用恢复基础存在，环境专用闭环未完成** | [manager](../../psrl/workers/agent_loop/manager.py#L629)已有失败组补采与连续失败熔断，但这不是“同一环境任务识别坏节点，清理后按明确尝试上限换到另一健康节点”。不应写成没有任何重试，也不能写成自动重派发已完整实现 |
| Phase 4：容器复用等延期功能 | **尚未实现，但原稿已明确延期** | 节点服务预热不等于跨任务保留并复用旧容器，主容器仍按任务创建和清理 |

**B1 到 B8 也不能笼统标记全修复。**

| 原稿问题 | 当前判断 |
|---|---|
| B1 驱动库版本、B2 PTX JIT 库 | [resolve_driver_libs](../../psrl/workers/env_service/overlay.py#L93)解析匹配库，旧 sandbox 路径也引用它。真实 CUDA 有通过证据，但原稿“特定不匹配版本必定报指定错误码”的观察不能当作现在所有节点的事实 |
| B3 固定映射 GPU 0 | 启用 EnvRouter 后按租约动态生成设备映射。用户固定 OFF 仍保留其静态调度语义，不将 OFF 或其他未启用路径说成也受这份账本管理 |
| B4 进程内信号量 | ON 绕过本地信号量并使用共享账本，OFF 分支保留原有路径，见 [agent_loop](../../examples/sciaccel_rl/agent_loop.py#L394) |
| B5 独立 verifier 泄漏、B6 无归属标签 | 已有 [环境标签覆盖](../../psrl/workers/env_service/docker.py#L158)与 [runner 收尾](../../examples/sciaccel_rl/runner.py#L550)，历史故障回收通过。标签覆盖不等于 sidecar 等整个集合都有独立的资源硬限制，也不保证所有 Docker 永久故障均能自动恢复 |
| B7 暂停路由后异常未恢复 | **原函数仍未彻底修复**。[check_and_migrate](../../psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py#L290)暂停后执行 ABORT 和等待，恢复操作仍未放进 `finally`。中途抛异常可能跳过恢复，这是当前源码风险，本次没有在实机注入该故障。新的借卡协议有自己的保护，不能替代修复这个函数 |
| B8 默认只尝试一次 | [默认 retry_limit](../../psrl/trainer/config/rollout/psrl_rollout.yaml#L154)仍为 1。[worker 重试循环](../../psrl/workers/agent_loop/worker.py#L391)与 manager 补采存在，但实验代理的受控重试不等于默认 science 配方已启用同样策略 |

**实现方向发生过变化。** 原稿将每次 exec 加锁和同卡并发计算列为范围外，因为 shell 可以遗留后台计算。当前 command 增加了设备撤权与物理归还确认，才允许在合作式前台阶段之间复用，并允许预算内多个命令并发。它不是只补一把锁，也没有解决任意后台任务的自动暂停恢复。原稿“Router 不参与 exec 路径”现在也应区分控制面与数据面：命令需要请求 Router 授权，但命令输出不经过 Router 转发。

**不应列为欠账的原稿非目标。** 活容器跨节点迁移、任意 CUDA 进程抢占恢复、MPS 共享、复制多份任务取最快结果，本来就不在这一阶段承诺中。预测空档、严格 FIFO 后的小请求回填、cgroup v2 支持是后续扩展，不是把原稿某一阶段勾选完成所必需的全部条件。当前应优先补后端统一、原生工具接入、活动归因、健康重派发与 B7 异常恢复。

## 10. Science 原生任务能否使用 command

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
| 1 | [demand_from_task](../../examples/sciaccel_rl/env_demand.py#L23)从任务清单与数据提示读取 GPU 需求。模型训练本身用 GPU，不等于环境任务也需要 GPU |
| 2 | [SciAccelAgentLoop._run_harbor_in_thread](../../examples/sciaccel_rl/agent_loop.py#L358)、[runner 注入 gpu_commands](../../examples/sciaccel_rl/runner.py#L401)负责资源接线，不会分析任意 shell 的内容 |
| 3 | [TruncatingTerminus2](../../examples/sciaccel_rl/agent.py#L34)没有通用 command 适配。Harbor [_execute_commands](env-service-verification/results/env-design-status-20261021/source/harbor/terminus_2.py#L1199)调用 `session.send_keys(block=False)`，由 [tmux 非阻塞发送](env-service-verification/results/env-design-status-20261021/source/harbor/tmux_session.py#L765)等待指定观察时间后返回 |
| 3P | [CommandPressureAgent](env-service-verification/scripts/science_fixed_work_agent.py#L27)显式调用 `exec_gpu`，仅覆盖注入的 CUDA 阶段 |
| 4 | 原生 CUDA 题的 [gpu-wrap](env-service-verification/results/env-design-status-20261021/source/cuda-task/gpu-wrap#L35)运行求解程序并记录整卡利用率、显存 CSV |
| 5 | 原生 CUDA 题的 [telemetry_ok](env-service-verification/results/env-design-status-20261021/source/cuda-task/grade.py#L68)读取 CSV，[compute_score](../../examples/sciaccel_rl/reward.py#L44)选取指定奖励键 |
| 6 | [runner finally](../../examples/sciaccel_rl/runner.py#L550)回收容器集合并释放租约 |

**答案要分三类。**

| 当前任务类型 | 是否需要环境 GPU | 能否用当前 command |
|---|---|---|
| 原生 LAPS 修复题，例如 `laps-repair-coef-2d-rktmod-l17k1` | [任务清单](env-service-verification/results/env-design-status-20261021/source/cpu-task/task.toml)申报 4 CPU、8192 MiB 主存，没有 GPU 声明，独立 verifier 也没有 GPU 声明 | 可以使用 EnvRouter 管理 CPU／主存和并发，但无需 GPU command。不能因训练模型用了 GPU，就把此题称为 GPU 容器任务 |
| 原生修复题加实验注入的 CUDA 压力阶段 | 注入阶段需要 GPU，后面的修复和评分仍是原任务逻辑 | **受控阶段已接入**，相关实机与训练实验验证的是这条显式路径，不代表普通模型生成的 GPU 命令都已接入 |
| 原生 `laps-accel-cuda` GPU 移植题 | [清单](env-service-verification/results/env-design-status-20261021/source/cuda-task/task.toml#L46)明确申报一张 GPU | **任务结构适合进一步适配，但当前不能仅切配置就视为完整支持**。默认终端工具未接入真实命令生命周期，活动归因也未完成 |

本地 `laps-accel-cuda` 位于任务仓库的 `blocked-gpu` 作者目录，本次核查的是该目录的任务契约，不能据目录名就推断它已经编译、构建或完成真实训练。本文结论只针对本地 PSRL 示例、任务快照及已读取的依赖，不代表另一个 ScienceInfra 仓库的当前实现。

**为什么原生 CUDA 题在结构上适合。** 它要求模型修改 Fortran／CUDA 求解器，编译后运行仿真，将数值轨迹和 `telemetry.csv` 保存到 `/logs/artifacts`。这可以划为“CPU 编辑编译 → GPU 仿真 → 文件保留 → CPU 检查”，如果求解进程及相关 GPU 访问都完整结束，便具备阶段归还条件。

这道题的 verifier 主要比较数值文件并检查遥测，不是重新在 GPU 上跑求解器。[清单的 verifier](env-service-verification/results/env-design-status-20261021/source/cuda-task/task.toml#L60)标为 separate，独立环境没有 GPU 声明。因此，此题当前最需要补的是 agent 执行阶段的命令生命周期和遥测归因，不能笼统说必须先把这个 CPU verifier 改成 GPU 命令。其他会在评分时重跑 CUDA 的任务，才需要另外适配其 GPU verifier。

**真正缺的适配不是改一个函数名。** 教学例子：模型向 tmux 输入一个需要 30 秒的 GPU 程序，工具只等待 1 秒就返回当前终端输出。若把这个“发送按键”的调用当成 `exec_gpu` 的全部阶段，1 秒后命令接口已返回，GPU 程序却还要运行 29 秒。当前归还检查应拒绝这种交接，无法完成确认时隔离相关 unit。正确适配需要一个可观察、可取消的 GPU 作业边界：获得授权后启动程序，允许模型读取其状态，程序及 GPU 访问结束后才发起归还。保持 tmux 的目录、变量和进程状态也要纳入适配，不能随意替换成一个全新的无状态 shell。

当前 [exec_gpu](../../psrl/workers/env_service/docker.py#L73)要求 GPU 工作显式走这个入口；[command overlay](../../psrl/workers/env_service/overlay.py#L183)默认不授予设备读写权限，获得命令授权时才开放。因此，把原生 GPU 题配置成 command，并不会自动识别普通 `exec` 中的 CUDA 程序或替它申请 GPU。实际可能表现为无法访问设备，不能据配置能够解析就宣称任务已经适配。

**遥测还涉及另一项独立问题。** 当前 `gpu-wrap` 采集整张可见 GPU 的活动，评分函数只要求每个检查至少 3 个有效样本且峰值利用率达到 10%。即使命令预算正确，两条合法并发命令仍可能使 CPU 求解器的遥测看起来活跃。必须在执行并采集遥测的窗口内做可靠进程归因或计算隔离。等到 CPU verifier 读取 CSV 时才加独占锁，不能修复此前已污染的样本。具体阈值来自本次源码读取，不是新一轮 GPU 测量结论。

**跟着图读源码。** 先在 1 核对 GPU 需求属于任务环境，沿 2 看配置变为租约和 Harbor 参数。随后分开读取 3 的默认终端路径与 3P 的实验代理路径，它们不是顺序调用。到 4 要辨别“工具返回”和“GPU 程序实际结束”，再到 5 确认评分使用的是哪些文件和奖励键，最后 6 核对清理。命令模式中的归还确认可与不依赖它的 CPU／模型工作并行，但下次 GPU 请求必须等待旧归还完成。这些代码如果有理解上的问题，可以随时追问。

**本次可复现的 CPU 静态审计。** 在项目 Python 环境、仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/audit_science_command_readiness.py
```

脚本读取保存的两个任务清单，并用 Python 抽象语法树查找调用位置，不导入 Harbor、不运行容器。实际输出见 [readiness.json](env-service-verification/results/env-design-status-20261021/readiness.json)：修复题环境 GPU 声明为 0，CUDA 题为 1，两者 verifier GPU 声明都为 0。这里读取的是清单声明，不是对实际容器设备挂载的测量。默认代理及 Harbor 终端、评分器文件没有 `exec_gpu` 调用，实验代理有显式调用。静态调用检查不是完整运行验收，不能证明任意输入一定成功或失败。文件来源与指纹见 [manifest.json](env-service-verification/results/env-design-status-20261021/manifest.json)。

本节新增脚本的 Ruff 与格式检查通过，本文及总览索引的本地链接路径、源码行号范围和本文目录锚点检查通过。全仓检查在隔离副本执行，仍因既有代码和注释风格问题失败，见 [本次 pre-commit.log](env-service-verification/results/env-design-status-20261021/pre-commit.log)；自动修正未写回工作区。
