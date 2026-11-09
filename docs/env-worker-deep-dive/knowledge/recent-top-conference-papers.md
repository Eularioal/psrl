# 环境执行与 GPU 资源调度：五篇近期顶会论文

> **核心结论**：优先读 RollArt 和 DynaRL，确定 Agent 强化学习环境调度已有工作的覆盖范围；再读 Murakkab、Prism、Weave，分别学习工作流优化、显存弹性与空闲窗口复用。五篇均已核对为 **OSDI 2026 正式论文**，不是仅有 arXiv 的预印本。课题可重点考察“有状态环境会话中的 GPU 命令调度，以及它与模型推理的资源竞争”；这是本次调研提出的定位建议，不是已经证实的新颖性结论。

检索日期：2026-11-02。按近期性、课题相关性和正式发表三项筛选，最终五篇均来自 2026 年 7 月 13–15 日举办的 OSDI（操作系统设计与实现研讨会，系统领域顶会）。这是一份相关论文精选，不声称穷尽全领域最新五篇。已核对官方出版条目并查阅论文问题定义与相关机制；未复现论文实验，也未完成各论文实现的逐项审计。

阅读补充：[RollArt 与 DynaRL 精读](rollart-dynarl-deep-reading.md)解释阶段解耦、动态迁移及其与本地研究的边界。该精读已计入 2026-11-05 的同机 command 池与准备前移实现，不把这些已有能力重新作为创新主张。

## 目录

- [1. 课题定位与本地依据](#1-课题定位与本地依据)
- [2. 五篇论文总表](#2-五篇论文总表)
- [3. RollArt：环境长尾与轨迹级解耦](#3-rollart环境长尾与轨迹级解耦)
- [4. DynaRL：随瓶颈变化重新分配资源](#4-dynarl随瓶颈变化重新分配资源)
- [5. Murakkab：按完整工作流优化](#5-murakkab按完整工作流优化)
- [6. Prism：让模型显存真正具有弹性](#6-prism让模型显存真正具有弹性)
- [7. Weave：利用依赖造成的空闲窗口](#7-weave利用依赖造成的空闲窗口)
- [8. 一个贯穿五篇论文的例子](#8-一个贯穿五篇论文的例子)
- [9. 跟着图读源码与可运行例子](#9-跟着图读源码与可运行例子)
- [10. 阅读顺序与研究验证建议](#10-阅读顺序与研究验证建议)
- [11. 证据边界与文档维护](#11-证据边界与文档维护)

## 1. 课题定位与本地依据

先看课题中的角色如何连接。下面是用于选论文的范围图，框架结构依据现有文档与本次抽查源码；不是五篇论文共同采用的架构。

```text
[1] Science runner 申请环境会话
          ↓
    Harbor 组织模型与环境的多轮交互
          ↓ GPU 命令准备就绪
[2] 统一作业入口：提交、等待、取消
          ├→ [3a] direct command：原 unit 内准入、执行、确认归还
          └→ [3b] MPS：按作业显存预算准入，并发执行
          ↓ 命令结果返回，容器与文件可继续保留
    下一轮交互 → 重复 [2]
          ↓ 任务结束
[4] 汇合作业清理、移除所属容器、释放会话租约

论文关注面：
RollArt → 多条轨迹如何独立推进；DynaRL → 各阶段怎样动态分资源
Murakkab → 整体依赖与执行选择；Prism → 模型占有的显存如何回收
Weave → 不同作业如何利用彼此等待形成的窗口
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责与函数 | 本地源码 |
|---|---|---|
| 1 | `run_harbor_episode` 申请环境资源，外部 Harbor 负责具体任务编排 | [runner.py:372](../../../examples/sciaccel_rl/runner.py#L372)；完整背景见[当前设计](current-design.md) |
| 2 | `start_gpu_job` 提交，`_execute_gpu_job` 选择执行分支 | [managed_execution.py:130](../../../psrl/workers/env_service/managed_execution.py#L130)、[执行分支:168](../../../psrl/workers/env_service/managed_execution.py#L168) |
| 3a | `GpuCommandScheduler.acquire` 按 `lease.unit_id` 排队；`_return` 等物理撤权后更新账本 | [command_gpu.py:58](../../../psrl/workers/env_service/command_gpu.py#L58)、[归还:215](../../../psrl/workers/env_service/command_gpu.py#L215) |
| 3b | `MpsDomain.admit` 结合实测显存、预约和作业状态准入 | [parallel_gpu.py:208](../../../psrl/workers/env_service/parallel_gpu.py#L208) |
| 4 | runner 的 `finally` 汇合 MPS 租约、清理容器，再释放会话 | [runner.py:605](../../../examples/sciaccel_rl/runner.py#L605) |

因此，本次把课题理解为：**Agent 强化学习（模型通过与环境交互和奖励来学习）中的环境执行基础设施，尤其是 GPU 命令的细粒度资源调度。** 一条 rollout 是采集到的交互轨迹，episode 是一次完整环境任务，command 是其中一次具体命令。三者不是同一个调度粒度。

本地对应专题是[命令阶段配额](command-gpu-reservations.md)、[MPS 并发](mps-parallel-execution.md)、[动态放置提案](dynamic-command-placement.md)及[模型与环境共享](vllm-env-opportunistic-sharing.md)。MPS 是 NVIDIA 的多进程 GPU 服务；它提供并发执行基础，但不能仅凭启用 MPS 就断定总任务耗时降低。这些代码如果有理解上的问题，可以随时追问。

## 2. 五篇论文总表

以下按建议阅读优先级排序，全部为 OSDI 2026；官方链接提供摘要、BibTeX 和论文下载。

| 顺序 | 论文及第一作者 | 页码 | 与课题的关系 | 官方来源 |
|---|---|---|---|---|
| 1 | **RollArt: Disaggregated Multi-Task Agentic RL Training at Scale**；Wei Gao 等 | 863–881 | 环境执行长尾、独立轨迹与训练流水线，最直接的问题域对照 | [出版页](https://www.usenix.org/conference/osdi26/presentation/gao) · [PDF](https://www.usenix.org/system/files/osdi26-gao.pdf) |
| 2 | **DynaRL: Flexible and Dynamic Scheduling of Large-Scale Reinforcement Learning Training**；Yuanqing Wang 等 | 847–862 | 动态瓶颈、资源迁移、多轮交互调度，最值得检验贡献重叠 | [出版页](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing) · [PDF](https://www.usenix.org/system/files/osdi26-wang-yuanqing.pdf) |
| 3 | **Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms**；Gohar Irfan Chaudhry 等 | 567–587 | 从整个 Agent 工作流决定执行配置和资源放置 | [出版页](https://www.usenix.org/conference/osdi26/presentation/chaudhry) · [PDF](https://www.usenix.org/system/files/osdi26-chaudhry.pdf) |
| 4 | **Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning**；Shan Yu 等 | 75–93 | 弹性收缩模型显存，启发模型与环境共享协议 | [出版页](https://www.usenix.org/conference/osdi26/presentation/yu-shan) · [PDF](https://www.usenix.org/system/files/osdi26-yu-shan.pdf) |
| 5 | **Weave: Efficient Co-Scheduling for Disaggregated RL Post-Training**；Tianyuan Wu 等 | 809–827 | 通过多作业协同利用等待窗口，同时计入状态驻留成本 | [出版页](https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan) · [PDF](https://www.usenix.org/system/files/osdi26-wu-tianyuan.pdf) |

选取同一届会议是本次相关性筛选的结果，并非认为只有 OSDI 才是顶会。检索也涉及 EuroSys、MLSys、NeurIPS 与 ATC；此处优先保留直接覆盖课题不同层面的近期正式论文。

## 3. RollArt：环境长尾与轨迹级解耦

**问题与方法。** 多轮 Agent 任务中，有些环境很快返回，有些需要启动容器、等待工具或处理失败。如果所有环境按轮同步，最快的轨迹也要等最慢者。RollArt 把生成、环境交互与评分解耦到轨迹粒度，各环境独立推进，并让不同阶段使用适合的硬件。它还将无状态奖励计算放到按需提供计算资源的平台，并通过限制数据使用的模型版本落后程度，让采样与训练重叠。[官方摘要](https://www.usenix.org/conference/osdi26/presentation/gao)

**对你的启发。** 重点阅读正文 §3 的环境耗时分析，以及 EnvManager 的独立交互循环。你的环境排队、执行、回收时间应分开统计：即使 GPU 命令更快，容器初始化或评分等待也可能仍支配整条轨迹。论文的环境侧主要映射到 CPU 集群，不能直接把其结果解读成已经解决任意 GPU 工具程序的显存准入与安全归还。[正文](https://www.usenix.org/system/files/osdi26-gao.pdf)

**阅读时要回答的问题。** PSRL 已有的异步交互消除了哪些同步屏障？在这些屏障已消除后，命令级 GPU 调度还有多少独立收益？比较时应保持异步方式一致，否则容易把“解除批同步”的收益归给 GPU 调度。

## 4. DynaRL：随瓶颈变化重新分配资源

**问题与方法。** 固定给采样和训练分配 GPU，不适合随时间变化的负载。DynaRL 用动态超图记录组件、依赖和运行信息；超图可以用一条关系连接多个参与者，适合表达组件组间的数据流。调度器识别资源过配的组件，将资源迁往瓶颈，并提供安全迁移接口与保持上下文关联的数据路由。[官方摘要](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing)

**最值得细读。** 正文 §5.1 的控制循环先观察持续低利用，再比较资源重分配候选的预计吞吐；§5 周边的数据调度讨论还考虑多轮交互与缓存复用。其价值是把“看起来空闲”变成“收回后整体更快”的决策，而不是每次看到低利用率就立即搬任务。[正文](https://www.usenix.org/system/files/osdi26-wang-yuanqing.pdf)

**与你的关系。** 这是最需要认真比较的动态调度工作之一。你的命令级资源申请可以提供更明确的可切换边界，但它是否足以形成独立贡献，需要比较状态类型、决策粒度、迁移成本和性能目标。不能仅凭“有 Router”或“支持动态分配”主张新颖性，也不能未经实现审计便断言 DynaRL 完全不支持某类 GPU 环境。

## 5. Murakkab：按完整工作流优化

**问题与方法。** Agent 应用包含多次模型和工具调用，逐个调用各自优化可能让整体更慢。Murakkab 用声明式抽象将“要完成什么工作流”与“选什么模型、硬件和运行配置”分离，再通过性能测量引导的优化器及自适应运行时满足服务目标。SLO 是服务等级目标，例如响应时间上限或质量要求。[官方摘要](https://www.usenix.org/conference/osdi26/presentation/chaudhry)

**对你的启发。** 调度优先级可以考虑命令的后继工作：某条环境命令完成后，是否马上解锁模型调用、评分或最后一条待收集轨迹？只看当前显存和先来先服务，无法表达这种端到端价值。阅读时重点找声明式接口、可选执行配置、性能分析和在线调整之间如何连接。[正文](https://www.usenix.org/system/files/osdi26-chaudhry.pdf)

**边界。** 它研究 Agent 工作流服务，并不等价于强化学习训练框架。它允许选择模型和执行配置，而你的调度对照通常需要固定模型、任务与评分规则。因此可以借鉴依赖建模，但不能把换模型带来的成本收益混入同模型调度加速。

## 6. Prism：让模型显存真正具有弹性

**问题与方法。** 推理引擎预留的显存，即使暂时没有有效数据，也不一定能被其他进程使用。Prism 通过 memory ballooning（显存弹性伸缩）回收并重新分配多个模型之间的显存，将同卡同时驻留与分时切换结合。其 kvcached 层分离 GPU 虚拟地址和实际物理显存，按需映射；这与只修改账本上的额度不同。[正文 §5](https://www.usenix.org/system/files/osdi26-yu-shan.pdf)

**对你的启发。** KV cache 是模型为后续生成保留的注意力中间结果。当前讨论中的“给环境留余量”，可以进一步拆成两个问题：哪些缓存页能真实归还，归还后推理延迟和并发容量会损失多少？重点读显存管理机制及请求调度如何保护延迟目标。[出版页](https://www.usenix.org/conference/osdi26/presentation/yu-shan)

**边界。** Prism 的对象是协作的多模型服务。把它推广到任意 Docker GPU 程序，还需要环境内存需求声明、真实释放确认和故障处理；显存能放下也不代表计算竞争可接受。它不是当前 EnvRouter 已接入的能力，也不能替代 MPS 的执行管理。

## 7. Weave：利用依赖造成的空闲窗口

**问题与方法。** 将 rollout 与训练放到不同集群后，两者的先后依赖仍可能造成空闲。Weave 让不同作业利用彼此的空窗，使用共同执行组组织资源，再分别做组间放置与组内调度。模型状态在主机内存中驻留，以降低再次启动执行时的切换代价。[官方摘要](https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan)

**对你的启发。** 你计划让环境使用推理空窗，首先要算清“空窗长度”和“借还总成本”。后者包括卸载、恢复、设备授权和确认归还，不只是命令本身的时间。阅读正文时重点查执行组、驻留约束及切换成本如何影响可行调度，而不是只看利用率曲线。[正文](https://www.usenix.org/system/files/osdi26-wu-tianyuan.pdf)

**边界。** 多个 RL 作业的阶段互补，与一个 Agent 内部未知时长、可能带副作用的命令不同。论文中组内调度的最优性依赖其模型与约束，不能照搬为任意命令队列的最优保证。

## 8. 一个贯穿五篇论文的例子

以下全部为**教学假设**，不是论文实验数字，也不是 PSRL 实测。一个 Agent 会话 A 需要运行 GPU 命令 A1，接着等待模型规划，再运行 A2；另一个会话 B 有独立 GPU 命令。容器内的代码与文件需要跨命令保留。

假设一张环境 GPU 上，A1 执行 2 秒，规划等待 6 秒，A2 执行 2 秒。B 在第 2 秒就绪，需运行 4 秒；两者不能同时运行。

| 策略 | 中间过程 | 所有工作完成 |
|---|---|---|
| A 全会话独占环境 GPU | A1：0–2 秒；空等：2–8 秒；A2：8–10 秒；B：10–14 秒 | 14 秒 |
| A 按命令借还，忽略借还开销 | A1：0–2 秒；B：2–6 秒；A2：8–10 秒 | 10 秒 |

这里保留的是 A 的容器会话，归还的是它暂时不需要的 GPU 使用权。收益来自 B 填补等待窗口，不是 A1 或 A2 本身算得更快。

五篇论文分别提醒你检查：RollArt——B 是否被 A 的环境同步屏障卡住；DynaRL——资源应从哪里转到哪里；Murakkab——哪条依赖决定最后完成时间；Prism——若借用的是模型 GPU，显存是否真的可以回收；Weave——切换成本是否吃掉整个空窗。

再设另一种独立条件：GPU 容量为 80 GiB，模型实际占用 60 GiB，环境命令需 24 GiB，那么直接并行需要 84 GiB，不能准入。如果模型经协议真实归还 8 GiB，容量才变为 52 + 24 = 76 GiB，尚余 4 GiB。这个算式仅说明容量可行；4 GiB 是否足够覆盖峰值、模型恢复会等多久、计算干扰是否拖慢关键路径，都仍需验证。

## 9. 跟着图读源码与可运行例子

下面是本次抽查的真实执行顺序；不同会话可以并发，归还还可能在后台进行，不能把它理解成全局单线程执行。

| 步骤 | 图中编号 | 文件与函数 | 输入、输出或状态变化 |
|---|---|---|---|
| 会话准入 | 1 | [runner.py:372](../../../examples/sciaccel_rl/runner.py#L372)，`run_harbor_episode` | 资源需求进入 `acquire`，得到会话租约 |
| 命令提交 | 2 | [managed_execution.py:130](../../../psrl/workers/env_service/managed_execution.py#L130)，`start_gpu_job` | 命令变成可查询、等待或取消的作业 |
| direct 分支 | 3a | [command_gpu.py:58](../../../psrl/workers/env_service/command_gpu.py#L58)，`acquire` | 请求在既有 unit 内等待或获准；本处没有跨 unit 自动改派 |
| MPS 分支 | 3b | [parallel_gpu.py:208](../../../psrl/workers/env_service/parallel_gpu.py#L208)，`admit` | 结合已批准作业与遥测判断能否接纳新作业 |
| direct 归还 | 3a | [command_gpu.py:215](../../../psrl/workers/env_service/command_gpu.py#L215)，`_return` | 完成撤权与模型侧释放后，清除活动 GPU 记账；失败则隔离 |
| 任务结束 | 4 | [runner.py:605](../../../examples/sciaccel_rl/runner.py#L605)，`finally` | 汇合、清理所属资源，最后释放会话 |

复用已有 [dynamic_command_placement_demo.py](../env-service-verification/scripts/dynamic_command_placement_demo.py)，从仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/dynamic_command_placement_demo.py
```

这是仅用 Python 标准库的 **CPU 教学模拟**，不调用 CUDA、不创建容器、不复现论文。脚本注释里的 C/P 编号对应[原动态放置文档](dynamic-command-placement.md)的现状与提案图。它演示本图 3a 的固定绑定和未来动态选择之间的区别：GPU0 被 B 占到第 22 秒，GPU1 第 5 秒空闲，A2 第 4 秒就绪、需 3 秒。固定 GPU0 时 A2 在第 25 秒完成；换卡成本为 0.2 秒时，A2 在第 8.2 秒完成，而全组仍需到第 22 秒。换卡成本为 20 秒时应继续留在 GPU0。

它假设预先知道作业耗时，并在结束时立即完成物理归还，没有模拟模型、缓存、实际并发干扰或故障。它说明调度与切换成本的关系，不能证明本地已实现动态改派。

本次已实际运行该脚本，固定绑定、低切换成本和高切换成本三组全组完成时间分别为 25、22、25 秒，与上述教学推导一致。文档与两个索引的 132 个本地链接路径检查通过，新文档目录锚点及源码链接行号检查通过。

## 10. 阅读顺序与研究验证建议

建议先读 **RollArt → DynaRL**：先确定环境长尾问题的已有解法，再明确动态调度贡献的重叠。随后读 **Murakkab → Prism → Weave**，分别补齐整体目标、资源释放机制和窗口经济性。

一个待验证的研究表述是：**在保留有状态环境会话的前提下，以 GPU 命令为执行单元，结合资源峰值、模型性能约束和回收确认，减少 Agent RL 的端到端环境等待。** “有状态”指下一轮仍要使用上一轮文件、进程或连接；具体能保留哪些状态，需要按后端能力注明。

| 待验证问题 | 建议比较 | 需要观察 |
|---|---|---|
| 命令粒度是否有独立收益 | 相同异步轨迹机制下，episode 与 command 持有 | 固定完成量总时长、空闲占用、排队与回收开销 |
| 动态换卡是否值得 | 固定 unit 与计入切换代价的候选选择 | 单会话延迟、全组完成时间、数据准备与恢复时间 |
| 并发是否真的有利 | 相同任务的串行、普通多进程、MPS | 任务时间、模型延迟、显存峰值、失败率 |
| 同卡借还是否可接受 | 独立环境卡、显式共享、睡眠借卡 | 推理延迟、吞吐、真实物理归还与恢复成本 |

这些是后续研究建议，本次没有启动训练。固定历史 baseline 仍遵循项目约定；机制消融应另列，不能用改弱基线的方法制造调度收益。已有异步或并发机制必须在对照双方保持一致，除非它本身就是待验证变量。

## 11. 证据边界与文档维护

- 本地核对基准为主开发目录 HEAD `affe7c2a39e40c3198423c1170614a54f34573c1`，包含现场未提交修改。本次只抽查图中接口、绑定和清理位置，不宣称重新验收完整架构或部署状态。
- 官方出版页已核对五篇的题名、作者、会议、年份和页码；论文来源均为 USENIX 官方全文。正文中的“对你的启发”和研究问题是本次分析，不是作者声称已经解决 PSRL 的问题。
- 本文未罗列论文最大加速比，以免不同硬件、任务、成本与延迟指标被误当成可直接比较的性能排名。
- 本文补充现有知识分类中的论文阅读入口，与 [GPU workload 调研](gpu-workload-benchmark-survey.md)互补：后者选实验任务，本文选系统研究参照。
- 本次没有改变架构、接口、配置或能力边界，因此不改写[当前设计](current-design.md)与既有机制专题；仅新增本调研并更新分类索引和总索引。现状入口原有核对版本不因这次文献调研而刷新。

[返回知识分类](README.md) · [返回专题入口](../README.md) · [返回总索引](../../README.md)
