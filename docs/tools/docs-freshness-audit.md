# docs 文档时效巡检

> **核心结论**：两轮审计均发现失效描述，已在原文修正。2026-11-05 新处理 worker 并发、GPU 工具分发、准入分支、direct/MPS 边界和远程缓存配置转发，删除被实现取代的实施待办与重复进度段。历史实验条件保留。目录结构清晰和链接可打开，都不能保证架构描述仍然正确。
>
> **范围与版本**：第 1 至 5 节记录 2026-11-02、基准 `affe7c2` 的巡检，第 6 节记录 2026-11-05、基准 `0326d46` 的增量审计。两轮都保留开始前已有的文档及实验未提交改动。全目录静态扫描加重点源码核验，不代表全部文档逐句验证，也不代表远端合并或集群部署。

## 目录

- [1. 覆盖范围与判定方法](#1-覆盖范围与判定方法)
- [2. 确认过时且已处理的内容](#2-确认过时且已处理的内容)
- [3. 抽查后仍成立的说明](#3-抽查后仍成立的说明)
- [4. 尚未覆盖与待核实项](#4-尚未覆盖与待核实项)
- [5. 检查结果与维护入口](#5-检查结果与维护入口)
- [6. 2026-11-05 再审计与清理](#6-2026-11-05-再审计与清理)

## 1. 覆盖范围与判定方法

巡检开始时 `rg --files docs -g '*.md'` 列出 114 篇可见 Markdown，覆盖设计、教程、运行指南、示例、环境专题及工具文档。逐文件扫描“当前”“尚未实现”“不支持”“TODO”等时效信号和本地链接，再对容易误导实现判断的内容查源码。语义重点是环境执行架构，同时抽查参数服务器、细粒度训练重叠、KV 缓存配置和 DAPO 快速开始参数。

| 文档区域 | 扫描篇数 | 核验深度 |
|---|---:|---|
| 环境执行专题 | 47 | 全部静态扫描，重点核验 GPU 执行、原生提供者、MPS 与项目介绍入口 |
| 设计 | 10 | 静态扫描，核验 EnvWorker 和参数服务器的部分能力声明 |
| 教程 | 5 | 静态扫描，抽查配置限制与快速开始参数 |
| PSRL 知识与运行指南 | 22 | 静态扫描，区分固定日期检查与现状断言，未重新运行集群操作 |
| 示例 | 16 | 静态扫描，抽查 DAPO 配方，未重新执行训练 |
| 开发工具 | 10 | 静态扫描和本地链接检查，外部产品当前行为未重新验证 |
| 总入口、概览与根目录笔记 | 4 | 静态扫描与入口语义检查 |

另外读取到的被忽略实验结果目录含源码或文档快照。这些是历史证据，不以当前源码重写。最初的路径扫描也把 MyST 的无 `.md` 页面引用列为候选，识别其页面目标后不算死链。不能将原始候选数量直接当作“过时文档数量”。

过时与否按内容判断：固定提交的实验报告与新代码不同不自动构成错误，明确说“当前不支持”却与当前实现相反才需要修订。文档记录的基准提交早于 HEAD，也不自动意味着其全部结论失效。

## 2. 确认过时且已处理的内容

| 问题 | 原描述的风险 | 当前依据与处理 |
|---|---|---|
| 原生后端仍被写成 CPU-only | [原生后端正文](../env-worker-deep-dive/knowledge/native-harbor-backend.md)仍说 GPU 接线未完成，并笼统列出 GPU 绑定拒绝 | [构造函数](../../psrl/workers/env_worker/harbor.py#L22)接受受管理 GPU 执行绑定，[runner](../../examples/sciaccel_rl/runner.py#L145)注入租约映射。修订接线、能力校验和后续工作，保留旧睡眠借卡与构建等限制 |
| 独立取消被写成所有路径都不支持 | 原生后端的普通 shell 失败清理，被扩展成整个后端都不能逐作业取消 | [MpsDomain.cleanup](../../psrl/workers/env_service/parallel_gpu.py#L342)管理单条 MPS 作业。正文现区分普通命令的环境失效和 MPS 正常取消，不宣称所有故障都隔离 |
| 同容器并发仍在待办列表 | [GPU 作业与遥测](../env-worker-deep-dive/knowledge/gpu-jobs-and-task-telemetry.md)开头说已实现，第 7 节却列“暂不实施” | [初始化](../../psrl/workers/env_service/managed_execution.py#L17)按后端设置并发，[MPS 提交](../../psrl/workers/env_service/managed_execution.py#L187)进入节点服务。移出整体未实施待办，分别说明已落地和剩余范围 |
| 流程比较表混淆 direct 和 MPS | [流程讨论](../env-worker-deep-dive/archive/env-router-current-flow.md)把同容器串行写成当前所有后端的限制 | 修正图例、能力表、数字例子和执行说明。direct 的串行限制保留，MPS 条目改成已有实现，历史 CPU 演示仍是历史证据 |
| 用旧参数判断受管理执行 | [命令配额](../env-worker-deep-dive/knowledge/command-gpu-reservations.md)称缺少 `gpu_commands` 就走普通执行 | [公共入口](../../psrl/workers/env_service/managed_execution.py#L53)通过执行策略判断，episode 也可以有受管理后端。改为核对 `gpu_execution`、租约模式及 runner 注入 |
| MPS 探索记录覆盖了后续能力 | [vLLM 共享](../env-worker-deep-dive/knowledge/vllm-env-opportunistic-sharing.md)仍笼统写 EnvRouter MPS 后端未完成 | 明确环境专用 MPS 后端已接入，探索阶段计划按历史阅读。vLLM 同卡 MPS、隔离与性能边界没有被顺带宣称完成 |
| 站点架构遗漏原生提供者 | [EnvWorker 架构](../design/env_worker.md)只写 Science 的 Compose 路径，调用方向也容易误读 | [后端选择](../../examples/sciaccel_rl/runner.py#L349)存在 `env_worker` 分支。补充 runner 申请租约、Harbor 编排、两种提供者与公共执行层的关系 |
| 历史汇总被当作最新介绍 | [设计简版](../env-worker-deep-dive/experiments/env-design-and-results-brief.md)和[路线图](../env-worker-deep-dive/milestones/env-status-and-roadmap.md)明确只核对到 2026-10-28，导航却推荐它们了解当前状态 | 标题与入口明确标为阶段报告，链接当前设计。保留原始数字、当时规划和未验证项，不把后续结果写回早期实验 |
| 重构后源码定位越界 | 两篇调度实验报告中有 7 处 `docker.py#L…` 超过现有文件总行数 | 对照现有 [Compose 属性](../../psrl/workers/env_service/docker.py#L108)、[启动](../../psrl/workers/env_service/docker.py#L84)、[集合预算](../../psrl/workers/env_service/docker.py#L122)修复定位，并区分历史调用路径与现有函数。关联的修复前侧车预算缺口也补充了修复章节边界 |

一个具体例子：读者想在同一个环境提交 A、B 两条 GPU 作业。旧 TODO 会让人以为需要从零实现并发。实际选择 direct 时两条作业串行，显式选择 MPS 且满足预算、身份和部署条件时，节点可以分别准入、执行和回收两条作业。因此应修改“整体未实现”的原句，同时保留 direct 限制，不能简单把所有“串行”替换为“并行”。本例是阅读与路径选择示例，本次没有运行 GPU 作业。

## 3. 抽查后仍成立的说明

| 文档声明 | 本地依据 | 本次结论 |
|---|---|---|
| `nixl_gpu` 尚未实现 | [trainer 初始化](../../psrl/trainer/ppo/ray_trainer.py#L2123)仍抛出 `NotImplementedError` | 参数服务器文档这条不需要修改 |
| micro-batch 的 `pre_step` 尚未实现 | [FineGrainOverlapStrategy](../../psrl/trainer/ppo/strategies/fine_grain_overlap.py#L45)仍拒绝对应有效粒度 | 配置文档这条不需要修改 |
| 两个 LMCache 选项应保持关闭 | [仓库配置](../../psrl/trainer/config/psrl/lmcache.yaml#L48)仍保留 bug 注释和关闭默认值 | 与仓库支持策略一致，没有重新证明外部依赖当前版本的 bug 状态 |
| Quick Start 的 2 节点、每节点 8 卡、默认 staleness=3 和上下文覆盖 | [DAPO 脚本](../../examples/dapo_trainer/qwen2.5_3b_fsdp.sh)仍使用对应参数 | 与配方一致，没有重新执行训练或验证安装命令 |
| Env Service 默认关闭，Science 默认 Docker 提供者 | [服务配置](../../psrl/trainer/config/psrl/env_service.yaml#L6)、[Science 配置](../../examples/sciaccel_rl/config.py#L22) | 默认值说明仍成立，原生和 MPS 能力存在不意味着默认启用 |

## 4. 尚未覆盖与待核实项

- **不能宣称 114 篇逐句准确。** 全目录完成的是静态扫描与阅读入口检查，语义核验集中于上表结论。算法推导、硬件机制和全部示例命令未逐条重证。
- **源码行号仍可能落在错误函数内。** 本次修复越界定位并核对新增定位，其他仍在文件范围内的旧链接不能仅靠行数检查判定语义正确。后续维护相关专题时继续核验。
- **产品与外部依赖的时效未全面复核。** Codex、Claude、第三方安装接口和远端服务行为需要按实际版本及官方资料另行检查，不能因本地链接通过就认定最新可用。
- **部署与实验不是静态文档检查。** 没有检查节点实时状态，没有执行 Docker、GPU 或训练任务。既有 MPS 训练报告和开始前的未提交实验由原任务维护，本次不重写其结果。
- **源码也存在旧措辞。** 原生 [EnvWorkerEnvironment](../../psrl/workers/env_worker/harbor.py#L19)的类说明仍写预构建 CPU 任务，但其构造函数已经接受受管理 GPU 绑定。本文核对实际分支而非仅引用说明字符串，本次没有修改生产代码。

## 5. 检查结果与维护入口

- 巡检前的 114 篇可见文档，本地文件链接在识别 MyST 无后缀页面后未发现缺失目标。这不覆盖外部 URL 和所有章节锚点。
- 确认过时的现状说明已在原位置修订。历史汇总明确限定版本，上级目录同步指向现状入口。
- 7 处越界源码行链接已修复，新增引用依据本次实际读取的源码。
- 修订后包含本报告的 115 篇可见 Markdown，本地文件目标与源码行号范围检查通过，本报告目录锚点和 `git diff --check` 通过。检查没有遍历验证所有旧章节锚点及行号语义。
- 按仓库要求在临时隔离副本执行 `pre-commit run --all-files`，因既有 Python 行长、格式与注释规范问题失败。自动修复仅发生在副本，未带回工作区，不将该结果写成全仓检查通过。
- 本次仅修改文档。检查覆盖与未覆盖项按本报告记录，不以更换日期表示整库已经验证。

Env 当前实现从[当前设计](../env-worker-deep-dive/knowledge/current-design.md)开始阅读，按[知识索引](../env-worker-deep-dive/knowledge/README.md)查机制。以后架构变化需同步现状入口和受影响正文，维护要求见仓库 [AGENTS.md](../../AGENTS.md)。

## 6. 2026-11-05 再审计与清理

**范围。** `rg --files docs -g '*.md'` 列出 120 篇可见 Markdown。检查项目设计、配置教程、示例和环境专题的时效信号与引用，重点读取 worker、Science runner、工具分发、统一 GPU 执行、原生提供者及 KV Cache 配置源码。对照 `affe7c2..0326d46` 的代码变化检查 MPS 控制通道、KernelBench command 和 RE-Bench 新接入的文档边界。开始前已有文档和实验脚本未提交改动，本次在其基础上修订，未修改生产代码，也未重写这些实验结果。

**确认并处理的内容。**

| 原描述 | 原位修订或删除 | 源码依据与边界 |
|---|---|---|
| 总架构称每个 worker 同时只处理一条轨迹 | [总架构](../design/architecture.md)改为每次调用接收一条请求，一个 worker 可跟踪多个并发 task | [generate_trajectory](../../psrl/workers/agent_loop/worker.py#L262)创建 task，[add_agent_program](../../psrl/workers/agent_loop/worker.py#L220)汇合批次任务。并发还受派发与准入限制 |
| 工具调用说明引用已不存在的 worker 队列与函数 | [工具调用正文及图](../psrl/psrl-agent-loop-harbor-tool-calling.md)删除旧 busy-loop、`pending_program_queue` 和 `_launch_agent_loop` 描述 | 当前 worker 入口是 `add_agent_program`，不应以旧函数名引导读者查源码 |
| PSRL 只有终端工具、完全不参与调用和观察拼接 | 同一文档的结论、流程图、角色表、命令解析和下一轮 prompt 说明均改为普通终端与受管 GPU 两条分支 | [agent._execute_commands](../../examples/sciaccel_rl/agent.py#L104)拦截 `psrl-gpu`，[dispatch_gpu_tool](../../examples/sciaccel_rl/gpu_tools.py#L32)调用环境接口，JSON 返回值参与观察拼接 |
| Science 进程内计数器是所有模式的并发上限 | 改为未启用 Env Service 时使用进程内计数器，启用时通过 Router 准入 | [SciAccel 准入分支](../../examples/sciaccel_rl/agent_loop.py#L391)区分两条路径 |
| 统一执行文档把串行与固定 lease 预算扩展成全部后端限制 | [统一执行层](../env-worker-deep-dive/knowledge/unified-gpu-execution.md)明确流程图与串行语义属于 direct，MPS 可以提交逐作业显存预算 | [初始化](../../psrl/workers/env_service/managed_execution.py#L17)区分并发数，[MPS 提交](../../psrl/workers/env_service/managed_execution.py#L187)读取作业 `gpu_memory_mb` |
| 已完成接入仍夹杂 CPU-only、GPU 未接入的页首阶段进展 | 删除统一执行与[原生后端](../env-worker-deep-dive/knowledge/native-harbor-backend.md)被后续实现取代的进度段，保留实验章节的日期与原始结果 | 当前两种提供者共用 `ManagedGpuExecution`，不意味着构建、网络和远程池已全部统一 |
| 旧缺口清单仍建议从零实现 Harbor 准入、心跳和后端统一 | 删除[旧实施待办](../env-worker-deep-dive/archive/implementation-and-gaps.md#153-旧待办的现状映射)，改为现有实现映射。文档标题、核心结论与索引明确历史属性，修复前 AIRS 证据保留 | [runner](../../examples/sciaccel_rl/runner.py#L369)已有准入，[EnvLeaseHandle](../../psrl/workers/env_service/client.py#L35)已有心跳。旧故障记录不能当成所有现有部署的状态 |
| KV Cache remote 被笼统称为完全未实现 | 同步修正 [KV 设计](../design/kv_cache.md#configuration)和[配置手册](../tutorial/configuration/transport-and-rollout.md#lmcache)：配置转发已存在，完整远程部署未在本次验证 | [to_env_vars](../../psrl/utils/kv_cache/config.py#L182)转发 URL，[apply_env_vars](../../psrl/utils/kv_cache/manager.py#L210)在 [vLLM 初始化](../../psrl/workers/gen/vllm_async_server.py#L235)前应用。YAML 仍标 TODO，不提升为端到端支持承诺 |
| 现状入口仍说 MPS 修复未提交，收尾链接指向旧位置 | [当前设计](../env-worker-deep-dive/knowledge/current-design.md)更新局部核对范围，修正 runner 收尾定位，明确 CPU 演示为 11 月 2 日的既有结果 | [runner finally](../../examples/sciaccel_rl/runner.py#L610)，第 4 节各局部实验和研究仍保留其原日期与边界 |

例如，读者想让一个 worker 同时处理两个请求。旧架构会让人误以为必须先增加 worker 数。源码实际把两个请求各自创建为 task，随后一起等待完成。如果两个任务都请求环境资源，它们仍需分别经过相应准入，不能从“可以创建两个 task”推导“GPU 上必定同时执行两个任务”。这是源码阅读示例，本次没有运行任务。

**删除原则。** 本轮删除的是失效的实施建议、旧调用机制和重复进度段，没有整篇删除仍含独立机制说明或历史证据的文档。旧待办的细分目录项同步移除，仓库内未发现其他文档引用这些被删小节。总索引、PSRL 索引和环境知识索引同步说明历史范围及工具分发范围。

**复核后保留。** `nixl_gpu` 的未实现限制、micro-batch `pre_step` 限制仍有源码拒绝分支，Env Service 默认关闭、Science 默认 Docker 仍与配置一致。固定日期的压力实验、AIRS 修复前 baseline 和尚未落地的动态放置提案按原条件保留。KernelBench command 属于显式接入的特例，不能概括成所有 benchmark 和 verifier 自动支持 command。

**验证与未覆盖项。** 全目录本地链接目标与源码行号范围检查通过，本轮修改文档的章节锚点检查通过。新增和调整的关键源码定位已阅读其函数或分支，未以行号未越界替代语义核验。按仓库要求在独立临时副本执行 `pre-commit run --all-files`，Ruff、格式和 prose-style 检查因既有 Python 文件问题失败，例如 `docs/conf.py` 和历史实验脚本的行长。自动改写只发生在副本，未带回工作区。没有运行 Docker、GPU、训练或远程缓存服务，也没有核验外部 Harbor 当前安装版本的全部内部接口。这里的全目录检查不等于全部正文、外部 URL 和部署可用性均已验证。
