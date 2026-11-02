# docs 文档时效巡检

> **核心结论**：存在已失效的现状描述，集中在原生 EnvWorker 的 GPU 接入、MPS 同容器并发、旧参数判断及阶段报告的阅读入口。本次直接修订相关正文和导航，保留历史实验条件。目录结构清晰和链接可打开，都不能保证架构描述仍然正确。
>
> **范围与版本**：2026-11-02 对主开发目录巡检，基准提交 `affe7c2`，保留开始前已有的文档及实验未提交改动。下面是静态扫描与重点源码核验记录，不代表全部文档已逐句验证，也不代表远端已合并或集群已部署。

## 目录

- [1. 覆盖范围与判定方法](#1-覆盖范围与判定方法)
- [2. 确认过时且已处理的内容](#2-确认过时且已处理的内容)
- [3. 抽查后仍成立的说明](#3-抽查后仍成立的说明)
- [4. 尚未覆盖与待核实项](#4-尚未覆盖与待核实项)
- [5. 检查结果与维护入口](#5-检查结果与维护入口)

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
| 流程比较表混淆 direct 和 MPS | [流程讨论](../env-worker-deep-dive/knowledge/env-router-current-flow.md)把同容器串行写成当前所有后端的限制 | 修正图例、能力表、数字例子和执行说明。direct 的串行限制保留，MPS 条目改成已有实现，历史 CPU 演示仍是历史证据 |
| 用旧参数判断受管理执行 | [命令配额](../env-worker-deep-dive/knowledge/command-gpu-reservations.md)称缺少 `gpu_commands` 就走普通执行 | [公共入口](../../psrl/workers/env_service/managed_execution.py#L53)通过执行策略判断，episode 也可以有受管理后端。改为核对 `gpu_execution`、租约模式及 runner 注入 |
| MPS 探索记录覆盖了后续能力 | [vLLM 共享](../env-worker-deep-dive/knowledge/vllm-env-opportunistic-sharing.md)仍笼统写 EnvRouter MPS 后端未完成 | 明确环境专用 MPS 后端已接入，探索阶段计划按历史阅读。vLLM 同卡 MPS、隔离与性能边界没有被顺带宣称完成 |
| 站点架构遗漏原生提供者 | [EnvWorker 架构](../design/env_worker.md)只写 Science 的 Compose 路径，调用方向也容易误读 | [后端选择](../../examples/sciaccel_rl/runner.py#L349)存在 `env_worker` 分支。补充 runner 申请租约、Harbor 编排、两种提供者与公共执行层的关系 |
| 历史汇总被当作最新介绍 | [设计简版](../env-worker-deep-dive/experiments/env-design-and-results-brief.md)和[路线图](../env-worker-deep-dive/experiments/env-status-and-roadmap.md)明确只核对到 2026-10-28，导航却推荐它们了解当前状态 | 标题与入口明确标为阶段报告，链接当前设计。保留原始数字、当时规划和未验证项，不把后续结果写回早期实验 |
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
