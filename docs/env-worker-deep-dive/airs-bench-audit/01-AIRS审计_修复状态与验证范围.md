# AIRS审计：修复状态与验证范围

> **核心结论**：AIRS 的主要接线、评分隔离、奖励和生命周期故障已有修复。19 个非 Kaggle 任务具备小样本 GPU 执行与独立评分，SICK 代表题打通策略训练链路。接下来主要补 APPS 全量吞吐与评分兼容性、最终测试协议和控制组件死亡恢复。

文档整理于 2026-11-16，只读既有材料。最近源码与 CPU 回归核对为 **2026-11-13**，基于 `6aa328d56eff746c2ae6d9fd766b2f3f76b4a670` 加当时未提交内容；GPU 与策略闭环实测为 **11-11**。下文 CPU 数字保留该次条件。

- 19 个非 Kaggle 任务身份均有真实数据小样本的 GPU 执行、独立评分和奖励转换证据。Kaggle Web Traffic 暂缓。
- SICK 代表题完成真实策略采样、一次 PSRL 优化器步骤、版本发布与新版本继续交互；逐题 GPU 快检与代表题策略闭环分别统计。
- 当前两批互不重叠的 CPU 回归分别 **155/155、9/9 通过，共 164 项，无失败或跳过**。19 题现存数据边界检查也通过。
- APPS 完整 5000 题的吞吐、评分等价性、严格独立最终测试协议与控制组件死亡恢复仍有明确缺口。

## 目录

- [当前可以确认到哪一步](#当前可以确认到哪一步)
- [执行路径与源码映射](#执行路径与源码映射)
- [跟着图读源码](#跟着图读源码)
- [旧问题逐项修复状态](#旧问题逐项修复状态)
- [用两个例子理解修复](#用两个例子理解修复)
- [验证结果与证据边界](#验证结果与证据边界)
- [尚未完成的能力](#尚未完成的能力)
- [如何快速复核](#如何快速复核)
- [历史证据与交付状态](#历史证据与交付状态)

## 当前可以确认到哪一步

AIRS-Bench 提供机器学习研究任务。agent 读取公开数据，编写训练、检索或预测程序，提交 CSV 表格。独立评分器读取隐藏答案计算成绩，PSRL 将成绩转换为奖励，用交互轨迹训练外层策略语言模型。

<!-- figure:evidence-levels -->
这张图沿同一个场景解释：一次检查通过，究竟能支持多大的结论？

[![验证证据：不同实验分别回答不同问题（1/2）](../../_static/img/document-figures/evidence-levels-screen.png)](../../_static/img/document-figures/evidence-levels.png)

[![验证证据：不同实验分别回答不同问题（2/2）](../../_static/img/document-figures/evidence-levels-screen-02.png)](../../_static/img/document-figures/evidence-levels.png)

同一套调度代码可能通过 CPU 回归，但容器 CUDA、真实评分和训练仍各有条件。 验证是结论与证据的对应关系，不是每次必须爬完的一整套阶梯。 证据类别示意，不把进程存活、历史成功或测试替身写成本次完整训练。 [查看矢量图](../../_static/img/document-figures/evidence-levels.svg)。
<!-- /figure:evidence-levels -->

“任务内训练”例如在 GPU 上训练文本分类器，“PSRL 后训练”则更新负责选择命令的语言模型。两层训练各自记录执行、参数和评分证据。CUDA 是调用 NVIDIA GPU 计算的平台。

| 范围 | 当前确认的能力 | 后续验收目标 |
|---|---|---|
| 19 个非 Kaggle 任务身份 | 每项真实 GPU 基线、提交、独立评分和有限数值奖励 | 按研究需要扩大数据或逐题策略闭环 |
| SICK 文本分类代表题 | 策略调用、验证/提交、有效文本 token、一次优化器步骤、版本 1 后续执行 | 开放研究提示下的自主完成与学习效果 |
| APPS 代码生成 | 五候选 CSV、私有判题器与无答案候选容器分离、有界真实题判分 | 5000 题预算内完成、全量评分行为兼容 |
| 资源生命周期 | 常规取消、命令超时、持有者心跳失联、删除失败保留占用 | 协调组件或执行 worker 死亡后的恢复 |
| 多后端协作 | AIRS 原生协调组件/EnvWorker 单独运行 | 与 Harbor/EnvRouter 混合训练、按命令借还 GPU |

其它 GPU 任务的覆盖统一见[接入现状与路线图](../milestones/02-Science现状_任务覆盖与接入路线.md)。

## 执行路径与源码映射

```text
[1] PSRL MLGymAgentLoop 读取任务，校验配置与资源规格
           |
[2] Coordinator 尝试准入，资源不足由客户端轮询
           | 获准后续期心跳
[3] EnvWorker 创建 agent sandbox，只挂公开数据
           |
[4] 创建策略 Session / TITO，启动独立 MLGym runner 子进程
           | BaseAgent 调模型生成操作，经桥接调用 EnvWorker.exec
           v
[5] GPU 程序产出 submission.csv，validate / submit 请求评分
           | 复制有界 CSV 快照
[6] 独立 verifier 读取隐藏答案，返回指标
           | APPS：judge 只发代码和输入给额外候选容器
           v
[7] 正常轨迹的 token、概率与奖励交给 PSRL 后训练
           |
[8] finally 终止 runner，删除所属容器后归还预算
           ^
    心跳和 watchdog 并发运行，失联走同一回收路径
    删除失败保留占用，不只改账本宣称物理资源已释放
```

图例：图里每个部分对应的源码位置。

| 节点 | 实际函数与职责 | 源码 |
|---|---|---|
| [1] | `MLGymAgentLoop.run`、环境 `reset`，读取任务并校验历史处理器 | [mlgym_agent_loop.py:109](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L109)、[mlgym_env.py:309](../../../psrl/environments/mlgym_env.py#L309) |
| [2] | `acquire_sandbox` 轮询，`reserve` 即时判断 | [coordinator.py:164](../../../psrl/workers/env_worker/coordinator.py#L164)、[coordinator.py:237](../../../psrl/workers/env_worker/coordinator.py#L237) |
| [3] | `build_sandbox_spec` 选择公开挂载，`create_sandbox` 创建容器 | [mlgym_env.py:60](../../../psrl/environments/mlgym_env.py#L60)、[worker.py:224](../../../psrl/workers/env_worker/worker.py#L224) |
| [4] | 先准入后开会话，`run_in_process` 启动子进程 | [mlgym_agent_loop.py:159](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L159)、[process_runner.py:57](../../../examples/airs_bench/process_runner.py#L57) |
| [4] → [5] | `run_mlgym_agent` 接上游 agent，`exec` 执行命令 | [runner.py:137](../../../examples/airs_bench/runner.py#L137)、[worker.py:278](../../../psrl/workers/env_worker/worker.py#L278) |
| [6] | `build_verifier_spec`、`evaluate` 创建私有评分环境 | [grading.py:32](../../../examples/airs_bench/grading.py#L32)、[worker.py:456](../../../psrl/workers/env_worker/worker.py#L456) |
| [6] APPS | `grade` 保留答案，`evaluate_isolated` 转发候选执行 | [judge.py:61](../../../examples/airs_bench/apps/judge.py#L61)、[isolated_code.py:135](../../../psrl/workers/env_worker/isolated_code.py#L135) |
| [7] | `get_training_data` 后构造 token 输出并计算奖励 | [mlgym_agent_loop.py:174](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L174) |
| [8] | `finally`、`release`、`destroy_sandbox` 回收 | [mlgym_agent_loop.py:200](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L200)、[coordinator.py:284](../../../psrl/workers/env_worker/coordinator.py#L284)、[worker.py:431](../../../psrl/workers/env_worker/worker.py#L431) |
| 并发 | `_beat` 续期，`reap_expired` 重试失联或未完成删除 | [coordinator.py:124](../../../psrl/workers/env_worker/coordinator.py#L124)、[coordinator.py:302](../../../psrl/workers/env_worker/coordinator.py#L302) |

Session 是单次策略交互的会话。TITO 是逐轮记录输入、输出 token 和概率的机制，后训练需要这些信息才能把奖励对应到模型操作。sandbox 指任务执行容器，verifier 指独立评分容器，Coordinator 是管理资源准入与占用的协调组件。这些代码如果有理解上的问题，可以随时追问。

## 跟着图读源码

按上表先看 [1]→[3]，理解任务行怎样变成资源申请与公开数据挂载；再看 [4]→[6]，跟随模型命令、提交快照和私有评分；最后看 [7]→[8]，理解哪些轨迹进入训练，以及何时确认删除、归还预算。

同一个 shell 的命令串行处理，心跳与 watchdog（超时监督器）同时运行。命令超时或完成状态不可信时销毁当前沙箱；协调组件与实际 worker 仍可达，才能沿原路径确认清理。这些分支的修复与验证见下表。

## 旧问题逐项修复状态

“已修复”表示修复存在于当前源码，并有对应回归或注明条件的实机证据，不表示消除了同类问题的所有可能原因。

| 原问题 | 当前状态与机制 | 实现 |
|---|---|---|
| agent 直接看到隐藏标签 | 已修复，只挂明确公开 split，拒绝公开 split 本身为软链接 | [build_sandbox_spec](../../../psrl/environments/mlgym_env.py#L60) |
| agent 修改自己的评分脚本或 Python 影响判分 | 已修复，独立 verifier 使用可信代码、私有数据和有界 CSV 快照 | [evaluate](../../../psrl/workers/env_worker/worker.py#L456) |
| APPS 候选代码接触答案 | 已修复，judge 持有答案，candidate 只收到代码与当前输入 | [APPS 隔离说明](02-APPS判题_代码隔离与验证边界.md)、[candidate_spec](../../../psrl/workers/env_worker/isolated_code.py#L116) |
| 一个坏候选中止全部后续评测 | 已修复，非数字字典键不再中止 judge，NumPy 返回值有适配 | [comparison.py](../../../examples/airs_bench/apps/comparison.py#L6)、[judge.grade](../../../examples/airs_bench/apps/judge.py#L61) |
| 评分器故障、NaN 或缺少指标算成坏提交 | 已修复，超时、非 0/2 退出与无效指标算基础设施错误，格式错误单独处理 | [grade_submission](../../../examples/airs_bench/grading.py#L106) |
| 最终缺失或非法提交沿用 validate 高分 | 已修复，失败终态清空奖励成绩副本，保留原始历史供诊断 | [collect_scores_from_info](../../../examples/airs_bench/runner.py#L83) |
| 超时串线、持续输出或 stdin 阻塞 | 已修复，独立命令标识、非阻塞 I/O、截止时间、8 MiB 输出上限，超时销毁 shell | [ContainerShell](../../../psrl/workers/env_worker/worker.py#L32)、[exec](../../../psrl/workers/env_worker/worker.py#L278) |
| runner 取消后仍后台执行 | 已修复，独立进程取消时 terminate、join，必要时 kill | [run_in_process](../../../examples/airs_bench/process_runner.py#L57) |
| 排队耗尽 actor 远程调用并发，阻塞 release | 已修复，准入立即判断，客户端等待，获准后才创建会话 | [reserve](../../../psrl/workers/env_worker/coordinator.py#L237) |
| 持有者退出不回收，删除失败仍还预算 | 常规路径已修复，心跳失联回收、删除确认后返还、失败保留占用并有界重试 | [release/reap_expired](../../../psrl/workers/env_worker/coordinator.py#L284) |
| MLGym 参数、任务 schema、模型 API 不匹配 | 已修复，按上游接口构造任务与模型配置，真实本地 HTTP 路由回归通过 | [prepare_mlgym_task](../../../examples/airs_bench/runner.py#L118)、[build_mlgym_model_config](../../../examples/airs_bench/runner.py#L33) |
| 历史裁剪破坏 TITO 对齐 | 有前置防护，要求 `DefaultHistoryProcessor` 保持原始历史 | [assert_default_history_processor](../../../psrl/environments/mlgym_env.py#L31) |
| 公开检索查询与隐藏行错位 | 已修复，公开列从同批选定隐藏行投影得到，不只比较行数 | [GPU 数据准备器](../../../examples/airs_bench/prepare/prepare_gpu_suite.py) |

评分退出码 2 的分类依赖可信评分脚本遵守格式错误约定，不能诊断所有脚本异常。超时销毁可避免复用污染状态，但会丢失当前 episode 工作，不是原地恢复功能。

## 用两个例子理解修复

**例一：不能用旧成绩奖励失败提交。** 假设分类任务第一次 `validate` 得 0.8，agent 随后覆盖坏 CSV，最终 `submit` 为 `invalid_submission`。直接取历史最后一个成绩会错误地用 0.8 奖励失败轨迹。现在失败终态会清空奖励使用的成绩副本。0.8 是教学假设，这个分支由真实上游 agent 回归覆盖。

**例二：APPS 必须继续尝试剩余候选。** 一题要求返回数字 1，第一个候选返回 `{"not_an_integer": 1}`，第二个返回正确的 1。第一个是错误答案，不应使评分器崩溃。当前比较器保留不能转为数字的字典键，judge 判错后继续第二个。历史真实 Docker 观察到这类“先错后对”的 Pass@5 为 1。这里 Pass@5 是每题最多五份候选中，一份通过该题全部选定测试就算成功，再计算成功题目的比例。

隐藏答案留在 judge，候选只接收输入。隐藏文件隔离已修复，但 validate/submit 仍反馈同一隐藏集合的指标，这是两项不同的能力。

## 验证结果与证据边界

| 证据层级 | 结果 | 覆盖与限制 |
|---|---|---|
| 2026-11-13 CPU 回归 | 155 项一般回归与 9 项真实上游回归分别全部通过，见[当前回归记录](cpu-regression-20261113.json) | 隔离、异常分类、奖励、真实 Bash/子进程、训练张量往返。上游 BaseAgent 和本地 HTTP 路由真实执行，容器、评分器或远程服务有替身，不是本次 GPU 重跑 |
| 2026-11-13 数据边界检查 | 19 项通过公开列、隐藏列排除和提交行数检查 | 只检查现存资产结构，不执行任务，不是容器逃逸审计 |
| 历史逐任务 GPU 快检 | 19/19，通过，合计 365.68 秒 | 每项真实 CUDA、提交、独立评分和有限奖励。逐项 submit=1、validate=0，没有逐项语言模型后训练 |
| 历史 SICK 策略闭环 | 672 有效生成 token，1 次优化器步骤，版本 1 后续 rollout | 真实 Qwen3.5-4B，有界验收提示下成功。同组奖励相同，loss、grad_norm 为 0，不证明学习收益 |
| 历史正式镜像 APPS 对照 | 4 题各前 3 个输入，共 12 输入判定一致，固定动作 agent 完成 validate/submit | 有限样本与真实容器，不是 5000 题或真实策略训练 |
| 2026-11-13 源码指纹 | GPU 记录保存的 14 个 AIRS 关键文件 SHA-256 全部匹配 | 不覆盖全部底层 EnvWorker、外部依赖及部署状态 |

GPU 快检通常至多 64 条公开训练、8 条测试。APPS 为 2 题、每题前 2 个隐藏案例、5 候选。代码检索为 8 查询与全部 19210 条公开候选代码。Rideshare 的 8 条输入展开成 120 行预测。SICK 的 8 条测试答对 6 条，准确率 `6/8=0.75`，归一化奖励约 0.5596034。

准确率低、Pass@5 为零或误差较大，若真实执行、评分和奖励合法，仍可通过功能验收。高分也不能替代资源清理或策略训练证据。SICK 普通研究提示曾缺少提交或超时，成功的三步验收提示不保证任意开放研究都能自主完成。

证据：[GPU 与策略记录](../experiments/gpu-benchmarks/airs-gpu-validation-20261111.json)、[当前指纹复核](current-source-review.json)。本次 164 项是两批互不重叠的运行，不与历史 103、145 或 123 项相加。

## 尚未完成的能力

| 问题 | 原因与影响 | 难度 |
|---|---|---|
| APPS 完整吞吐 | 按题与候选串行，每候选新容器。5000 题最多 25000 次生命周期，缺省评分期限 3600 秒，未全量验收 | 中高，需有界并行、资源预算与全量验收，不只加大超时 |
| APPS 全量评分等价及 Python 契约 | 自有比较器与限制，历史正式镜像 Python 3.11.15、任务文字 3.10，仅有有限题目对照 | 中等，固定解释器契约并扩大行为对照 |
| 严格独立最终测试 | validate/submit 共用隐藏集合并返回指标，允许反复反馈间接试探测试集，用户此前明确要求暂缓修改 | 协议改动中等，需决定验证集/最终集合、次数和奖励语义，当前不改规则 |
| 控制面死亡恢复 | reservation/watchdog 在内存，回收依赖原 worker，组件或节点死亡无独立持久核对 | 高，需持久身份、节点回收和重启后对账 |
| 命令级 GPU 借还与混合后端 | AIRS 按整个 episode 持有 Ray GPU 槽位，未走 Harbor/EnvRouter command 路径 | 中高，需统一生命周期后实测混训，不能互换 YAML 代替 |
| 全任务逐题策略闭环、长期并发 | 逐题 GPU 快检与 SICK 代表闭环是两层证据，未测每题后训练和长期稳定性 | 工作量较大，应按研究需求有界验证 |
| Kaggle Web Traffic | 凭据与竞赛授权待用户提供 | 授权后中等，仍需准备与评分验证 |

3600 秒来自 [per_action_timeout_s](../../../examples/airs_bench/config.py#L52)，由 [grader 回调](../../../examples/airs_bench/runner.py#L182)传给独立评分。`3600/25000=0.144` 秒只是最坏规模下每候选可用预算的算术，不是全量实测速度。

## 如何快速复核

从 PSRL 仓库根目录执行。先激活项目环境，在 source 后恢复本仓库的 `PSRL_WORKSPACE`，固定 `PYTHONPATH`、现有 `MLGYM_REPO` 和隔离依赖路径，不要向共享环境安装包。详细依赖规则见[运行指南](../../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md#集群踩坑总表与最小复现2026-10-13)。

标准库 CPU 实验无需模型、下载、Docker 或 GPU：

```bash
python docs/env-worker-deep-dive/airs-bench-audit/apps-pass5-demo.py
python docs/env-worker-deep-dive/airs-bench-audit/apps-isolation/review-error-classification.py
python -m examples.airs_bench.prepare.audit_gpu_suite \
  --data ../benchmark-runs/airs-gpu-20261111/bounded
```

第一个是**教学模拟**，两题各五个固定函数，只有求和题第三候选通过全部输入，输出 Pass@5=0.5，不执行不可信代码或模拟容器边界。第二个运行当前真实 judge 和比较器，仅把候选传输替换为固定错误输出，观察五次 begin/case/end 后正常返回零分，而不是第一份坏结果就崩溃。第三个检查现有 19 题公开列边界，不读取隐藏标签值。三者均于2026-11-13实际运行通过；本次文档整理只读取这些记录。

对应图中，第一个模拟 [6] 的汇总，第二个检查 [6] 的异常处理，第三个检查 [1]/[3] 的数据结构。完整生命周期使用以下现有回归，替身与真实进程的差别见测试说明：

```bash
python -m pytest -q \
  tests/examples/airs_bench \
  tests/env_worker/test_airs_regressions.py \
  tests/env_worker/test_shell.py \
  tests/env_worker/test_coordinator.py \
  tests/env_worker/test_worker.py \
  tests/env_worker/test_sandbox.py \
  tests/env_worker/test_manager_placement.py \
  tests/env_service/test_position_ids_roundtrip.py
```

确认 `test_upstream_mlgym.py` 没有因缺少上游而跳过，并核对 `psrl`、`verl`、`transfer_queue` 的导入来源。最后一项需现有隔离 veRL 张量补丁副本，覆盖单条、等长、变长 token 坐标存储往返，不需要 GPU。本次为明确区分真实上游覆盖，将其拆成 155 项与 9 项两个运行，原始命令、来源与日志保存在回归 JSON。

## 历史证据与交付状态

历史实验保留原条件，不把旧测试数相加为一次新运行：

| 资料 | 阅读用途 |
|---|---|
| [baseline/](baseline)、[results.json](results.json)、[baseline-replay.json](baseline-replay.json)、[reproduce.py](reproduce.py) | 修复前故障快照与重放，不能加入当前训练 PYTHONPATH |
| [after-results.json](after-results.json) | 早期 CPU 汇总，保留整组与补充定向回归的区别 |
| [remote-docker-results.json](remote-docker-results.json) | 历史真实容器的标签不可见、独立判分、超时销毁和清理 |
| [APPS 隔离说明](02-APPS判题_代码隔离与验证边界.md)、[正式镜像四题结果](apps-followup/remote-b.json) | 真实容器隔离与有限题目对照 |
| [review-round-two/summary.json](review-round-two/summary.json) | 历史 145 项 CPU 回归，不等于当前重测 |
| [AIRS GPU recipe](../knowledge/03-任务接入/03-AIRSBench_接入流程与有界验收.md) | 当前数据准备、GPU 入口与代表闭环配置 |

2026-11-13核对没有安装依赖或修改共享Python包。该次conda激活未在预算内完成，终止所属进程组后，使用现有项目解释器和显式隔离依赖检查。该次没有新建GPU容器或Ray集群。

该次十份文档的本地路径、章节锚点与源码行号检查通过。全库pre-commit在一次性源码快照中运行，仍有范围外既有长行、格式与prose违规，自动修改未写回主目录，见[11月13日静态检查记录](../../../../benchmark-runs/airs-status-review-20261113/precommit.json)。

证据对应11月13日核对的主目录源码与当时未提交内容。11月16日只整理文档；文档日期不表示代码已经提交、推送、合并或更新部署。
