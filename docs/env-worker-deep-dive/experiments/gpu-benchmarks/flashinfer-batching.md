# FlashInfer 评估实验：先复用进程，再看筛选与分段

> **核心结论**：这个短评估任务的大头成本在进程与准备工作。四项评估共用一个进程，整组均值从 **125.55 秒降到 25.86 秒**，而资源准入与归还只有几十毫秒。之后的候选筛选没有超过简单固定预算；分段评估可行，但首段之后只剩约 **0.33 秒**计算，当前子集适合优先完善复用。

本文合并讲清三次相关实验：2026-11-09 的准备成本、同日的候选筛选，以及 11-11 的评估器分段。它们各有固定协议与原始证据。2026-11-16 整理本文只读已有结果，各次代码版本和范围见最后一节。

## 目录

- [先看懂实验具体在做什么](#先看懂实验具体在做什么)
- [Results：先省掉重复准备](#results先省掉重复准备)
- [候选筛选后续实验](#候选筛选后续实验)
- [保留评估器分段的 Results](#保留评估器分段的-results)
- [研究判断与边界](#研究判断与边界)
- [执行链路与复现](#执行链路与复现)
- [实验条件与证据](#实验条件与证据)

## 先看懂实验具体在做什么

FlashInfer-Bench 给出 GPU 算子、输入和评分程序，任务是提交完成相同计算的实现，再比较正确性与耗时。本实验用 **RMSNorm**，即按一行数值的均方根调整幅度再乘权重的归一化运算。每行有 2048 个数，两种输入分别为 1 行和 16254 行。

**候选**是同一个计算的不同程序实现；**workload** 是一份具体输入规格。候选写好后，评分程序还要启动 Python、导入 GPU 库、准备输入、编译、检查结果和计时。算子本体可能只需微秒，整次准备却可能持续数十秒。因此要先分清哪个环节最值得优化。

三次实验回答三个顺序相接的问题：

| 问题 | 改什么 | 固定什么 |
|---|---|---|
| 重复准备多贵 | 每项新进程、合并命令、单进程批量 | 相同四个输入与候选组合 |
| 测几轮才能选出赢家 | 固定四轮、固定八轮、先筛选再追加 | 相同候选和完整实测数据，离线回放决策 |
| 评估能否分段 | 一次运行全部输入，或同一评估器分两次 | 相同输入与候选、正确性和计时配置 |

GPU **command** 是一段受资源管理的执行区间；合并 command 可以减少申请次数，保留进程还可以复用解释器和评估器状态。二者省掉的工作不同。

<a id="results"></a>

## Results：先省掉重复准备

第一项实验固定两个输入和两个候选，共四项评估。候选为官方 FlashInfer 实现与未修改数学参考的本地封装。三种方式各执行两轮，顺序为 A、B、C、C、B、A，第二轮反向候选次序。

| 方式 | GPU command 数 / Python 主进程数 | 首轮整组 | 次轮整组 | 两轮均值 |
|---|---:|---:|---:|---:|
| 每项分别执行 `separate` | 4 / 4 | 147.58 秒 | 103.51 秒 | **125.55 秒** |
| 合并 command、逐项新进程 `one_command_fresh` | 1 / 4 | 107.83 秒 | 96.64 秒 | **102.23 秒** |
| 同进程批量 `one_process` | 1 / 1 | 24.04 秒 | 27.68 秒 | **25.86 秒** |

**整组时间从第一条命令提交开始，到最后一条返回且资源归还结束**，包含导入、初始化、评估和进程退出。它说明完成这一组评估要等多久。

单进程批量约为分别执行时间的 `1/4.86`，约为仅合并 command 的 `1/3.95`。资源准入与归还均值分别只有 74.45、19.50、19.72 毫秒，无法解释数十秒差异。主要机会在复用进程、GPU 初始化与官方评估器准备；各项独立贡献未完全分离。

一个具体读法：四个评估结果仍全部产生，改变的是它们是否重复启动软件。比较“125.55 秒到25.86秒”是在比较整组工作等待，算子本体速度需看逐项微秒计时。

全部六组、24 项正确性通过，每组结束时命令与显存申报归零，最终所属容器为 0、GPU 无计算进程。数据见[批量证据](flashinfer-batching-evidence.json)。首轮明显更慢，说明冷启动和运行状态参与了差异，两轮样本仅支持当前机制判断。

## 候选筛选后续实验

第二项实验沿用两个输入，候选换为官方 FlashInfer baseline 以及 Gemini 2.5 Pro、GPT-5、o3 生成并由上游发布的三个 Triton 实现。Triton 是编写 GPU 计算程序的语言与工具；这些名称标识候选代码来源，实测中直接执行固定程序。

先预热一轮，再正式测 12 轮，每轮两输入 × 四候选，合计 `(1+12)×8=104` 次。候选与输入顺序轮换，全部结果都在实机采集，之后用前八轮离线比较决策，最后四轮留作核对。

| 选择方法 | 每输入用于决策的评估数 | 与留出四轮的赢家一致 |
|---|---:|---|
| 全部候选固定八轮 | 32 | 两输入均一致 |
| 前四轮留下最快两个，再加四轮 | 24 | 两输入均一致 |
| 全部候选固定四轮 | **16** | 两输入均一致 |

筛选比固定八轮少用 25% 观测，但固定四轮少用 50% 且得到同样赢家。**这组数据优先支持简单固定预算。** 表中节省的是离线决策使用的观测数，实机实际完成全部 104 次。

| 候选 | 1 行输入，12 轮中位数 | 16254 行输入，12 轮中位数 |
|---|---:|---:|
| FlashInfer | 5.619 微秒 | 44.104 微秒 |
| Gemini Triton | 5.811 微秒 | 51.707 微秒 |
| GPT-5 Triton | 12.104 微秒 | 100.765 微秒 |
| o3 Triton | 12.299 微秒 | 176.301 微秒 |

中位数是排序后的中间值，减轻个别慢样本的影响。FlashInfer 在两个输入的全部 12 轮均第一。1 行输入时，GPT-5 与 o3 先后次序各出现六次，并与测试历史相关：先测大输入或小输入时次序不同。候选轮换也随轮次变化，温度、频率等未分别控制，原因仍待定位。

如果只选第一名，这两名落后候选互换没有改变选择；若需要完整排名或所有候选的数值奖励，就需要另外检查这些变化。完整反馈与选优是两类目标。

包含初始化和清理总计约 **102.09 秒**，采集 command 约 65.19 秒，首次预热 22.87 秒，后续 12 轮共 4.61 秒，约占 command 的 7.1%。这个短任务中，减少正式轮数可节省的总时间有限。全部 104 次正确性通过并清理，见[筛选证据](flashinfer-screening-evidence.json)。

## 保留评估器分段的 Results

第三项实验检查：两种必做输入能否分两段评估，复用同一个评估器并保留完整结果。`whole` 一次 `run_all` 处理两种输入；`split` 用同一个 Benchmark 和内部 runner 分两次处理。每次外层 Python 都重新启动。

四次顺序为 whole、split、split、whole，两输入各比较 FlashInfer 与数学参考，共 **16 项结果全部正确**。

| 顺序 | 方式 | 脚本总时间 | 第一输入内部时间 | 第二输入内部时间 | 含 command 返回 |
|---|---|---:|---:|---:|---:|
| 1 | whole | 32.10 秒 | 5.83 秒 | 0.385 秒 | 33.39 秒 |
| 2 | split | 34.61 秒 | 5.55 秒 | 0.334 秒 | 35.76 秒 |
| 3 | split | 29.48 秒 | 4.91 秒 | 0.331 秒 | 30.60 秒 |
| 4 | whole | 34.09 秒 | 4.36 秒 | 0.330 秒 | 35.01 秒 |

脚本总时间从导入依赖前到关闭评估器后，含初始化与退出。均值为 whole **33.09 秒**、split **32.05 秒**，重复间波动大于均值差，应作为可行性结果阅读。

**首项之后只剩约 0.33 秒输入工作。** 首项包括首次执行成本，两种形状也不同，所以首项耗时无法乘剩余项数来预测总量。当前子集尾部很短，新排队、交接或准备容易吃掉重排空间。

split 改的是输入列表分组，使用私有 `_runner`、`_trace_set` 接口，尚未形成稳定生产接口。实验一直持有 GPU，分段处保留评估器与 GPU 状态；安全交还、迁移和插入竞争任务属于下一阶段能力。全部流程约 **173.5 秒**，在 360 秒主体预算内结束，清理通过，见[分段证据](flashinfer-split-evidence.json)。

## 研究判断与边界

三项结果共同指向一条主线：**先建立进程与准备复用这个基线，再寻找额外调度空间。** 当前筛选没有胜过固定四轮；当前分段尾部太短；单纯少申请几次资源也解释不了主要时间变化。

下一步可以选择含多个较长必做单元的任务，分别测准备、稳定执行和状态持有成本。需要同时保持相同评估质量与工作量，才能判断复杂策略是否值得。关于候选比较与竞争干扰，另有[性能反馈压力实验](feedback-pressure.md)；相关研究背景见 [KREX 阅读](../../literature-reviews/krex-region-exclusivity.md)和[command 调度提案](../../research-proposals/command-scheduling-research.md)。

本页固定候选，比较当前真实 Env 上的执行方式，研究范围为评估过程。完整 agent 任务、策略训练、并发干扰和线上调度收益各自另有验收目标。

## 执行链路与复现

```text
准备固定输入与候选 → EnvRouter 准入 → 原生 EnvWorker 容器
                                           ↓
                              command 内调用官方评估器
                                           ↓
                              保存完整结果 → 清理与物理核验
```

| 环节 | 真实入口与职责 |
|---|---|
| 批量任务准备 | [prepare_flashinfer_batching](../../env-service-verification/scripts/prepare_flashinfer_batching.py#L12)筛选两个输入，封装数学参考 |
| Env 与 GPU 预检 | [flashinfer_env_probe.run](../../env-service-verification/scripts/flashinfer_env_probe.py#L22)建立私有 Ray/Router、原生容器并检查 CUDA |
| 批量驱动与评分 | [batch_experiment](../../env-service-verification/scripts/flashinfer_batch_experiment.py#L14)、[batch_worker](../../env-service-verification/scripts/flashinfer_batch_worker.py#L9)运行三方式并核对四项结果 |
| 筛选准备与采集 | [prepare_racing](../../env-service-verification/scripts/prepare_flashinfer_racing.py)、[racing_probe](../../env-service-verification/scripts/flashinfer_racing_probe.py)固定候选并保存全部轮次 |
| 筛选分析 | [analyze_racing](../../env-service-verification/scripts/analyze_flashinfer_racing.py)离线回放固定预算与筛选选择 |
| 分段 | [split_experiment](../../env-service-verification/scripts/flashinfer_split_experiment.py)、[split_worker](../../env-service-verification/scripts/flashinfer_split_worker.py)改变分组并核对结果完整性 |

运行前先按[集群预检](../../../psrl/psrl-run-guide/cluster-preflight.md)检查实际节点与依赖。`FIB_SOURCE`、`FIB_TRACE` 为固定上游源码与 trace，`FIB_OUTPUT` 为新输出目录，设备、节点和镜像也使用现场值。第一项入口示例：

```bash
FIB_SCRIPTS=docs/env-worker-deep-dive/env-service-verification/scripts
python "$FIB_SCRIPTS/prepare_flashinfer_batching.py" \
  --upstream "$FIB_SOURCE" --trace "$FIB_TRACE" --output "$FIB_OUTPUT"
PYTHONPATH="$PWD" python "$FIB_SCRIPTS/flashinfer_batch_experiment.py" \
  --root "$FIB_OUTPUT" --upstream "$FIB_OUTPUT/upstream" \
  --node-ip "$FIB_NODE" --gpu "$FIB_GPU" --image "$FIB_IMAGE"
```

准备步骤写入筛选后的资产；第二步通过真实 Env 执行并保存 `groups.json`、`status.json`。筛选实验使用 `prepare_flashinfer_racing.py` 和 `flashinfer_racing_probe.py` 的同类参数；完成后可只用 CPU 分析：

```bash
python "$FIB_SCRIPTS/analyze_flashinfer_racing.py" \
  --root "$FIB_OUTPUT" --output "$FIB_OUTPUT/analysis.json"
```

## 实验条件与证据

| 实验 | 固定版本与预算 | 条件与核验 |
|---|---|---|
| 准备成本 | `0326d46` 加冻结运行代码；主体 900 秒、监督 1200 秒 | 空闲 H20、4 CPU、16 GiB；六组全完成 |
| 候选筛选 | `bc5665f` 加冻结运行代码；command 420 秒、Env 600 秒、监督 900 秒 | 同类设备与配额；104 次全通过 |
| 评估器分段 | 启动 HEAD `ab74120`，整理时 `6aa328d`；主体 360 秒 | 实际 worker 哈希、镜像与依赖已记录，整个工作区未完整冻结 |

前两项 unit 按整卡容量，单任务申报 8 GiB；资源声明为准入预算。官方容差、随机输入与内部预热计时配置保留；缺少 CUPTI 性能分析接口时，三项均使用官方 CUDA Event 回退计时，即 GPU 起止时间标记。筛选候选原声明 B200，本次在 H20 检查所选输入的运行与正确性。

首次工具启动在准备脚本落盘前失败，当次尚无 GPU 工作；正式数据与失败分别保留。各次目标任务完成后停止，所属容器删除，GPU 无计算进程，收尾约 0.19 MiB。历史定向检查通过，全仓 pre-commit 仍有范围外格式问题。整理报告期间没有重跑实验。
