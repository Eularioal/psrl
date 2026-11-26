# Env 阶段成果与 benchmark 导览

> **核心结论**：这里集中查阅 bench 介绍、任务代码、题库位置和接入方式。了解任务先读 [Science GPU 简版](01-Science介绍_已接入任务与研究价值.md)，找代码直接看下表，需要覆盖数量与验证结果再读[路线图](02-Science现状_任务覆盖与接入路线.md)。

研究全景及交流版于 2026-11-24 同步当前分区结果和后续优先级，任务题库条目保留各自核对日期。本页核对主目录的目录、任务准备器和训练启动入口，并关联本轮新增 GPU 验证。接入代码与分组修复已在共享主目录提交 `8755e20`、`30997ea`，后续修改仍含未提交内容；具体版本与各题库验收范围见路线图和固定证据。

回顾整个 Env，先从[专题首页](../README.md)选择[架构现状](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)、[实验结论](../experiments/01-实验总览_调度收益与证据范围.md)或[下一步优先级](05-调度研究_研究全景与证据详解.md#接下来可以做什么优先级与验收边界)。本目录的“Science现状”只负责任务覆盖，10月阶段报告只供历史追溯。

<!-- figure:map-e3e99949ff -->
先按当前问题选择阅读入口，再在下方目录点击对应文档。

[![Env 阶段成果与 benchmark 导览：阅读地图（1/2）](../../_static/img/document-figures/map-e3e99949ff-screen.png)](../../_static/img/document-figures/map-e3e99949ff.png)

[![Env 阶段成果与 benchmark 导览：阅读地图（2/2）](../../_static/img/document-figures/map-e3e99949ff-screen-02.png)](../../_static/img/document-figures/map-e3e99949ff.png)

每张彩色书页列出对应入口中的实际章节，帮助按问题定位阅读内容。编号只用于图中定位，不要求顺序通读。 [查看矢量图](../../_static/img/document-figures/map-e3e99949ff.svg)。
<!-- /figure:map-e3e99949ff -->

## 目录

- [阶段报告](#阶段报告)
- [各 bench 的任务与代码入口](#各-bench-的任务与代码入口)
- [上游源码与已准备题库在哪](#上游源码与已准备题库在哪)
- [两条接入方式与查看示例](#两条接入方式与查看示例)
- [阅读与维护](#阅读与维护)

## 阶段报告

| 阶段 | 报告 | 内容 |
|---|---|---|
| 当前研究与下一步 | [Env 调度探索 Big Map](05-调度研究_研究全景与证据详解.md) | 先读优先级与验收边界；2026-11-26 补充[论文证据缺口与自适应持卡](05-调度研究_研究全景与证据详解.md#论文还缺什么自适应持卡的研究切口)，性能由实验总览维护 |
| 交流精简版 | [Env 调度研究 Big Map](04-调度研究_研究全景交流版.md) | 用于交流的整体结构、证据边界与当前研究顺序 |
| 先读介绍 | [Science介绍：已接入任务与研究价值](01-Science介绍_已接入任务与研究价值.md) | 从数字分类的完整例子理解目标、任务类型、已有结果与研究方向 |
| 查看任务覆盖 | [Science现状：任务覆盖与接入路线](02-Science现状_任务覆盖与接入路线.md) | 题库数量、GPU 与策略闭环覆盖、[多样性与场景缺口](02-Science现状_任务覆盖与接入路线.md#多样性覆盖与缺口)、剩余接入工作及研究推进顺序 |
| 理解接入 | [Science GPU 接入与验收](03-Science验收_接入流程与环境隔离.md) | 解释执行与训练流程、环境隔离和模型/代码交付，保留 2026-11-10 六类代表题验收记录 |
| 2026-10-28 | [Env 实现、实测收益与后续规划](06-历史阶段_2026年10月28日实现与规划.md) | EnvRouter 阶段架构、固定 baseline、压力实验收益与自动策略规划，按历史版本阅读 |

## 各 bench 的任务与代码入口

**接入代码放在仓库的 `examples/`，已经准备好的题库放在运行资产目录**，两者分开查找。下表同时列出任务内容、准备入口与题库位置。

| Bench | agent 具体做什么、交付什么 | 接入代码目录 | 任务准备入口 |
|---|---|---|---|
| KernelBench | 将 PyTorch 运算改写成 GPU 实现，例如矩阵乘法，提交代码供正确性和计时检查 | [examples/kernel_bench/](../../../examples/kernel_bench) | [prepare_suite.py](../../../examples/kernel_bench/prepare_suite.py) 扫描多题，[prepare.py](../../../examples/kernel_bench/prepare.py) 准备单题 |
| FlashInfer-Bench | 实现推理算子，例如归一化或注意力，提交官方格式的 Solution JSON，包含实现代码 | [examples/flashinfer_bench/](../../../examples/flashinfer_bench) | [prepare_suite.py](../../../examples/flashinfer_bench/prepare_suite.py) 按算子定义准备 |
| RE-Bench | 完成偏好训练、算子、模型修复等研究工程题，按题交代码、权重或预测 | [examples/re_bench/](../../../examples/re_bench) | [prepare.py](../../../examples/re_bench/prepare.py) 聊天训练；其他子题入口见 [RE recipe](../knowledge/03-任务接入/05-REBench_聊天模型训练接入.md) |
| InferenceBench | 启动并改进语言模型推理服务，提交服务方案，检查输出与请求性能 | [examples/inference_bench/](../../../examples/inference_bench) | [prepare.py](../../../examples/inference_bench/prepare.py) 导入上游任务，[scenarios.py](../../../examples/inference_bench/scenarios.py) 准备本地服务场景 |
| MLE 本地任务与 MLE-bench | 用训练数据学习分类器，预测测试集并交 CSV；本地含手写数字等数据，官方竞赛数据待 Kaggle 授权 | [examples/mle_bench/](../../../examples/mle_bench) | [local.py](../../../examples/mle_bench/local.py) 准备本地题，[prepare.py](../../../examples/mle_bench/prepare.py) 接官方已准备竞赛 |
| autoresearch | 修改训练程序，做有预算的实验，交训练代码供独立重跑与评分 | [examples/autoresearch/](../../../examples/autoresearch) | [prepare.py](../../../examples/autoresearch/prepare.py) |
| AIRS-Bench | 做文本分类、代码生成、检索、分子或时间序列预测，交预测 CSV | [examples/airs_bench/](../../../examples/airs_bench) | [prepare_gpu_suite.py](../../../examples/airs_bench/prepare/prepare_gpu_suite.py) 准备非 Kaggle GPU 任务；[专用说明](../knowledge/03-任务接入/03-AIRSBench_接入流程与有界验收.md) |
| MLGym | 训练游戏策略、图像或文本模型，交权重供独立加载评分 | [examples/mlgym_bench/](../../../examples/mlgym_bench) | [prepare.py](../../../examples/mlgym_bench/prepare.py) 强化学习、[prepare_vision.py](../../../examples/mlgym_bench/prepare_vision.py) 视觉、[prepare_nlp.py](../../../examples/mlgym_bench/prepare_nlp.py) 文本 |
| PostTrainBench | 继续训练给定语言模型以适应数学、代码、写作或健康问答，本地协议提交更新的权重 | [examples/posttrain_bench/](../../../examples/posttrain_bench) | [prepare.py](../../../examples/posttrain_bench/prepare.py)，[judged_model.py](../../../examples/posttrain_bench/judged_model.py) 实际生成与独立裁判评分 |
| FastKernels | 优化真实模型里的算子、模块或整模型，保持接口和数值正确，交实现代码 | [examples/fastkernels_bench/](../../../examples/fastkernels_bench) | [capture_local.py](../../../examples/fastkernels_bench/capture_local.py)、[capture_cases.py](../../../examples/fastkernels_bench/capture_cases.py) 捕获输入，[prepare.py](../../../examples/fastkernels_bench/prepare.py) 准备选定目标 |
| SWE-Serve | 修改真实推理服务源码，交代码修改并运行独立测试；本地补丁子集会重放提交的补丁 | [examples/swe_serve/](../../../examples/swe_serve) | [prepare.py](../../../examples/swe_serve/prepare.py)；[接入说明](../knowledge/03-任务接入/04-MLGym与SWE及Serve_任务协议与评分训练.md) 区分导入题与已验证协议 |
| SciAccel 原有 science 入口 | 修复科学计算代码并运行测试，具体题目决定是否需要 GPU | [examples/sciaccel_rl/](../../../examples/sciaccel_rl) | [准备说明](../../../examples/sciaccel_rl/prepare/README.md) 与[运行 README](../../../examples/sciaccel_rl/README.md) |

PostTrain、FastKernels、SWE 的完成范围分别统计。选择可用题目时，以[路线图状态表](02-Science现状_任务覆盖与接入路线.md#目前有哪些任务及可用边界)为准。AIRS 与 MLGym 自身题库也分开统计，AIRS 只是复用了 MLGym 驱动 agent 的框架。

## 上游源码与已准备题库在哪

下面路径以 PSRL 仓库根目录为基准，`../` 表示当前工作区的上一级目录。

| 目录 | 内容与用途 |
|---|---|
| [examples/](../../../examples) | PSRL 接入代码，含任务准备器、agent 配置、评分适配和启动脚本 |
| [../benchmark-sources/](../../../../benchmark-sources) | 多数 bench 的上游源码。实际来源由准备命令的 `--repo` 指定，AIRS/MLGym 还使用显式配置的外部源码路径 |
| [../benchmark-assets/](../../../../benchmark-assets) | 下载的数据、模型权重和独立依赖环境 |
| [../benchmark-runs/](../../../../benchmark-runs) | 已生成的任务、训练数据表、实验输出及证据。名字带 runs 的目录也保存可复用题库 |

优先复用已有题库，从以下入口定位：

- 公共六类：看 [manifest.json](../../../../benchmark-runs/complete-library-20261111/published/manifest.json) 的任务清单，以及配套的 [tasks.parquet](../../../../benchmark-runs/complete-library-20261111/published/dataset/tasks.parquet) 和 [agent.yaml](../../../../benchmark-runs/complete-library-20261111/published/dataset/agent.yaml)。
- AIRS：看 [manifest.json](../../../../benchmark-runs/airs-gpu-20261111/bounded/manifest.json)、[tasks.parquet](../../../../benchmark-runs/airs-gpu-20261111/bounded/tasks.parquet) 与 [GPU agent 配置](../../../examples/airs_bench/config/gpu_agent_config.yaml)。
- MLGym、PostTrain、FastKernels、SWE：看路线图的[题库位置表](02-Science现状_任务覆盖与接入路线.md#题库数量位置与逐题覆盖)，按题库进入固定产物或逐题审计记录。

`manifest.json` 是可直接阅读的清单，通常列出任务身份或实际目录。Parquet 是 PSRL 加载的表格数据格式，每行描述一项任务。数据表与对应 agent 配置要配套使用。

## 两条接入方式与查看示例

**公共 GPU 路径**用于表中 AIRS 以外的 GPU 扩展。各准备器把任务指令、环境、独立测试和资源需求封装为任务目录，再生成 PSRL 数据表。[公共 train.sh](../../../examples/gpu_benchmarks/train.sh#L5) 根据任务配置名，如 `kernel_bench`，选择 `examples/<recipe>/agent_config.yaml`，通过 Harbor 驱动任务、EnvRouter 管理资源准入。准备器共用的[write_dataset](../../../examples/gpu_benchmarks/prepare.py#L47) 将任务目录和 GPU 预算写入数据行。具体配置与运行方法见[公共接入说明](../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)。SciAccel 原有科学修复题按自己的 README 使用独立配置与启动入口。

**AIRS 原生 sandbox 路径**使用 [gpu_train.sh](../../../examples/airs_bench/gpu_train.sh#L14) 和 [gpu_agent_config.yaml](../../../examples/airs_bench/config/gpu_agent_config.yaml)，选择 `MLGymAgentLoop`，通过 Coordinator/EnvWorker 创建 GPU 任务容器和独立评分环境。它有自己的数据字段与启动配置，详见[AIRS 接入说明](../knowledge/03-任务接入/03-AIRSBench_接入流程与有界验收.md)。Harbor 是任务执行编排框架，EnvRouter 和 Coordinator 分别负责各自路径的资源准入，sandbox 指任务执行沙箱。

**以查找 KernelBench 矩阵乘法题为例**：先在上表找到 `kernel_bench/`，读 [prepare_suite.py 的 discover](../../../examples/kernel_bench/prepare_suite.py#L13)，了解它怎样按 level，即题目分级，扫描上游题目。接着打开已有公共题库清单，找到 `level1-1_Square_matrix_multiplication_` 的 `task_path`，进入[该题实际目录](../../../../benchmark-runs/quick-verified-library-20261111/tasks/kernel_bench/level1-1_Square_matrix_multiplication_)，查看任务指令、环境与测试。接入的数据表把这个任务交给公共 GPU 路径，策略操作任务、提交实现、接收独立评分，随后交互记录进入 PSRL 训练。完整执行与训练过程已有[流程说明](03-Science验收_接入流程与环境隔离.md#验收目标与执行链)。这里只导航到已有任务，无须重新下载或执行实验。

## 阅读与维护

阶段成果与 roadmap 统一放在本目录。详细实验结果放在[实验目录](../experiments)，组件原理与当前设计放在[知识目录](../knowledge/README.md)，持续维护的架构现状以[当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)为入口。任务数量与通过范围集中维护于路线图，本页负责导航到对应代码、题库和说明。
