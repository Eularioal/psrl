# Science现状：任务覆盖与接入路线

> **核心结论**：现有任务已经覆盖算子、训练、模型交付和常驻服务，可以据此开展资源调度研究。基础库 **495 项**均有 GPU 快检，MLGym **9 个 GPU 任务身份**均有策略闭环。扩展库的独立验证覆盖为 PostTrainBench **15/24 个组合**、FastKernels **77/458 个目标**、SWE-Serve **21/41 个测试全过身份**。三个扩展库仍有明确缺口，不能按整库完成使用。

最新核对：**2026-11-16**。本轮实际运行新增任务的 GPU 快检与 PSRL 策略训练验证，历史已通过任务沿用各自固定证据。AIRS 修复说明沿用此前源码核对与回归范围。实验起始工作区为 `6aa328d` 加未提交内容；验证期间共享主目录已有提交 `30997ea`（单样本分组修复）、`8755e20`（bench 扩展）与 `c5f2d65`（文档整合），本页按这些提交加后续未提交修改维护。本任务未发起提交、推送、合并或共享服务部署，远端状态未核验。每次实际使用的任务版本和文件指纹保留在固定证据中。

第一次了解这项工作，可先读[简版介绍](01-Science介绍_已接入任务与研究价值.md)；需要接线、环境与复现细节，读[接入与验收详情](03-Science验收_接入流程与环境隔离.md)。本页集中回答“有什么任务、验收到哪一步、接下来先做什么”。

本篇负责解释任务场景和接入覆盖。回顾整个Env时，分别看[架构现状](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)、[当前基线与实验效果](../experiments/01-实验总览_调度收益与证据范围.md)和[下一步研究优先级](05-调度研究_研究全景与证据详解.md#接下来可以做什么优先级与验收边界)，不把任务题库覆盖当成调度收益。

## 目录

- [目前有哪些任务及可用边界](#目前有哪些任务及可用边界)
- [题库数量、位置与逐题覆盖](#题库数量位置与逐题覆盖)
  - [公共六类任务的本地协议](#公共六类任务的本地协议)
- [AIRS 任务协议](#airs-任务协议)
- [MLGym 与 SWE 任务协议](#mlgym-与-swe-任务协议)
- [PostTrainBench 与 FastKernels 任务协议](#posttrainbench-与-fastkernels-任务协议)
- [多样性覆盖与缺口](#多样性覆盖与缺口)
  - [已覆盖的任务与资源行为](#已覆盖的任务与资源行为)
  - [优先补充的运行场景](#优先补充的运行场景)
  - [可补充的具体 benchmark](#可补充的具体-benchmark)
  - [用显存变化理解为什么还需要补场景](#用显存变化理解为什么还需要补场景)
  - [补充顺序与验收](#补充顺序与验收)
- [环境是否隔离及为何反复下载](#环境是否隔离及为何反复下载)
- [Results：验证覆盖与证据索引](#results验证覆盖与证据索引)
- [可用性验收与复现](#可用性验收与复现)
- [未完成接入与下一步工作](#未完成接入与下一步工作)
- [后续研究方向与否证条件](#后续研究方向与否证条件)
- [继续阅读与交流问题](#继续阅读与交流问题)

## 目前有哪些任务及可用边界

这里的 GPU 任务包括 agent 在工具调用中训练模型、运行算子或启动推理服务。**任务内模型**负责具体计算，**PSRL 的策略语言模型**负责决定调用哪些工具。接入要把任务执行、独立评分和策略训练串起来。

<!-- figure:evidence-levels -->
这张图沿同一个场景解释：一次检查通过，究竟能支持多大的结论？

[![验证证据：不同实验分别回答不同问题（1/2）](../../_static/img/document-figures/evidence-levels-screen.png)](../../_static/img/document-figures/evidence-levels.png)

[![验证证据：不同实验分别回答不同问题（2/2）](../../_static/img/document-figures/evidence-levels-screen-02.png)](../../_static/img/document-figures/evidence-levels.png)

同一套调度代码可能通过 CPU 回归，但容器 CUDA、真实评分和训练仍各有条件。 验证是结论与证据的对应关系，不是每次必须爬完的一整套阶梯。 证据类别示意，不把进程存活、历史成功或测试替身写成本次完整训练。 [查看矢量图](../../_static/img/document-figures/evidence-levels.svg)。
<!-- /figure:evidence-levels -->

阅读状态表时，用下面三层区分进展：

| 层次 | 具体完成了什么 | 能回答的问题 |
|---|---|---|
| 已导入 | 任务指令、源码、配置和训练数据行已准备 | 框架知道要运行什么 |
| GPU 快检通过 | 在真实 GPU 上执行固定程序，交付结果，获得独立评分，并检查资源清理 | 这份任务协议能否运行 |
| 策略闭环通过 | 策略模型实际生成操作，评分进入 PSRL，执行一次优化器步骤，发布策略版本，再用新版本交互并评分 | 任务能否接入策略训练链路 |

快检检查实际 GPU 计算；闭环另检查真实模型交互和训练步骤。使用小数据、短训练的**本地协议**，是为了在固定预算内检查功能。下面的覆盖范围以该协议为单位，官方完整规模、学习效果与调度收益各自另行评测。

| 家族 | agent 的主要工作 | GPU 快检覆盖 | 策略闭环覆盖 |
|---|---|---|---|
| KernelBench | 改写 GPU 计算代码 | 270/270 题身份，含原尺寸与缩小输入 | 方阵乘法代表题 |
| FlashInfer-Bench | 实现推理算子 | 190/190 个算子定义，每定义一种输入 | RMSNorm 代表题 |
| RE-Bench | 完成训练、算子或模型修复等研究工程任务 | 7/7 类本地协议 | 聊天偏好训练代表题 |
| InferenceBench | 启动模型服务并处理请求 | 4/4 个本地服务场景 | 服务代表题 |
| MLE 本地任务 | 训练分类器、提交预测表格 | 4/4 个公开数据任务 | digits 代表题 |
| autoresearch | 修改并重跑训练程序 | 1/1 个固定任务 | 该任务 |
| AIRS-Bench | 训练、检索、代码生成或时间序列预测 | 19/19 个非 Kaggle 任务 | SICK 代表题 |
| MLGym | 训练游戏、视觉或文本模型并提交权重 | 9/9 个 GPU 任务身份 | 全部 9 个任务身份 |
| PostTrainBench | 更新预训练模型并交付权重 | 15/24 个模型与目标组合 | 全部 15 个已快检组合，各自有更新前后评分 |
| FastKernels | 优化真实模型中的算子或模块 | 77/458 个目标 | Linear 与 SigLIP2 整模型代表题 |
| SWE-Serve | 修改推理服务源码并运行测试 | 已导入 41 题，去重 21 题测试全过，17 题观测到 GPU 工作 | score 本地服务题、DSv32 cache、dLLM 采样协议 |

**495 项基础库 = 公共六类 476 项 + AIRS 19 项。** MLGym、PostTrain、FastKernels 与 SWE 分别单列。题目身份、算子定义和模型组合是不同计数单位，因此这里保留各题库的分母，不用一个总数表达整库完成度。

公共六类、MLGym、PostTrain、FastKernels 和 SWE 主要沿用 Harbor 执行任务、EnvRouter 管理资源准入的路径。AIRS 使用独立的 Coordinator/sandbox 路径，即环境协调组件与任务沙箱。已有闭环按各自执行路径成立，两条路径的统一混合训练仍待验证。任务接入与调度能力的对应关系见[当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)。

## 题库数量、位置与逐题覆盖

下表用于找到训练数据和实际任务。路径中的日期标识固定产物。**任务数据表与对应 agent 配置要成对使用**，因为数据行决定任务身份，配置决定由哪个执行器处理。

| 范围 | 任务清单与数据 | 配套说明 |
|---|---|---|
| 公共六类 476 项 | [manifest.json](../../../../benchmark-runs/complete-library-20261111/published/manifest.json)、[tasks.parquet](../../../../benchmark-runs/complete-library-20261111/published/dataset/tasks.parquet) | [agent.yaml](../../../../benchmark-runs/complete-library-20261111/published/dataset/agent.yaml)、[公共 recipe](../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md) |
| AIRS 19 项 | [manifest.json](../../../../benchmark-runs/airs-gpu-20261111/bounded/manifest.json)、[tasks.parquet](../../../../benchmark-runs/airs-gpu-20261111/bounded/tasks.parquet) | [GPU agent 配置](../../../examples/airs_bench/config/gpu_agent_config.yaml)、[AIRS recipe](../knowledge/03-任务接入/03-AIRSBench_接入流程与有界验收.md) |
| MLGym | [RL 4 题](../../../../benchmark-runs/heterogeneous-onboarding-20261112/mlgym-rl-v3/tasks.parquet)、[视觉 3 题](../../../../benchmark-runs/heterogeneous-onboarding-20261112/mlgym-vision/tasks.parquet)、[NLP 2 题](../../../../benchmark-runs/complete-onboarding-20261112/mlgym-nlp-tasks/tasks.parquet) | [MLGym/SWE recipe](../knowledge/03-任务接入/04-MLGym与SWE及Serve_任务协议与评分训练.md) |
| PostTrain 15 个通过组合 | [manifest.json](../../../../benchmark-runs/complete-bench-onboarding-20261116/verified-library/posttrain/manifest.json)、[tasks.parquet](../../../../benchmark-runs/complete-bench-onboarding-20261116/verified-library/posttrain/tasks.parquet) | [agent.yaml](../../../../benchmark-runs/complete-bench-onboarding-20261116/verified-library/posttrain/agent.yaml)，清单区分答案概率与本地裁判两种评分 |
| FastKernels 77 个通过目标 | [manifest.json](../../../../benchmark-runs/complete-bench-onboarding-20261116/verified-library/fastkernels/manifest.json)、[tasks.parquet](../../../../benchmark-runs/complete-bench-onboarding-20261116/verified-library/fastkernels/tasks.parquet) | [agent.yaml](../../../../benchmark-runs/complete-bench-onboarding-20261116/verified-library/fastkernels/agent.yaml)，每目标关联实际输入与通过证据 |
| SWE 全部 41 题与 21 个通过身份 | [全部导入清单](../../../../benchmark-runs/complete-onboarding-20261112/swe-all/manifest.json)、[通过题清单](../../../../benchmark-runs/complete-bench-onboarding-20261116/verified-library/swe-serve/manifest.json)、[通过题数据](../../../../benchmark-runs/complete-bench-onboarding-20261116/verified-library/swe-serve/tasks.parquet) | [agent.yaml](../../../../benchmark-runs/complete-bench-onboarding-20261116/verified-library/swe-serve/agent.yaml)，原测试和本地协议分别记录 |

汇总目录复用各题已验证的实际任务版本，清单中的 `task_path` 可直接找到指令、环境和测试。三个汇总 `agent.yaml` 使用 `BENCH_JOBS_DIR` 环境变量指定新建的输出目录。训练前仍须设置模型、硬件布局与资源准入预算。

AIRS 另有固定的 [13 题训练划分](../../../../benchmark-runs/airs-gpu-20261111/bounded/train.parquet)与 [6 题验证划分](../../../../benchmark-runs/airs-gpu-20261111/bounded/val.parquet)。历史闭环输入另保存在 [六类 agent-suite](../../../../benchmark-runs/full-onboarding-20261110/agent-suite/manifest.json)，用途是复现当时六个代表题。

### 公共六类任务的本地协议

| 家族 | 固定本地范围与关键改动 | 分数怎样理解 |
|---|---|---|
| KernelBench | 四个等级为 100、100、50、20 题。22 题原尺寸；193 题缩小输入；35 题缩小且保留形状约束；20 题实际加载预训练模型并限制批量 | 本地训练奖励主要检查数值正确性，耗时另看 |
| FlashInfer | 190 个定义各测一种输入，95 个原始输入档、95 个本地协议。原题库有 3267 条输入案例 | 验证所选输入的正确性与执行，计时按所选协议阅读 |
| RE | 聊天偏好训练、条件累加、受限模型训练、词嵌入修复、模型规模规律预测、训练脚本优化、Rust 解题脚手架共七类 | 各题分别用偏好、正确性、损失或任务指标评分；模型、数据和预算按本地协议固定 |
| Inference | 同一真实 Mistral 服务与真实提示，分别改变上下文、生成长度或请求并发。并发场景为 4 次请求、并发度 2 | 先比较输出与冻结参考是否一致，再记录请求耗时 |
| MLE 本地任务 | digits、wine、iris、breast-cancer 四个公开分类数据集，真实 CUDA 训练与预测 | 隐藏集准确率；与官方 Kaggle 竞赛分别登记 |
| autoresearch | 固定真实训练/验证分片，每次任务至多两次实验，提交训练程序供独立重放 | BPB，即每字节预测编码代价，越低越好；奖励为 `1/(1+BPB)` |

**为什么缩小输入还要保留形状约束？** 例如程序先定义输出宽度为 8192，再据此创建同样长度的偏置向量。若只在文件末尾把宽度改为 128，偏置仍长 8192，程序就无法计算。保约束协议在原赋值位置修改宽度，让依赖它的张量一起重建。这样减小工作量，同时保留需要检查的计算关系。

逐题输入档与失败修复保存在[完整快检证据](../experiments/gpu-benchmarks/complete-validation-20261111.json)，RE 各子协议的具体模型、步数和评估方法见[固定实验记录](../experiments/gpu-benchmarks/01-任务验收_结果与证据导航.md)与[RE recipe](../knowledge/03-任务接入/05-REBench_聊天模型训练接入.md)。

## AIRS 任务协议

AIRS 给 agent 公开数据与目标指标，agent 提交预测 CSV，独立评分器读取隐藏结果。当前 19 个非 Kaggle 任务全部快检通过，总用时 **365.68 秒**。通用快检至多使用 64 条公开训练记录和 8 条隐藏测试记录；代码生成、检索和时间序列分别保留必要结构：

- **APPS 代码生成**：2 道真实题，每题 5 份候选、2 个隐藏案例，候选在没有隐藏标签的容器中执行。
- **代码检索**：8 个查询，保留 19210 条完整公开测试候选代码。
- **Rideshare 时间序列**：8 条多变量输入展开为 120 行提交，每行预测 48 步。

SICK 句对分类代表题完成三次任务内 CUDA 更新，独立评分为 `6/8=0.75`。奖励函数再按该题的最优值、参考成绩和较差成绩，把准确率换算到 0 至 1，得到约 0.5596，具体规则见[奖励转换实现](../../../examples/airs_bench/reward.py#L42)。随后真实策略模型采样、一次 PSRL 优化器步骤、版本 1 后续评分贯通。该次策略训练有 672 个有效生成 token；同组采样奖励相同，观测到的梯度为 0。训练链路已执行，策略学习收益仍待专门测量。

AIRS 的接线、数据隔离、评分、奖励和取消路径已有修复，最近固定 CPU 回归为 **155 项一般回归 + 9 项真实上游 MLGym 回归 = 164 项通过**。仍需解决 APPS 完整 5000 题的吞吐与评分等价、重复反馈同一隐藏集合的评估协议，以及控制组件死亡后的持久恢复。具体任务与已修复问题统一见 [AIRS recipe](../knowledge/03-任务接入/03-AIRSBench_接入流程与有界验收.md)和[接入审计](../airs-bench-audit/01-AIRS审计_修复状态与验证范围.md)。

## MLGym 与 SWE 任务协议

**MLGym 的 9 个任务身份**由 4 个强化学习配置、3 个视觉任务和 2 个自然语言处理任务组成。4 个强化学习配置对应 3 种游戏环境，各做两次任务内更新。视觉任务使用 64 个训练、16 个测试样本，训练两轮。

文本任务包括 FineWeb 语言建模与 MNLI 句对分类。FineWeb 用真实文本训练小型 GPT 风格网络两次；MNLI 使用真实 BERT-base 编码器并固定其参数，只更新分类头两次。两者提交权重，由独立评分器使用可信网络结构重新加载，检查参数名称、形状和有限数值后评分。

例如，MLGym Breakout 参考提交回报为 0.46875，CIFAR-10 参考准确率为 0.125。这些小预算结果说明训练、权重交付与评分能运行，当前验收重点是链路。

**SWE-Serve 的状态需要把两种协议分开看。** 原测试全过 12 题，其中 9 题观测到 GPU 工作；本地快速协议全过 10 题，其中 9 题观测到 GPU 工作。两组共享一个 score 题，所以去重为 **21 个通过身份、17 个 GPU 工作身份**。还有 20 题缺少完整测试通过证据。

21 个测试全过身份中，另有 4 个尚缺足够的实际 GPU 工作证据，当前按测试通过统计。选取 GPU 资源调度实验任务时，应优先使用清单里已记录 GPU 执行的身份或有可信 CUDA 运算记录的协议。

本地快速协议一般选取原测试中的 2 至 4 个；Indexer 另用真实投影模块在 1、8、32 个 token 上做 CUDA 计算，再与独立矩阵乘法参考比较。score 策略闭环题使用两个上游 GPU 测试，并公开参考补丁，让 agent 执行修复和服务探测。其功能证据对应这份明确协议，原题 26 个测试的结果单独保存。

本轮新增三项可运行协议：**DSv32 cache 4/4**，验证缓存索引、页边界等行为；**Embedding LoRA 3/3**，检查低秩适配配置、输出变化与原服务行为；**dLLM 6/6**，用上游固定 logits 测试真实扩散采样算法。dLLM 的六项测试强制使用 CUDA 张量，逐项记录实际 CUDA 操作，覆盖阈值筛选、分块推进和保留提示词。它的完整模型服务、质量、前缀复用与 CUDA graph 测试保留在 `official-test.sh`，不计入这六项快检。DSv32 cache 与 dLLM 已补策略更新和新版策略评分；dLLM 的参考补丁通过 6/6，策略模型自己生成的补丁在更新前后均为 0/6，产生合法零奖励，保留这项真实解题失败。

需要理解代码修改怎样进入干净评分环境，见[接入详情中的 SWE 补丁例子](03-Science验收_接入流程与环境隔离.md#例子二swe-怎样交付代码修改)。

## PostTrainBench 与 FastKernels 任务协议

**PostTrainBench** 的固定上游清单是 6 个目标 × 4 个模型，共 24 个组合。已有独立 GPU 验证的 15 个组合来自 AIME2025、GSM8K、HumanEval、ArenaHard Writing、HealthBench 五个目标与 Qwen3-1.7B-Base、Qwen3-4B-Base、SmolLM3-3B-Base 三个模型。

前三个目标的 9 个组合加载真实基础模型，固定主体权重，只训练最终归一化层的一个参数向量，做两次更新并提交权重。准备数据有 8 条公开记录，起始方案使用其中两条；评分器重载模型与提交权重，在 4 条隐藏记录上计算标准答案的预测概率，再转换为奖励。因此这里评估的是**给标准答案分配的概率**；答案生成正确率、代码执行通过率和完整微调另有评测方法。只读模型为何仍能训练，见[真实模型交付例子](03-Science验收_接入流程与环境隔离.md#例子一posttrain-怎样训练只读模型)。

ArenaHard Writing 与 HealthBench 使用真实上游数据，固定 **2 条公开训练、2 条隐藏评分记录**。agent 同样提交归一化层权重，评分器重载后实际生成回答，再由只读冻结的本地 Qwen3.5-4B 裁判评分。写作题把候选与上游参考答案交换位置各比较一次，平均两次偏好概率；健康题逐项检查原始标准中的两个正分项和两个负分项，用带符号权重合成分数。最多生成 64 个新 token，属于本地功能协议，适合验证模型生成、独立裁判和奖励接线。

例如一项健康题的四条标准若权重为 `+2、+2、-1、-1`，裁判认为四项都满足时，得分是 `(2+2-1-1)/4=0.5`。负分项描述有害行为，满足它会降低成绩。该例为教学数字；真实权重、回答与裁判概率保存在每次 verifier 的 `judge-details.json` 中。

剩余 **9 个组合**为 Gemma-3 的 6 个模型组合与其他三个模型的 GPQA 组合。本地没有相应授权凭据和所需资产；对模型权重、GPQA 数据的匿名 HTTP 探测均返回 **401 Unauthorized**，因此需要有相应访问权限的凭据后才能准备与验收，见[访问检查](../../../../benchmark-runs/complete-bench-onboarding-20261116/restricted-assets-http.json)。

**FastKernels** 已覆盖 **77 个目标**，包括推荐模型 DLRMv2、图推荐 LightGCN、视觉网络 ConvNeXt、CLIP 文本编码器，以及独立算子和小配置的 SigLIP2、SwinV2、MobileNetV4。捕获来自真实 CUDA 前向执行，使用小模型配置与随机权重；独立评分器每目标检查一种真实捕获的输入形状、两轮正确性，并要求有效 CUDA 计时。目标层次涵盖单算子、模型模块、网络阶段与整模型。Linear 与 SigLIP2 整模型代表题已有策略闭环。

剩余 **381 个目标**需要继续补合法输入、依赖、编译与运行上下文，部分 FP4 算子要求 Blackwell GPU。已通过的原生 CUDA 扩展包括 **MoeSum、SiLU-and-Mul、RMSNorm**：任务一起交付 Python 与 `.cu` 文件，使用专用 CUDA 12.8 镜像编译。RMSNorm 的候选实现使用独立算子名称空间，避免和参考实现注册冲突；专用 H20 镜像将编译目标固定为已采集的 `sm_90`，避免运行时探测切换目标后重复编译。OneHot 的整数输入现按类别数生成合法索引。以上均有独立 GPU 回归，初始失败保留在审计记录。上游捕获配置另含 41 个单卡、4 个双卡、1 个四卡、1 个八卡场景，后续按硬件条件分组。准备器默认要求全目标捕获，显式使用 `--targets` 才准备子集，并在清单保留 `complete_scope=false`。

实现入口：[PostTrain 准备](../../../examples/posttrain_bench/prepare.py)、[训练与答案概率评分](../../../examples/posttrain_bench/local_model.py)、[裁判数据准备](../../../examples/posttrain_bench/prepare_judge_data.py)、[生成与裁判评分](../../../examples/posttrain_bench/judged_model.py)；[FastKernels 模型捕获](../../../examples/fastkernels_bench/capture_local.py)、[新增算子与视觉捕获](../../../examples/fastkernels_bench/capture_cases.py)、[子集准备](../../../examples/fastkernels_bench/prepare.py)、[独立评分](../../../examples/fastkernels_bench/verify.py)。

## 多样性覆盖与缺口

**任务内容已经较丰富，足够支持第一轮调度研究。当前更需要补充 GPU 使用方式的难例，以及它们在真实 agent 交互中的组合。** 本节依据当前源码、已验收任务协议和固定实验评估覆盖，没有新增 GPU 实验。表中的缺口是选题依据，新增场景仍需单独接入和验收。

### 已覆盖的任务与资源行为

“多样性”有两个维度。**任务内容**回答 agent 在做什么，例如优化算子、训练游戏策略、预测分子性质。**资源行为**回答它怎样使用 GPU，例如短时计算、持续训练、模型常驻、同时需要多张卡。两种不同内容的任务可能有相同的资源行为，同一道任务改变批量或执行阶段，也可能产生不同的调度问题。

仅看基础库的条目构成，KernelBench 的 270 项与 FlashInfer 的 190 项合计 **460/495，约 93%**，题量明显偏向 GPU 程序优化。这个比例描述题库构成，实际 rollout 采样比例由训练配置决定，任务耗时也需单独统计。继续增加同类算子能扩大正确性覆盖，对新增 GPU 生命周期场景的帮助较小。

| 已覆盖的场景 | 对应任务与已有证据 | 当前覆盖的主要深度 |
|---|---|---|
| 编译、运行并比较 GPU 程序 | KernelBench、FlashInfer、FastKernels，包含 Triton、原生 CUDA 扩展及整模型前向 | 覆盖较充分。快检一般选一种输入，跨尺寸、跨硬件的性能变化仍需另设协议 |
| 训练后交付权重或预测 | MLE、AIRS、autoresearch、MLGym、PostTrain | 已有真实更新和独立评分。PostTrain 当前只更新最终归一化层，短训练协议对优化器状态和大批量激活的显存压力覆盖有限 |
| 常驻模型处理请求 | Inference、SWE，以及[常驻服务压力实验](../experiments/gpu-benchmarks/06-显存压力_提前排队与失败重试.md) | 已覆盖真实请求、模型驻留、争用、排队与清理。标准题的请求规模较小，持续高并发与长上下文压力覆盖较浅 |
| 游戏交互、视觉、文本、推荐与分子预测 | MLGym 的 GPU 游戏与视觉任务、AIRS 的分子和时间序列任务、FastKernels 的视觉与推荐模型 | 任务主题已有异质性。MLGym 的[游戏交互实现](../../../examples/mlgym_bench/rl.py#L28)使用 JAX 计算框架在 GPU 上执行，已有仿真交互类工作 |
| 多个模型承担不同角色 | PostTrain 的提交模型生成回答，再由独立冻结裁判评分 | 已覆盖模型切换和独立裁判。当前实现先释放提交模型，再加载裁判，采用顺序执行 |
| 环境模型与 rollout 模型共享 GPU | [长任务共享实验](../experiments/06-训推共享_常驻vLLM与环境并行实验.md#长生成与-4b-环境模型的-12-组对照)实测环境模型更新与 vLLM 生成重叠，以及合作式暂停 | 已有组件级真实计算证据。请求流固定，尚未由环境结果动态决定下一轮 agent 生成与工具分支 |

原有 [SciAccel](../../../examples/sciaccel_rl/README.md#the-task-bank)已经涉及 LAPS、MITgcm、Athena++ 等科学仿真代码修复。需要按题核对其中哪些仿真计算实际使用 GPU，再纳入本页的 GPU 快检与策略闭环清单。因此科学仿真修复已有基础，**GPU 仿真任务的统一验收范围仍待补齐**。

### 优先补充的运行场景

| 待补场景 | 为什么现有覆盖还不够 | 建议的具体任务形态 |
|---|---|---|
| **运行中显存增长与状态恢复** | 已有显存争用实验主要检查任务准入。短协议较少覆盖训练阶段变化、长上下文增长，以及释放显存后恢复同一任务 | 复用 PostTrain 或 Inference，设置真实批量、上下文或更新范围变化，保存任务状态、释放 GPU、重新加载并继续，核对最终产物与评分 |
| **agent 根据反馈发起有依赖的并行实验** | 已有批量候选评估和固定请求共享实验。还需让中间结果实际决定后续工作，才能研究调度怎样影响 agent 的决策等待 | agent 同时启动两个候选，先完成者返回结果；agent 据此继续、取消或调整另一项，再对选中产物调用独立评分器，保留命令依赖和版本身份 |
| **一个工具任务同时使用多张 GPU** | PSRL 策略模型采用多卡训练，与任务内作业使用多卡分别属于两层。当前公共快检启动器明确限制每个环境一张 GPU | 接入一个有真实跨卡数据交换的双卡训练或模型切分任务，检查同时取得两张卡、设备拓扑、通信、评分和整组回收，再考虑跨节点 |
| **真实故障后的任务恢复** | 取消与清理已有实现和回归，整组任务经历服务退出、评分失败或显存不足后的恢复证据仍不完整，AIRS 还缺控制组件死亡后的持久恢复 | 在有检查点的短任务里有界注入一次故障，核对旧进程释放、恢复位置、提交身份与独立评分，避免重复更新或读取旧结果 |
| **完整多模态生成与 GPU 科学仿真** | 已有图像训练、编码和文本生成。当前验收清单尚缺图像、视频或语音生成的完整链路，GPU 物理求解器的范围也待专项核对 | 应用覆盖优先时，选择一个小规模图像生成流程，或复用 SciAccel 中能实际运行 GPU 求解器的题目，验证数据准备、计算、产物与数值评分 |

多卡缺口可以直接定位到[公共快检启动器](../../../examples/gpu_benchmarks/run.py#L70)。PostTrain 的训练范围见[参数选择与更新](../../../examples/posttrain_bench/local_model.py#L66)，提交模型与裁判的顺序切换见[独立评分实现](../../../examples/posttrain_bench/judged_model.py#L46)。这些是当前接入边界，新增任务要连同执行入口一起改造。SWE 的 dLLM 快检覆盖扩散语言模型的采样算法，图像生成需要独立的端到端任务与验收。

### 可补充的具体 benchmark

以下候选已核对官方入口和任务机制，**尚未新增 PSRL 接入或 GPU 通过记录**。详细源码入口、任务计数和实施方案集中在[GPU 工作负载调研](../literature-reviews/05-基准调研_GPU任务类型与接入顺序.md#按场景选择新增-benchmark)。

| 候选 | 补充的场景 | 当前判断 |
|---|---|---|
| [Automated LLM Speedrunning](https://github.com/facebookresearch/llm-speedrunner) | 从头训练 GPT、完整参数更新、多卡通信，以及根据候选实验结果继续搜索 | **优先新增，中高接入难度**。官方使用 8 张 H100，本地可设计保留真实通信的双卡档。论文定义 19 个纪录迁移任务，提示级别另计 |
| [GenEval 2](https://github.com/facebookresearch/GenEval2) | 图像生成、图片交付、独立视觉模型评分 | 中等难度。已有提示和视觉问答评测，需增加 agent 可修改的生成程序及 PSRL 接线，准备生成与评分模型 |
| [ManiSkill 3](https://github.com/mani-skill/ManiSkill) | GPU 物理仿真、并行机器人任务与可选渲染 | 中高难度。封装成 agent 修改控制或训练程序、提交控制器或权重的任务，先验证状态仿真，渲染另查 Vulkan 能力 |
| [MLPerf Inference / LoadGen](https://github.com/mlcommons/inference/tree/master/loadgen) | 多种请求到达模式、服务排队、吞吐与响应时间 | 用于扩展现有 Inference 任务。LoadGen 是负载与测量组件，模型、质量评分及 agent 任务封装需自行提供 |

Speedrunner 的[默认任务配置](https://github.com/facebookresearch/llm-speedrunner/blob/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/config/task/nanogpt_speedrun/default_config.yaml)要求关闭训练检查点，当前[候选搜索](https://github.com/facebookresearch/llm-speedrunner/blob/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/core/runners/bon_science_runner.py#L255)每轮等待本轮实验完成。因此，释放显存后恢复和某个分支先返回就立即触发下一项，仍需分别增加协议。它能提供这些研究所需的真实训练任务基础。

### 用显存变化理解为什么还需要补场景

下面采用**教学假设**：一张 GPU 给任务池的显存预算为 24 GiB，两个任务 A、B 共享这个池。

1. A 加载训练模型后占 8 GiB，B 的推理模型常驻占 10 GiB，合计 `8+10=18 GiB`，此时可以放下。
2. A 开始训练，激活和工作区使它的峰值达到 18 GiB。两项合计变成 `18+10=28 GiB`，超过预算 4 GiB。启动时的空闲快照不足以描述后续需求。
3. 调度器可以一开始按峰值承诺让其中一项排队，也可以通过明确的协议保存 B 的状态、释放显存，等 A 的高峰结束后重载 B。只暂停 B 的计算仍会保留其 10 GiB，显存压力需要真正释放状态才能缓解。

现有准入与常驻压力实测，为第一种方法提供了基础。第二种方法还要验证保存和恢复的成本、状态一致性，以及 agent 是否确实存在值得利用的等待区间。[长任务共享实验](../experiments/06-训推共享_常驻vLLM与环境并行实验.md#5-验证边界与研究判断)的合作式暂停保留模型显存，研究的是计算让路，释放显存后恢复需要另一份任务协议。

### 补充顺序与验收

**面向 GPU 调度创新，建议先补显存阶段变化和真实 agent 分支。** 已有任务可复用 PostTrain、Inference、KernelBench，新增 benchmark 优先选择 Speedrunner，设计保留完整训练和跨卡通信的本地档。故障恢复另加检查点与恢复协议。研究目标转向科学应用或多模态应用时，再优先接入 ManiSkill 3 或 GenEval 2。已有题库剩余目标的接入继续按[未完成清单](#未完成接入与下一步工作)推进，与场景补充分别统计。

新增场景沿用现有验收要求：真实 GPU 快检、独立评分，以及 PSRL 策略更新和更新后 rollout。还要保存能证明该场景实际发生的证据，例如双卡均参与计算与通信、显存实际释放后恢复、并行分支的结果确实触发后续命令。运行时间变长主要扩大重叠机会，判断计算干扰还需核对真实批量、并发和观测结果。

## 环境是否隔离及为何反复下载

安装主要写入任务镜像、专用虚拟环境或独立依赖目录；部分任务通过只读挂载复用主 Python。已有审计检查 **2540 个包元数据文件**和 **90 份任务配置**，未发现接入期间的主环境安装变更，匹配到的主 Python/JAX 前缀挂载均为只读。

镜像是软件模板，容器是运行实例；容器自身的修改、宿主挂载和显存各有不同生命周期。依赖只读复用能保护宿主包目录，但宿主后续升级仍会影响复现，因此专用镜像固化列入下一步。环境机制、SWE/PostTrain/MLGym 的真实例子及 CPU 教学脚本集中在[接入详情](03-Science验收_接入流程与环境隔离.md#环境与依赖怎样隔离)。

反复下载主要用于补真实模型和数据、配齐各题不同版本的 GPU 扩展，以及处理不完整资产。**下载 → 安装 → 实际导入 → GPU 执行**是四个独立环节。下载文件保存到各 bench 的专用资产目录，随后才安装进指定镜像或虚拟环境；快检负责确认最后一环能实际运行。

本轮预检共享存储约剩余 **140 TiB**、节点本地约剩余 **8 TiB**，新增任务复用已有模型和裁判，**没有下载新模型**。仅尝试下载 SDAR 缺少的一个依赖包，触及预算后保留未完成文件，没有安装到宿主环境。原生 CUDA 工具链由本机已有安装复制到专用镜像。新增 **57 项协议、114 份 Compose 配置**的挂载检查通过，共享 Python 前缀均只读，agent 不挂载私有数据或裁判，见[挂载审计](../../../../benchmark-runs/complete-bench-onboarding-20261116/task-mount-audit.json)与[现场预检](../../../../benchmark-runs/complete-bench-onboarding-20261116/preflight.json)。

## Results：验证覆盖与证据索引

下面按固定实验列出证据。它们验证各自的功能范围，公开起始程序和参考补丁用于降低解题差异。多次独立运行的覆盖，在表中保留各自的完成状态。

| 固定实验 | 实际取得的结果 | 证据 |
|---|---|---|
| 公共六类 476 项快检 | 逐题或逐定义执行、独立评分与清理 | [complete-validation-20261111.json](../experiments/gpu-benchmarks/complete-validation-20261111.json) |
| 公共六类代表题闭环 | 11525 个有效训练 token，一次策略更新；六类均取得版本 1 有效评分 | [六类验收记录](03-Science验收_接入流程与环境隔离.md#results-与验证边界)、[JSON](../experiments/gpu-benchmarks/full-onboarding-20261110.json) |
| AIRS 19 题 | 19 题快检通过，SICK 代表题闭环 | [airs-gpu-validation-20261111.json](../experiments/gpu-benchmarks/airs-gpu-validation-20261111.json) |
| MLGym RL/视觉 7 题 + SWE score | 32508 个有效训练 token，一次策略更新；8 题均取得版本 1 评分，版本 1 前增加了公开工具指引 | [heterogeneous-validation-20261112.json](../experiments/gpu-benchmarks/heterogeneous-validation-20261112.json) |
| PostTrain 9 组合 + MLGym NLP 2 题 | 独立快检 11/11；一次优化器步骤后，预算内 10/11 取得版本 1 评分，缺 AIME Qwen3-4B，因此该次 11 题整体验收未完成 | [首组结果](../../../../benchmark-runs/complete-onboarding-20261112/policy-functional-progress.json) |
| AIME Qwen3-4B + FastKernels Linear | 单独启动另一组策略，两题均完成更新前后评分；3486 个有效 token，梯度范数约 0.34655 | [两题补验证](../../../../benchmark-runs/complete-onboarding-20261112/final-policy/policy-functional-progress.json) |
| 当前扩展库独立验证 | PostTrain 15 个组合、FastKernels 77 个目标通过；SWE 去重 21 题测试全过、17 题观测 GPU | [逐题状态审计](../experiments/gpu-benchmarks/onboarding-status-audit-20261116.json) |
| 新增写作/健康 6 组合、SigLIP2、DSv32 cache | 8 项更新前均有评分；真实执行两次优化器更新，新版策略 7/8 有效评分，缺 HealthBench Qwen3-1.7B 的权重交付 | [8 项策略结果](../../../../benchmark-runs/complete-bench-onboarding-20261116/policy-retry/policy-functional-progress.json) |
| HealthBench Qwen3-1.7B 与 dLLM | 独立两项补验证；一次实际优化器更新、版本 1 发布后，两项均完成交互和有效评分，达标后受控停止 | [2 项补验证](../../../../benchmark-runs/complete-bench-onboarding-20261116/policy-completion/policy-functional-progress.json) |

当前证据合起来覆盖 PostTrain 全部 **15 个**已快检组合、MLGym 的 9 个任务身份、FastKernels 两个代表题与 SWE 三个代表协议。新增 8 项策略实验第一步有 6457 个有效训练 token；两项补验证第一步有 2831 个，梯度范数约 0.14526。这些数字确认实际训练路径，策略效果改善另设实验。部分任务在达到轮数或输出预算后提交已有产物并完成评分；每次的自然结束和预算结束分别记录，因此此处统计功能链路覆盖。

策略链路还修复了一项实际故障：单次采样的数据缺少 `parent_id`，导致按问题分组时出现零维数组错误。worker 现在从 `uid` 补齐分组身份，两项真实数据转换与优势计算回归通过；失败运行保存在[初始策略证据](../../../../benchmark-runs/complete-bench-onboarding-20261116/policy/policy-functional-progress.json)，机制见[训练分组说明](../../psrl/02-强化学习_PPO与GRPO及DAPO.md#psrl-怎样保存分组身份)。新增实验使用每题一次采样及本地 veRL 的单样本计算分支。

较早的 11 题运行有 19658 个有效训练 token，loss 与梯度范数均为 0，随后超时退出；AIME Qwen3-4B 的独立补验证保留对应证据。上述未完成运行均在验收表中保留，不计为整组通过。

各组均有独立清理记录；达到目标后的受控停止会使子进程返回 `-15`。验收结合实际优化器步骤、版本和后续评分判断。[新增 8 项清理](../../../../benchmark-runs/complete-bench-onboarding-20261116/policy-retry/policy-cleanup.json)、[两项补验证清理](../../../../benchmark-runs/complete-bench-onboarding-20261116/policy-completion/policy-cleanup.json)、[较早第一组清理](../../../../benchmark-runs/complete-onboarding-20261112/policy-cleanup.json)和[较早补验证清理](../../../../benchmark-runs/complete-onboarding-20261112/final-policy/policy-cleanup.json)记录了所属容器与 GPU 的物理释放。

对应版本和文件指纹保存在各 JSON 中。11-12 状态审计基于 `6aa328d` 加当时未提交内容，六类 11-10 验收基于 `bc5665f` 加当时未提交内容。之后的 CPU 回归按自己的代码快照记录：近期 12 项定向检查、较早 35 项扩题检查和 32 项 MLGym/SWE 检查分别保留，不累加成一次测试总数。静态检查的完整状态见[原始记录](../../../../benchmark-runs/complete-onboarding-20261112/status-precommit.json)，失败尝试与实验条件见[实验记录](../experiments/gpu-benchmarks/01-任务验收_结果与证据导航.md)。

## 可用性验收与复现

先按对应 recipe 准备任务，再用独立执行器检查一次任务，最后才按研究目标决定是否需要策略训练。各步骤需要的输入和验收含义见[接入详情](03-Science验收_接入流程与环境隔离.md#运行入口)。

例如，在已完成节点、Docker、GPU 与依赖预检的环境中，可从仓库根目录复验一个条件累加任务：

```bash
python -m examples.gpu_benchmarks.run \
  --task ../benchmark-runs/complete-library-20261111/re-cumsum-task \
  --output ../benchmark-runs/recheck-cumsum-new \
  --node-ip "$PSRL_NODE_IP" --gpu "$PSRL_ENV_GPU" \
  --gpu-memory-mb 4096 --timeout 180
```

`PSRL_NODE_IP`、`PSRL_ENV_GPU` 使用现场确认的节点地址和空闲 GPU；输出目录须新建。`--timeout 180` 设置任务执行预算，运行器另有评分与收尾时间。结果 `evidence.json` 应包含任务评分、所属剩余容器与 GPU 进程检查。这条命令使用固定参考程序，策略训练由另一条入口启动。

当前评分脚本的针对性 CPU 回归入口为：

```bash
python -m pytest tests/examples/gpu_benchmarks/test_task_boundaries.py \
  tests/examples/gpu_benchmarks/test_local_mle.py \
  tests/examples/gpu_benchmarks/test_re_bench.py -q
```

这里给出复现入口，本轮实际执行范围以逐题证据和 CPU 检查日志为准。正式运行还要对齐模型与隐藏划分、任务来源标记、镜像、全部所属容器的预算，以及包括工具输出的上下文长度。

## 未完成接入与下一步工作

如果目的是开展调度研究，现有已验收范围已提供多种 GPU 使用方式。若要完成更多题库，应按下面顺序逐项补齐，分别记录资产、快检和策略闭环。

| 对象 | 下一项具体工作 | 需要什么才算完成 |
|---|---|---|
| FastKernels 381 个目标 | 中高难度。补合法输入、扩展编译与模型上下文，多卡、Blackwell 限定题按资源另排 | 每目标独立正确性和有效 GPU 计时，新增执行形态再补代表闭环 |
| SWE-Serve 20 题未全过，另 4 题待补 GPU 证据 | 中高难度。逐题对齐 SGLang 源码、扩展版本、模型及服务参数，保留有界快检 | 明确每题采用原测试还是本地协议，完整测试通过并记录实际 GPU 工作，再扩闭环 |
| PostTrain Gemma-3 6 个组合 | 取得模型许可与可访问权重 | 逐组合准备、独立快检和闭环 |
| PostTrain GPQA 3 个组合 | 取得受限数据许可，固定公开/隐藏划分 | 数据协议明确后逐组合验收 |
| 官方 Kaggle | 准备凭据并接受具体竞赛规则，再逐竞赛准备数据 | 数据解析、提交格式、隐藏评分与 CUDA 起始程序分别通过 |
| 依赖固化 | 将当前只读复用的包与任务脚本固化到专用镜像，记录 image ID | 按修改范围复验代表任务，确认跨环境可重复 |

SWE 的 SDAR 候选需要 `flashinfer-jit-cache==0.6.3+cu130`，对应 8B 与 30B 模型已有缓存。官方包入口与约 **1.82 GB** 的发布文件可访问；下载前已核实空间，180 秒预算内下载约 **1.6 GB** 后停止，保留独立 `.part` 文件，尚未安装。当前阻塞仍是固定运行时依赖未准备完整，见[下载与预算记录](../../../../benchmark-runs/complete-bench-onboarding-20261116/sdar-wheel-download.json)。原服务质量测试还包含 200 道 GSM8K 题，后续需选择明确的短协议，避免把完整质量评估塞进快速验证。其他候选仍有不同的源码、扩展、模型或服务测试缺口，镜像准备后须继续经过逐题回归。

Kaggle 的 [75 个 MLE 竞赛名单](../../../../benchmark-sources/mle-bench/experiments/splits/split75.txt)和 AIRS Web Traffic 分开准备。凭据使用本机受控目录，传给下载准备流程，任务容器与评分器不挂载凭据。具体方法见[已有 Kaggle 配置](../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md#kaggle-下载权限配置)。

## 后续研究方向与否证条件

**优先问题是一组候选怎样更快得到可靠反馈。** agent 常常要等几个候选的评分齐备，才决定下一轮尝试。并发可以缩短等待，也可能因 GPU 争用影响计时。

用一组教学数字说明：4 个候选独占评估各需 10 秒，串行共 40 秒。两两并行后，每组受争用影响需 16 秒，整组共 32 秒。若评分仍保持真实的优劣顺序，这个节省有价值；若候选排名反转，agent 可能据此选错。因此要同时衡量**整组完成时间和反馈可靠性**。这些数字用于解释方法。

| 优先级 | 研究问题与任务 | 应比较的基线和指标 | 何时停止该方向 |
|---|---|---|---|
| P1 | KernelBench/FlashInfer 多候选评估 | 固定候选与输入，比较独占、直接共享、受控测量；记录总耗时、正确性、排序变化 | 等待很少，或控制开销抵消时间收益 |
| P2 | 准备、编译与 GPU 执行怎样协同 | 与单进程批量复用、静态放置和现有池比较；分开计初始化、编译、拷贝、计算 | 简单复用已经解释全部收益 |
| P3 | MLE/autoresearch 训练阶段的资源共享 | 保持相同数据、训练步数与评分，比较整组耗时、失败和质量 | 收益依赖减少工作量或降低质量 |
| P4 | RE 或服务任务在等待时保留还是释放模型 | 先测等待分布和重载成本，再比较常驻、全卸载、阈值策略 | 等待普遍短于恢复成本，或拖慢主要任务 |

其中，反馈可靠性需要先定义速度怎样进入奖励。目前本地 KernelBench 与 FlashInfer 的训练奖励主要检查正确性，`reward=1` 表示正确性通过。

已有[FlashInfer 批量成本实验](../experiments/gpu-benchmarks/07-FlashInfer_进程复用与任务合批.md)可作为进程复用基线，[热卡与冷卡模型放置实验](../experiments/gpu-benchmarks/04-动态放置_热卡等待与冷卡加载对照.md)研究等待和加载之间的取舍。后续调度贡献要在这些基线之上比较同一工作量。GPU 选择沿用用户指定的优先级：先用独立可用 GPU，再评估经过确认的共享余量。

建议先从已验收任务选少量代表，冻结候选、模型版本、评分与生命周期，测清等待和干扰是否明显，再实现更复杂策略。质量评估还要有足够稳定的数据；例如 AIRS SICK 只有 8 条隐藏记录，一条预测变化就是 12.5 个百分点，适合快检，效果研究需另设规模。

## 继续阅读与交流问题

| 想理解什么 | 阅读入口或可继续追问的问题 |
|---|---|
| 一次任务怎样变成策略训练样本 | [接入与验收详情](03-Science验收_接入流程与环境隔离.md#验收目标与执行链)；可追问“任务内训练与策略训练分别更新什么” |
| 任务为什么要重新加载模型评分 | [环境与模型交付例子](03-Science验收_接入流程与环境隔离.md#环境与依赖怎样隔离)；可追问“独立评分保护了什么” |
| 资源何时申请与释放 | [Env 当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md)；可追问“episode 预算和按命令预算怎样选” |
| 调度问题的研究背景 | [command 调度研究](../research-proposals/01-调度研究_Rollout完成时间与反馈保真.md)、[KREX 阅读](../literature-reviews/03-论文精读_KREX共享GPU测量保护.md)；可追问“已有工程基线之后还剩哪些空间” |
