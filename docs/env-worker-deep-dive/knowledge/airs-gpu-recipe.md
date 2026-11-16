# AIRS-Bench GPU 接入与有界验收

> **核心结论**：AIRS-Bench 的 20 个任务身份中，19 个非 Kaggle 任务已经准备为 PSRL 可加载的数据行，并通过原生 EnvWorker 的真实 CUDA 执行、独立评分和奖励转换。Kaggle Web Traffic 单列暂缓。这里使用真实数据的小样本协议，不是官方全量成绩；原有 CPU recipe 默认行为保持为不申请 GPU。
>
> 最新核对：2026-11-13，基于 `6aa328d` 与主目录未提交修改。本次核对 AIRS 修复状态和历史证据，164 项 CPU 回归、19 题现存数据边界检查通过，GPU 记录保存的 14 个关键源码指纹一致。GPU 与策略闭环仍是 2026-11-11 实测：逐题 365.68 秒，所属容器与 GPU 计算进程清零，SICK 完成一次优化器步骤和版本 1 后续评分。详见[实机证据](../experiments/gpu-benchmarks/airs-gpu-validation-20261111.json)和[接入审计](../airs-bench-audit/fixes-and-results.md)。本次没有重跑 GPU、提交或部署。


## 目录

- [当前接入范围](#当前接入范围)
- [任务怎样进入 PSRL](#任务怎样进入-psrl)
- [19 个任务的输入与结果](#19-个任务的输入与结果)
- [用 SICK 任务理解一次执行](#用-sick-任务理解一次执行)
- [真实策略闭环结果](#真实策略闭环结果)
- [运行入口与复现预算](#运行入口与复现预算)
- [CPU 数据边界检查](#cpu-数据边界检查)
- [跟着图读源码](#跟着图读源码)
- [限制与后续方向](#限制与后续方向)

## 当前接入范围

AIRS-Bench 给 agent 一项机器学习研究任务，例如根据两句话判断它们是否矛盾，或根据分子结构预测热容量。agent 自己编写训练、检索或预测代码，交出 CSV 预测文件。评分器使用 agent 看不到的测试答案计算指标，PSRL 再把指标转换为策略训练的奖励。

PSRL 使用 MLGym 驱动 AIRS agent 调用模型、执行 shell 命令和提交结果。MLGym agent 与 MLGym 自身题库不是同一概念。当前接入包含 GPU 资源声明、可执行 CUDA 基线、19 题真实数据的有界准备、训练入口以及明确范围的验证。

“完整接入”与“全量官方成绩”分别描述接口是否通、实验规模是否相同。这里每个非 Kaggle 任务都有数据、任务配置、公开起始程序、独立评分器、奖励函数和 PSRL agent 配置。快速检查的目的，是确认这些接口能接起来，不能据此声称解决了这些研究任务。

## 任务怎样进入 PSRL

```text
[1] 真实数据与任务描述 → prepare_gpu_suite → Parquet 数据行
                                             │ task_config_path / dataset_data_path
                                             ▼
[2] PSRL MLGymAgentLoop ←→ Session / TITO ←→ 策略模型生成下一条操作
           │ 先申请 sandbox，再运行 MLGym
           ▼
[3] EnvWorkerCoordinator → Ray 分配的 EnvWorker → GPU agent 容器
           │ 仅挂公开 train / validation / test，保留工作目录
           ▼
[4] agent 执行 CUDA 训练或检索 → submission.csv
           │ 提交物快照
           ▼
[5] 独立 CPU verifier 读取隐藏标签 → 原始指标
           └─ APPS：另起无标签的候选执行容器，只把输入送进去
           │ 指标交回 PSRL
           ▼
[6] reward 归一化 → 有效 token 进入后训练 → 参数更新 → 下一轮 rollout
           │ 无论成功、失败或取消
           └─ destroy：回收本 episode 的容器及 GPU 槽位
```

图例：图里每个部分对应的源码位置。

| 图中位置 | 作用与函数 | 源码 |
|---|---|---|
| 1 | `prepare` 选择真实数据、生成配置和 Parquet | [prepare_gpu_suite.py](../../../examples/airs_bench/prepare/prepare_gpu_suite.py#L87) |
| 2 | `MLGymAgentLoop.run` 创建会话、运行 episode、提取 TITO 数据 | [mlgym_agent_loop.py](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L109) |
| 2 → 3 | `reserve` 同时检查 CPU、RAM 和 GPU 槽位 | [coordinator.py](../../../psrl/workers/env_worker/coordinator.py#L237) |
| 3 | `build_sandbox_spec` 只挂公开 split，声明 `sandbox_gpus` | [mlgym_env.py](../../../psrl/environments/mlgym_env.py#L60) |
| 3 → 4 | `run_mlgym_agent` 将上游 MLGym 的 shell 操作接到远程 sandbox | [runner.py](../../../examples/airs_bench/runner.py#L137) |
| 4 | `predict` 在 CUDA 上训练线性模型或执行向量检索 | [gpu_baseline.py](../../../examples/airs_bench/gpu_baseline.py#L23) |
| 5 | `build_verifier_spec` 单独挂隐藏数据 | [grading.py](../../../examples/airs_bench/grading.py#L32) |
| 5 APPS | `grade` 将待执行代码和输入发送到隔离执行器，答案留在 judge | [judge.py](../../../examples/airs_bench/apps/judge.py#L61) |
| 6 奖励 | `normalized_score` 按距离最优值的对数转换奖励 | [reward.py](../../../examples/airs_bench/reward.py#L42) |
| 6 轨迹与回收 | `compute_reward_score` 之后在 `finally` 中销毁 sandbox | [mlgym_agent_loop.py](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L180) |

TITO（Token-In-Token-Out，按 token 记录输入和输出）是保存模型逐轮输入、输出 token 和概率的会话机制，用来把工具调用轨迹交给后训练。GPU 子任务中的小模型负责做预测；PSRL 的策略语言模型负责决定执行什么命令。两者参数不同，不能把小模型的三步训练当成 PSRL 策略已更新。

这条 AIRS 路径使用原生 sandbox 的 Coordinator 和 Ray GPU 分配，**不经过 Harbor/EnvRouter，也没有接入 `psrl-gpu` command 借还**。GPU 在整个 episode 内保留，评分器本身使用 CPU。GPU recipe 的 240 秒 episode 期限包含排队时间；同一环境卡串行处理两条采样时，第二条的可执行时间会扣除等待，研究实验需按实际动作预算调整配置。它与原有 476 题的 Harbor recipe 可以分别训练，本次没有验证两种入口混在同一个训练作业中。

这些代码如果有理解上的问题，可以随时追问。

## 19 个任务的输入与结果

完整清单见 [manifest.json](../../../../benchmark-runs/airs-gpu-20261111/bounded/manifest.json)。普通任务使用至多 64 条公开训练数据、8 条测试输入；测试标签只提供给独立评分器。其余特殊规模列在表后。下面指标来自 2026-11-11 小样本基线实测，不是论文效果或本次重新运行的结果。

| 任务身份 | GPU 基线做什么 | 独立评分结果 |
|---|---|---:|
| CodeGenerationAPPSPassAt5 | 根据题目向量检索公开训练代码，生成五个候选 | Pass@5 = 0 |
| CodeRetrievalCodeXGlueMRR | 在完整公开测试代码库中做 CUDA 向量检索 | MRR = 0 |
| CoreferenceResolutionSuperGLUEWSCAccuracy | 线性分类器，三次 CUDA 参数更新 | Accuracy = 0.5 |
| CoreferenceResolutionWinograndeAccuracy | 线性分类器，三次 CUDA 参数更新 | Accuracy = 0.625 |
| CvMolecularPropertyPredictionQm9MeanAbsoluteError | 分子特征回归，三次 CUDA 参数更新 | MAE = 3.8621 |
| GMolecularPropertyPredictionQm9MeanAbsoluteError | 分子特征回归，三次 CUDA 参数更新 | MAE = 624129250 |
| GraphRegressionZincMae | 图特征回归，三次 CUDA 参数更新 | MAE = 1.4286 |
| MathQuestionAnsweringSVAMPAccuracy | CUDA 检索公开训练问答 | Accuracy = 0 |
| QuestionAnsweringDuoRCAccuracy | CUDA 检索答案并处理无答案标记 | Accuracy = 0.5 |
| QuestionAnsweringEli5RougeL | CUDA 检索公开训练答案 | ROUGE-L = 0.08818 |
| QuestionAnsweringFinqaAccuracy | CUDA 检索公开训练问答 | Accuracy = 0 |
| R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError | 分子特征回归，三次 CUDA 参数更新 | MAE = 176.2069 |
| ReadingComprehensionSquadExactMatch | CUDA 检索公开训练答案 | ExactMatch = 0 |
| SentimentAnalysisYelpReviewFullAccuracy | 线性分类器，三次 CUDA 参数更新 | Accuracy = 0.25 |
| TextualClassificationSickAccuracy | 线性分类器，三次 CUDA 参数更新 | Accuracy = 0.75 |
| TextualSimilaritySickSpearmanCorrelation | 相似度回归，三次 CUDA 参数更新 | Spearman = -0.04791 |
| TimeSeriesForecastingRideshareMAE | 多变量序列预测，三次 CUDA 参数更新 | MAE = 9.0915 |
| TimeSeriesForecastingSolarWeeklyMAE | 五步序列预测，三次 CUDA 参数更新 | MAE = 2575.2271 |
| U0MolecularPropertyPredictionQm9MeanAbsoluteError | 分子特征回归，三次 CUDA 参数更新 | MAE = 624100500 |

Accuracy/ExactMatch 表示预测正确的比例；MAE 是平均绝对误差，越小越好；MRR 衡量正确代码排在检索结果中的位置；ROUGE-L 衡量文本匹配；Spearman 衡量预测与答案的排序相关性；Pass@5 表示五份代码中至少一份通过所有选定测试的题目比例。

特殊规模与边界：

- **APPS**：2 道真实测试题，每题保留前 2 个隐藏案例，每题交 5 个代码候选。沿用已有隔离 judge，候选代码不能读取评分答案。本地缩小了案例数，不等于官方完整 APPS 验收。
- **代码检索**：8 个查询，但保留 **19210 条完整公开测试候选代码**，没有通过只挑正确候选来降低题目难度。其公开训练查询保留 64 条。
- **Rideshare**：8 条多变量输入，每条有 15 个序列，提交 **120 行**，每行预测 48 个值。64 条训练记录展开、过滤无效未来值后，基线使用 945 条训练序列。
- **分数为零、很大的误差或负相关仍是有效结果**：这些反映基线质量。接入验收要求真实执行、提交格式正确、评分正常和奖励有限，不要求战胜论文基线。

归一化奖励沿用 AIRS 公式与已准备元数据，但小测试集上的 1 分不等于达到论文 SOTA（已发表的最好结果）。例如 DuoRC 的本地归一化奖励达到 1，是 8 条样本与既定归一化常数共同产生的数值，不能当成研究结论。

## 用 SICK 任务理解一次执行

历史 GPU 快检输入是 64 对有标签的训练句子和 8 对不带标签的测试句子。公开 GPU 基线将文本转为 128 维词哈希特征，用三次随机梯度下降更新一个线性分类器，然后在 CUDA 上预测测试句子的类别，写出 8 行 CSV。

独立 verifier 在另一容器读取这 8 行预测和隐藏标签，得到 **6/8 = 0.75** 的准确率。奖励函数把 0.75 转成 **0.5596034**。这些数字来自逐题 GPU 检查，并非教学假设。

随后 PSRL 策略闭环使用同一个提交和评分契约：语言模型生成执行基线、提交结果的操作，TITO 保存操作文本对应的有效 token，PSRL 使用轨迹奖励执行后训练。闭环配置使用专门的 `acceptance/` 提示，让模型快速执行公开起始方案；普通 `tasks/` 提示仍允许改代码和优化指标，二者的评分器相同。

代码检索的公开测试输入由选定隐藏数据的公开字段投影生成，输入和标签逐行对应，投影移除答案字段。这修复了公开查询与隐藏查询顺序不同的问题，不能把“两边行数相同”当成配对正确。

## 真实策略闭环结果

SICK 代表题已完成真实 Qwen3.5-4B 采样、1 次 PSRL 优化器更新和版本 1 后续 GPU 评分。有效生成 token 为 672，奖励约 0.5596；组内奖励相同，梯度为 0。达到验收目标后受控停止剩余采样，所属容器和 GPU 计算进程已清零。

| 训练阶段 | 有效生成 token | 归一化奖励 | 策略 loss / 梯度范数 |
|---|---:|---:|---:|
| 第 1 步 | 672 | 0.5596035 | 0 / 0 |

版本 0 完成 2 个 episode（完整任务尝试），版本 1 完成至少 1 个有效 episode 后立即停止剩余采样，不为追加第二次训练而等待。训练子进程退出 `-15` 是验收通过后的受控收尾，启动记录与 [controlled-stop.json](../../../../benchmark-runs/airs-gpu-20261111/policy-acceptance/controlled-stop.json) 保留其条件。每个 episode 可拆成多条 TITO token 片段，不能把片段数当成任务数。模型实际按三步验收提示生成命令、执行 CUDA 基线、调用独立 `validate` 并主动 `submit`。记录为 `finished`。版本 1 轨迹中同时保留真实 CUDA 遥测和 `Accuracy: 0.75` 的验证观察。

同组两条尝试采用同一公开起始方案，奖励相同，组内相对优势为 0，因而策略 loss 和梯度范数为 0。实测证明有效 token 进入优化器、版本发布与拉取、更新后执行和评分均可用，**不证明权重产生有意义变化、模型学得更好或自主研究能力提高**。本次没有为追求非零梯度继续增加实验。

最终策略验收运行 931.82 秒（约 15.5 分钟，包含分布式模型初始化），其中一次策略优化器更新为 17.39 秒。它与前面的 19 题 GPU 快检 365.68 秒分别计时，不能合称整个接入闭环只用 6 分钟。

策略验证只选 SICK 代表题覆盖公共 agent/会话/后训练路径。全部 19 题各自的 GPU 与评分接口另由逐题快检覆盖，不表示每题都做过语言模型训练。当前 CPU 复测为两批互不重叠的 155 项一般回归和 9 项真实上游回归，全部通过且无跳过，见[当前记录](../airs-bench-audit/cpu-regression-20261113.json)。GPU 证据文件中的 123 项属于当时全套与修复后定向运行的覆盖统计，不能当成一次当前整组实测或与 164 项相加。

原始[训练日志](../../../../benchmark-runs/airs-gpu-20261111/policy-acceptance/train.log)、[启动与退出记录](../../../../benchmark-runs/airs-gpu-20261111/policy-acceptance/launch.json)、[策略证据](../../../../benchmark-runs/airs-gpu-20261111/policy-evidence.json)与[物理清理记录](../../../../benchmark-runs/airs-gpu-20261111/policy-cleanup.json)均保留。

## 运行入口与复现预算

从 PSRL 仓库根目录使用这些入口。`AIRS_ORIGINAL_DATA` 指已准备的原始 AIRS 数据根目录，`AIRS_GPU_DATA` 指本轮生成的新目录；`AIRS_REPO` 是上游源码，`NODE_IP` 是现场核实的执行节点地址，`GPU_INDEX` 是空闲物理卡号，`RESULT_DIR` 必须是新的输出目录。

```bash
python -m examples.airs_bench.prepare.prepare_gpu_suite \
  --repo "$AIRS_REPO" --data-root "$AIRS_ORIGINAL_DATA" --output "$AIRS_GPU_DATA"

python -m examples.airs_bench.gpu_quick_check \
  --data "$AIRS_GPU_DATA" --output "$RESULT_DIR" \
  --node-ip "$NODE_IP" --gpu "$GPU_INDEX" --python "$AIRS_SANDBOX_PYTHON"
```

第一条只准备数据，不使用 GPU；已有输出会拒绝覆盖。第二条在执行节点本机启动隔离 Ray，逐题申请原生 GPU sandbox，运行 CUDA 基线并调用独立评分器。每题最多 240 秒、每个动作最多 120 秒；本次外层还使用 1500 秒超时，正常 19 题合计约 6.1 分钟。`--tasks` 可只重验指定失败题，不需要整批重跑。

| 入口 | 用途 |
|---|---|
| [tasks.parquet](../../../../benchmark-runs/airs-gpu-20261111/bounded/tasks.parquet) | 全部 19 个任务身份 |
| [train.parquet](../../../../benchmark-runs/airs-gpu-20261111/bounded/train.parquet)、[val.parquet](../../../../benchmark-runs/airs-gpu-20261111/bounded/val.parquet) | 按既有任务划分保留 13 个训练任务与 6 个验证任务；排除 Kaggle |
| [gpu_agent_config.yaml](../../../examples/airs_bench/config/gpu_agent_config.yaml) | GPU agent 的镜像、Python、CPU/RAM、超时等配置 |
| [gpu_train.sh](../../../examples/airs_bench/gpu_train.sh) | PSRL 后训练入口，默认 2 张训练卡、1 张策略推理卡，另给 EnvWorker 1 张独立卡 |
| [policy-acceptance.sh](../../../../benchmark-runs/airs-gpu-20261111/policy-acceptance.sh) | 本机实际有界闭环启动参数，需为再次实验改用新的输出目录 |
| [quick/evidence.json](../../../../benchmark-runs/airs-gpu-20261111/quick/evidence.json) | 每题 CUDA、指标、耗时、导入来源、Ray 身份及清理记录 |

训练还需要激活项目环境并固定 `PYTHONPATH`、`LOCAL_IP`、`HF_MODEL_PATH`、`OUTPUT_DIR`、`AIRS_DATA_ROOT`、`MLGYM_REPO`、`TRAIN_FILES` 和 `VAL_FILES`，遵循[集群预检](../../psrl/psrl-run-guide/cluster-preflight.md#集群踩坑总表与最小复现2026-10-13)。本轮 MLGym 缺失依赖装在独立 `benchmark-assets/airs-psrl-deps` 中，没有修改共享 Python 安装。

历史验收镜像的 PyTorch 为 CUDA 13.0，验证节点驱动报告 CUDA 12.8。因此该 GPU 配置使用 `sandbox_python` 指向兼容的宿主 Python 前缀并只读挂载，CPU verifier 与 APPS 候选继续使用镜像自己的 Python。`sandbox_python` 默认空，`sandbox_gpus` 默认 0，GPU recipe 才显式改为 1。配置 GPU 槽位不等于 CUDA 可执行，启动新实验仍须现场检查兼容性，不能复用历史节点状态作为保证。

CPU/GPU 两份 YAML 均显式用 `_target_` 指向 `MLGymAgentLoop`，模型前缀为 `litellm:openai/...`，本地端点使用占位 API key 和零费用元数据。真实本机 HTTP 回归验证请求路由、TITO header 和费用统计，不调用外部模型。这些契约修复对应历史初始化与 API 接线故障，没有修改评分规则来绕过失败。

通用研究提示不能保证在预算内产出合法提交。保存的一次运行中，版本 0 完成训练，但版本 1 缺少提交或超时，作业退出 1，不计为新版评分成功。代表闭环成功证据来自独立三步验收提示与明确输出预算，历史失败日志和清理记录保留在实验目录。

三步验收配置为 [acceptance_agent_config.yaml](../../../examples/airs_bench/acceptance_agent_config.yaml)，通过 `AIRS_AGENT_CONFIG` 显式选择。`gpu_train.sh` 默认仍选普通研究配置。验收使用每轮最多 512 个输出 token、4 步动作上限，研究 recipe 保留原有较宽预算。

## CPU 数据边界检查

这不是模拟训练，而是读取本轮真实准备资产的 schema（字段结构）做静态检查。使用标准库即可运行，不需要 GPU 或模型：

```bash
python -m examples.airs_bench.prepare.audit_gpu_suite --data "$AIRS_GPU_DATA"
```

预期输出 `passed: true` 与 19 个任务。以 SICK 为例，公开测试只有 `sentence_A` 和 `sentence_B` 等允许字段，隐藏 schema 另有 `label`；脚本检查公开特征不含隐藏字段、题目身份无重复、提交行数与协议相符。Rideshare 应显示 120 行，而普通任务为 8 行，APPS 为 2 行。

现存资产的 19 项检查已通过，当前复核记录见[CPU 与数据边界结果](../airs-bench-audit/cpu-regression-20261113.json)，原始 GPU 准备记录为 [schema-audit.json](../../../../benchmark-runs/airs-gpu-20261111/schema-audit.json)。这只能证明静态边界与计数，不能代替容器隔离或 GPU 实测，对应代码为 [audit_gpu_suite.py](../../../examples/airs_bench/prepare/audit_gpu_suite.py)。

## 跟着图读源码

| 步骤 | 图中编号 | 函数与源码 | 输入、输出与状态 |
|---|---|---|---|
| 准备题库 | 1 | `prepare`，[prepare_gpu_suite.py](../../../examples/airs_bench/prepare/prepare_gpu_suite.py#L87) | 原始真实 split → 有界公开 split、隐藏 split、Parquet |
| 读取一题 | 2 | `MLGymEnvironment.reset`，[mlgym_env.py](../../../psrl/environments/mlgym_env.py#L309) | 数据行 → 任务路径、指标与 episode 身份 |
| 准入 | 3 | `EnvWorkerCoordinator.reserve`，[coordinator.py](../../../psrl/workers/env_worker/coordinator.py#L237) | GPU 槽位从空闲变占用；无容量时客户端等待 |
| 运行 agent | 2、4 | `run_mlgym_agent`，[runner.py](../../../examples/airs_bench/runner.py#L137) | MLGym 子进程通过异步 RPC 调用远程 shell，并向会话接口请求模型输出 |
| 执行 GPU | 4 | `predict`，[gpu_baseline.py](../../../examples/airs_bench/gpu_baseline.py#L23) | CUDA 分类、回归或检索 → CSV，不读取隐藏答案 |
| 提交评分 | 5 | `grade_submission`，[grading.py](../../../examples/airs_bench/grading.py#L106) | 提交物快照 → 独立 verifier → 有限指标；APPS 额外启动无标签的代码容器 |
| 交给训练 | 6 | `MLGymAgentLoop.run`，[mlgym_agent_loop.py](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L180) | 轨迹和指标 → TITO token、奖励 → PSRL 后训练；这是跨组件异步流程 |
| 清理 | 6 | `finally`，[mlgym_agent_loop.py](../../../psrl/workers/agent_loop/loops/mlgym_agent_loop.py#L200) | 销毁 sandbox 后退出；超时也经过该路径 |

## 限制与后续方向

非 Kaggle 任务身份已接齐，验证使用小数据与短程序。逐项快检只直接提交一次，未逐项做 agent 的 validate 交互或后训练，后两者由 SICK 代表闭环覆盖。完整 APPS 5000 题的吞吐与评分等价、反复反馈同一隐藏集合的评测协议、Coordinator/worker 死亡后的持久恢复仍未解决。具体修复和缺口见[接入审计](../airs-bench-audit/fixes-and-results.md#尚未完成的能力)，其他 benchmark 的当前数量与状态统一见[路线图](../milestones/science-gpu-bench-status-and-roadmap.md)。

后续如果用 AIRS 做创新研究，先扩大足以稳定比较的测试数据，再讨论调度收益。SICK 只有 8 条测试时，改对或改错 1 条就会让准确率变化 12.5 个百分点，足以淹没很多算法改进。最小接入测试只解决“能不能运行与评分”，不能拿这种噪声奖励证明策略学习有效。

一个可检验方向是把 CPU 数据处理、GPU 训练、CPU 独立评分分开计量，研究在评分阶段释放 GPU 的收益及重新加载代价。当前实现会一直保留 episode 的 GPU 槽位；只有补齐阶段边界与物理回收证据，才能与已有 Harbor command 路径做同条件比较。跨两套入口的统一混训和共享准入仍须单独设计和验证。
