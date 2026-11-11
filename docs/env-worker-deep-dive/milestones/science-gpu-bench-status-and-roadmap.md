# Science GPU 任务接入现状与下一步研究规划

> **核心结论**：PSRL 目前已有 **6 类 GPU 代表任务通过本地完整闭环**：KernelBench、autoresearch、MLE 本地竞赛、Inference 本地服务、RE 有界偏好训练和 FlashInfer 算子子集。它们都完成了真实策略采样、独立 GPU 任务验收、一次 PSRL 策略更新，以及更新后模型的继续采样和有效评分。
>
> 下一阶段的重点是用这些真实任务研究“怎样更快取得可靠反馈”。接入可用、agent 每次成功、模型学得更好、调度更快是四个不同结论。已完成的 GPU 验收证明接入可用，其他三项需要独立实验。
>
> 更新日期：2026-11-11。六类策略闭环各覆盖一个代表题；本轮新增快速 GPU 验证，发布 249 题可加载任务池，其中 KernelBench 原尺寸 22 题、小输入档 193 题。逐题覆盖与位置见题库统计节，闭环证据见 Results，研究方向与最小对照见后两节。详细协议与运行条件见[完整接入报告](science-gpu-full-onboarding.md)。

## 目录

- [目前有哪些任务及可用边界](#目前有哪些任务及可用边界)
- [题库数量、位置与逐题覆盖](#题库数量位置与逐题覆盖)
- [Results：扩题与快速 GPU 验证](#results扩题与快速-gpu-验证)
- [任务与框架如何串联](#任务与框架如何串联)
- [Results：已完成的 PSRL 闭环验证](#results已完成的-psrl-闭环验证)
- [具体任务例子与评分含义](#具体任务例子与评分含义)
- [失败处理与本轮边界修复](#失败处理与本轮边界修复)
- [可用性验收与复现](#可用性验收与复现)
- [相对旧路线图完成了什么](#相对旧路线图完成了什么)
- [后续研究方向与否证条件](#后续研究方向与否证条件)
- [建议的推进顺序与交流问题](#建议的推进顺序与交流问题)

## 目前有哪些任务及可用边界

这里说的 GPU 任务，是 agent 操作工具时需要运行的计算，例如训练分类器、执行 GPU 算子、启动推理服务。任务中的小模型与 PSRL 的策略语言模型是两份模型：前者是任务产物，后者负责决定下一步调用什么工具。

本文的“完整接入”标准是：**真实 rollout → 独立 verifier → PSRL 后训练更新 → 更新后 rollout**。rollout 是策略模型实际生成操作并与任务环境交互的过程。verifier 是独立检查提交物、计算得分的程序。只提交已知答案的 oracle 检查可以证明评分器可运行，但不能代替真实模型采样与策略更新。

| 任务家族 | agent 要做什么，GPU 做什么 | 已通过的本地任务与评分 | PSRL 闭环 | 保证范围 |
|---|---|---|---|---|
| **KernelBench** | 提交等价的 GPU 计算实现，独立运行正确性与计时检查 | 矩阵乘法代表题，奖励取正确性，速度另报 | 已通过 | 270 题已准备，闭环实测 1 题；不表示候选变快 |
| **autoresearch** | 提交语言模型训练脚本，独立 verifier 重新训练与评估 | 真实训练分片、约 300 秒训练预算，按验证 BPB 计分 | 已通过 | 固定缓存与单训练分片，不是默认全部数据 |
| **MLE 本地竞赛** | 在 CUDA 上训练数字分类器并生成预测 CSV | 真实 digits 数据，隐藏测试集准确率，`mle_bench_local` | 已通过 | MLE 风格本地竞赛，不是原 MLE-bench 的 Kaggle 竞赛成绩 |
| **Inference 本地服务** | 提交前台服务启动脚本，独立 verifier 重启服务并发请求 | 两个真实 MMLU-Pro 请求，输出匹配冻结基线后按耗时计分，`inference_bench_local` | 已通过 | 固定服务协议，不是官方完整质量、合规与性能评测 |
| **RE 有界偏好训练** | 用 GPT-2 XL 生成回答、获取本地裁判偏好并训练，提交模型 | 保留真实训练与评价函数，评估两条隐藏提示，`re_bench_local` | 已通过 | 显式 `bounded-local`，不是原 512 条完整评估，两种 smoke 仍为零奖励 |
| **FlashInfer-Bench** | 提交官方格式的 Solution JSON，独立执行生产算子评估 | RMSNorm 固定子集，三项官方 workload 全部正确才得 1 分 | 已通过 | 公共 Harbor / PSRL recipe 已接入，非全题库或性能收益证明 |

这六类任务的“可用”限定于已准备资产、镜像、依赖和本地协议。换模型、换数据、换节点或扩大规模，需要重新检查条件，不承诺任意部署都能直接运行。

以下项目单列，不混入六类已通过结果：

| 项目 | 当前定位 |
|---|---|
| **SciAccel** | 原有 science 家族与公共会话执行来源，已有历史训练记录。是否需要 GPU 取决于具体题目，CPU 编译修复题不重复计入新增 GPU 覆盖 |
| **AIRS-Bench / MLGym** | 有独立 recipe，但本轮未找到完整本地准备资产，GPU 子集和真实 PSRL 闭环未纳入保证范围 |
| **PostTrainBench、SWE-Serve、FastKernels 等** | 调研或环境探针候选。源码下载、介绍文档、组件实验都不自动等于已完成 PSRL 接入 |

## 题库数量、位置与逐题覆盖

**2026-11-11 核对：此前“六类完整闭环”指六类各一个代表任务，不是六套题库全部通过。** 本次扩展了 KernelBench 的全量任务准备入口：270 个源题全部生成了独立 Harbor 任务与统一 PSRL 数据表。随后完成本节下方登记的快速 GPU 验证，并发布只包含通过项的独立任务池；没有重复启动策略训练。

先把四个数字分开看：**源题数**是题库中存在的题目；**适配能力**是代码能为哪些题生成任务；**已准备数**是确实生成了任务资产与数据行；**闭环验证数**还必须有真实 GPU 评分、PSRL 更新和更新后采样证据。将同一题复制到多个实验目录，或对同一题采样多次，都不能增加题目覆盖。

| bench | 固定本地版本的题库规模 | 当前适配与准备范围 | 已完成 PSRL GPU 闭环的题目覆盖 | 尚缺什么 |
|---|---|---|---|---|
| **KernelBench** | **270 题**：L1 100、L2 100、L3 50、L4 20 | 原先可逐题指定；本次批量准备 **270/270**，统一数据表 270 行 | **1/270**：L1 的方阵乘法 | 快速 GPU 检查原尺寸通过 22 题、小输入档通过 193 题；35 题未通过，20 题缺预训练资产未运行 |
| **MLE-bench** | 官方 `split75.txt` 为 **75 个竞赛**；注册目录有 82 个配置，不能将两者混为 82 个正式题 | 官方 recipe 可逐竞赛准备，但指定本地资产根目录内完整官方 public/private 数据为 **0 套**；另有 **4 个本地竞赛**：digits、Iris、Wine、乳腺癌，均已快速 GPU 验证 | 官方竞赛 **0/75**；策略闭环仍是本地 digits 代表题，新三题仅快速任务验收 | 官方数据与准备流程仍需补齐。本地 digits 不算官方题库覆盖 |
| **RE-Bench** | **7 个任务目录** | 当前专属准备器只实现聊天偏好训练 `ai_rd_nanogpt_chat_rl/main`，本地有界正式协议 **1 个** | **1 个本地有界任务**；其余 6 个任务没有适配，不等于完整官方 1/7 成绩 | 其他任务需各自接入提交格式、资产与评分器，不是换目录就能复用聊天训练 recipe |
| **InferenceBench** | **4 个官方场景**：输入重、输出重、高负载、通用；不是四条请求 | 官方生成任务的导入接口存在，但完整官方场景验收 **0 个**；已有 **1 个本地服务任务、2 条请求** | 本地服务 **1/1**；官方完整场景 **0/4** | 质量与速度基线、模型和数据、请求规模及独立验收需逐场景准备 |
| **FlashInfer-Bench / Trace** | **190 个算子定义**，其中 **111 个有 workload 文件**，共 **3267 条输入案例** | 公共 recipe 已支持逐定义准备，新增 **30 个定义各一条原始随机 workload**，均经官方 evaluator 快速通过 | **1 个算子题、3 条 workload** | 其余 160 个定义未进入本轮通过池；全量输入引用的 2570 个张量文件仍为 LFS 指针，需按 workload 准备实体数据 |
| **autoresearch** | **1 个固定训练优化任务**，不是多题题库 | **1 个任务**；缓存 1 个训练分片，保留固定验证分片 | **1/1 个本地固定任务** | 更多训练分片是扩大数据规模，不是新增题目；默认完整训练数据覆盖未验证 |

这些统计限于本地固定源码版本与 `benchmark-assets/` 资产根目录，不代表整个共享集群没有其他数据。FlashInfer 的张量指针只是几行下载元信息，不能当作可加载的 `.safetensors` 数据。该阻塞影响需要这些文件的 workload，不意味着所有随机输入案例都不可扩展。本次没有自动下载全量大数据。

**题库和生成产物在哪儿？** 以下位置以 PSRL 仓库根目录为起点，`../` 表示同工作区的上一级目录。题库源码不是已经能训练的数据表。

| 用途 | 本地入口 |
|---|---|
| KernelBench 270 道源题 | [level1](../../../../benchmark-sources/KernelBench/KernelBench/level1)、[level2](../../../../benchmark-sources/KernelBench/KernelBench/level2)、[level3](../../../../benchmark-sources/KernelBench/KernelBench/level3)、[level4](../../../../benchmark-sources/KernelBench/KernelBench/level4) |
| 本次 KernelBench 全量任务 | [kernel-library-20261111](../../../../benchmark-runs/kernel-library-20261111)：`tasks/` 每题一个目录，`inventory.json` 记录题目、来源指纹和未验收状态 |
| 全量准备表（含尚未通过题，不作为默认训练池） | [dataset/tasks.parquet](../../../../benchmark-runs/kernel-library-20261111/dataset/tasks.parquet) 与 [dataset/agent.yaml](../../../../benchmark-runs/kernel-library-20261111/dataset/agent.yaml)，必须成对使用 |
| MLE 官方题目名单与描述 | [split75.txt](../../../../benchmark-sources/mle-bench/experiments/splits/split75.txt)、[competitions](../../../../benchmark-sources/mle-bench/mlebench/competitions) |
| RE 七类任务与说明 | [RE-Bench](../../../../benchmark-sources/RE-Bench)，当前适配的是 `ai_rd_nanogpt_chat_rl` |
| Inference 官方场景 | [src/eval/tasks](../../../../benchmark-sources/InferenceBench/src/eval/tasks) |
| FlashInfer 定义与输入案例 | [definitions](../../../../benchmark-sources/flashinfer-trace/definitions)、[workloads](../../../../benchmark-sources/flashinfer-trace/workloads) |
| autoresearch 任务与缓存 | [train.py](../../../../benchmark-sources/autoresearch/train.py)、[autoresearch-data](../../../../benchmark-assets/autoresearch-data) |
| 已验证的六类代表题清单 | [11-10 agent-suite/manifest.json](../../../../benchmark-runs/full-onboarding-20261110/agent-suite/manifest.json)，每类一题，不能拿它代表全量题库 |

KernelBench 的四个 level 也有不同执行条件：

| level | 源题 / 本次准备 | 内容 | 运行前重点检查 |
|---|---|---|---|
| L1 | 100 / 100 | 单算子，例如矩阵乘法、归约 | PyTorch、编译器、CUDA / Triton 与原始输入显存 |
| L2 | 100 / 100 | 多算子融合，例如计算后加激活 | 融合实现正确性与中间张量显存 |
| L3 | 50 / 50 | 完整模型结构 | 除 PyTorch 外，部分题导入 `einops`、NumPy，需镜像依赖齐全 |
| L4 | 20 / 20 | Hugging Face 预训练模型题 | `transformers`、模型配置和权重缓存、原题加载兼容性，以及较大的模型和输入内存 |

**L4 的任务文件已生成，不表示权重已准备。** 例如 [GPT-2 原题](../../../../benchmark-sources/KernelBench/KernelBench/level4/16_gpt2_bs1_seq1023.py#L10) 调用 `from_pretrained` 加载权重，配置也从缓存加载。任务容器默认离线，没缓存时会在执行参考题之前失败。批量入口支持 `--hf-cache` 挂载包含 `hub/` 的准备缓存，本次没有额外下载这些权重，也没有验证 20 题的实际模型加载。因此 270 是准备覆盖，不能称作 270 个已保证可执行的 GPU 任务。

例如，KernelBench 的 L1 方阵乘法即使有四条成功 rollout，仍只覆盖一道题；本次生成 L1 100 道、L2 100 道、L3 50 道、L4 20 道任务后，准备覆盖变成 `270/270=100%`，策略闭环覆盖仍为 `1/270≈0.37%`，后续快速 GPU 检查另有 215 题通过。这两个百分比回答的是不同问题。

**全量原尺寸准备阶段与复现。** 批量入口默认复用原评分器和 EnvRouter 路径，不修改上游输入尺寸。小输入档需显式传入 `--bounded-inputs`，使用不同数据源身份，见下一节。新数据中的每题仍有独立任务路径、GPU 声明与评分配置。

```text
[A] 固定题库 → AST 检查 Model / get_inputs / get_init_inputs
                         ↓
[B] 每题调用既有准备器 → 独立任务、参考题指纹与 verifier
                         ↓
[C] manifest → 混合准备器 → tasks.parquet + agent.yaml
                         ↓
[D] PSRL 任务执行与独立 GPU 检查（见快速验证节）
```

图例：图里每个部分对应的源码位置。

| 编号 | 实现、输入与输出 |
|---|---|
| A | [discover](../../../examples/kernel_bench/prepare_suite.py#L13) 用 AST（抽象语法树，即 Python 解析后的代码结构）检查三项入口是否存在，不执行题目或分配张量 |
| B | [prepare_suite](../../../examples/kernel_bench/prepare_suite.py#L29) 按 level 与文件名区分题目，调用 [prepare](../../../examples/gpu_benchmarks/prepare.py#L66) 生成原有任务协议 |
| C | [批量清单发布](../../../examples/kernel_bench/prepare_suite.py#L65) 调用 [mix.prepare](../../../examples/gpu_benchmarks/mix.py#L12)，写出统一数据表和对应 agent 配置 |
| D | 使用下一节图中的原策略采样、独立评分、更新与清理路径；此准备命令只完成到 C，D 的逐题快速结果见下一节 |

跟着图读源码：A 枚举源题并检查接口，B 保留题目身份并逐题准备，C 在所有准备步骤结束后发布数据集；全量准备器不会自动跳过失败题；D 的快速验证结果在下一节登记。语法接口存在只能说明“长得像可评估题目”，无法发现 CUDA 算子缺失、模型太大或运行超时。这些代码如有理解上的问题，可以随时追问。

在项目 Python 环境中执行下列 CPU 准备命令。先设置 `BENCH_REPO` 为干净、固定版本的 KernelBench 源码目录，`TASK_IMAGE` 与 `VERIFIER_IMAGE` 为已准备镜像名，`TASK_PYTHON` 为镜像内可用的 Python 路径，`KERNEL_SUITE` 为尚不存在的输出目录。

```bash
python -m examples.kernel_bench.prepare_suite \
    --repo "$BENCH_REPO" --output "$KERNEL_SUITE" \
    --image "$TASK_IMAGE" --verifier-image "$VERIFIER_IMAGE" \
    --python "$TASK_PYTHON" --levels 1 2 3 4 \
    --timeout 900 --gpu-memory-mb 65536
```

使用宿主独立 Python 环境时另加 `--mount-python-prefix`。预训练模型题还需 `--hf-cache "$HF_CACHE"`，其中 `HF_CACHE/hub/` 包含所需完整配置和权重；空缓存或只有下载指针不满足离线执行条件。这里的 65536 MiB 和 900 秒是统一声明预算，不是逐题实测需求，更不是所有 L4 模型都能容纳的保证。不要直接启动 270 题全量长训练；先按 level 和依赖选择有界样本验证。训练入口继续使用公共启动器，将 `TRAIN_FILES`、`VAL_FILES` 以及 agent 配置路径指向生成产物，正式研究时另做训练/验证题目划分，不能默认拿同一表验证泛化。

本次数量统计来自 [inventory.py](../../../examples/gpu_benchmarks/inventory.py#L14)，逐题清单、固定源码版本和 FlashInfer 数据实体检查见[统计 JSON](../experiments/gpu-benchmarks/task-inventory-20261111.json)。在仓库根目录可复核：

```bash
python -m examples.gpu_benchmarks.inventory \
    --sources ../benchmark-sources --assets ../benchmark-assets \
    --kernel-suite ../benchmark-runs/kernel-library-20261111 \
    --output /tmp/science-task-inventory.json
```

上述 inventory 命令只读取本地文件，不执行 GPU。后续真实 GPU 检查与成功任务池见下一节，原统计 JSON 保留准备阶段条件，不将其指纹覆盖为后续结果。

**后续扩题顺序**：先修复未通过题目的具体形状约束和依赖，再有界补齐 L4 缓存、FlashInfer 张量及剩余定义；MLE 官方竞赛、Inference 完整场景与 RE 其余任务按资产和专属评分器逐项推进。接入数量与性能创新分开记录。

## Results：扩题与快速 GPU 验证

本轮按“尽量多接题、验证简洁快速”的要求执行，基于 `ab74120` 加主目录未提交修改。**快速验证不重复策略训练**，使用公开参考实现作为提交控制，经真实 PSRL EnvRouter、Harbor 容器和独立 GPU 评分执行。它验证任务接口与计算路径，不证明 agent 会优化，也不证明新题已经完成策略更新。

| 任务 | 本轮实际验证 | 发布到快速任务池 | 保留边界 |
|---|---|---|---|
| KernelBench | L1–L3 共 250 题均被尝试，22 题原尺寸通过、193 题小输入档通过 | **215 题**，其中 193 题使用独立 `kernel_bench_local` 身份 | 35 题失败/异常/超时，不进入通过池；20 道 L4 题未准备预训练资产，未运行 |
| FlashInfer | **30 个算子定义**，每个取一条原始随机输入 workload，官方 evaluator 全部 PASSED | **30 题**，覆盖 RMSNorm、融合归一化和 GEMM 等定义 | 是有界 workload 子集，不是全部 190 个定义、3267 条案例；不报告速度收益 |
| MLE 本地任务 | digits、Iris、Wine、乳腺癌四套真实数据，均完成 CUDA 训练、提交和隐藏标签评分 | **4 题**，其中新增 3 题 | `mle_bench_local`，不冒充官方 Kaggle 竞赛 |
| RE / Inference / autoresearch | 保留此前已完成接入的代表任务，本轮未扩成更多任务 | 不混入本轮新增的 249 题池 | RE 其余任务需要专属资产与评分器，Inference 缺四场景完整准备；autoresearch 本身是一个固定任务 |

KernelBench 分层结果如下，每道源题只计一次，原尺寸已通过题不再重测小输入档。

| 层级 | 原尺寸通过 | 小输入档通过 | 未通过快速检查 | 未运行 |
|---|---|---|---|---|
| L1，100 题 | 22 | 74 | 4 | 0 |
| L2，100 题 | 0 | 78 | 22 | 0 |
| L3，50 题 | 0 | 41 | 9 | 0 |
| L4，20 题 | 0 | 0 | 0 | 20，需预训练资产 |

**小输入档到底改了什么？** 原题的 ReLU 输入是 `4096 × 393216`，约 16 亿个 float32 元素，仅输入约需 6 GiB，还没有计算输出和比较开销。逐题完整尺寸检查很容易消耗十几秒。因此新增独立诊断档，只在参考代码副本中覆盖显式模块维度，例如把 `batch_size` 改为 2、`dim` 改为 32；原始题库文件不修改，容差和轴参数不改。具体覆盖值记录在逐题证据和任务 provenance 中。

这里的 `4096×393216` 是原题源码值，`2×32` 是本地验证档的配置，不是优化收益。小输入档通过不能证明原尺寸能运行、显存够用或候选更快。反过来，小输入报错也不能直接判定原题不可用，例如通道数缩成 32 后与原分组数 64 不整除，可能是诊断档破坏了原形状约束。35 个未通过样本中有 29 个异常、1 个失败和 5 个超时，保留具体记录，不称作“全部接入成功”。

```text
[Q1] 原题或显式小输入档 + 公开参考提交
                      ↓
[Q2] PSRL EnvRouter 准入 → Harbor → 真实 GPU / 官方评分器
                      ↓
[Q3] 逐题结果 + profile + 清理证据
                      ↓
[Q4] 只发布通过题 → 独立任务目录 + Parquet + agent 配置
```

图例：图里每个部分对应的源码位置。

| 节点 | 实现、输入与输出 |
|---|---|
| Q1 | [bounded_source](../../../examples/kernel_bench/quick_check.py#L13) 生成可追踪的维度覆盖；[任务准备](../../../examples/gpu_benchmarks/prepare.py#L66) 将小输入档写成 `kernel_bench_local` 任务，并附独立 verifier 与公开参考控制 |
| Q2 | [公共运行器](../../../examples/gpu_benchmarks/run.py) 走真实准入和容器路径；[Kernel evaluator](../../../examples/gpu_benchmarks/evaluate.py#L10) 调用上游正确性检查；[FlashInfer 批量检查](../../../examples/flashinfer_bench/quick_check.py#L8) 在一次持久评估器中检查多个定义 |
| Q2，MLE 分支 | [CUDA baseline](../../../examples/mle_bench/local_baseline.py#L10) 只使用训练集计算标准化参数、训练分类器；[grade](../../../examples/mle_bench/local_grade.py#L9) 独立比对隐藏标签 |
| Q3 | [持久检查 worker](../../../examples/kernel_bench/quick_check.py#L79) 在超时后回收所属进程组；[汇总器](../../../examples/gpu_benchmarks/summarize_quick_checks.py#L10) 保留原尺寸与小输入档身份，不把超时或缺资产记为成功 |
| Q4 | [混合任务准备器](../../../examples/gpu_benchmarks/mix.py#L12) 生成对应数据与 agent 配置；[奖励门槛](../../../examples/gpu_benchmarks/reward.py#L6) 要求本地 Kernel 评分携带本地档位标记 |

跟着图读源码：Q1 固定题目身份与输入档，Q2 执行独立计算与评分，Q3 逐条保存结果并回收资源，Q4 仅收录成功项。批量诊断复用容器和 worker 来避免重复初始化，不能等同于每题都重新创建过一次容器。另对数据池中的 `level1-19_ReLU` 做封装后标准任务验收：普通 runner 返回 `reward=1`、`correctness=1`、`local_evaluation=1`、`bounded_inputs=1`，无异常，所属容器清空、GPU 计算进程为空。首次验收发现 Harbor 奖励字段不能携带字符串，已改为数值标记，原失败记录与修复后通过记录均保留。输入档名称仍保存在任务 spec 与来源记录中。以上没有策略模型采样或优化器更新。这些代码如有理解上的问题，可以继续追问。

本轮实际可加载的数据入口为：

- [249 题 manifest](../../../../benchmark-runs/quick-verified-library-20261111/manifest.json)。
- [PSRL tasks.parquet](../../../../benchmark-runs/quick-verified-library-20261111/dataset/tasks.parquet) 与 [agent.yaml](../../../../benchmark-runs/quick-verified-library-20261111/dataset/agent.yaml)，必须成对使用。
- [完整逐题证据](../experiments/gpu-benchmarks/quick-validation-20261111.json)，包含未通过项、输入覆盖、代码指纹和资源回收结果。

池中数据源分别为 `kernel_bench` 22 行、`kernel_bench_local` 193 行、`flashinfer_bench` 30 行、`mle_bench_local` 4 行。生成时统一使用含 tmux、asciinema 的 agent 镜像层，独立 verifier 保留任务镜像。新执行节点仍须预热镜像与只读资产，不能只复制 Parquet。

复用该任务池时，在项目环境中设置 `TRAIN_FILES` 和 `VAL_FILES`，并使用生成的 agent 配置覆盖公共启动器。正式研究必须另划训练/验证题目，不把同一份数据当成泛化验证。单个任务的快速独立验收可执行：

```bash
python -m examples.gpu_benchmarks.run \
    --task "$CHECK_TASK" --output "$CHECK_OUTPUT" \
    --node-ip "$NODE_IP" --gpu "$GPU_INDEX" \
    --gpu-memory-mb 4096 --timeout 130
```

`CHECK_TASK` 为 manifest 中带 `solution/solve.sh` 的已准备任务路径（例如 `kernel-local/level1-19_ReLU`），`CHECK_OUTPUT` 为尚不存在的结果目录，`NODE_IP` 为实际执行节点地址，`GPU_INDEX` 为现场核对的空闲卡序号；4096 MiB 仅适合本地小任务，原尺寸题按实际声明设置。该命令默认使用 oracle 参考控制，不启动语言模型训练。22 个原尺寸任务沿用原 recipe，其中除方阵乘法外未附 oracle 脚本，应使用真实 agent 提交，不能直接套用默认 oracle 命令。

本轮新增及相关 CPU 回归 26 项通过，覆盖真实 Harbor 数值奖励 schema、失败元信息序列化、数据隔离和任务准备；本次修改文件的 Ruff、格式和 prose 检查通过。

本轮先尝试逐题冷启动，因开销过高受控停止并保留已通过项，再对未通过部分复用进程；原尺寸整批预算 240 秒，小输入整批预算 180 秒，每题有独立超时，工作进程首次初始化另给有界预算。FlashInfer 合并评估只做一次计时试验，不把测得时间用于性能结论。MLE 隐藏准确率分别为 digits 0.97556、Iris 0.94737、Wine 1.0、乳腺癌 0.99301。首轮超时、受控取消与后续通过结果同时保留，不以“运行器正常退出”代替逐题成功。

## 任务与框架如何串联

```text
[1] 真实资产 → 准备器 → Harbor 任务目录 + Parquet 数据表
                                 ↓
[2] 策略模型 v0 产生工具调用 ← 任务观察 ← [3] EnvRouter 准入、容器与 GPU 工作
        │                                          ↓ agent 提交
        │ 实际输入输出 token                [4] 独立 verifier 评分
        └──────────────────┬───────────────────────┘
                           ↓
                  [5] TITO 轨迹 + 奖励
                           ↓
                  [6] PSRL 计算 loss、执行优化器
                           ↓
                  [7] 发布参数版本 v1 → 回到 [2] 继续真实采样

每次任务退出（含异常）→ [8] 清理所属容器与进程 → 确认物理释放 → 归还资源
```

Harbor 负责驱动任务与验收。Parquet 是存放训练任务的数据表格式。EnvRouter 决定是否准许任务占用资源，租约记录这次占用。TITO 是 Token-In-Token-Out，即记录真实模型输入、输出 token 和生成概率的机制，用来把多轮工具交互转成训练样本。图中的 GPU 工作既可能发生在 agent 的实验里，也可能发生在独立 verifier 中。

图例：图里每个部分对应的源码位置。

| 编号 | 实现与职责 |
|---|---|
| 1 | [write_dataset](../../../examples/gpu_benchmarks/prepare.py#L47) 生成单任务数据，[混合任务 prepare](../../../examples/gpu_benchmarks/mix.py#L12) 生成数据表和对应 agent 配置 |
| 2 | [SciAccelAgentLoop.run](../../../examples/sciaccel_rl/agent_loop.py#L144) 创建真实模型会话，[GpuBenchmarkAgentLoop](../../../examples/gpu_benchmarks/agent_loop.py#L14) 强制使用受管 GPU 任务路径 |
| 3 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L275) 申请资源并执行 Harbor，[原生 Env](../../../psrl/workers/env_worker/harbor.py#L19) 和 [Docker 提供者](../../../psrl/workers/env_service/docker.py#L25) 提供两条执行路径 |
| 4 | [RE verifier](../../../examples/re_bench/verify.py)、[MLE grader](../../../examples/mle_bench/local_grade.py)、[Inference verifier](../../../examples/inference_bench/local_verify.py)、[FlashInfer verifier](../../../examples/flashinfer_bench/verify.py) 检查提交物，[compute_score](../../../examples/gpu_benchmarks/reward.py#L6) 执行奖励身份门槛 |
| 5 | [SciAccelAgentLoop.run](../../../examples/sciaccel_rl/agent_loop.py#L144) 取回实际会话轨迹并关联任务奖励 |
| 6 | [完整批次更新阶段](../../../psrl/trainer/ppo/strategies/full_batch.py#L110) 进入 [_update_actor](../../../psrl/trainer/ppo/ray_trainer.py#L3222)，调用训练 worker 更新策略 |
| 7 | [模型版本推送](../../../psrl/workers/ps/ps_manager.py#L1194) 登记新版本，[模型拉取](../../../psrl/workers/ps/ps_manager.py#L1269) 协调推理端取回参数 |
| 8 | [runner 收尾](../../../examples/sciaccel_rl/runner.py#L610) 与[独立验证运行器](../../../examples/gpu_benchmarks/run.py) 检查所属资源和物理释放 |

**跟着图读源码**：先读 1，确认数据行中的任务路径、后端和 `data_source`。再读 2、3，跟随“创建模型会话 → 等待准入 → 启动容器 → 多次模型与工具交互”。随后沿 4、5 看提交物怎样变成奖励和训练轨迹。6 更新的是策略语言模型，7 把新参数交回推理端，必须再看到真实采样才能闭环。8 是每个任务的退出分支，不是等所有训练结束才执行。模型、工具、评分和监控存在异步执行，不能把图误读成单线程调用顺序。这些源码如有理解上的问题，可以随时按编号追问。

CPU 教学实验可在仓库根目录运行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/explain_resident_command_lifetime.py
```

它用常驻 HTTP 服务和两个客户端模拟 3、8 的生命周期。本次实际运行观察到：两个客户端均退出，但服务仍持有状态，模拟归还被拒绝。停止服务后才允许归还。这个 CPU 模拟不调用 CUDA 或真实 EnvRouter，真实回收还检查容器和 GPU 进程。

## Results：已完成的 PSRL 闭环验证

以下数字来自 **2026-11-10 已完成的 GPU 实测**，不是本次文档修订重新启动的训练。框架代码基准为 `bc5665f`，验收收尾 HEAD 为 `ab74120`，两者已提交差异仅为会话归档文档。验收包含主目录未提交实现，尚未提交、推送或部署共享服务。策略模型为 Qwen3.5-4B，设备为 H20，策略训练、策略推理、环境任务和本地裁判使用独立 GPU。

| 验收环节 | 已取得的证据 | 能说明什么 |
|---|---|---|
| 真实初始 rollout 与任务评分 | 六类任务各采样两条，均至少有一条有效独立评分 | 不是用固定动作代替策略模型，也不是只检查容器存活 |
| PSRL 后训练 | 首批 12 条轨迹进入一次更新，11525 个有效训练 token，没有整条样本被屏蔽为零 token | 训练确实执行，loss 约 0.00335、梯度范数约 0.799 |
| 新参数回传 | 发布模型版本 1，推理端完成参数拉取 | 不是训练完仍使用旧权重 |
| 更新后 rollout | 六类任务均至少有一条版本 1 轨迹得到有效独立 GPU 正分 | 已覆盖“更新后继续做任务”的闭环 |
| 物理回收 | 所属主容器、verifier、sidecar 均为 0，GPU 计算进程为空，裁判已关闭 | 任务结束后的资源没有继续占用 |

本轮保存 20 条已结束采样，其中 17 条有效正分、3 条失败。验收达到“一次策略更新 + 六类新版本任务评分”后主动停止，余下 4 条采样取消，没有等待第二次优化器更新。训练子进程 `-15` 是这次受控停止的退出码，不是“两步训练正常结束”。

具体覆盖如下。分母是已结束的采样数，分子是其中取得有效正分的数量，不把被取消的采样计入成功。

| 任务 | 更新前 v0 | 更新后 v1 |
|---|---|---|
| KernelBench | 2/2 | 2/2 |
| autoresearch | 2/2 | 1/1 |
| MLE 本地竞赛 | 2/2 | 1/1 |
| Inference 本地服务 | 1/2 | 1/1 |
| RE 有界偏好训练 | 1/2 | 1/2 |
| FlashInfer RMSNorm | 2/2 | 1/1 |

这是一轮采用公开基线提示的功能验收，不是 agent 优化能力评测。它没有证明收敛、长期稳定性、跨任务泛化或调度性能提升。达到轮数或 token 上限时，已有提交物仍可能获得有效评分，逐条记录保留真实终止原因。

逐任务评分、版本、失败原因和代码指纹见[机器可读证据](../experiments/gpu-benchmarks/full-onboarding-20261110.json)，运行准备与详细条件见[完整接入报告](science-gpu-full-onboarding.md#results-与验证边界)。11-09 的 oracle 和 smoke 结果保留在 [FINDINGS 历史章节](../experiments/gpu-benchmarks/FINDINGS.md#2026-11-09-接入复核)，不覆盖为本轮策略训练结果。

## 具体任务例子与评分含义

**MLE：从训练到隐藏评分，再到策略更新。** 任务包含 1797 张真实的 8×8 数字图像，固定划分为 1347 个训练样本与 450 个测试样本。agent 能看到训练标签和测试图像，隐藏测试标签只交给独立 verifier。agent 调用 CUDA 分类器训练，提交每个测试 ID 的预测类别。verifier 先检查是否缺行、重复、非法类别或多列，再按 ID 与隐藏标签比较。参考实现答对 437 个，准确率为 `437/450≈0.9711`。

这个分类器的训练属于图中 3。PSRL 随后用 agent 的命令生成轨迹和该任务奖励更新策略模型，属于图中 6。版本 1 再次完成数字分类任务，才覆盖图中 7 回到 2 的过程。分类准确率不等于策略模型的学习提升，这轮没有做这种效果对照。

**KernelBench 与 FlashInfer：正确不等于更快。** 当前训练奖励主要检查计算结果是否正确。复制一个正确的公开基线也可以得到 1 分，因此正分只能说明本地正确性门槛通过，不能说明 agent 发明了更快的 kernel。速度是单独指标，若下一步研究性能反馈，需要先定义可靠的计时与比较规则。

**autoresearch 与 Inference：两种分数的含义不同。** autoresearch 的 BPB 是模型对每字节数据的预测编码代价，越低越好，当前奖励为 `1/(1+val_bpb)`。Inference 本地分要求输出与冻结参考一致，再用 `1/(1+批请求秒数)` 计分。前者包含模型质量，后者检查固定运行时的输出一致性与服务耗时，不能把它说成题目答对率。

## 失败处理与本轮边界修复

先区分三类问题：提交无效应得到任务零分，评分器或依赖损坏应报告验收异常，agent 不遵守任务协议则可能产生失败轨迹。把它们全部写成“bench 不可用”或“任务正常得零分”都会误导后续训练分析。

| 边界 | 当前处理 | 验证与限制 |
|---|---|---|
| RE 缺少训练模型或 `model.py` | 返回明确的无效提交零分，`judge_valid=0`，不冒充裁判评分 | 本次修复做 CPU 回归，原 GPU 记录中的缺失奖励异常保留，不改历史结果 |
| MLE CSV 多列、超额行、引号损坏、非法编码 | 拒绝提交并返回零分，隐藏标签错误仍作为基础设施异常暴露 | 本次 CPU 回归覆盖这些输入，合法隐藏标签评分不变 |
| 本地任务被标成官方数据源 | 混合准备器对已有 provenance 来源标记及 RE 裁判身份做一致性检查，拒绝明显错标 | 未标记来源的历史任务不能靠文件名推断官方身份，仍需核对准备资产 |
| Inference verifier 的前台 shell 已退出，但子进程仍存活 | 清理整个所属进程组并回收主进程，缺少脚本也能产生明确零分 | 本次 CPU 进程与接口回归，不代表重新做过 GPU 服务验证 |
| agent 试运行常驻服务却不结束 | 仍属于 agent 生命周期错误。实验必须显式取消受管服务作业，独立 verifier 再重启并评分 | 不自动把仍在运行的服务判为完成，不保证每次采样成功 |
| smoke、本地协议与官方协议混用 | smoke 保持零训练奖励，本地数据源与官方评分门槛分开 | 两条隐藏提示的有效本地评分不表示完成 512 条官方评估 |
| 运行器 `passed=true` 被解读成“优化成功” | 必须同时核对提交评分标记、真实轨迹、训练 step、模型版本和清理 | 单个布尔值不证明效果、速度或全题库可用 |

本次边界修复的代码不回写已完成 GPU 实验的镜像或资产。新准备任务会携带当前评分脚本，RE 的独立 verifier 镜像需要重建后使用。这些边界修复尚未重跑六类混合策略训练；11-11 的 GPU 快速检查仅覆盖上方扩题节列出的路径，不能把它写成六类新代码又完整训练了一遍。具体 CPU 检查结果和影响文件在[接入报告](science-gpu-full-onboarding.md#本轮边界修复与回归)登记。

## 可用性验收与复现

从[公共 recipe](../knowledge/gpu-benchmark-recipes.md)准备任务，再按[完整接入报告的运行入口](science-gpu-full-onboarding.md#运行入口)生成混合数据与 agent 配置。RE 使用[专属 recipe](../knowledge/re-bench-recipe.md)，不能把普通 runtime 镜像当成含隐藏资产的 verifier。

一次复现至少需要对齐以下条件：

1. **任务身份**：数据、隐藏划分、`data_source`、本地或官方模式一致，不能只更名来获得官方分。
2. **实际运行依赖**：镜像、模型、Python/CUDA、裁判和任务资产准备齐全。真实 Terminus agent 还需要 `tmux`，oracle 通过不能证明这一项齐全。
3. **资源预算**：明确节点和独立空闲 GPU，主容器、verifier、sidecar 的 CPU 与主存都计入，显存声明不是硬隔离。
4. **上下文预算**：包含多轮模型回复和工具输出。训练打包预算与推理窗口分别核对，不能只增加 `MAX_TOKENS_PER_GPU` 就认定窗口变大。
5. **停止与清理**：根据目标设步骤和超时，保存失败记录，到达结论立即收尾，核实容器和物理 GPU 释放。

此前边界修复针对已识别问题做 CPU 回归，30 项全部通过。11-11 扩题及数值奖励接口另有 26 项定向回归通过，二者存在重叠，不相加。修复后的混合准备器也实际接受了现有六类任务清单，能生成配套数据与 agent 配置；这一步不启动 GPU。回归脚本位于 [test_task_boundaries.py](../../../tests/examples/gpu_benchmarks/test_task_boundaries.py)，使用当前项目 Python 环境运行：

```bash
python -m pytest tests/examples/gpu_benchmarks/test_task_boundaries.py \
    tests/examples/gpu_benchmarks/test_local_mle.py \
    tests/examples/gpu_benchmarks/test_re_bench.py -q
```

## 相对旧路线图完成了什么

[2026-10-28 阶段报告](env-status-and-roadmap.md)把“闭合原生 science、接入更多 GPU 任务”列为下一步。当前进展已经超出仅有任务入口的阶段。

| 阶段目标 | 当前完成情况 | 后续仍需处理 |
|---|---|---|
| 丰富 GPU 任务 | 六类代表题覆盖算子、训练、服务、偏好反馈 | 扩大题目覆盖与跨环境复现，AIRS 等候选独立推进 |
| 接通 PSRL 策略训练 | 六类任务完成真实采样、策略更新与更新后任务评分 | 效果提升与长期稳定性需要单独实验，不由功能验收代替 |
| 统一环境资源管理 | 六类公共 recipe 使用 EnvRouter，FlashInfer 已进入公共 Harbor / PSRL 路径 | AIRS 仍有独立接线，不宣称全部后端已统一 |
| 缩短持卡范围 | KernelBench 有显式 command verifier，FlashInfer 有原生 command 探针 | 本轮混合策略验收使用 episode，不能外推为所有任务 command 借还通过 |
| 自动模式选择与状态让位 | 已有机制分析与成本实验 | 通用自动卸载恢复及收益仍未完成 |
| 形成研究贡献 | 已具备真实任务、闭环和可观测证据 | 下一步要证明超越强工程基线的收益，不以接入数量充当创新 |

## 后续研究方向与否证条件

**优先研究：一组候选怎样更快获得可靠反馈。** agent 往往需要比较几个候选，等评分齐备后才能决定下一轮。共享 GPU 可以提高并发，但也可能改变测得的快慢顺序。研究目标应同时约束反馈误差和整组完成时间，而不是只看 GPU 是否更忙。

教学假设：四个候选独占评估各需 10 秒，串行共 40 秒。两两并行后，每组受争用影响变成 16 秒，总共 32 秒，看起来节省了时间。但若真实更快的候选 A 被干扰得比 B 更慢，agent 就会选错。因而必须同时测“整组多久完成”和“比较结果是否可信”。这些数字是教学假设，不是本轮性能实测。

| 优先级 | 方向与适合任务 | 最小有意义的对照 | 否证条件 |
|---|---|---|---|
| **P1** | 反馈可靠性与整组时间联合优化，KernelBench + FlashInfer | 固定候选，比较独占、直接共享、测量窗口隔离与有界重测，记录排序错误和总耗时 | 误差没改善，或控制成本使整组更慢，就不能声称有效 |
| **P2** | 准备阶段与 GPU 执行协同，FlashInfer + KernelBench | 与单进程批量复用、静态同机放置、现有动态池比较，分开统计初始化、编译、拷贝、计算 | 若收益已被简单进程复用解释，就不是新调度策略的贡献 |
| **P3** | 非计时评分任务的预算共享，autoresearch + MLE | 固定任务和实际工作量，记录完成时间、失败、训练步骤、质量与整组等待 | 少训练或降低质量换来的结束更快，不能算等价提速 |
| **P4** | 等待裁判时有代价的状态让位，RE + 常驻服务 | 先测真实等待分布与释放、恢复成本，再比较常驻、全卸载与阈值策略 | 等待短于恢复成本，或拖慢关键路径，停止该分支 |

已有[FlashInfer 批量成本实验](../experiments/gpu-benchmarks/flashinfer-batching.md)说明进程与准备复用是强工程基线。不能把这类收益全部归因于资源账本，也不能仅凭“共享 GPU”或“卸载恢复”声称新颖性。相关工作和更细的研究边界见 [command 调度研究](../research-proposals/command-scheduling-research.md)及 [KREX 分析](../literature-reviews/krex-region-exclusivity.md)。本报告提出可验证的问题，不宣称已证明新的调度算法。

## 建议的推进顺序与交流问题

1. **先固定任务与协议。** 保留这六类最小代表题，明确版本、评分和资源生命周期。新题进入实验前先通过独立 verifier，防止把数据或依赖错误归因于调度器。
2. **优先选 P1 做最小实验。** 固定同一组真实候选、策略版本和工作量，把独占测量、现有共享方式、简单批量复用作为基线，确定是否存在足够大的等待和评分干扰。
3. **存在可观收益再实现策略。** 决策依据使用任务比较关系、测量状态与资源成本，分别报告初始化、排队、计算和清理。没有触发争用或等待的实验不能证明调度价值。
4. **最后讨论状态卸载。** 只有真实等待足够长且状态能物理释放，才投入恢复机制。优先独立 GPU，再评估经过确认的共享余量，不默认打断模型推理。

对外交流时可以围绕三个问题展开：我们最想缩短哪一段反馈等待，现有强工程基线还留下多少空间，以及怎样证明并发后的奖励仍足以支持 agent 正确选择下一步。任务接入已经提供实验基础，后续创新需要由这些问题的对照结果支撑。
