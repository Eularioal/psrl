# FlashInfer 评估实验：进程复用成本与候选筛选

> **核心结论**：第一轮比较逐条 command、仅合并租约、单进程批量评估，六组均完成。单进程批量模式的两轮均值约为逐条命令的 1/4.86，主要体现进程与准备复用，账本借还仅几十毫秒。第二轮完成四个官方候选的 104 次评估：固定四轮、固定八轮与二阶段筛选选出相同赢家，尚不支持复杂筛选的额外收益。小 batch 的第三、第四名出现与执行顺序相关的互换，原因尚未确定。两轮都走真实 Env，没有实现比较关系感知调度，也不能称为 Env ON/OFF。

## 目录

- [先看懂实验具体在做什么](#先看懂实验具体在做什么)
- [问题与停止条件](#问题与停止条件)
- [协议与对照](#协议与对照)
- [执行链路与复现](#执行链路与复现)
- [Results](#results)
- [研究判断与边界](#研究判断与边界)
- [候选筛选后续实验](#候选筛选后续实验)

## 先看懂实验具体在做什么

本节于 2026-11-10 对照本地主目录实验脚本补充讲解，解释 2026-11-09 的历史实验，没有新增 GPU 测量。两轮的实际代码版本分别见后文，不能把本次文档核对日期当作重跑日期。

```text
[1] 准备固定代码和输入规格
    同一个 RMSNorm 运算，两种输入大小，若干种实现
              ↓
[2] 驱动脚本通过真实 Env 提交 GPU command
              ↓
[3] 容器内运行官方评估器
    生成输入 → 检查计算结果 → 对正确实现重复计时
              ↓
[4] 保存结果：正确性、算子耗时、外层命令耗时
              ↓
[5] 进程退出、归还预算、检查物理 GPU 释放
              ↓
[6] CPU 分析结果表，比较执行方式或筛选规则
```

图例：图里每个部分对应的源码位置。

| 步骤 | 对应函数与源码 | 数据或状态变化 |
|---|---|---|
| 1 | [prepare main](../../env-service-verification/scripts/prepare_flashinfer_batching.py#L12) | 筛出输入规格，加入数学参考对照 |
| 2 | [experiment](../../env-service-verification/scripts/flashinfer_batch_experiment.py#L14) | 生成命令，调用 `environment.exec_gpu`，记录整组时间 |
| 3、4 | [worker main](../../env-service-verification/scripts/flashinfer_batch_worker.py#L9) | 导入库、调用官方 Benchmark，保存每项评估结果 |
| 5 | [experiment](../../env-service-verification/scripts/flashinfer_batch_experiment.py#L14)、[run](../../env-service-verification/scripts/flashinfer_env_probe.py#L22) | 每组查账本与进程，最后清理本次环境 |
| 6 | [analyze](../../env-service-verification/scripts/analyze_flashinfer_racing.py#L11) | 第二轮已采集数据 → 三种决策规则的离线结果 |

跟着图读源码：第一轮从 `run → experiment → exec_gpu` 进入容器，容器调用 worker 的 `main → Benchmark.run_all`，返回后 experiment 核对结果与释放，最终 run 收尾。第二轮走同一个 Env 入口，但容器改为 [collect](../../env-service-verification/scripts/flashinfer_racing_probe.py#L12) 持续执行 13 个评估块，采集结束后单独调用 analyze。Ray 与官方工作进程存在并发，图示是控制依赖，实验没有同时提交多个竞争候选。上述代码有不清楚的地方，可以随时追问。

**先理解被测对象。** RMSNorm 是模型中的归一化运算，会按一行数值的均方根调整幅度，再乘权重。本任务每行有 2048 个数。`batch=1` 输入有一行，`batch=16254` 有 16254 行。不同候选是完成同一计算的不同程序，比较谁算得正确、谁用时少。候选不是四个同时运行的大模型，名称里的 GPT-5、o3、Gemini 指官方发布代码的生成来源。这两轮都没有让 agent 现场写代码，也没有训练完整模型。

官方协议用随机张量提供数值输入，使用真实 GPU 运行与官方正确性检查。这不等于下载了真实用户文本，或运行了完整 Qwen 模型。评估器会把候选输出与参考结果按容差比较，通过后记录性能。一次外层“评估”内部还包含官方预热与多次执行计时，所以 104 次评估不表示只启动了 104 次 GPU kernel。

**第一轮只换执行方式。** 固定四项工作：FlashInfer 实现算小输入、参考实现算小输入、FlashInfer 实现算大输入、参考实现算大输入。三种方式都完成这四项，而且都按顺序执行：

```text
A：申请 → 启动进程评估一项 → 退出归还，重复四次
B：申请 → 新进程评估一项并退出，重复四次 → 最后归还
C：申请 → 启动一个驱动进程，批量评估四项 → 退出归还
```

“进程”是运行着的程序实例。把四次 `python evaluate.py` 写在一条 shell 命令里，仍然会启动四个新 Python 进程，不能自动保留上一次的解释器或 CUDA 状态。C 才是通过同一次官方 Benchmark 批量调用复用工作进程和准备状态，内部仍可能有官方子进程，不能理解成整个系统只有一个进程。

每种方式重复两次，一共六组、24 项评估。每组时间从第一条 GPU 命令提交前开始，到最后一条返回且完成归还为止，包含导入、初始化、评估和进程退出，不含组后的额外遥测核对。它不是单个算子的运行时间。结果分别为 125.55、102.23、25.86 秒，说明复用准备状态对这个短实验很有帮助。不能把约 4.86 倍的时间比解释为算子本体加速或 EnvRouter 新算法收益。

**第二轮先采集，再分析。** 将候选换成四个官方实现，仍测两个输入大小。一次评估块包含 `4 × 2 = 8` 个候选与负载组合。先跑一个不参与筛选的预热块，再跑十二个正式块，所以总数是 `(1 + 12) × 8 = 104`。这里额外排除一个外层预热块，不代表关闭了每次评估内部的官方预热。

采集阶段实际把所有组合都跑完。分析阶段再问：如果只允许自己看其中部分数据，选出的最快实现会不会改变？对每个负载分别处理，前四轮用于初选，第五至第八轮用于追加，最后四轮留作核对，不能反过来参与初选。

| 分析规则 | 看哪些数据 | 使用的观测数 |
|---|---|---:|
| 固定八轮 | 四个实现，前八轮全部使用 | 32 |
| 筛选再细化 | 前四轮每个都看，选两个更快者，再看它们后四轮，最终按这两个的前八轮中位数选优 | 24 |
| 固定四轮 | 四个实现，只看前四轮 | 16 |

“中位数”是把测得的耗时排序后取中间位置的值，偶数个值则取中间两个的平均数，用来减少个别极端值的影响。“留出核对”是用没参与选择的数据再检查结果，本次它只是带噪声的参照，不是绝对真值。

三个规则在两个负载上都选中 FlashInfer，因此尚未证明复杂筛选优于简单少测。24 相对 32 少用 25% 的观测，16 相对 32 少用 50%，这些是 CPU 回放计算出的数据使用量，不能写成实机已经省下相应 GPU 时间。

小输入的第三、第四名各有六次领先对方，且与执行顺序相关。它说明重复测量可能影响完整排名，但本轮未分开控制负载顺序和候选顺序，还没有证明原因。只找第一名与要求所有候选可靠评分，也不能使用同一种验收标准。

**为什么先做这些。** 第一轮排查是否只是重复启动太贵，避免把进程复用包装成新调度算法。第二轮排查是否值得追加复杂的测量分配策略。两轮都是 YARN 启发方案的前置实验，尚未实现评分协调者与 Env 的联合调度，没有竞争任务、动态多 GPU 放置或 episode 卸载实验。

可以在仓库根目录用 CPU 重读保存的结果：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/explain_flashinfer_results.py
```

脚本只读已保存的 JSON，输出三种方式的平均组耗时、104 的构成及各规则的观测数和赢家，不重新测 GPU，也不模拟性能。预期三种均值为 125.55、102.23、25.86 秒，两个负载的三种规则赢家均为 FlashInfer。

## 问题与停止条件

先回答：若把多个候选评估放进一个 GPU command，省下的是准入和归还，还是 Python、CUDA 上下文及评估器初始化？两者对研究路线的含义不同。前者可能要求调度协议优化，后者首先要求建立进程复用的强工程基线。

每组固定四个逻辑评估，三种执行策略各两轮，合计 24 个 workload-candidate 结果。按 A、B、C、C、B、A 顺序执行，第二轮反向候选顺序。实验主体 900 秒期限，监督进程 1200 秒硬期限含初始化与清理，不因结果不漂亮而增加轮数。全部完成或遇到失败后收尾，不追加训练。

## 协议与对照

| 项目 | 条件 |
|---|---|
| 硬件 | 一张现场确认空闲的 H20，单容器 4 CPU、16 GiB 主存 |
| 数据 | 官方 FlashInfer-Trace 的 RMSNorm h2048，batch=1 和 16254，随机输入协议，保留原始 workload UUID |
| 候选 | 官方 FlashInfer baseline，以及把未修改的官方 PyTorch 数学参考封装成的本地研究对照 |
| 代码 | PSRL `0326d46` 加运行时主目录代码的冻结副本，记录文件哈希，不能称为纯 HEAD |
| 评估器 | 官方 `BenchmarkConfig.default`，只改任务筛选和每候选超时，不改容差、预热或计时循环 |
| GPU 预算 | unit 按整卡容量配置，单任务申报 8 GiB，二者含义不同 |
| 计时边界 | 整组从第一条命令提交到最后一条确认归还，另记进程导入、评估和每次准入归还 |

| 模式 | GPU command 数 | 新 Python 主进程数 | 作用 |
|---|---:|---:|---|
| `separate` | 4 | 4 | 每个候选和负载单独执行 |
| `one_command_fresh` | 1 | 4 | 只合并一次资源持有窗口，候选仍逐次启动进程 |
| `one_process` | 1 | 1 | 一次官方 Benchmark 批量评估，复用官方工作进程和内部状态 |

每组逻辑工作量相同，不表示内部重复准备工作相同。官方批量模式可复用参考侧准备、输入与工作进程，这正是需要识别的成本。此对照不能把全部减少的工作归因于 CUDA 上下文，更不能归因于 Env 调度算法。

全部候选在组开始时已就绪，没有模拟到达时间，也没有其他等待者。缓存可延续，第一轮可能包含冷启动，反向执行只能减少部分顺序混淆，不能消除环境漂移。两次重复不足以报告统计显著性或可靠的尾延迟。

## 执行链路与复现

```text
[1] 官方任务准备：两种形状、两个固定候选
       ↓
[2] 原生 Env 探针：私有 Ray → EnvRouter → EnvWorker
       ↓
[3] 实验驱动：三策略、反向两轮 → exec_gpu
       ↓
[4] 官方评估器：完整性、正确性、计时
       ↓
[5] 每组核对账本与物理释放 → 最后关闭容器和租约
```

图例与跟着图读源码：

| 步骤 | 输入、输出和实际入口 |
|---|---|
| 1 | [prepare main](../../env-service-verification/scripts/prepare_flashinfer_batching.py#L12) 从官方源筛出两个 UUID，并封装参考对照。不会把本地封装冒充官方 agent 候选 |
| 2 | [run](../../env-service-verification/scripts/flashinfer_env_probe.py#L22) 现场预检、私有 Ray 身份核对、会话准入、容器 CUDA 预检 |
| 3 | [experiment](../../env-service-verification/scripts/flashinfer_batch_experiment.py#L14) 调用真实 `exec_gpu`，保存每次命令的时间与返回记录 |
| 4 | [worker main](../../env-service-verification/scripts/flashinfer_batch_worker.py#L9) 调用官方 Benchmark，检查完整 UUID/候选集合和所有正确性状态 |
| 5 | experiment 检查每组 GPU 回到空闲，run 的 finally 清理本次 actor 标签的容器、租约和私有节点服务 |

Ray actor、心跳和评估工作进程具有并发，图示表示控制依赖。这里复用现有生产接口，不引入新的调度算法。代码有不清楚的地方可以继续追问。

在已激活项目环境且完成节点预检后，于实际执行节点的仓库根目录运行：

```bash
read -r -p 'FlashInfer source checkout: ' FIB_SOURCE
read -r -p 'Official trace checkout: ' FIB_TRACE
read -r -p 'New output directory: ' FIB_OUTPUT
read -r -p 'Execution node IP: ' FIB_NODE
read -r -p 'Idle GPU index: ' FIB_GPU
read -r -p 'Prepared image: ' FIB_IMAGE
FIB_SCRIPTS=docs/env-worker-deep-dive/env-service-verification/scripts
python "$FIB_SCRIPTS/prepare_flashinfer_batching.py" \
  --upstream "$FIB_SOURCE" --trace "$FIB_TRACE" --output "$FIB_OUTPUT"
PYTHONPATH="$PWD" python "$FIB_SCRIPTS/flashinfer_batch_experiment.py" \
  --root "$FIB_OUTPUT" --upstream "$FIB_OUTPUT/upstream" \
  --node-ip "$FIB_NODE" --gpu "$FIB_GPU" --image "$FIB_IMAGE"
```

输入依赖与容器系统库要求沿用[已有探针说明](../../literature-reviews/ai4infra-performance-benchmarks.md#运行与观察)。`groups.json` 按已完成组落盘，`status.json` 记录最终状态。正式长期运行还应使用外部监督进程处理初始化或同步调用卡住的情况，本轮保留了该监督日志。

## Results

| 策略 | 第一轮整组时间 | 第二轮整组时间 | 两轮均值 | 准入与归还合计的两轮均值 |
|---|---:|---:|---:|---:|
| `separate` | 147.58 s | 103.51 s | 125.55 s | 74.45 ms |
| `one_command_fresh` | 107.83 s | 96.64 s | 102.23 s | 19.50 ms |
| `one_process` | 24.04 s | 27.68 s | 25.86 s | 19.72 ms |

2026-11-09 全部六组完成，24 个 workload-candidate 结果全部通过官方正确性检查，各组结束时 GPU 账本无活动命令、显存预算归零，物理 GPU 无计算进程。最终本次容器为 0，GPU 显存回到约 0.19 MiB，清理无异常。结果和源码哈希见[可移植证据](flashinfer-batching-evidence.json)。

单进程批量模式相对逐条命令的两轮均值比为 4.86 倍，相对只合并租约为 3.95 倍。这是本条件下的组完成时间比，不是 kernel 本体加速，也不是稳定吞吐或置信区间。第一轮逐条命令明显慢于第二轮，存在冷启动和运行环境变化。

准入与归还记录只有几十毫秒，无法解释几十秒的差异。单进程模式同时复用解释器、CUDA 和官方 worker，并减少部分重复准备，不能把全部收益单独归因于 CUDA 上下文。只合并租约的差异也不能全部归因于少调用 Router，顺序、进程启动和系统状态都有混淆。

本轮两候选在所有组的两个形状上均由 FlashInfer 实现更快，但它与数学参考差距大，不能据此证明接近候选的排序可靠性。计时使用官方缺少 CUPTI 时的 CUDA Event 后备路径。输入按官方随机协议产生，未要求跨进程张量逐元素相同。完整容差、计时配置与逐项 latency 保存在结果中。

第一轮工具启动曾在准备脚本尚未落盘时失败，未开始 GPU 工作，日志保留。确认资产完成后重新启动正式实验，没有重跑已完成的 GPU 组。新增脚本 Ruff 检查通过，文档链接检查通过。全库 pre-commit 在隔离快照运行，因已有 E501 和 prose-style 问题失败，未改写主工作区的无关文件。

## 研究判断与边界

本轮结论是先把进程与评估准备复用设为强基线，暂不以“减少准入次数”或“批量评估”单独立项。比较关系、自适应测量和放置是否有额外价值仍未验证。结论明确后已停止本轮 GPU 任务，没有扩大压力或训练实验。

2026-11-09 阅读 [KREX 正文](https://arxiv.org/html/2609.30057v1)后，进一步确认其已有持续上下文复用、候选排名翻转分析，并讨论短 kernel 的原生重复波动。[GPU Forecasters](https://arxiv.org/html/2605.31464v1)也已采用批量候选评估。因此本轮即使看到明显加速，也只能先当作工程基线收益。

本轮不包含竞争注入、接近性能的优化候选、采样自适应、异步到达或 agent 决策，不验证独立创新，也不验证排名可靠性改善。所有候选都通过是正确性结论，不等于测量误差满足任意概率约束。下一阶段是否值得做，取决于超过这个进程复用基线的剩余成本与真实反馈误差。

## 候选筛选后续实验

**结论**：2026-11-09 的第二轮研究验证，在复用评估进程后没有发现必须采用复杂候选筛选的收益证据。最优候选稳定，但小 batch 的第三、第四名随测量历史相关条件变化。后者值得作为受控研究问题保留，不能把前者的省测量次数当成新调度算法收益。

这次使用 `bc5665f` 加运行时主目录代码的冻结副本，与上文 `0326d46` 成本实验分开。生产调度器没有为本次实验修改。原生 EnvRouter、EnvWorker 与 command 借还沿用上文实际链路，单张空闲 H20、4 CPU、16 GiB 主存、整卡 unit、8 GiB 单任务申报。

在两个相同的官方负载形状上，候选改为官方 FlashInfer baseline、Gemini 2.5 Pro、GPT-5、o3 三个官方发布的 Triton 解法。三个解法原声明面向 B200，源码和声明均未修改。本次验证它们在 H20 上可运行且正确，不是官方 B200 成绩，也不证明全部 B200 任务可移植。

协议在运行前写入 manifest：先跑一轮预热，再测 12 轮，每轮包含两个负载乘四个候选，合计 104 次正确性和性能评估。轮换候选顺序并交替负载顺序，保留官方默认容差、预热和计时配置。实机采集使用一次 command 和持续评估进程，不部署自适应调度器。

离线比较三种决策，最后四轮只用于核对，不能拿它们决定淘汰谁：

| 策略 | 决策使用的数据 | 每个负载使用的候选评估次数 | 与留出四轮的赢家一致 |
|---|---|---:|---|
| 固定八轮 | 四个候选均测八轮 | 32 | 两个负载均一致 |
| 筛选再细化 | 前四轮选最快两个候选，仅这两个使用后四轮 | 24 | 两个负载均一致 |
| 固定四轮 | 只看前四轮，选其中位数最快者 | 16 | 两个负载均一致 |

筛选相对固定八轮少用 25% 的决策观测，但固定四轮少用 50%，也得到相同赢家。因此本例不能证明筛选策略优于简单固定预算。这里的计数是离线回放，实机实际执行了全部 104 次，不能说已经节约了 25% 的 GPU 时间。留出中位数本身也有噪声，不是无误差真值。

| 候选 | batch=1，12 轮中位数 | batch=16254，12 轮中位数 |
|---|---:|---:|
| FlashInfer | 5.619 微秒 | 44.104 微秒 |
| Gemini 2.5 Pro Triton | 5.811 微秒 | 51.707 微秒 |
| GPT-5 Triton | 12.104 微秒 | 100.765 微秒 |
| o3 Triton | 12.299 微秒 | 176.301 微秒 |

FlashInfer 在两个负载的全部 12 轮均排名第一。batch=1 时 GPT-5 和 o3 的先后次序各出现六次，不能用整体中位数掩盖这种变化：奇数轮先测大 batch 再测小 batch，此时 o3 比 GPT-5 快；偶数轮小 batch 先测，此时 GPT-5 比 o3 快。候选轮换也随轮次变化，设备频率和温度未做因果控制，因此不能断言是缓存、时钟、候选顺序或带宽竞争中的哪一个原因。这里也没有注入竞争负载。

例如，只选第一名时这两个落后候选的互换没有改变决策；要求完整排名时则无法忽略。要求 GRPO 数值奖励时，还要另测组内优势偏差，不能把落后候选互换直接等同于优势符号翻转。普通候选筛选适合选优目标，不能直接用于删除训练所需的完整组反馈。

时间分解：包含初始化与清理的整次试验约 102.09 秒，采集 command 约 65.19 秒，其中首次预热评估 22.87 秒，后续 12 轮合计 4.61 秒，每轮约 0.37 至 0.49 秒。后续正式测量约占该 command 的 7.1%，这是本轮成本占比，不是提前停止策略的实测加速。对于持续运行很多轮的搜索，初始化会被摊薄，不能把 7.1% 推广为长期比例。

所有 104 次评估通过，command GPU 配额归零，最后没有本次容器和 GPU 计算进程，显存回到约 0.19 MiB。未放松校验，官方因缺少 CUPTI 使用 CUDA Event 后备计时。完成后停止，不为追求正结果扩大负载或次数。完整数据、协议与源码哈希见[后续实验记录](flashinfer-screening-evidence.json)。

新增三个脚本通过 Ruff 检查与格式校验，文档本地链接通过检查。全库 pre-commit 在一次性快照运行，因既有 E501、格式及 prose-style 问题失败，未将自动改写带回主工作区。本轮仅新增实验与离线分析，没有改变生产接口、配置或资源生命周期，因此无需改写 `current-design.md` 的能力说明。

准备、采集与分析的源码路径如下，接在上文流程图的第 1、3、4、5 步，不是新增生产调度链路：

| 入口 | 输入与输出 |
|---|---|
| [prepare_flashinfer_racing.py](../../env-service-verification/scripts/prepare_flashinfer_racing.py) | 官方 checkout → 保留 UUID、原始候选和固定筛选协议的目录 |
| [flashinfer_racing_probe.py](../../env-service-verification/scripts/flashinfer_racing_probe.py) | 原生 Env → 一次 command 内 13 轮官方评估 → 完整性、账本与物理清理 |
| [analyze_flashinfer_racing.py](../../env-service-verification/scripts/analyze_flashinfer_racing.py) | 已完成实测 → 固定预算与筛选离线回放、留出比较、完整排名次数 |

复现时使用上文交互输入的 `FIB_SOURCE`、`FIB_TRACE`、`FIB_NODE`、`FIB_GPU`、`FIB_IMAGE`，为本轮指定尚不存在的新输出目录，然后执行：

```bash
read -r -p 'New screening output directory: ' FIB_OUTPUT
FIB_SCRIPTS=docs/env-worker-deep-dive/env-service-verification/scripts
python "$FIB_SCRIPTS/prepare_flashinfer_racing.py" \
  --upstream "$FIB_SOURCE" --trace "$FIB_TRACE" --output "$FIB_OUTPUT"
PYTHONPATH="$PWD" python "$FIB_SCRIPTS/flashinfer_racing_probe.py" \
  --root "$FIB_OUTPUT" --upstream "$FIB_OUTPUT/upstream" \
  --node-ip "$FIB_NODE" --gpu "$FIB_GPU" --image "$FIB_IMAGE"
python "$FIB_SCRIPTS/analyze_flashinfer_racing.py" \
  --root "$FIB_OUTPUT" --output "$FIB_OUTPUT/analysis.json"
```

采集 command 限时 420 秒，Env 主体 600 秒，本轮另设 900 秒外部监督期限。分析步骤只使用 CPU。候选逐步淘汰属于已有资源分配思路，可对照 [Jamieson 与 Talwalkar，AISTATS 2016](https://proceedings.mlr.press/v51/jamieson16.html)。本轮简单二阶段筛选不是该论文完整实现，也不继承其理论保证。
