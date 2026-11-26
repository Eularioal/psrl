# 基准调研：AI4Infra性能反馈与验证

> **核心结论**：性能反馈研究要同时回答“结果是否可靠”和“整批是否更快”。共享 GPU 的竞争可能把较快候选测成较慢，导致 Agent 朝错误方向优化。KernelBench 与 FlashInfer 适合先验证算子测量，InferenceBench 与 SWE-Serve 性能子集用于检验服务和工程层面的适用性。

例如真实耗时为 10 ms 和 11 ms 的两个候选，若前者测量时遭遇 20% 干扰，就会被记录为 12 ms，从而错选 11 ms 的候选。这是教学数字。研究目标是降低这类错误，同时计入独占、重复测量和排队的完整成本。

本文的原生 Env 实验基于 2026-11-05 的 `0326d466` 加运行快照，保留原配置与证据。此后 FlashInfer 和 FastKernels 已有本地接入与有界验证，具体完成范围查[当前路线图](../milestones/02-Science现状_任务覆盖与接入路线.md)。下面的候选顺序表示性能反馈研究的使用顺序，任务可用状态以该入口为准。[非计时场景](../research-proposals/02-研究场景_非计时评分的GPU任务.md)则按功能与质量目标单独验收。


## 目录

- [现有 benchmark 够做什么](#现有-benchmark-够做什么)
- [新增候选如何选择](#新增候选如何选择)
- [为什么可靠反馈会改变调度目标](#为什么可靠反馈会改变调度目标)
- [实验怎样开展](#实验怎样开展)
- [本次原生 Env 验证链路](#本次原生-env-验证链路)
- [跟着图读源码](#跟着图读源码)
- [运行与观察](#运行与观察)
- [Results 与验证边界](#results-与验证边界)
- [官方资料与固定版本](#官方资料与固定版本)

## 现有 benchmark 够做什么

AI4Infra 在这里指让 AI agent 改进计算基础设施，例如生成 GPU 算子、优化推理引擎或调整服务参数。只有评分依赖执行时间、吞吐等资源敏感指标的任务，才直接对应本文的研究问题。使用 GPU，但只以功能正确性或模型精度评分的任务，不能自动充当性能反馈实验。

| 现有任务 | 可以支撑什么 | 尚不能据此声称什么 |
|---|---|---|
| KernelBench | 同一数学问题的多个算子候选、正确性检查、运行时间测量，适合初期干扰与重测实验 | 当前公共 recipe 默认 reward 是正确性，不能说已经用 speedup 做 RL 奖励或实现反馈保真调度 |
| InferenceBench | 官方协议含质量门槛与服务性能指标，适合服务级外部验证 | 既有 resident smoke 仅检查常驻执行链路，不能替代完整官方质量门槛、请求协议与评分验收 |
| autoresearch、MLE 类 | 模型训练和有限预算优化，可作为调度工作负载 | 限时内训练工作量随竞争变化，分数变化可能是实际训练更少，不只是测速误差 |
| RE-Bench、PostTrainBench 候选 | 长状态、外部裁判等待、训练质量反馈 | 本地裁判可用或 GPU 占用高，不等于适合性能排名可信性研究 |

因此，**当前足够验证“问题是否存在”，还不足以证明“对生产 AI4Infra 广泛有用”**。后者至少需要两种不同层次：短算子测速与有模型状态、队列或请求并发的服务测量。还要有多个候选及不同差距，不能只跑一个已知正确的官方 baseline。

## 新增候选如何选择

| 优先级 | 候选及任务 | 为什么值得加入 | command 与 H20 边界 |
|---|---|---|---|
| 第一 | [FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench)，从生产推理捕获的算子定义和 workload | 带形状、数据类型、参考实现、候选与官方评估器，能补充 KernelBench 的任务来源，并覆盖不同 batch 形状 | 单次评估进程可以完整包入一个 command。本次 RMSNorm baseline 明确列 H20，但不能把这推广到所有低精度或 Blackwell 专用任务 |
| 第二 | [SWE-Serve](https://github.com/NVIDIA/swe-serve) 的 speedup 子集，修改实际 SGLang 工程代码 | 从孤立函数走到真实工程修改，适合检查方法是否仍有价值 | 必须逐题核验。官方 GPU 平台与 H20 不同，硬件专用测试不能只改设备名字就算兼容 |
| 服务级验证 | 当前 [InferenceBench](https://github.com/aisa-group/InferenceBench) 的正式任务，或 FastKernels 的端到端负载 | 补齐模型初始化、请求调度、缓存和服务时延，验证跨层次适用性 | 常驻服务更适合 episode，或将服务启动、预热、测量、关闭包为一个完整 command。不能仅在客户端请求结束时归还服务仍持有的 GPU |
| 储备 | [FastKernels](https://github.com/Snowflake-AI-Research/fastkernels)，单算子、融合算子、层/块、端到端四层 | 提供从局部改进到模型级效果的桥梁 | 已有 29/458 个本地目标快检与 Linear 代表闭环，整库及官方完整协议仍待验收，详见当前路线图 |
| 暂缓 | [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench/blog/introducing-sol-execbench) | 生产模型来源与硬件性能上界视角有参考价值 | 官方 B200 条件不能直接转换成 H20 官方成绩，本轮不作为首个接入目标 |

SWE-Serve 不能整个算作“性能评分 benchmark”。在已审计的固定版本中，GPU 任务的 speedup 子集只有三题：`ss-sgl23856-use-torch-torch-mm-for-deepseek-v3-2-indexer`、`ss-sgl25265`、`ss-sgl25311-perf-mla-tma-bulk-store-set-mla-kv-buffer-up`。其余 GPU correctness 任务属于另一条研究路线。任务清单审计见[既有说明](../research-proposals/02-研究场景_非计时评分的GPU任务.md)。本轮未运行这三题的完整评分器。

三个名字相似的套件并不等于三个独立的适用领域。KernelBench 加 FlashInfer-Bench 仍主要是算子层证据，要声称适用于服务性能，需要服务层实验。

## 为什么可靠反馈会改变调度目标

<!-- paper-figure:paper-krex-fig01 -->
性能反馈的成本不止图中真正计时的那一小段。先看KREX原图如何拆开一次评估，再讨论下面的奖励翻转算例。

[![KREX: Concurrent Kernel Benchmarking on Shared GPUs via Region-Granular Exclusivity，图1：整命令与测量区间的保护范围](../../_static/img/literature-reviews/paper-krex-fig01-screen.png)](../../_static/img/literature-reviews/paper-krex-fig01.png)

**原图来源：**Tianyu Feng et al.，[原文](https://arxiv.org/html/2609.30057v1)，图1，PDF第2页，arXiv 2609.30057v1。版权归原作者，作为本文讨论对象节选。仅截取图形区域与等比例显示，中文说明置于图外，访问于2026-11-23。

**怎样读这张图。** 上半部把导入、编译、准备和计时分开，下半部对比整命令与区域两种独占范围。允许区外准备重叠，可以减少等待，但目标计时区仍须具有一致测量条件。图里的时间分解来自该论文，不能当作所有benchmark通用的占比。

**与本节的联系。** 调度服务的整批完成时间应包含这些准备和保护成本，候选性能分数则按规定计时区间计算。只有把这两个时钟分开，才能判断服务变快的同时反馈是否仍可靠。
<!-- /paper-figure:paper-krex-fig01 -->


<!-- chapter-figure:chapter-local-service-time -->
任务本身有多快，与服务多久返回可信答案，应该分别衡量。

[![任务本身有多快，与服务多久返回可信答案，应该分别衡量](../../_static/img/document-figures/chapter-local-service-time-screen.png)](../../_static/img/document-figures/chapter-local-service-time.png)

图中24 ms包含等待与隔离成本，1 ms是候选测量时间。提升执行服务不能靠改变测量目标或减少规定工作量获得。 [放大查看 SVG](../../_static/img/document-figures/chapter-local-service-time.svg)。
<!-- /chapter-figure:chapter-local-service-time -->


下面是**教学假设，不是本次 GPU 实测**。同一题有两个候选，独占测量时 A 为 10 微秒、B 为 10.1 微秒。参考时间固定为 12 微秒，以参考时间除以候选时间作为奖励：A 得 1.2，B 约 1.188。

现在为了提高 GPU 利用率，让测量与别的任务共享 GPU。若 A 受竞争影响慢 10%，B 只慢 1%，测到的是 11 和 10.201 微秒，奖励约为 1.091 和 1.176。代码没有变，B 却被判得更好。

[![相同候选程序因测量时的竞争不同而发生奖励排序翻转（1/3）](../../_static/img/document-figures/explained-reward-distortion-5865243-screen.png)](../../_static/img/document-figures/explained-reward-distortion-5865243.png)

[![相同候选程序因测量时的竞争不同而发生奖励排序翻转（2/3）](../../_static/img/document-figures/explained-reward-distortion-5865243-screen-02.png)](../../_static/img/document-figures/explained-reward-distortion-5865243.png)

[![相同候选程序因测量时的竞争不同而发生奖励排序翻转（3/3）](../../_static/img/document-figures/explained-reward-distortion-5865243-screen-03.png)](../../_static/img/document-figures/explained-reward-distortion-5865243.png)

<!-- explained-source:explained-reward-distortion-5865243 -->
**读图结论：**随机噪声可通过重测缩小，系统偏差要先统一测量条件，不能仅增加重复次数。 教学数字不是外部 benchmark 数据，不由排名不变推断梯度不变。 [完整解释图（SVG）](../../_static/img/document-figures/explained-reward-distortion-5865243.svg) · [原始机制图或实验图](../../_static/img/literature-reviews/reward-distortion.png)。
<!-- /explained-source:explained-reward-distortion-5865243 -->

[查看可放大的 SVG](../../_static/img/document-figures/explained-reward-distortion-5865243.svg)。上下两组是同一份 A、B 代码和相同输入，蓝、橙柱表示耗时，越短越快；右侧奖励越高越好。第二组只改变测量期间的竞争条件，没有更换候选。

这会影响两类用途：优化 agent 下一轮可能保留错误版本，RL 训练可能鼓励错误候选。GRPO，即同题组相对策略优化，还会把分数相对组均值进行标准化，因此不是排名不变就一定保住了学习信号。

有三件事需要分开：

1. **随机波动**：重复测量可能缩小不确定区间。
2. **系统偏差**：竞争对两个候选影响不同，多测可能只是更确信错误排名。
3. **真实工作量变化**：同样限时训练中少跑了步骤，质量分数下降是任务结果改变，不能用测速校正直接修复。

对一批候选，直接追求最快返回会牺牲第二项。全程独占虽容易解释，却可能浪费初始化、CPU 编译和 agent 等待期间的 GPU。值得研究的是：哪些阶段能共享，哪些测量需要隔离，哪些候选值得重测，以及这些选择是否缩短整组获得可信结果的时间。

把三类变化连起来看，才能确定下一步动作。如果只是偶发波动，可以追加采样；如果 A 一直和带宽密集程序共跑、B 一直独占，应先统一测量条件，再谈重测；如果训练题以固定时间为预算，竞争使它少训练了几步，变化已经进入真实产物，需要控制工作量或把竞争写入评估协议。三者都会表现为“分数变了”，处理方式却不同。

这是**研究提案**。不能声称业界没有做过测量噪声、序贯采样或性能隔离。差异必须落在这些手段与真实 agent 反馈决策、组级完成目标、容器借还成本的联合设计，并通过对照证实。

## 实验怎样开展

应用侧反馈需求如何与资源侧调度协作，以及比较批次、保障切换和状态保留的候选机制，见 [YARN 对 GPU 工具调度的启发](04-系统借鉴_YARN资源分配与反馈调度.md)。该文为研究设计，没有改变本页记录的接入状态。

先固定候选代码，避免“模型生成了更容易的候选”干扰调度比较，再接入 agent 闭环。建议分三步，后两步不是本次已经启动的任务。

**第一步：功能与测量协议检查。** 每类先选一个官方任务，核对正确性、计时单位、参考实现、预热、重复次数、返回值以及清理。这就是本次 FlashInfer 最小任务。它只回答能否测、走的是否为 Env，不回答新调度器是否更快。

**第二步：固定候选的可靠性与成本。** 选择有明显性能差距及接近性能的候选，在同一硬件与软件版本上建立重复独占参考。参考也有误差，需要记录其分布。分别加入计算竞争、内存带宽竞争和不同缓存状态，不能只制造显存 OOM。记录：

- 整组时间 `T_group = 最后一条有效反馈返回时刻 - 整组释放时刻`，包含准入、初始化、测量、交接和必要重测。
- 正确性通过率、计时偏差、候选选择错误率。若用于 GRPO，再检查组内优势偏差。
- 额外测量成本、GPU 使用时间及命令借还成本。初始化与 kernel 本体计时分开报告。

至少比较四种方法：全程独占、直接共享、每个候选独立自适应测量加测量区间独占、拟议组级方法。最后一项若不能超越第三项，不能仅凭胜过粗糙共享方案认定创新成立。后续 Env 系统主基线统一为同次冻结实现的静态环境卡池、episode 持卡、满池排队。Command、共享和其他机制单列消融，旧提交结果仅保留历史含义。

**第三步：有限的 agent 闭环与跨层次检查。** 用相同题目、模型、候选或生成预算，检查更早返回的可信反馈能否提高最终所选实现质量或缩短固定组完成时间。再在服务级任务复核。如果只有短 kernel 上有效，就把结论限定为算子优化，不宣称所有性能指标都通用。

测量次数应按 pilot 方差和最小关心差距制定有界预算。一次 3 个 workload 的通过不能估计稳定的排名错误率，也不能证明速度收益。到验收问题已有答案便停止，不无限补采。

## 本次原生 Env 验证链路

下图是本次试验的实际调用链。它没有调用 agent、裁判模型或 PSRL 优化器，不能称为完整训练接入。

```text
[1] 官方定义、baseline、三个 workload → 固定子集与版本清单
                       ↓
[2] 私有 Ray 内 EnvRouter → 会话租约与单卡 command unit
                       ↓
[3] EnvWorkerEnvironment → 原生 EnvWorker 创建无网络容器
                       ↓
[4] exec_gpu：CUDA 预检 → 确认归还 → 官方评估命令
                       ↓
[5] FlashInfer Benchmark：正确性 + 参考与候选计时
                       ↓
[6] 检查结果集合完整、全部通过、GPU 配额归还
                       ↓
[7] 关闭容器、会话租约、私有 Router，检查物理 GPU
```

先沿编号理解控制依赖：准备固定输入、拿到资源许可、创建环境、运行评估、核对结果，最后完成清理。每一步的源码与状态变化合在下一节同一张表中，避免反复对照两份入口清单。

## 跟着图读源码

| 步骤 → 图中编号 | 函数 | 输入、输出或状态变化 |
|---|---|---|
| 准备 → 1 | [prepare](../env-service-verification/scripts/prepare_flashinfer_probe.py#L18) | 官方 checkout → 小型目录，保留 UUID 与哈希。没有改写参考数学函数，也不下载整个模型 |
| 准入 → 2 | [run](../env-service-verification/scripts/flashinfer_env_probe.py#L22) → [acquire](../../../psrl/workers/env_service/client.py#L159) | CPU、内存与 GPU 需求 → 会话租约。unit 容量覆盖整张卡，8 GiB 仅为该任务申报量，不是整卡准入上限 |
| 创建 → 3 | [EnvWorkerEnvironment.start](../../../psrl/workers/env_worker/harbor.py#L146) | 预构建镜像、只读依赖和任务挂载 → 真实容器。GPU 尚未作为普通命令的常开权限 |
| 执行 → 4、5 | [exec_gpu](../../../psrl/workers/env_service/managed_execution.py#L53) → [main](../env-service-verification/scripts/flashinfer_official_eval.py#L8) | 命令阶段持有访问权。调用官方 `TraceSet`、`BenchmarkConfig.default`、`Benchmark.run_all`，所有权需覆盖评估器内部工作进程 |
| 反馈 → 6 | 官方评估 [main](../env-service-verification/scripts/flashinfer_official_eval.py#L8) → `run` | UUID 与 solution 集合必须匹配预期三条结果且全部通过，避免空结果被当作成功 |
| 收尾 → 7 | [run 的 finally](../env-service-verification/scripts/flashinfer_env_probe.py#L165) | 不论成功或异常都关闭本次环境、释放租约并检查 GPU，写入状态文件。遥测异常报告为未确认，不记作零占用 |

Ray actor、租约心跳和官方评估工作进程具有并发性，图中的上下顺序描述控制依赖，不表示整个系统只有一个线程。

## 运行与观察

先看无需 GPU 的教学模拟，在仓库根目录执行：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/command_group_objective_demo.py
```

2026-11-05 实际运行得到：假设的 A/B 优势符号在竞争后翻转，固定偏差场景重复 2、20、200 次仍选错 B。脚本还演示同一时间区间足以判断是否小于 20 ms，却不足以确定两候选排名。输入是人为构造，不能当作 GPU 测量或统计覆盖率验证。

真实验证需要已激活项目环境、官方 checkout、准备好的镜像和空闲 GPU。在仓库根目录交互指定这些路径，输出目录必须尚不存在：

```bash
read -r -p 'FlashInfer-Bench checkout: ' FIB_SOURCE
read -r -p 'FlashInfer-Trace checkout: ' FIB_TRACE
read -r -p 'New output directory: ' FIB_OUTPUT
read -r -p 'Actual execution node IP: ' FIB_NODE
read -r -p 'Idle GPU index: ' FIB_GPU
read -r -p 'Prepared container image: ' FIB_IMAGE
FIB_SCRIPTS=docs/env-worker-deep-dive/env-service-verification/scripts
python "$FIB_SCRIPTS/prepare_flashinfer_probe.py" \
  --upstream "$FIB_SOURCE" --trace "$FIB_TRACE" --output "$FIB_OUTPUT"
PYTHONPATH="$PWD" python "$FIB_SCRIPTS/flashinfer_env_probe.py" \
  --root "$FIB_OUTPUT" --upstream "$FIB_OUTPUT/upstream" \
  --node-ip "$FIB_NODE" --gpu "$FIB_GPU" --image "$FIB_IMAGE"
```

脚本在实际执行节点运行，不要仅传一个远程 IP 却在无 Docker 的工具容器内启动。镜像需提供运行所挂载 Python 的系统库。当前 pilot 明确使用 CUDA 12.8 工具链，所有任务在无网络容器中运行。该命令是 11-05 探针复现入口，公共 recipe、任务 manifest 与独立 Harbor verifier 已在后续接入，见[任务 recipe](../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)。

`status.json` 看总状态与清理结果，`official-results.json` 看官方正确性和时间，`benchmark.stderr` 看编译或评估失败。官方 `speedup_factor` 比较的是数学参考实现，不能解释成相对当前生产最优 FlashInfer 实现的加速。

## Results 与验证边界

本节保存 2026-11-05 的官方任务与 Env 联通性探针：从申请环境到正确性评分，再到命令归还和容器清理。另一项研究——多次候选评估能否共用准备、降低成本——见[批量评估对照](../experiments/gpu-benchmarks/07-FlashInfer_进程复用与任务合批.md)。

探针选用官方 RMSNorm（按均方根归一化）任务，隐藏维度为 2048，批量大小为 1、64、16254。定义中的形状来自 Qwen3-30B-A3B，数据采用 BF16（16 位浮点表示）。评估器按这些形状生成官方规定的随机输入，用于检查算子正确性与耗时。

离线任务使用原生 EnvWorker command 后端，一张空闲 H20、4 CPU、16 GiB 容器内存。命令主体有 600 秒异步期限，离线监督进程设 900 秒总期限。固定工作区代码快照保留了 HEAD、未提交覆盖文件清单及源码哈希，避免其他会话修改正在运行的代码。共享 Python 依赖只读使用，未重新安装。

初次工具容器内 Docker 预检失败，未启动 GPU 工作，证据保留。随后改为在指定节点 SSH 启动，任务已经完成并停止。可移植的[结果记录](../experiments/gpu-benchmarks/flashinfer-env/evidence.json)保留了上游版本、运行时源码哈希、官方评估结果和原始日志哈希。

| 验收项 | 2026-11-05 探针结果 |
|---|---|
| 实际链路 | 私有 Ray EnvRouter → 原生 EnvWorker → direct command GPU |
| 容器 CUDA | 小型张量运算通过 |
| 官方任务 | 三个 workload 均为 PASSED，结果 UUID 与候选集合完整 |
| 命令间归还 | CUDA 预检后 GPU 无计算进程，显存回到约 0.19 MiB，GPU 配额为 0 |
| 评估命令后账本 | 会话仍保留，活动 GPU 命令与 GPU 配额为 0 |
| 最终清理 | 本次容器为 0、GPU 计算进程为 0，显存回到约 0.19 MiB，无清理异常 |
| 时间边界 | 整次约 75 秒，官方 Benchmark 内部约 5.43 秒，两者包含的阶段不同，不能混作 kernel 耗时 |

当次环境缺少 CUPTI（CUDA 性能分析工具接口），官方评估器采用 CUDA Event，在 GPU 流上记录时间戳。评分器与正确性容差保持官方设置。做稳定的性能比较时，需要固定计时后端、预热与缓存条件，并测量重复运行的波动。

这次探针验证了三项官方工作负载能够走完 Env 执行、评分和清理链路。FlashInfer 现有的公共 Harbor / PSRL 接入、代表题策略闭环和题库覆盖范围，统一见[有效接入结果](../experiments/gpu-benchmarks/01-任务验收_结果与证据导航.md)。研究资源调度时，还应独立比较启用与关闭调度器的相同工作量、同卡计算干扰和完整服务完成时间；功能联通性是这些研究的基础证据。

## 官方资料与固定版本

- [FlashInfer-Bench 官方代码](https://github.com/flashinfer-ai/flashinfer-bench)，本轮 `15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee`。
- [FlashInfer-Trace 官方数据](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)，本轮 `da915083d4c7c5e61aa3005e3d17ae488e0fc71c`。
- [RMSNorm 定义](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/blob/da915083d4c7c5e61aa3005e3d17ae488e0fc71c/definitions/rmsnorm/rmsnorm_h2048.json)，核对模型来源、数据类型与参考函数。
- [SWE-Serve 固定版本](https://github.com/NVIDIA/swe-serve/tree/84d8fd92739a4a76ceddba56e5a70605921c4ad2)，区分 correctness 与 speedup。
- [FastKernels 官方代码](https://github.com/Snowflake-AI-Research/fastkernels)，历史资料核查版本为 `9acebaa1349c78af8bb0ecb7e0719b5372d26fd8`；本地接入与验证范围见[当前任务 recipe](../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)。
