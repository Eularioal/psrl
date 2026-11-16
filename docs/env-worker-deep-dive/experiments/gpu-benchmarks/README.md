# GPU benchmark 导览：任务目标与 command 适用性

> **核心结论**：先按研究问题选报告，再看相应任务和执行路径。任务接入现状统一看[路线图](../../milestones/science-gpu-bench-status-and-roadmap.md)；本页重点解释“GPU在做什么、何时可以归还”，帮助选用整任务预约或按命令预约。

## 目录

- [按任务目标选择](#按任务目标选择)
- [更多任务与扩展方向](#更多任务与扩展方向)
- [command 究竟要求什么](#command-究竟要求什么)
- [推荐任务与真实执行路径](#推荐任务与真实执行路径)
- [不占 GPU 的生命周期演示](#不占-gpu-的生命周期演示)
- [接入状态与阅读入口](#接入状态与阅读入口)

对外交流与本地完整接入请读[阶段成果目录](../../milestones/README.md)和[完整接入报告](../../milestones/science-gpu-full-onboarding.md)，逐项验收见 [FINDINGS](FINDINGS.md#2026-11-10-本地完整接入验收)。

扩展库的当前逐题状态见[固定状态审计](onboarding-status-audit-20261116.json)，包含 PostTrain、FastKernels、SWE 的通过题、失败尝试、所用任务协议与策略验证。任务介绍及可复用题库入口统一在[当前路线图](../../milestones/science-gpu-bench-status-and-roadmap.md)。

## 按任务目标选择

| 想研究的问题 | 先读哪份报告 |
|---|---|
| 显存不够时，提前排队是否比失败重试更快？ | [常驻模型压力](resident-model-pressure.md)：相同7项任务、14请求，总时间减少10.6% |
| 模型所在GPU忙时，等待还是换卡加载？ | [热卡等待与冷卡加载](warm-placement-matrix.md)：实测选择过程、时间和显存代价；[代码指南](../../knowledge/warm-placement-code-guide.md)讲实现 |
| 在两张卡之间怎样兼顾模型复用和排队？ | [联合放置探索](joint-gpu-placement.md)：同执行器内比较策略 |
| 评分是否能复用模型，省掉反复加载？ | [PostTrain评分复用](posttrain-verifier-reuse.md)：请求变快，但完整短批次还未胜过普通进程复用 |
| 同卡竞争会不会让算子速度测错？ | [性能反馈压力](feedback-pressure.md)：解释如何误选更慢的候选及保护手段 |
| 多个候选能否一起评估？ | [FlashInfer批量与分段](flashinfer-batching.md)：进程复用、筛选、分段分别测量 |

这里的 benchmark 是给 agent 做的任务和验收规则。GPU 可能用来训练任务里的模型，也可能用来运行 agent 写的 kernel 或服务。**任务内训练模型，不等于训练 PSRL 中负责生成代码的 agent 模型。** 下面先看任务本身，接入路径在第三节串联。

| Bench | 输入与 agent 要完成的工作 | 交付物与验收 | 从任务形态判断 command 适用性 |
|---|---|---|---|
| **KernelBench** | 给一个 PyTorch 计算程序，改写为结果相同、运行更快的 GPU kernel，即在 GPU 上执行的计算函数 | `submission.py` 的 `ModelNew`，检查随机输入结果及运行速度 | **适合按一次编译、正确性与性能评估来预约**。评估进程退出后通常无需保留 GPU 状态 |
| **MLE-bench** | 给竞赛说明与公开训练数据，自行做数据处理、选模型、训练、验证、预测 | `submission.csv`，由独立验收器用隐藏标签评分 | **通常适合按训练或批量预测作业预约**。不是所有题或解法都必须用 GPU，需挑选实际使用 GPU 的训练方案 |
| **AIRS-Bench 本地 GPU** | 真实 NLP、代码、分子与时间序列数据，训练或检索后预测 | CSV，由原生 sandbox 独立 verifier 评分；19/19 非 Kaggle 任务快检通过 | **当前按 episode 保留 GPU**。分阶段借还仍需接入，不属于 Harbor command 路径 |
| **autoresearch** | 改训练代码里的模型结构、优化器或超参数，执行有固定预算的实验，按验证指标选择方案 | `train.py`，独立进程重新训练并调用官方验证函数 | **适合每次训练实验一条长 command**。一次训练持续数分钟，不代表两次实验之间必须保留显存 |
| **InferenceBench 官方任务** | 给基础语言模型和请求场景，优化推理服务实现与启动配置，提高服务性能并保持回答质量 | 服务启动脚本，重新启动服务测速度、质量和方案合规性 | **取决于边界**。启动服务、跑完整批请求、停服务，可以整体是一条 command。单次请求结束通常不能归还后台服务的 GPU |
| **本地常驻服务 workload** | 固定 InferenceBench 的服务和 Mistral 模型，让同一服务跨多条环境命令处理真实请求 | 请求输出、服务复用、显存、排队、失败及清理记录 | **不适合按每次请求借还，推荐 episode**。这是本地框架调度实验，不是第五个独立官方 benchmark |
| **FlashInfer-Bench** | 为固定生产算子与真实 workload 提交官方格式 Solution JSON | 独立官方 evaluator 检查完整子集，本地奖励为正确性，时延另报 | **可按独立评估进程划分**。公共 Harbor 入口使用 episode，不扩大 KernelBench command 白名单 |

官方任务定义参考 [KernelBench](https://github.com/ScalingIntelligence/KernelBench#-task-description)、[MLE-bench](https://github.com/openai/mle-bench#dataset)、[autoresearch](https://github.com/karpathy/autoresearch#how-it-works)、[InferenceBench](https://github.com/aisa-group/InferenceBench)。本地实际接入版本以 [upstream.json](../../../../examples/gpu_benchmarks/upstream.json) 为准，command 判断来自任务状态和本地回收协议分析，不是这些项目发布的模式兼容认证。

四个具体例子帮助区分它们：

1. **KernelBench：矩阵乘法优化。** 输入是计算 `A @ B` 的参考实现，agent 写一个 Triton 实现。Triton 是编写 GPU 计算程序的一种语言与工具。验收器运行两种实现，先比较数值，再计时。当前 recipe 的 `reward` 取正确性，速度提升另存为指标，不能把 reward=1 解读成已经加速。见 [本地提交示例](../../../../examples/kernel_bench/solutions/matmul.py) 和 [evaluate](../../../../examples/gpu_benchmarks/evaluate.py#L10)。
2. **MLE-bench：航拍图像是否含仙人掌。** `aerial-cactus-identification` 给公开图像与标签。agent 可以训练图像分类模型，对待预测图像输出概率 CSV。独立验收器读取隐藏标签评分，当前 recipe 的奖励为是否超过参考竞赛成绩中位数。训练结束且 GPU 进程退出后，可以释放资源，再用 CPU 分析错误。数据权限尚待配置，不把这个例子写成已运行成功。见 [任务准备](../../../../examples/gpu_benchmarks/prepare.py#L147) 与 [评分](../../../../examples/gpu_benchmarks/evaluate.py#L37)。
3. **autoresearch：比较两种训练方案。** agent 改 `train.py`，训练一次，读取 `val_bpb`，再决定保留还是修改。BPB 是验证文本每字节的预测编码代价，越低越好。官方每次训练预算为 300 秒，不包含启动和编译。当前 recipe 限制一个 episode 内最多两次实验，最后独立重放提交，奖励为 `1 / (1 + val_bpb)`。它属于自动研究迭代场景，不能仅凭它就声称已经实现 agent 自身能力的递归提升。见 [准备指令](../../../../examples/gpu_benchmarks/prepare.py#L175) 和 [重放](../../../../examples/gpu_benchmarks/autoresearch_replay.py)。
4. **InferenceBench：让同一语言模型更高效地回答请求。** 输入是模型与请求场景，产物是更好的服务方案，不是训练出新模型。它可以研究推理基础设施优化，较接近这里所说的 AI4Infra，即用 AI 改进计算基础设施。当前本地常驻实验固定服务实现和短请求，只测试 Env 的资源管理，不能当作完成了官方自由优化任务或质量评分。见 [官方任务导入](../../../../examples/gpu_benchmarks/prepare.py#L77) 与 [本地固定服务驱动](../../../../examples/inference_bench/resident_ab.py#L82)。

<a id="图片中-high-gpu-的新增候选"></a>

## 更多任务与扩展方向

**RE-Bench 的聊天模型训练任务已有原生 Env/PSRL 接线**，见 [运行 recipe](../../knowledge/re-bench-recipe.md)。其官方起始代码在 GPU 生成与训练之间等待外部裁判，期间仍持有 CUDA 模型。PostTrainBench 的 9 个公开本地组合已完成策略闭环，整库范围见[路线图](../../milestones/science-gpu-bench-status-and-roadmap.md#posttrainbench-与-fastkernels-任务协议)。

| 任务项目 | 选型用途与 command 边界 |
|---|---|
| AIRS-Bench | 19 个非 Kaggle 任务已完成原生 GPU 快检，SICK 代表题完成 PSRL 策略闭环；按训练和预测拆作业仍是后续调度工作，见 [接入说明](../../knowledge/airs-gpu-recipe.md) |
| Automated LLM Speedrunning | 优化达到目标损失的训练时间。官方配置 8 张 H100，完整训练可作为长 command |
| MLE-bench | GPU 机器学习竞赛方案。官方 Kaggle 数据需凭据，本地真实 digits 竞赛提供不依赖 Kaggle 的入口 |
| MLGym-Bench | 可选 FineWeb 语言模型训练，保存模型后能退出训练进程 |
| RE-Bench | 可选训练与外部裁判交替的任务。整次训练可作为 command，但中间归还需实现状态卸载恢复 |
| PostTrainBench | 优先补充单卡后训练。完整训练后保存模型、退出，再评估，是自然的 command 边界 |

表中选型不能代替逐项接入与评分验证，RE-Bench 已有本地 Qwen 裁判链路，完整 512 条评分仍未验收。high GPU 也不表示持续满显存，官方链接和具体选题见 [六项专项调研](../../literature-reviews/gpu-workload-benchmark-survey.md#图片中六项-high-gpu-的具体选题)。

**看 InferenceBench 分数前先分清来源**：官方在质量达标后按延迟或吞吐评分，本地 PSRL 还检查裁判标记。常驻实验中的 `reward=1` 只是请求与服务复用检查。公式、手算例子及 CPU 演示见 [InferenceBench 评分说明](../../knowledge/inferencebench-scoring.md)。

## command 究竟要求什么

在当前 Env 中，episode 是一次任务的完整生命周期，GPU 预算保持到环境清理。command 是一段显式申请 GPU 的执行区间，结束时必须确认资源实际归还，才能让其他任务使用这份预算。普通 shell 命令能结束，并不意味着它启动的后台服务也退出了。

最有用的判断是：**在拟定的 command 边界，能否让本任务的 GPU 进程和设备句柄退出，并在下次工作开始时正确恢复？** 设备句柄是进程与 GPU 驱动保持的访问连接。若它仍存在，当前 Env 不能仅凭显存减少就认定资源已归还，见 [revoke_gpu_command](../../../../psrl/workers/env_service/command_gpu_node.py#L207)。

| 划分方式 | 资源归还与适用性 |
|---|---|
| 完整训练进程作为一条 command | 可以很长。进程结束并完成回收后归还，无需因为训练超过几分钟就改用 episode |
| 常驻服务的每次 HTTP 请求作为一条 command | 客户端退出后服务仍持有模型，不满足归还条件。HTTP 是客户端向服务发送请求的协议 |
| 启动服务、全部请求、停服务作为一条长 command | 可以设计这种适配，但预约区间仍覆盖全部请求间隙，无法在间隙借出 GPU |
| 请求间卸载 GPU 状态，下次恢复 | 需要额外实现卸载与恢复协议。offload 指将状态移出 GPU，reload 指重新加载。当前回收还检查设备句柄，通常需要退出 GPU 工作进程并重建，不能只调用 `empty_cache()` |

当前 [GpuBenchmarkAgentLoop](../../../../examples/gpu_benchmarks/agent_loop.py#L23) 默认要求 episode，但配置 KernelBench 专属 verifier 时允许经过约束校验的 direct command。KernelBench 已有该入口，autoresearch 尚需接线 agent 和独立 verifier 的全部 GPU 操作及回收，不能只改配置字符串。verifier 是独立验收任务结果的程序。

## 推荐任务与真实执行路径

**选 `examples/inference_bench/resident_ab.py` 对应的常驻 Mistral 服务 workload。** 它使用官方 Transformers 服务、真实模型和真实 MMLU-Pro 问题。同一个服务完成两次请求，中间另起 CPU 等待命令，模型在 GPU 上持续驻留。`resident_agent.py` 是固定动作 agent，不调用语言模型来优化代码。该路径已完成原生 Env 的真实验证，完整对照看[常驻模型压力报告](resident-model-pressure.md)。

```text
[1] resident_ab 提交 episode → production runner 向 EnvRouter 申请 lease
                                   ↓
[2] 原生 EnvWorker 创建 GPU 容器 → resident_agent 后台启动模型服务
                                   ↓
[3] 请求命令一 → client 发题目 → 同一服务推理 → client 退出
                                   ↓
[4] 独立 CPU 等待命令结束，后台模型与 GPU 状态仍在
                                   ↓
[5] 请求命令二 → client 发下一题 → 同一服务推理 → client 退出
                                   ↓
[6] 停服务 → verifier 检查结果 → 清理容器 → 释放 lease
```

图例：图里每个部分对应的源码位置。

| 编号 | 职责、函数与源码 |
|---|---|
| 1 | [resident_ab.run](../../../../examples/inference_bench/resident_ab.py#L82) 配置原生 Env 和 episode 预算，[run_harbor_episode](../../../../examples/sciaccel_rl/runner.py#L275) 驱动任务，[EnvServiceClient.acquire](../../../../psrl/workers/env_service/client.py#L159) 请求租约，即已获准的资源记录 |
| 2 | [EnvWorkerEnvironment.start](../../../../psrl/workers/env_worker/harbor.py#L146) 创建环境，[ResidentInferenceAgent.run](../../../../examples/inference_bench/resident_agent.py#L10) 后台启动官方服务，等待模型可接受请求 |
| 3、5 | [resident_client.main](../../../../examples/inference_bench/resident_client.py#L11) 发送真实样本并保存响应及服务 PID，即进程编号 |
| 4 | [独立等待命令](../../../../examples/inference_bench/resident_agent.py#L37) 只暂停 agent 的下一步，不会卸载后台服务权重 |
| 6 | [agent finally](../../../../examples/inference_bench/resident_agent.py#L45) 停服务，[prepare 中的 verifier](../../../../examples/inference_bench/resident_ab.py#L64) 检查两次输出及服务复用，[EnvWorkerEnvironment.stop](../../../../psrl/workers/env_worker/harbor.py#L182) 清理环境，[EnvLeaseHandle.release](../../../../psrl/workers/env_service/client.py#L69) 归还租约 |

**跟着图读源码：** 从 1 看任务配置如何进入 runner，再沿租约进入 2 的环境创建。2 启动的服务与 3 到 5 的短命令是并发存在的两条执行线，所以不能把短命令返回当成服务退出。最后从 6 检查异常路径是否也停服务、清理容器。官方服务源码不随 PSRL 分发，本地核对了上游固定版本的 `main` 将模型保存到服务状态，以及每次请求复用该状态，见 [固定版本服务源码](https://github.com/aisa-group/InferenceBench/blob/108ec747fab0bd92fd9de501f62157d074c50ed5/src/eval/inference/servers/transformers_openai_server.py#L282)。这些代码如有理解上的问题，可以继续按图中的步骤追问。

**数字例子，属于教学假设。** 假设模型加载后占 14 GiB。请求一的客户端退出时，后台服务仍占这 14 GiB。等待 30 秒不会把权重搬走，请求二继续复用它。此时若把请求一结束解释成“归还 14 GiB”，账本会宣布一块实际上仍被占用的容量可用。保持 episode 预约，直到第 6 步释放，符合这项任务的生命周期。

**推荐范围要准确。** 不适合的是“服务保持常驻，却按每次请求归还 GPU”的划分。整项 workload 仍可包装为一条长 command。请求间的等待由本地压力实验明确设置，不是官方 InferenceBench 强制所有任务这样运行。如果要求找到一个“官方完整 benchmark 在任何 command 粒度下都不成立”的例子，当前四项没有这样的证据，不能以本地适配器拒绝 command 作为证明。

## 不占 GPU 的生命周期演示

从 PSRL 仓库根目录运行，只使用 Python 标准库、回环网络和 1 MiB CPU 数据：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/explain_resident_command_lifetime.py
```

[演示脚本](../../env-service-verification/scripts/explain_resident_command_lifetime.py) 用常驻 HTTP 服务线程模拟图中第 2 步，用两个真正退出的客户端子进程模拟第 3、5 步。每个客户端限时 10 秒，服务只监听本机自动分配的端口，结束后关闭本次服务。它没有 GPU、CUDA 或 EnvRouter，因此不是 GPU 归还协议测试。

预期两次都是 `client_exit_code: 0`，但 `resident_server_alive: true`，模拟的 `simulated_release_allowed: false`。两次回复中的 `server_pid` 相同。只有停止服务并清除 CPU 状态后，`after_stop_simulated_release_allowed` 才为 true。这展示的是“请求结束”和“服务生命周期结束”的区别。真实 Env 还需检查 GPU 设备句柄、进程归属和物理清理，不能用线程状态替代。

## 接入状态与阅读入口

| 入口 | 当前可用范围与下一步 |
|---|---|
| [KernelBench 接入](../../../../examples/kernel_bench) | 已有真实 kernel 编译、正确性与性能评估。已新增独立 verifier 的 command 适配，同节点动态选卡仍待实现，见[必要性评估](../../research-proposals/dynamic-command-placement.md#9-已接入-benchmark-的必要性评估) |
| [MLE-bench 接入](../../../../examples/mle_bench) | 已有任务和评分适配，官方数据下载待 Kaggle 认证。暂不以它作为马上可运行的 GPU 对照 |
| [autoresearch 接入](../../../../examples/autoresearch) | 已有真实任务内训练与验证。适合研究较长 GPU 作业，不适合仅凭运行时长证明 episode 必要性 |
| [InferenceBench 接入](../../../../examples/inference_bench) | 常驻服务框架 workload 已真实验证。官方完整优化与评分尚需对应基线、judge 及资源配置 |
| [RE-Bench 接入](../../knowledge/re-bench-recipe.md) | 原生 Env、独立 verifier、PSRL 数据与奖励入口已接线。smoke 和正式评分分离，支持独立卡本地 Qwen，原 Replicate 分支仍需宿主凭据 |
| [准备、下载与运行 recipe](../../knowledge/gpu-benchmark-recipes.md) | 具体命令、依赖、数据隔离与 Env/PSRL 接线，避免在本导览重复维护启动步骤 |
| [记账与归属机制](../../knowledge/gpu-memory-admission.md#9-episode-归属记账修复) | 解释模型驻留时如何抵扣已兑现预约，以及何时仍需等待 |
| [Results 与验证边界](FINDINGS.md) | 查看实际工作量、ON/OFF 条件、失败、耗时和清理，不将功能验收分数当作官方成绩 |

本导览按当前本地代码说明适用性。CPU 演示可直接运行，GPU 实验需先按 recipe 核对现场资源和官方资产，不会因阅读导览自动启动。

AIRS 的非 Kaggle 任务已有 [19/19 原生 GPU 快检与运行入口](../../knowledge/airs-gpu-recipe.md)，GPU 配置、数据与奖励使用独立 recipe。MLGym 本地 GPU 范围为 4 个 RL、3 个视觉和 2 个 NLP 任务，均有闭环。SWE-Serve 41 题已导入，FastKernels 29 个目标及 PostTrain 9 个组合有快检。各层覆盖与整库缺口见[路线图](../../milestones/science-gpu-bench-status-and-roadmap.md#目前有哪些任务及可用边界)。不能将 AIRS 使用 MLGym agent 记作 MLGym-Bench 完整接入。

性能反馈调度研究另见 [159 组压力实验](feedback-pressure.md)：官方算子与候选在当前 Env 中运行，比较测量保护、固定预算及自适应采样，不改变上述公共 recipe 的接入方式。
