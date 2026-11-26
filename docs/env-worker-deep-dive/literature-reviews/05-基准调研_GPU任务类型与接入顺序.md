# 基准调研：GPU任务类型与接入顺序

> **核心结论**：补充 GPU 场景时，优先考虑 Automated LLM Speedrunning，它能提供完整语言模型训练、多卡通信和根据实验反馈继续搜索的任务。图像生成可选 GenEval 2，物理仿真可选 ManiSkill 3，持续服务压力可引入 MLPerf Inference 的 LoadGen。后三者需要封装成 agent 任务。本页列出选型和接入方案，实际可用范围统一查[路线图](../milestones/02-Science现状_任务覆盖与接入路线.md)。
>
> 最新核对：2026-11-16。本次读取上述四个项目的官方论文、源码或文档，并清点 Speedrunner 上游 `ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9` 的文件树。PSRL 核对范围为主目录 `c5f2d65` 加未提交修改，没有安装候选依赖、下载模型或启动 GPU 实验。其他项目的官方规模沿用已有调研，接入数量按当前路线图更新。


## 目录

- [目标与全局流程](#目标与全局流程)
- [比较维度](#比较维度)
- [候选对照表](#候选对照表)
- [图片中六项 high GPU 的具体选题](#图片中六项-high-gpu-的具体选题)
- [按场景选择新增 benchmark](#按场景选择新增-benchmark)
- [Automated LLM Speedrunning 接入评估](#automated-llm-speedrunning-接入评估)
  - [任务内容、评分与完整例子](#任务内容评分与完整例子)
  - [题目数量与文件入口](#题目数量与文件入口)
  - [能补充什么、还需要什么](#能补充什么还需要什么)
  - [PSRL 接入与快速验收方案](#psrl-接入与快速验收方案)
- [已接入任务的研究用法](#已接入任务的研究用法)
- [具体例子与评分干扰](#具体例子与评分干扰)
- [本地实现与源码阅读路径](#本地实现与源码阅读路径)
- [可运行的教学模拟](#可运行的教学模拟)
- [验收与未验证事项](#验收与未验证事项)
- [非计时评分场景的当前优先级](#非计时评分场景的当前优先级)

## 目标与全局流程

这里的 workload 指实际交给计算资源执行的一段工作。我们关心 agent 在环境中启动的训练、计算核和模型服务。驱动 agent 思考的大语言模型也需要算力，但它属于 PSRL 的策略模型训推资源，需要另记。

下图描述本地 science 的公共执行结构，具体 benchmark 的完成数量以路线图和对应 recipe 为准。PostTrain 的 15 个本地组合与 RE 的有界协议已有执行及策略闭环证据，官方完整规模另行评测。

[![任务申请资源、循环实验、独立评分和清理归还的执行关系（1/3）](../../_static/img/document-figures/explained-gpu-task-lifecycle-336b197-screen.png)](../../_static/img/document-figures/explained-gpu-task-lifecycle-336b197.png)

[![任务申请资源、循环实验、独立评分和清理归还的执行关系（2/3）](../../_static/img/document-figures/explained-gpu-task-lifecycle-336b197-screen-02.png)](../../_static/img/document-figures/explained-gpu-task-lifecycle-336b197.png)

[![任务申请资源、循环实验、独立评分和清理归还的执行关系（3/3）](../../_static/img/document-figures/explained-gpu-task-lifecycle-336b197-screen-03.png)](../../_static/img/document-figures/explained-gpu-task-lifecycle-336b197.png)

<!-- explained-source:explained-gpu-task-lifecycle-336b197 -->
**读图结论：**benchmark 选型还在选择状态寿命与提交边界，不只是选择一个 GPU 计算函数。 不是所有任务都默认 command，接入和验证层次按正文逐项确认。 [完整解释图（SVG）](../../_static/img/document-figures/explained-gpu-task-lifecycle-336b197.svg) · [原始机制图或实验图](../../_static/img/literature-reviews/gpu-task-lifecycle.png)。
<!-- /explained-source:explained-gpu-task-lifecycle-336b197 -->

[查看可放大的 SVG](../../_static/img/document-figures/explained-gpu-task-lifecycle-336b197.svg)。先沿上排看任务怎样获得会话，再沿右侧的双向箭头看多轮实验，最终提交后从右下向左读评分与清理。这里的 GPU 命令持有方式由配置决定，图不要求所有任务都采用 command 模式。

例如训练题在同一容器中多次修改脚本，磁盘上的代码和 checkpoint 可以保留。某次训练命令运行时申请 GPU，结束后按持有策略处理资源；随后 verifier 重新加载最终产物，给出分数。因此，选择 benchmark 不只是在选计算函数，也是在选“反复做什么、哪些状态必须留下、最后以什么标准判断成功”。

图例：图里每个部分对应的源码位置。以下链接是当前工作目录中的实现，不代表所有外部 benchmark 已经能复用。

| 编号或箭头 | 职责 | 对应函数及源码 |
|---|---|---|
| 1 → 2 | science 任务转换资源需求并申请租约 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L372) |
| 2 | 向路由器申请资源 | [EnvServiceClient.acquire](../../../psrl/workers/env_service/client.py#L159) |
| 3 → 4 | 创建并运行 Harbor job，由 Harbor 内部执行 agent 与 verifier | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L550) |
| 4 | 提取奖励，按条件处理 GPU 作业报告 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L574) |
| 5 | finally 清理容器，随后归还资源 | [run_harbor_episode](../../../examples/sciaccel_rl/runner.py#L610)、[EnvLeaseHandle.release](../../../psrl/workers/env_service/client.py#L69) |
| 候选分支 | 训练、kernel、服务的专属任务适配 | [已有 recipe 与边界](../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)，新增候选仍需适配 |

租约表示某个任务获准使用一份资源预算，不自动保证显存硬隔离。verifier 是负责重新检查结果的评分程序，不能直接相信 agent 打印出来的分数。一个 episode 是一次从任务开始到评分和清理结束的完整尝试。若这些源码有理解上的问题，可以随时沿图中编号追问。

## 比较维度

<!-- paper-figure:paper-kernelbench-fig01 -->
用KernelBench原图说明为什么“任务需要GPU”和“评分依赖时间”必须分别核对。

[![KernelBench: Can LLMs Write Efficient GPU Kernels?，图1：生成代码、正确性检查与性能测量](../../_static/img/literature-reviews/paper-kernelbench-fig01-screen.png)](../../_static/img/literature-reviews/paper-kernelbench-fig01.png)

**原图来源：**Anne Ouyang et al.，[原文](https://arxiv.org/html/2502.10517v1)，图1，PDF第2页，arXiv 2502.10517v1。[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。仅截取图形区域与等比例显示，中文说明置于图外，访问于2026-11-23。

**怎样读这张图。** 左侧给出PyTorch参考任务，中间模型生成CUDA候选及调用代码，右侧分成Correctness和Performance两条检查：先比较随机输入下的输出，再比较候选与参考实现的运行时间。一次工具评估可以同时包含编译、正确性和计时，不能只从最后打印的耗时反推整条命令都在计算。

**与本节的联系。** 分类benchmark时应问清最终采用哪条评分路径。本地recipe若只用正确性作为reward，就不能把论文中的性能评价默认当作本地训练奖励。
<!-- /paper-figure:paper-kernelbench-fig01 -->


<!-- chapter-figure:chapter-warm-layers -->
比较任务之前，先辨认每次命令重新准备的是哪一层。

[![比较任务之前，先辨认每次命令重新准备的是哪一层](../../_static/img/document-figures/chapter-warm-layers-screen.png)](../../_static/img/document-figures/chapter-warm-layers.png)

同样叫“模型加载”，可能是新建容器、启动Python，也可能是把权重重新搬到GPU。只有真实命令链重复支付的成本，才构成该benchmark的复用空间。 [放大查看 SVG](../../_static/img/document-figures/chapter-warm-layers.svg)。
<!-- /chapter-figure:chapter-warm-layers -->


先回答评分问题，再看资源形态。**任务使用 GPU，并不意味着它必须以 GPU 耗时评分。** 固定步数训练后的验证分数、生成图像质量、功能测试通过率，都可以用于研究资源调度；但具体协议仍要核对数值行为和工作量。

[![按结果评分标准区分 GPU 调度实验的两类约束（1/3）](../../_static/img/document-figures/explained-benchmark-choice-c6aaac2-screen.png)](../../_static/img/document-figures/explained-benchmark-choice-c6aaac2.png)

[![按结果评分标准区分 GPU 调度实验的两类约束（2/3）](../../_static/img/document-figures/explained-benchmark-choice-c6aaac2-screen-02.png)](../../_static/img/document-figures/explained-benchmark-choice-c6aaac2.png)

[![按结果评分标准区分 GPU 调度实验的两类约束（3/3）](../../_static/img/document-figures/explained-benchmark-choice-c6aaac2-screen-03.png)](../../_static/img/document-figures/explained-benchmark-choice-c6aaac2.png)

<!-- explained-source:explained-benchmark-choice-c6aaac2 -->
**读图结论：**选择非时间指标还要固定实际工作，限时准确率任务仍可能受设备竞争影响。 图用于研究选型，不把外部 benchmark 的公开描述当作本地已接入。 [完整解释图（SVG）](../../_static/img/document-figures/explained-benchmark-choice-c6aaac2.svg) · [原始机制图或实验图](../../_static/img/literature-reviews/benchmark-choice.png)。
<!-- /explained-source:explained-benchmark-choice-c6aaac2 -->

[查看可放大的 SVG](../../_static/img/document-figures/explained-benchmark-choice-c6aaac2.svg)。左支更适合先研究相同工作量下的整批时间，右支还需保护测速条件或核对限时内真正完成的工作。固定时间训练虽然可能以准确率评分，竞争减少训练步数后仍会改变真实结果，应归入右支的考虑范围。

再看状态怎样延续：独立命令执行完后可以只留下文件；常驻服务则会把模型和连接留在进程里。前者适合在命令边界重新选卡，后者需要明确的保存、卸载或重启协议。这个区别决定调度动作是否合法，评分类型决定该怎样比较收益。下面的表将两类条件放在一起看。

图片适合比较研究能力覆盖范围，例如任务来源、研究数据、任务跨度、科学研究环节和计算需求。对本次资源调度目标，还需要以下字段。图片没有给 H/I/E/A 的完整定义，本文不凭缩写补造含义。

| 字段 | 为什么影响接入 |
|---|---|
| 任务来源、版本、固定 commit | 任务数量与评测协议会变化，不能把截图当作最新事实 |
| GPU 使用者与阶段 | agent 模型推理、环境训练、最终评分可能分别需要 GPU |
| 单卡或多卡、硬件绑定 | CUDA 是 NVIDIA 的计算平台，ROCm 是 AMD 的计算平台，不能仅换镜像名就互通 |
| 显存峰值、空闲保留显存 | 利用率低不代表显存已释放，服务进程可能一直保留模型 |
| 占用时间、调用频率、生命周期 | 短命令适合阶段借还，常驻服务需要覆盖整个进程生命周期 |
| 评分与计时边界 | 时延和吞吐分数会受同卡竞争影响，不能把调度干扰算作代码质量 |
| 数据、模型、镜像、许可证 | 提前确认获取条件和可再分发范围，不统一打包传播未知许可数据 |
| 反馈、最终提交、可信评分 | 公开反馈与隐藏测试隔离，避免 agent 改评分器获利 |
| 接入方式与证据等级 | 区分文档宣称、代码审核、容器通过、真实 CUDA 执行和正式评测 |

RSI 在这里按 Recursive Self-Improvement，即递归自我改进理解。研究 agent 自动改训练代码属于相关工作负载，但若它改的是另一个模型，而没有把改进反馈到自身能力中，就不能仅凭这个循环声称实现了严格的递归自我改进。MLE 是机器学习工程，包括处理数据、训练模型、调参和提交预测。AI4Infra 指利用 AI 改进基础设施，可覆盖推理服务、训练框架、通信与底层计算。

## 候选对照表

GPU 形态是根据任务机制作的接入判断，不是本地测得的资源曲线。未列具体显存时表示尚未核定，不能解释为低显存。

| 候选与一手来源 | 任务和来源 | GPU 使用形态 | 结果与接入判断 |
|---|---|---|---|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | PyTorch 程序改写为更快的 GPU kernel，基础三级为 100/100/50 个任务，另有第 4 级 | 编译后反复在 GPU 上检查正确性和计时，通常可按单次评测组织 | 270 个题身份已有本地 GPU 快检。适合短作业与计时隔离研究，输入缩放条件见路线图 |
| [MLE-bench](https://github.com/openai/mle-bench) | 75 个 Kaggle 机器学习竞赛，Lite 为 22 个低复杂度任务 | 数据准备、训练、预测交替，选图像任务可形成持续训练负载 | 本地四个公开分类任务已快检，官方竞赛待凭据。官方参考资源为单张 24GB A10、24 小时，部分表格解法可以只用 CPU |
| [InferenceBench](https://github.com/aisa-group/InferenceBench) | agent 优化模型推理服务，涵盖首 token 时延、后续 token 时延、吞吐及综合场景 | 模型服务常驻显存，外部请求产生 GPU 压力 | 四个本地场景已快检。官方仓库公开 Mistral-7B 和每次 2 小时设置，当前继续用于服务生命周期与性能隔离研究 |
| [autoresearch](https://github.com/karpathy/autoresearch) | agent 修改训练代码，运行实验后保留或放弃修改 | 单张 NVIDIA GPU，每次训练固定 5 分钟，另计启动、编译和评估 | 固定任务已有闭环。它提供研究循环，运行时另设总轮数与总期限 |
| [MLGym](https://github.com/facebookresearch/MLGym) | 13 个开放研究任务，覆盖监督学习、强化学习等 | 不同任务差异大，训练类可用 GPU | 本地 9 个 GPU 任务已有逐题闭环，RL、视觉和 NLP 使用明确的小预算协议，源码与依赖需固定 |
| [RE-Bench](https://github.com/METR/RE-Bench) | 7 类研究工程任务，包括训练脚本优化、模型修复、Triton 前缀和 | 同时覆盖长训练与短 kernel，逐任务审核卡数和评分环境 | 7 类本地协议已有逐题 GPU 快检，聊天训练代表题闭环，官方模型规模和完整裁判协议仍未复现 |
| [PostTrainBench](https://github.com/aisa-group/PostTrainBench) | 自动对基础语言模型做后训练 | 官方设置为一张 H100、10 小时，训练和评估均可能占 GPU | 本地 15/24 个组合已有闭环，含真实生成与裁判评分。当前只更新最终归一化层，完整优化器状态的资源行为仍需补充 |
| [TritonBench](https://github.com/thunlp/TritonBench) | 生成 Triton 算子，Triton 是便于编写 GPU 并行计算的语言与编译系统 | GPU 正确性与性能评测 | KernelBench 接口稳定后扩充 kernel 题型，初期无需两个近似负载同时接 |
| [InfraDev 论文](https://arxiv.org/abs/2610.36589) | 2026-10-27 发布，36 个源自 Megatron-LM、vLLM、verl 的真实修改任务 | 多 GPU 执行与隐藏验证 | **跟进候选**。论文定位明确，公开任务包、运行入口、每题硬件预算本次尚未核实，不能宣称可立即运行 |
| [sammysun0711/InfraBench](https://github.com/sammysun0711/InfraBench) | 20 个 Harbor 格式 AI 基础设施任务 | 目标为 AMD MI35X，任务表包含 1 卡和 2 卡 | Harbor 格式接近已有路径，但硬件不匹配时不宜首接。与其他同名论文不自动视为同一项目 |
| [MLSysBench](https://github.com/haiyuanwan/MLSysBench) | 配置搜索、调度和跨规模迁移，包含模拟器与原型任务 | 多个任务只运行模拟，真实 CUDA 路径另有边界 | **暂不作为真实 GPU 负载证明**。其仓库明确说明尚非完整正式评测套件 |

这张表关注可运行环境，故没有把图片中所有论文复现或科学问答 benchmark 一并列为 GPU 候选。是否使用研究论文数据，与是否实际启动 GPU 任务是两回事。

## 图片中六项 high GPU 的具体选题

图片的 high GPU 是计算需求类别，不是实测显存、GPU 利用率或跨命令驻留保证。下面按官方任务定义选择，不把套件中所有题都当成 GPU 任务。command 判断是对进程与状态边界的分析，不是官方发布的 Env 模式兼容声明。

| 图中项目与官方来源 | 真正要做的任务 | GPU 形态与 command 判断 | 当前建议 |
|---|---|---|---|
| [AIRS-Bench](https://github.com/facebookresearch/airs-bench) | 20 个来自论文的研究任务，自行提出方法、训练、评估 | 按具体题选择 GPU 训练。训练进程保存结果并退出，可作为 command | 19/19 非 Kaggle 任务有 CUDA 快检，SICK 代表题闭环，原生 sandbox 分支尚无 command 借还 |
| [Automated LLM Speedrunning](https://github.com/facebookresearch/llm-speedrunner/blob/main/config/task/nanogpt_speedrun/default_config.yaml) | 修改 GPT-2 训练代码，在验证损失不高于 3.28 的条件下缩短训练时间 | 官方配置使用单节点 8 张 H100，通过 `torchrun` 启动分布式训练 | 当前优先补充候选，增加完整训练和多卡场景。本地双卡档需保持真实通信，并重新校准质量门槛与参考耗时，详见[接入评估](#automated-llm-speedrunning-接入评估) |
| [MLE-bench](https://github.com/openai/mle-bench) | 75 个 Kaggle 竞赛，做特征、训练、预测和提交 | 选择 GPU 图像或语言模型方案。训练和预测可以分别作为 command | 保留现有接入，按用户安排暂缓 Kaggle 权限和数据 |
| [MLGym-Bench](https://github.com/facebookresearch/MLGym/blob/main/configs/tasks/languageModelingFineWeb.yaml) | 例如 `languageModelingFineWeb`，训练 GPT-2 风格模型做下一个 token 预测 | 官方要求保存 `model.pt` 与 `model_config.pt`，天然有落盘边界。单次训练超时 2400 秒，不等于必须跨命令保留模型 | 本地 4 个 RL、3 个视觉、2 个 NLP 任务均有闭环，CPU 博弈题不计入 GPU 范围 |
| [RE-Bench](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/ai_rd_nanogpt_chat_rl.py) | 例如用强化学习把 GPT-2 XL 训练成聊天模型，按外部模型偏好训练 | GPU 生成、外部裁判等待、GPU 更新交替。官方起始代码在等待时保留 CUDA 模型 | 已有原生接入、本地 Qwen 反馈观测及有界评分闭环，官方全量偏好评测未复现 |
| [PostTrainBench](https://github.com/aisa-group/PostTrainBench/blob/main/src/harbor_adapter/README.md) | 对基础语言模型后训练，提高数学、代码、写作或健康问答等目标能力 | 官方单 H100、默认 10 小时，训练后交付最终模型。完整训练或评估可作为 command | 五个目标配三个公开模型的 15 个本地组合均有闭环，整库 24 组合未完成 |

**PostTrainBench 提供的工作形态。** 后训练是在已有基础模型上继续优化参数。当前本地协议使用真实基础权重和真实数据，仅训练最终归一化层，提交 safetensors 权重，再由独立 verifier 重载基础模型。数学和代码目标评分隐藏答案似然，写作和健康目标实际生成回答、交给冻结本地裁判评分。Qwen3-1.7B × GSM8K 包含在通过的 15 个组合中。

固定上游的范围为 4 个模型、6 个目标，共 24 个组合。现有 15 个组合对应三个公开模型和 AIME2025、GSM8K、HumanEval、ArenaHard Writing、HealthBench。其余 9 个组合涉及 Gemma-3 与 GPQA 的访问权限。单卡 H20 上的已选协议已有实际闭环，逐组合数据入口、边界与证据见[任务协议](../milestones/02-Science现状_任务覆盖与接入路线.md#posttrainbench-与-fastkernels-任务协议)。

**RE-Bench 的驻留证据更具体。** 下面是已查阅的上游起始代码流程，本地接入复用该训练程序。它描述正式裁判链的状态生命周期，不是无裁判 smoke 已测得的等待曲线：

```text
[R1] GPT-2 XL 放到 CUDA，建立 Adam 优化器
    ↓
[R2] GPU 为每道提示生成两份回答
    ↓
[R3] await 外部裁判比较回答，模型仍由当前进程持有
    ↓
[R4] 根据偏好选回答，GPU 反向传播并更新参数
    └→ 下一轮 R2，最终保存模型
```

图例：此图是外部实现，未复制进本地仓库，因此给官方源码位置。

| 编号 | 官方函数与状态 |
|---|---|
| R1 | [rl_finetune.main:49](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L49)，`model.to(device)` 和 Adam 实例在循环外创建 |
| R2 | [main:74](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L74)，模型生成两组候选回答 |
| R3 | [main:90](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L90)，等待 `evaluate` 返回偏好，没有先把模型移出 GPU |
| R4 | [main:119](https://github.com/METR/RE-Bench/blob/main/ai_rd_nanogpt_chat_rl/assets/solution/rl_finetune.py#L119)，计算 loss、backward、optimizer step，循环后保存模型 |

跟着图读源码：R1 建立跨迭代状态，R2 将提示变为文本，R3 通过异步 API 取得偏好，R4 用这些偏好更新同一模型，随后重复。这是外部异步等待，不是本地 GPU 在等待期间一定释放。Adam 优化器保存参数更新所需的历史统计，第一次更新后这些状态也会占资源。只保存模型权重不能完整恢复同一训练轨迹，还要考虑优化器、缩放器和随机数状态。以上源码若有不清楚的地方，可以随时追问。

**它仍能包成一条长 command。** 但这样 R3 的等待仍占预约。若想在每次 R3 把 GPU 真正借走，就要适配卸载和恢复，并满足 Env 的实际回收要求。仅在 Python 中 `await` 不会搬走模型。此任务为研究这种机制提供了自然切入点，不等于官方禁止 command，也不等于当前 Env 已自动实现状态迁移。无需 GPU 的相近生命周期演示见 [导览脚本](../env-service-verification/scripts/explain_resident_command_lifetime.py)，它只模拟客户端结束而服务仍在，不模拟 RL 算法或优化器恢复。

后续接入应沿本文全局图，把 task、agent、独立 verifier 与最终清理都纳入同一套 Env 资源管理。对照保持相同真实数据和工作量，分别记录常驻准入与状态卸载恢复的成本，不能以换 benchmark 本身作为 ON 必然更快的保证。

## 按场景选择新增 benchmark

**当前优先增加完整训练、多卡协作和有依赖的实验反馈。** 下表把[路线图中的场景缺口](../milestones/02-Science现状_任务覆盖与接入路线.md#多样性覆盖与缺口)对应到具体项目。它们处于调研与适配方案阶段，本次没有产生新的本地 GPU 通过记录。

| 场景与推荐项目 | 官方已有内容 | 接入 PSRL 时 agent 做什么 | 接入代价与优先级 |
|---|---|---|---|
| 完整训练、多卡与候选反馈：[Automated LLM Speedrunning](https://github.com/facebookresearch/llm-speedrunner) | GPT 训练脚本、改进提示、训练指标和多候选搜索执行器 | 修改训练实现，运行实验、读取损失与耗时，再基于结果选择后续代码版本，提交脚本供独立重跑 | **优先新增，中高难度**。重点在多卡资源组、不同记录的依赖和可信评分，详见下节 |
| 图像生成与独立视觉评分：[GenEval 2](https://github.com/facebookresearch/GenEval2) | 生成提示、分解问题与视觉问答评分。官方评分代码加载 Qwen3-VL-8B-Instruct | 调整固定图像生成程序的合法配置或实现，提交图片与配置，独立视觉模型检查对象、数量和关系 | **应用异质性优先时选择，中等难度**。需要生成模型和视觉评分模型。它本身是生成质量 benchmark，agent 操作与训练接线需新增 |
| GPU 物理仿真与可选视觉观测：[ManiSkill 3](https://github.com/mani-skill/ManiSkill) | 机器人操作任务、GPU 并行物理、成功反馈，以及仿真与渲染性能测试 | 修改任务内控制或训练程序，提交控制器或策略权重，在独立随机种子下重放任务并评分 | **仿真异质性优先时选择，中高难度**。先使用状态观测和真实 GPU 物理，再单独验收渲染。原项目是机器人环境，需新增研究 agent 的任务封装 |
| 持续到达、排队与服务压力：[MLPerf Inference / LoadGen](https://github.com/mlcommons/inference/tree/master/loadgen) | 负载生成、请求完成日志和场景设置，可对接具体推理系统 | 优化受限制的服务配置，由受保护负载发生器驱动真实请求，分别评价输出质量、吞吐和响应时间 | **作为现有 Inference 任务的扩展，中等难度**。LoadGen 是测量组件，模型、数据和质量评分另行提供 |
| 显存释放后恢复、故障后继续 | 上述项目可提供真实计算，恢复协议需要按任务定义 | 保存模型与优化器等必要状态、退出 GPU 进程，再恢复同一任务，并核对产物与更新位置 | **直接扩展已有训练任务**。Speedrunner 默认关闭检查点，需要显式增加恢复档 |

GenEval 2 的[评分实现](https://github.com/facebookresearch/GenEval2/blob/main/evaluation.py)独立加载视觉语言模型，对图片中的事实提问并计算答案概率。原始 [GenEval](https://github.com/djghosh13/geneval)也可作为备选，其对象检测评分依赖 `mmdetection` 2.x 与 Mask2Former，需要另一个固定环境。

ManiSkill 的[入门接口](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)提供 `PickCube-v1` 等任务，增加并行环境数可以启用 GPU 仿真。[安装说明](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)区分状态仿真和渲染依赖，渲染还要求 Vulkan 图形驱动。当前节点的 CUDA 能用，只能证明计算路径的前提，渲染能力仍需单独预检。

LoadGen 的[职责说明](https://github.com/mlcommons/inference/blob/master/loadgen/README.md)允许用于自定义短测试，但质量评分由调用方提供。若希望一份资产同时覆盖服务压力和图像生成，也可选 [MLPerf SDXL 参考任务](https://github.com/mlcommons/inference/tree/master/text_to_image)，它已提供图像生成、质量评估与多种请求场景。采用短协议时明确记录本地规模。

这些候选补充任务形态。GPU 生命周期的研究价值还要通过实际轨迹判断，例如模型是否跨轮常驻、是否发生真实等待、分支结果是否改变后续命令。新增一个项目名称本身不会自动提供这些条件。

## Automated LLM Speedrunning 接入评估

### 任务内容、评分与完整例子

**可以补充，而且比继续增加近似算子题更能丰富当前训练负载。** 它把 NanoGPT 的连续训练加速纪录转成研究任务：给 agent 前一条纪录的 `train_gpt2.py`，可提供下一项改进的提示，让 agent 修改程序，实际训练到规定质量，再比较耗时。官方使用 FineWeb 验证损失不高于 **3.28** 的质量门槛，运行条件为单节点 **8 张 H100**。依据：[论文任务定义](https://arxiv.org/html/2506.22419#S3)、[官方任务配置](https://github.com/facebookresearch/llm-speedrunner/blob/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/config/task/nanogpt_speedrun/default_config.yaml)。

以一项训练加速题为例，执行关系如下。PSRL 的策略模型负责编辑与决策，任务里的 GPT 模型负责实际训练。

```text
前一条训练纪录的源码 + 改进提示
  → agent 产生候选训练脚本
  → 获得任务所需 GPU，执行 torchrun 分布式训练
  → 返回训练质量、耗时和错误信息
  → agent 选择继续改进哪个候选，或先修复失败版本
  → 提交最终脚本
  → 独立环境重跑并核验质量与耗时
  → PSRL 收集交互和奖励，更新策略，再用新策略 rollout
```

上图是建议的 PSRL 接线。官方实现已有训练与候选搜索，独立评分边界和外层策略更新需要接入。

用**教学数字**解释其评分：旧纪录用 100 秒达到目标质量，新的人类纪录用 60 秒，agent 的候选用 80 秒。论文的 FSR，即“恢复了多少比例的已知加速”，是 `(100-80)/(100-60)=0.5`，表示恢复了已知时间节省的一半。候选首先要达到相同质量门槛。硬件、数据或本地质量门槛变化后，应重新测两个参考耗时，不能直接套用论文时间。

### 题目数量与文件入口

论文定义 **19 个纪录迁移任务**，提示可以是无提示、伪代码、文字说明或简短论文。截图中的 **76** 可以对应 `19×4` 种任务与提示配置，这是依据任务定义的推断。提示组合还能形成其他实验条件，计数时应分别保存基础任务数与提示变体数。

本次对固定上游文件树清点得到 **21 份训练模板、20 份任务启动配置**。这些文件提供纪录起点和后继参考，论文还排除了纯升级 PyTorch 带来的纪录变化。准备题库时需要先建立起点、目标、提示和排除理由的映射，再按 19 个基础任务验收。

| 官方目录或文件 | 内容与接入用途 |
|---|---|
| [workspace_templates/nanogpt_speedrun](https://github.com/facebookresearch/llm-speedrunner/tree/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/workspace_templates/nanogpt_speedrun) | `record_1` 至 `record_21` 的训练程序，提供初始代码与参考实现 |
| [config/task/nanogpt_speedrun](https://github.com/facebookresearch/llm-speedrunner/tree/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/config/task/nanogpt_speedrun) | `speedrun_record_1.yaml` 至 `speedrun_record_20.yaml`，以及公共配置，描述入口与资源 |
| [data/nanogpt_speedrun_knowledge_in_levels](https://github.com/facebookresearch/llm-speedrunner/tree/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/data/nanogpt_speedrun_knowledge_in_levels) | 各纪录的改进提示，准备时按基础任务与提示级别关联 |
| [conda_envs](https://github.com/facebookresearch/llm-speedrunner/tree/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/conda_envs) | 不同纪录组的依赖。README 将记录分为 1 至 11、12 至 18、19 至 21，最后一组还提到打包的 nightly 环境 |
| [core/runners/bon_science_runner.py](https://github.com/facebookresearch/llm-speedrunner/blob/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/core/runners/bon_science_runner.py) | 生成候选、提交实验、收集结果与选择后续版本。作为交互行为参考，GPU 执行改走 PSRL 的受管入口 |

### 能补充什么、还需要什么

| 能补充的行为 | 源码依据与接入边界 |
|---|---|
| 完整模型训练 | 起始程序从随机初始化训练 GPT，包含前向、反向和优化器更新。相较当前只更新归一化层的 PostTrain 快检，它更适合检查训练状态和激活的资源成本 |
| 多卡协作 | 官方配置通过 8 进程 `torchrun` 启动，训练代码使用 NCCL，即 NVIDIA GPU 之间的集合通信库。后期纪录还包含异步集合通信，需保留真实跨卡工作 |
| 根据实验反馈继续搜索 | 搜索执行器会建立多个代码版本，收集结果后选择下一轮起点，适合提供有依赖的实验轨迹 |
| 编译与执行阶段差异 | 训练模板使用 `torch.compile`，可以分别记录环境准备、编译预热和有效训练的耗时 |

这里仍有两个需要单独补的场景。第一，当前[搜索执行器](https://github.com/facebookresearch/llm-speedrunner/blob/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/core/runners/bon_science_runner.py#L255)在一轮提交后等待该轮实验完成，之后选择下一轮版本。要研究某候选先返回后立即触发下一项或取消其他候选，需要增加异步反馈协议。第二，[任务配置](https://github.com/facebookresearch/llm-speedrunner/blob/ec3e4afd6dcf7c99f7b0be35898e40df6e1257c9/config/task/nanogpt_speedrun/default_config.yaml#L21)明确要求 `save_checkpoint=False`，释放显存后的训练恢复要另加保存和重载协议。

整体接入难度评估为**中高**。已有训练程序、提示和任务配置降低了内容准备成本，主要工作落在多卡资源、依赖固化、数据与可信评分上。后期记录的打包环境还需确认可获取入口，或重建并逐题验证。官方示例的 API key 用于其内置研究 agent，采用 PSRL 策略可以替换这一层。GPT 任务本身从头训练，资产准备重点是 FineWeb 数据分片与对应软件依赖。

### PSRL 接入与快速验收方案

以下是待实施方案，当前仓库尚无 Speedrunner 适配器。

1. **冻结全任务清单。** 为 19 个基础任务建立起点与目标映射，登记提示级别、依赖组、硬件要求和本地变更。逐基础任务检查 GPU 执行，提示变体另检查指令与文件映射。
2. **提供保留多卡行为的本地档。** 可以先设计双卡、小批量和短步骤协议，逐题核验批量整除、通信与模型约束。单卡档适合调试接线，双卡实际参与训练和通信后才计入多卡覆盖。质量门槛与参考成绩按本地条件重新固定。
3. **复用已有受管执行与隔离。** 任务环境使用专用镜像或依赖副本，FineWeb 分片只读挂载。用 PSRL 资源租约替代官方 Slurm 作业提交，覆盖独立评分和整组资源回收。当前[公共快检入口](../../../examples/gpu_benchmarks/run.py#L70)限制每环境一张 GPU，需要同步扩展资源声明、执行和验收入口。
4. **建立可信独立评分。** 在干净环境重跑提交，由受保护评测代码核验数据、质量和耗时。agent 打印的损失仅作开发反馈。若为独立评估或恢复新增权重交付，记录为本地协议变更。编译初始化、训练计时和整个任务等待分别保存。
5. **有界完成闭环。** 预先固定步骤、任务期限和重试次数，完成真实 GPU 快检、独立评分、PSRL 优化器更新、版本发布及更新后 rollout。功能快检使用的短步骤只验执行与接线，正式训练加速研究另按已固定的质量门槛比较。

“完整接入”需要覆盖固定任务清单，并让其执行、评分与策略训练均可复现。先跑通一个纪录可以验证适配方向，整库状态按逐任务证据填写。关于同机动态换卡是否值得优先做，仍按[动态放置评估](../research-proposals/03-动态放置_会话与GPU执行分离.md#9-已接入-benchmark-的必要性评估)判断，任务接入优先级与调度机制原型优先级分别安排。

## 已接入任务的研究用法

已有入口和运行条件见 [GPU benchmark recipe](../knowledge/03-任务接入/01-任务接入_公共GPU任务与PSRL配方.md)，完成范围以[当前路线图](../milestones/02-Science现状_任务覆盖与接入路线.md)为准。研究选题应从已通过的协议中取代表任务，避免重复处理依赖和数据错误。

| 研究顺序 | 已有目录 | 当前可用选题 | 适合检验的机制 |
|---|---|---|---|
| 1 | `examples/kernel_bench/` | 270 题身份中的矩阵运算代表题 | 短 GPU 工作、编译成本、计时争用与反馈可靠性 |
| 2 | `examples/mlgym_bench/`、`examples/posttrain_bench/` | MLGym 9 题，PostTrain 已闭环的 15 个组合 | 训练与 CPU 阶段交替、权重交付、独立重载的成本 |
| 3 | `examples/inference_bench/`、`examples/swe_serve/` | 4 个本地服务场景与 SWE score 本地闭环任务 | 常驻服务状态、重启、自测和物理回收 |
| 4 | `examples/autoresearch/`、`examples/re_bench/` | 固定训练起点与有界偏好任务 | 多次实验、外部反馈等待和有代价的状态恢复 |

MLE 官方 Kaggle 题等待凭据和逐竞赛准备，不作为当前可直接执行的研究输入。任务自己的目标服务与 agent 使用的策略模型服务继续分别核对端口、权重和 GPU 归属。对照必须保持真实数据、工作量和评分边界一致。

当前适配器负责任务准备、提交和评分，资源申请、容器生命周期及 GPU 清理复用 [GpuBenchmarkAgentLoop](../../../examples/gpu_benchmarks/agent_loop.py#L14)与公共 science runner。不要为每个 benchmark 复制一套调度器。AIRS 使用另一条原生 Coordinator/sandbox 分支，后端统一混训尚未验证。

## 具体例子与评分干扰

<!-- chapter-figure:chapter-measurement-ranking -->
评分依赖时间时，要先排除“调度把较快代码评成较慢”的情况。

[![评分依赖时间时，要先排除“调度把较快代码评成较慢”的情况](../../_static/img/document-figures/chapter-measurement-ranking-screen.png)](../../_static/img/document-figures/chapter-measurement-ranking.png)

图中数字是毫秒教学反例，说明反馈条件的重要性。准确率、正确性与固定训练步数任务则按各自协议验收，不自动需要同样的测速保护。 [放大查看 SVG](../../_static/img/document-figures/chapter-measurement-ranking.svg)。
<!-- /chapter-figure:chapter-measurement-ranking -->


**教学假设，不是测量结果**：一个 KernelBench episode 花 40 秒读题和生成代码、10 秒编译、2 秒 GPU 运行与计时，三个阶段共 52 秒。若整个 episode 独占一张 GPU，预留为 52 GPU 秒，其中只有 2 秒执行该任务的 GPU 计算。按评测阶段申请理论上可缩短预留，但还要计算初始化、数据传输、切换和等待成本，不能据此直接宣称 26 倍吞吐收益。

输入是一段 PyTorch 矩阵计算和固定输入规范。agent 生成 CUDA 或 Triton 程序，编译器生成可执行代码，evaluator 在 GPU 上运行参考与候选。只有输出符合容差才比较速度，最终再销毁评测进程并核实释放。GPU kernel 是由大量 GPU 线程协作执行的计算函数，其速度来自并行计算与访存组织，不是把 Python 文件改个后缀。

假设参考为 100 微秒，候选为 80 微秒，加速比是 100/80=1.25。若另一个任务竞争显存带宽，使候选测到 120 微秒，则测出的 0.83 会把正确优化误判为退步。因此正式性能计时应使用受控独占窗口，同样约束参考与候选。开发阶段是否允许共享可以另外评估。

对比 MLE：agent 可能先用 CPU 处理数据，再训练 10 分钟，然后用 GPU 预测，最终提交 CSV。GPU 分配应覆盖实际训练和预测进程，不能在 shell 返回但后台训练仍活着时归还。对比 InferenceBench：服务在请求之间利用率可能很低，但模型仍占显存，不能将请求间空隙直接解释为可借出的整卡。

## 本地实现与源码阅读路径

本次核对的是带有未提交修改的主开发目录，而不是一个干净的已发布版本。

- science 的 [runner.py](../../../examples/sciaccel_rl/runner.py#L275) 已有 `needs_gpu`、资源租约和 GPU 作业报告路径。其 [README](../../../examples/sciaccel_rl/README.md) 也提示 ScienceInfra 为后续开发入口。任务是否实际执行 GPU 仍需逐题证明。
- AIRS 的 [run_mlgym_agent](../../../examples/airs_bench/runner.py#L137) 复用 MLGym。原 CPU recipe 保持零 GPU，新增 [GPU 配置](../../../examples/airs_bench/config/gpu_agent_config.yaml) 与 [训练入口](../../../examples/airs_bench/gpu_train.sh) 已覆盖 19 个非 Kaggle 任务，并有 SICK 代表题的策略闭环。实际协议与证据见 [AIRS GPU 接入](../knowledge/03-任务接入/03-AIRSBench_接入流程与有界验收.md)。这是 AIRS 分支的证据。MLGym 的 4 个 RL、3 个视觉和 2 个 NLP 任务已有独立 GPU 快检与策略闭环，实际协议及未完成范围见[当前接入状态](../milestones/02-Science现状_任务覆盖与接入路线.md#目前有哪些任务及可用边界)，不等于整个 MLGym 全部任务已完成。
- 这篇文档补充的是候选选择，公共调度机制继续以 [当前设计](../knowledge/01-架构与执行/01-Env架构_现状与完整执行流程.md) 和 [统一 GPU 执行后端](../knowledge/01-架构与执行/02-统一执行_EnvRouter与容器及预算策略.md) 为入口，避免重复维护架构说明。

跟着图读源码，按真实 science 路径查看：

| 步骤 → 图中编号 | 文件、函数 | 输入输出或状态 |
|---|---|---|
| 申请 → 1、2 | [runner.py:372](../../../examples/sciaccel_rl/runner.py#L372)，`run_harbor_episode` | 任务路径和需求转换为租约，可能异步等待 |
| 维持租约 → 2 | [client.py:50](../../../psrl/workers/env_service/client.py#L50)，`start_heartbeat` | 后台心跳与任务并发，不是任务执行完才调用 |
| 执行 → 3、4 | [runner.py:550](../../../examples/sciaccel_rl/runner.py#L550)，`Job.create`、`job.run` 调用点 | Harbor 执行任务并返回结果，内部 agent/verifier 顺序由 Harbor 管理 |
| 评分处理 → 4 | [runner.py:589](../../../examples/sciaccel_rl/runner.py#L589)，`run_harbor_episode` | command 模式读取 GPU 报告，LAPS 条件分支检查活动证据 |
| 清理 → 5 | [runner.py:610](../../../examples/sciaccel_rl/runner.py#L610)，`run_harbor_episode` 的 finally | 正常、异常、取消均进入收尾，再调用 lease release |

## 可运行的教学模拟

下面是自包含 CPU 脚本，在仓库根目录执行即可。只展示“episode 预留”和“实际计算阶段”的差异，不调用 PSRL、Docker 或 GPU，也不模拟抢占、显存分配和排队收益。

```bash
python - <<'PY'
# 对应图 1：人为定义三个阶段，单位为秒。
stages = [("edit", 40, False), ("compile", 10, False), ("evaluate", 2, True)]
# 对应图 2、3、4：比较整个任务预留和仅计算阶段预留。
episode_reserved = sum(seconds for _, seconds, _ in stages)
phase_reserved = sum(seconds for _, seconds, gpu in stages if gpu)
for name, seconds, gpu in stages:
    print(f"{name}: {seconds}s, GPU={gpu}")
print(f"episode_reserved={episode_reserved} GPU-s")
print(f"phase_reserved={phase_reserved} GPU-s")
# 对应图 5：模型假设进程退出后释放，真实系统必须额外核验。
print("simulation_end: assumed resources released")
PY
```

预期分别输出 52 和 2 GPU 秒。差值代表本例可避免的预留空窗，不是已经证实的加速。真实容器可能在编译或初始化时建立 GPU 上下文，真实训练和服务也可能跨命令保留状态，必须按实际进程与遥测修正阶段边界。

本次已实际运行此 CPU 模拟，输出为 52 和 2 GPU 秒。本篇及两个更新索引的本地链接检查通过，新增目录锚点和源码行号范围检查通过。全仓 `pre-commit run --all-files` 在临时副本执行，因既有 Python 行长、格式和注释规范问题失败，自动格式化仅发生在副本，没有回写主开发目录。

## 验收与未验证事项

接入验收分开记录：任务包准备、固定动作或已知解通过、真实 CUDA 执行、agent 完整 episode、资源物理释放、正式 benchmark 协议评测。前一项通过不能替代后一项。

建议首批每类只选一题、最多两个尝试。KernelBench 的集成检查可先给 10 分钟总期限，MLE 的小样本路径给 30 分钟，InferenceBench 的服务启动和固定请求检查给 20 分钟，autoresearch 两轮给 20 分钟含启动评估。这些是拟议工程检查预算，需要结合实际任务可行性核定，不是官方评测预算。到期证据不足即记录未验证，不自动扩容重跑。

资源紧张时首先选择满足要求的独立 GPU。若研究共享机制，单列共享实验与正式分数实验。多 GPU 任务需确认同时分配的设备集合、互联要求以及失败后整个进程组的清理，不能用单卡通过代替。

路线图已标注的通过范围有本地 GPU 和策略证据，不能扩展为逐题完整许可审计、峰值显存上界、漏洞审计或全部硬件兼容保证。未接入的候选仍按公开入口和调研结论阅读，InfraDev 只核验到论文。执行任何真实实验前按 [集群检查指南](../../psrl/psrl-run-guide/02-集群预检_依赖地址与GPU资源核验.md) 做现场预检。

## 非计时评分场景的当前优先级

非计时研究按[场景与策略](../research-proposals/02-研究场景_非计时评分的GPU任务.md)进入，测速反馈按[AI4Infra 选择](06-基准调研_AI4Infra性能反馈与验证.md)进入。固定 SWE 清单包含 41 个 GPU 题，其中 3 个按 speedup 评分、38 个按正确性评分，适用问题不同。后续 score 服务题已完成原始评分与相同工作量准备前移对照；fused-kernels 的六项原始正确性测试完成 11 组 H20 运行，观察到跨命令同机换卡仍可利用编译缓存。这些分别支持任务功能、准备取舍和缓存机制，完整接入范围统一见[当前路线图](../milestones/02-Science现状_任务覆盖与接入路线.md)。
